import argparse
import os
import pprint
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

VITIS_SETTINGS = Path('/tools/Xilinx/Vivado/2025.2/Vitis/settings64.sh')
BASE_DIR = Path(__file__).resolve().parent
TB_SAMPLES = 8
ACT_BITS = 8
ACT_INT = 0
WEIGHT_BITS = 8
WEIGHT_INT = 0
BIAS_BITS = 32
BIAS_INT = 0
BIT_EXACT = False


def _prepend_env_path(var_name, value):
    current = os.environ.get(var_name, '')
    if current:
        os.environ[var_name] = f'{value}:{current}'
    else:
        os.environ[var_name] = value


def load_vitis_environment():
    if not VITIS_SETTINGS.is_file():
        raise FileNotFoundError(f'Cannot find Vitis settings script: {VITIS_SETTINGS}')

    proc = subprocess.run(
        ['bash', '-lc', f'source "{VITIS_SETTINGS}" >/dev/null 2>&1 && env -0'],
        capture_output=True,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(f'Failed to source Vitis settings: {VITIS_SETTINGS}\n{stderr}')

    for entry in proc.stdout.split(b'\x00'):
        if not entry or b'=' not in entry:
            continue
        key, value = entry.split(b'=', 1)
        os.environ[key.decode('utf-8', errors='ignore')] = value.decode('utf-8', errors='ignore')

    env_prefix = Path(sys.prefix)
    _prepend_env_path('PATH', str(env_prefix / 'bin'))
    _prepend_env_path('LD_LIBRARY_PATH', str(env_prefix / 'lib'))

    check = subprocess.run(
        ['bash', '-lc', 'command -v vitis-run && vitis-run --version | head -n 1'],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        raise RuntimeError(f'vitis-run is not available after sourcing {VITIS_SETTINGS}')

    print(check.stdout.strip())


def shell_safe_name(value):
    return ''.join(ch if ch.isalnum() else '_' for ch in value)


def format_clock_tag(clock_period):
    if float(clock_period).is_integer():
        return str(int(clock_period))
    return str(clock_period).replace('.', 'p')


load_vitis_environment()

import hls4ml
import numpy as np
import tensorflow as tf
from qkeras.qlayers import QActivation, QDense
from qkeras.quantizers import quantized_bits, quantized_relu
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential


def copy_if_exists(src, dst):
    if src.exists():
        shutil.copy(src, dst)


def write_text(path, text):
    path.write_text(text, encoding='utf-8')


def copy_result_artifacts(output_dir, result_dir, proj_name, vsynth, cosim):
    result_dir.mkdir(parents=True, exist_ok=True)
    copy_if_exists(
        output_dir / 'myproject_prj/solution1/syn/report/myproject_csynth.rpt',
        result_dir / f'{proj_name}_csynth.rpt',
    )
    if cosim:
        copy_if_exists(
            output_dir / 'myproject_prj/solution1/sim/verilog/myproject.performance.result.transaction.xml',
            result_dir / f'{proj_name}_transaction.xml',
        )
        copy_if_exists(
            output_dir / 'myproject_prj/solution1/sim/verilog/myproject.result.lat.rb',
            result_dir / f'{proj_name}_latency.rb',
        )
    if vsynth:
        copy_if_exists(output_dir / 'vivado_synth.rpt', result_dir / f'{proj_name}_vsynth.rpt')


def patch_generated_jet_project(output_dir):
    mult_path = output_dir / 'firmware/nnet_utils/nnet_mult.h'
    params_path = output_dir / 'firmware/parameters.h'

    mult_text = mult_path.read_text(encoding='utf-8')
    if 'class mult_fabric : public Product' not in mult_text:
        old = """template <class x_T, class w_T> class mult : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        // 'Normal' product
        #pragma HLS INLINE
        return a * w;
    }
};
"""
        new = """template <class x_T, class w_T> class mult : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        // 'Normal' product
        #pragma HLS INLINE
        return a * w;
    }
};

template <class x_T, class w_T> class mult_fabric : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        #pragma HLS INLINE
        auto prod = a * w;
        #pragma HLS BIND_OP variable=prod op=mul impl=fabric
        return prod;
    }
};
"""
        if old not in mult_text:
            raise RuntimeError(f'Could not find mult product template in {mult_path}')
        mult_text = mult_text.replace(old, new, 1)
        write_text(mult_path, mult_text)

    params_text = params_path.read_text(encoding='utf-8')
    old = 'using product = nnet::product::mult<x_T, y_T>;'
    new = 'using product = nnet::product::mult_fabric<x_T, y_T>;'
    if old not in params_text:
        raise RuntimeError(f'Could not find default product binding in {params_path}')
    params_text = params_text.replace(old, new)
    write_text(params_path, params_text)


def prepare_testbench_data(model, proj_name):
    tb_dir = BASE_DIR / 'tb_data'
    tb_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2026)
    tb_input = rng.uniform(-1.0, 1.0, size=(TB_SAMPLES, 16)).astype(np.float32)
    tb_output = model.predict(tb_input, verbose=0)

    input_path = tb_dir / f'{proj_name}_input.npy'
    output_path = tb_dir / f'{proj_name}_output.npy'
    np.save(input_path, tb_input)
    np.save(output_path, tb_output.astype(np.float32))
    return input_path, output_path


def dense_block(model, in_features, out_features, index, with_batchnorm=True):
    dense_kwargs = {}
    if index == 0:
        dense_kwargs['input_shape'] = (in_features,)

    model.add(
        QDense(
            out_features,
            name=f'fc{index}',
            kernel_quantizer=quantized_bits(WEIGHT_BITS, WEIGHT_INT, alpha=1),
            bias_quantizer=quantized_bits(BIAS_BITS, BIAS_INT, alpha=1),
            kernel_initializer=RandomUniform(minval=-1, maxval=1, seed=np.random.randint(0, 100)),
            bias_initializer=RandomUniform(minval=-1, maxval=1, seed=np.random.randint(0, 100)),
            **dense_kwargs,
        )
    )
    if with_batchnorm:
        model.add(BatchNormalization(name=f'bn{index}'))
    model.add(QActivation(quantized_relu(ACT_BITS, ACT_INT), name=f'relu{index}'))


def build_jet_model_raw():
    model = Sequential(name='jet_raw')
    model.add(
        QActivation(
            quantized_bits(ACT_BITS, ACT_INT),
            name='input_quant',
            input_shape=(16,),
        )
    )

    dense_block(model, 16, 64, 0, with_batchnorm=True)
    dense_block(model, 64, 32, 1, with_batchnorm=True)
    dense_block(model, 32, 32, 2, with_batchnorm=True)
    dense_block(model, 32, 5, 3, with_batchnorm=False)

    return model


def build_jet_model_fused(raw_model):
    fused_model = Sequential(name='jet')
    fused_model.add(
        QActivation(
            quantized_bits(ACT_BITS, ACT_INT),
            name='input_quant',
            input_shape=(16,),
        )
    )
    fused_model.add(
        QDense(
            64,
            name='fc0',
            kernel_quantizer=quantized_bits(WEIGHT_BITS, WEIGHT_INT, alpha=1),
            bias_quantizer=quantized_bits(BIAS_BITS, BIAS_INT, alpha=1),
        )
    )
    fused_model.add(QActivation(quantized_relu(ACT_BITS, ACT_INT), name='relu0'))
    fused_model.add(
        QDense(
            32,
            name='fc1',
            kernel_quantizer=quantized_bits(WEIGHT_BITS, WEIGHT_INT, alpha=1),
            bias_quantizer=quantized_bits(BIAS_BITS, BIAS_INT, alpha=1),
        )
    )
    fused_model.add(QActivation(quantized_relu(ACT_BITS, ACT_INT), name='relu1'))
    fused_model.add(
        QDense(
            32,
            name='fc2',
            kernel_quantizer=quantized_bits(WEIGHT_BITS, WEIGHT_INT, alpha=1),
            bias_quantizer=quantized_bits(BIAS_BITS, BIAS_INT, alpha=1),
        )
    )
    fused_model.add(QActivation(quantized_relu(ACT_BITS, ACT_INT), name='relu2'))
    fused_model.add(
        QDense(
            5,
            name='fc3',
            kernel_quantizer=quantized_bits(WEIGHT_BITS, WEIGHT_INT, alpha=1),
            bias_quantizer=quantized_bits(BIAS_BITS, BIAS_INT, alpha=1),
        )
    )
    fused_model.add(QActivation(quantized_relu(ACT_BITS, ACT_INT), name='relu3'))

    dummy_input = np.zeros((1, 16), dtype=np.float32)
    raw_model.predict(dummy_input, verbose=0)
    fused_model.predict(dummy_input, verbose=0)

    for index in range(3):
        kernel, bias = raw_model.get_layer(f'fc{index}').get_weights()
        gamma, beta, moving_mean, moving_var = raw_model.get_layer(f'bn{index}').get_weights()
        epsilon = raw_model.get_layer(f'bn{index}').epsilon
        scale = gamma / np.sqrt(moving_var + epsilon)
        fused_kernel = kernel * scale
        fused_bias = beta + (bias - moving_mean) * scale
        fused_model.get_layer(f'fc{index}').set_weights([fused_kernel, fused_bias])

    fused_model.get_layer('fc3').set_weights(raw_model.get_layer('fc3').get_weights())

    probe = np.random.default_rng(123).uniform(-1.0, 1.0, size=(4, 16)).astype(np.float32)
    raw_out = raw_model.predict(probe, verbose=0)
    fused_out = fused_model.predict(probe, verbose=0)
    max_abs_diff = np.max(np.abs(raw_out - fused_out))
    if max_abs_diff > (1.0 / 128.0 + 1e-6):
        raise RuntimeError(f'BatchNorm folding changed the model outputs (max abs diff {max_abs_diff})')

    return fused_model


def build_jet_model():
    raw_model = build_jet_model_raw()
    return build_jet_model_fused(raw_model)


def run_jet_model(REUSE_FACTOR, PART, CLK_PERIOD, VSYNTH=True, COSIM=True):
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)

    clock_tag = format_clock_tag(CLK_PERIOD)
    proj_name = shell_safe_name(f'jet_rf{REUSE_FACTOR}_clk{clock_tag}')
    output_dir = BASE_DIR / 'nn_prj' / proj_name
    result_dir = BASE_DIR / 'nn_result' / proj_name

    model = build_jet_model()

    # Keep per-layer precision inference from the explicit quantizers, but do
    # not force model-wise bit-exact propagation here. With BatchNormalization
    # in this network, the bit_exact path generates pathological internal
    # precisions and breaks C/RTL co-sim on this flow.
    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Vitis')
    config['Model']['ReuseFactor'] = REUSE_FACTOR
    config['Model']['Strategy'] = 'Resource'

    print('-----------------------------------')
    pprint.pprint(config, sort_dicts=False)
    print('-----------------------------------')

    tb_input_path = None
    tb_output_path = None
    if COSIM:
        tb_input_path, tb_output_path = prepare_testbench_data(model, proj_name)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        backend='Vitis',
        bit_exact=BIT_EXACT,
        output_dir=str(output_dir),
        part=PART,
        clock_period=CLK_PERIOD,
        io_type='io_parallel',
        input_data_tb=str(tb_input_path) if tb_input_path is not None else None,
        output_data_tb=str(tb_output_path) if tb_output_path is not None else None,
    )
    hls_model.write()
    patch_generated_jet_project(output_dir)
    hls_model.build(csim=COSIM, synth=True, cosim=COSIM, validation=COSIM, vsynth=VSYNTH, log_to_stdout=False)

    copy_result_artifacts(output_dir, result_dir, proj_name, VSYNTH, COSIM)


def time_block(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f'{fn.__name__} took {dt:.2f}s')
    return out, dt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-factor', type=int, default=1)
    parser.add_argument('--part', default='xcve2802-vsvh1760-2MP-e-S')
    parser.add_argument('--clock-period', type=float, default=3.2)
    parser.add_argument('--cosim', dest='cosim', action='store_true', default=True)
    parser.add_argument('--no-cosim', dest='cosim', action='store_false')
    parser.add_argument('--vsynth', dest='vsynth', action='store_true', default=True)
    parser.add_argument('--no-vsynth', dest='vsynth', action='store_false')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    _, _ = time_block(
        run_jet_model,
        REUSE_FACTOR=args.reuse_factor,
        PART=args.part,
        CLK_PERIOD=args.clock_period,
        VSYNTH=args.vsynth,
        COSIM=args.cosim,
    )
