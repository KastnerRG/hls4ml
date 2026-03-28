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
BITS = 8
INT = 0


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
from tensorflow.keras.models import Sequential


def write_text(path, content):
    path.write_text(content, encoding='utf-8')


def copy_if_exists(src, dst):
    if src.exists():
        shutil.copy(src, dst)


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


def prepare_testbench_data(model, proj_name, input_size):
    tb_dir = BASE_DIR / 'tb_data'
    tb_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(4343)
    tb_input = rng.uniform(-1.0, 1.0, size=(TB_SAMPLES, input_size)).astype(np.float32)
    tb_output = model.predict(tb_input, verbose=0)

    input_path = tb_dir / f'{proj_name}_input.npy'
    output_path = tb_dir / f'{proj_name}_output.npy'
    np.save(input_path, tb_input)
    np.save(output_path, tb_output.astype(np.float32))
    return input_path, output_path


def patch_generated_dense_project(output_dir, reuse_factor):
    mult_path = output_dir / 'firmware/nnet_utils/nnet_mult.h'
    params_path = output_dir / 'firmware/parameters.h'
    dense_resource_path = output_dir / 'firmware/nnet_utils/nnet_dense_resource.h'

    mult_text = mult_path.read_text(encoding='utf-8')
    if 'class mult_dsp : public Product' not in mult_text:
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

template <class x_T, class w_T> class mult_dsp : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        #pragma HLS INLINE
        auto prod = a * w;
        #pragma HLS BIND_OP variable=prod op=mul impl=DSP
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
    new = 'using product = nnet::product::mult_dsp<x_T, y_T>;'
    if old not in params_text:
        raise RuntimeError(f'Could not find default product binding in {params_path}')
    params_text = params_text.replace(old, new)
    write_text(params_path, params_text)

    dense_text = dense_resource_path.read_text(encoding='utf-8')
    old = '#pragma HLS RESOURCE variable=weights core=ROM_nP_BRAM'
    new = '#pragma HLS BIND_STORAGE variable=weights type=ROM_NP impl=BRAM'
    #if old in dense_text:
    #    dense_text = dense_text.replace(old, new)

    old_loop = """ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind

        int w_index = ir;
        int in_index = ir;
        int out_index = 0;
        int acc_step = 0;

    MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL

            acc[out_index] += static_cast<typename CONFIG_T::accum_t>(
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]));

            // Increment w_index
            w_index += rufactor;
            // Increment in_index
            in_index += rufactor;
            if (in_index >= nin) {
                in_index = ir;
            }
            // Increment out_index
            if (acc_step + 1 >= multscale) {
                acc_step = 0;
                out_index++;
            } else {
                acc_step++;
            }
        }
    }
"""
    new_loop = """ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind
    OutputLoop:
        for (int out_index = 0; out_index < nout; out_index++) {
            #pragma HLS UNROLL
        MultLoop:
            for (int im = 0; im < multscale; im++) {
                #pragma HLS UNROLL
                const int in_index = ir + im * rufactor;
                const int w_index = out_index * nin + in_index;

                acc[out_index] += static_cast<typename CONFIG_T::accum_t>(
                    CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]));
            }
        }
    }
"""
    if reuse_factor == 2 and old_loop in dense_text:
        dense_text = dense_text.replace(old_loop, new_loop)

    write_text(dense_resource_path, dense_text)


def build_cascade_model(input_size, output_size, num_layers):
    model = Sequential()

    current_input = input_size
    for i in range(num_layers):
        dense_kwargs = {}
        if i == 0:
            dense_kwargs['input_shape'] = (current_input,)

        model.add(
            QDense(
                output_size,
                name=f'fc{i}',
                kernel_quantizer=quantized_bits(BITS, INT, alpha=1),
                bias_quantizer=quantized_bits(2 * BITS, 2 * INT, alpha=1),
                kernel_initializer=RandomUniform(minval=-1, maxval=1, seed=np.random.randint(0, 100)),
                bias_initializer=RandomUniform(minval=-1, maxval=1, seed=np.random.randint(0, 100)),
                **dense_kwargs,
            )
        )
        model.add(QActivation(quantized_relu(BITS, INT), name=f'relu{i}'))
        current_input = output_size

    return model


def run_scale_dense(IN_SIZE, OUT_SIZE, REUSE_FACTOR, NUM_LAYERS, PART, CLK_PERIOD, VSYNTH=True, COSIM=True):
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)

    clock_tag = format_clock_tag(CLK_PERIOD)
    proj_name = shell_safe_name(
        f'scale_db_in{IN_SIZE}_out{OUT_SIZE}_l{NUM_LAYERS}_rf{REUSE_FACTOR}_clk{clock_tag}'
    )
    output_dir = BASE_DIR / 'scale_db_prj' / proj_name
    result_dir = BASE_DIR / 'scale_db_result' / proj_name

    model = build_cascade_model(IN_SIZE, OUT_SIZE, NUM_LAYERS)

    config = hls4ml.utils.config_from_keras_model(model, granularity='model', backend='Vitis')
    config['Model']['ReuseFactor'] = REUSE_FACTOR
    config['Model']['Precision'] = f'ap_fixed<{BITS},{INT + 1}>'
    config['Model']['Strategy'] = 'Resource'
    print('-----------------------------------')
    pprint.pprint(config, sort_dicts=False)
    print('-----------------------------------')

    tb_input_path = None
    tb_output_path = None
    if COSIM:
        tb_input_path, tb_output_path = prepare_testbench_data(model, proj_name, IN_SIZE)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        backend='Vitis',
        output_dir=str(output_dir),
        part=PART,
        clock_period=CLK_PERIOD,
        io_type='io_stream',
        input_data_tb=str(tb_input_path) if tb_input_path is not None else None,
        output_data_tb=str(tb_output_path) if tb_output_path is not None else None,
    )
    hls_model.write()
    patch_generated_dense_project(output_dir, REUSE_FACTOR)
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
    parser.add_argument('--in-size', type=int, default=96)
    parser.add_argument('--out-size', type=int, default=96)
    parser.add_argument('--reuse-factor', type=int, default=1)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--part', default='xcve2802-vsvh1760-2MP-e-S')
    parser.add_argument('--clock-period', type=float, default=5.0)
    parser.add_argument('--cosim', dest='cosim', action='store_true', default=True)
    parser.add_argument('--no-cosim', dest='cosim', action='store_false')
    parser.add_argument('--vsynth', dest='vsynth', action='store_true', default=True)
    parser.add_argument('--no-vsynth', dest='vsynth', action='store_false')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    _, _ = time_block(
        run_scale_dense,
        IN_SIZE=args.in_size,
        OUT_SIZE=args.out_size,
        REUSE_FACTOR=args.reuse_factor,
        NUM_LAYERS=args.layers,
        PART=args.part,
        CLK_PERIOD=args.clock_period,
        VSYNTH=args.vsynth,
        COSIM=args.cosim,
    )
