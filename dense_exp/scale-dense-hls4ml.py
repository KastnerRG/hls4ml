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


def build_cascade_model(input_size, output_size, num_layers):
    model = Sequential()
    model.add(
        QActivation(quantized_bits(BITS, INT), name="input_quant", input_shape=(input_size,)),
    )
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


def normalize_strategy(strategy):
    value = strategy.strip().lower()
    if value == 'resource':
        return 'Resource', 'R'
    if value == 'latency':
        return 'Latency', 'L'
    raise ValueError(f"Unsupported strategy '{strategy}'. Use 'Resource' or 'Latency'.")


def run_scale_dense(IN_SIZE, OUT_SIZE, REUSE_FACTOR, NUM_LAYERS, PART, CLK_PERIOD, STRATEGY='Resource', VSYNTH=True, COSIM=True):
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)

    strategy_name, strategy_tag = normalize_strategy(STRATEGY)
    clock_tag = format_clock_tag(CLK_PERIOD)
    proj_name = shell_safe_name(
        f'dense_{strategy_tag}_in{IN_SIZE}_out{OUT_SIZE}_l{NUM_LAYERS}_rf{REUSE_FACTOR}_clk{clock_tag}'
    )
    output_dir = BASE_DIR / f'scale_hls4ml_{strategy_tag}_prj' / proj_name
    result_dir = BASE_DIR / f'scale_hls4ml_{strategy_tag}_result' / proj_name

    model = build_cascade_model(IN_SIZE, OUT_SIZE, NUM_LAYERS)

    config = hls4ml.utils.config_from_keras_model(model, granularity='model', backend='Vitis')
    config['Model']['ReuseFactor'] = REUSE_FACTOR
    config['Model']['Strategy'] = strategy_name
    config['Model']['Precision'] = f'ap_fixed<{BITS},{INT + 1}>'
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
    parser.add_argument('--strategy', default='Resource')
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
        STRATEGY=args.strategy,
        VSYNTH=args.vsynth,
        COSIM=args.cosim,
    )
