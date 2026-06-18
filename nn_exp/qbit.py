import argparse
import pprint
import shutil

import hls4ml
import numpy as np
import tensorflow as tf
from qkeras.qlayers import QActivation, QDense
from qkeras.quantizers import quantized_bits, quantized_relu
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.models import Sequential

from jet import (
    ACT_BITS,
    ACT_INT,
    BASE_DIR,
    BIAS_BITS,
    BIAS_INT,
    BIT_EXACT,
    TB_SAMPLES,
    WEIGHT_BITS,
    WEIGHT_INT,
    copy_result_artifacts,
    format_clock_tag,
    patch_generated_jet_project,
    shell_safe_name,
    time_block,
)


def prepare_testbench_data(model, proj_name):
    tb_dir = BASE_DIR / 'tb_data'
    tb_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2026)
    tb_input = rng.uniform(-1.0, 1.0, size=(TB_SAMPLES, 256)).astype(np.float32)
    tb_output = model.predict(tb_input, verbose=0)

    input_path = tb_dir / f'{proj_name}_input.npy'
    output_path = tb_dir / f'{proj_name}_output.npy'
    np.save(input_path, tb_input)
    np.save(output_path, tb_output.astype(np.float32))
    return input_path, output_path


def add_dense_relu(model, in_features, out_features, index):
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
    model.add(QActivation(quantized_relu(ACT_BITS, ACT_INT), name=f'relu{index}'))


def build_qbit_model():
    model = Sequential(name='qbit')
    model.add(
        QActivation(
            quantized_bits(ACT_BITS, ACT_INT),
            name='input_quant',
            input_shape=(256,),
        )
    )

    add_dense_relu(model, 256, 128, 0)
    add_dense_relu(model, 128, 128, 1)
    add_dense_relu(model, 128, 128, 2)
    add_dense_relu(model, 128, 128, 3)
    add_dense_relu(model, 128, 5, 4)

    return model


def run_qbit_model(REUSE_FACTOR, PART, CLK_PERIOD, VSYNTH=True, COSIM=True):
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)

    clock_tag = format_clock_tag(CLK_PERIOD)
    proj_name = shell_safe_name(f'qbit_rf{REUSE_FACTOR}_clk{clock_tag}')
    output_dir = BASE_DIR / 'nn_prj' / proj_name
    result_dir = BASE_DIR / 'nn_result' / proj_name

    model = build_qbit_model()

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
        # Use io_parallel locally for this fixed Dense network to avoid very wide
        # packed inter-layer stream words in io_stream mode.
        io_type='io_parallel',
        input_data_tb=str(tb_input_path) if tb_input_path is not None else None,
        output_data_tb=str(tb_output_path) if tb_output_path is not None else None,
    )
    hls_model.write()
    patch_generated_jet_project(output_dir)
    hls_model.build(csim=COSIM, synth=True, cosim=COSIM, validation=COSIM, vsynth=VSYNTH, log_to_stdout=False)

    copy_result_artifacts(output_dir, result_dir, proj_name, VSYNTH, COSIM)


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
        run_qbit_model,
        REUSE_FACTOR=args.reuse_factor,
        PART=args.part,
        CLK_PERIOD=args.clock_period,
        VSYNTH=args.vsynth,
        COSIM=args.cosim,
    )
