import argparse
import io
import json
import os
import pprint
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')

import numpy as np
import tensorflow as tf
from qkeras.qlayers import QActivation, QDense
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.utils import _add_supported_quantized_objects
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Activation
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import to_categorical

VITIS_SETTINGS = Path('/tools/Xilinx/Vivado/2025.2/Vitis/settings64.sh')
DATASET_NAME = 'hls4ml_lhc_jets_hlf'
DATA_SPLIT_SEED = 42
DEFAULT_SEED = 0
TB_SAMPLES = 8
BITS = 8
INT = 0
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'jet_data'
RUN_ROOT = BASE_DIR / 'jet_train_runs'
DATA_FILES = ('X_train_val.npy', 'X_test.npy', 'y_train_val.npy', 'y_test.npy', 'classes.npy')


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


load_vitis_environment()

import hls4ml


def configure_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.set_visible_devices([], 'GPU')
    except (RuntimeError, ValueError):
        pass
    if hasattr(tf.config.experimental, 'enable_op_determinism'):
        tf.config.experimental.enable_op_determinism()
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except RuntimeError:
        pass


def dataset_ready(data_dir):
    return all((data_dir / name).is_file() for name in DATA_FILES)


def fetch_and_cache_dataset(data_dir):
    print(f'Fetching dataset {DATASET_NAME} from OpenML', flush=True)
    data = fetch_openml(
        DATASET_NAME,
        as_frame=False,
        parser='liac-arff',
        data_home=str(data_dir / 'openml_cache'),
    )
    x_all = np.asarray(data['data'], dtype=np.float32)
    y_all = np.asarray(data['target'])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_all)
    y_one_hot = to_categorical(y_encoded, len(label_encoder.classes_)).astype(np.float32)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x_all,
        y_one_hot,
        test_size=0.2,
        random_state=DATA_SPLIT_SEED,
    )

    scaler = StandardScaler()
    x_train_val = scaler.fit_transform(x_train_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / 'X_train_val.npy', x_train_val)
    np.save(data_dir / 'X_test.npy', x_test)
    np.save(data_dir / 'y_train_val.npy', y_train_val)
    np.save(data_dir / 'y_test.npy', y_test)
    np.save(data_dir / 'classes.npy', label_encoder.classes_)
    np.savez(data_dir / 'standard_scaler.npz', mean=scaler.mean_, scale=scaler.scale_)

    metadata = {
        'dataset_name': DATASET_NAME,
        'data_split_seed': DATA_SPLIT_SEED,
        'num_features': int(x_train_val.shape[1]),
        'num_classes': int(y_train_val.shape[1]),
        'train_val_samples': int(x_train_val.shape[0]),
        'test_samples': int(x_test.shape[0]),
    }
    (data_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


def load_dataset(data_dir):
    if not dataset_ready(data_dir):
        fetch_and_cache_dataset(data_dir)

    x_train_val = np.load(data_dir / 'X_train_val.npy').astype(np.float32)
    x_test = np.load(data_dir / 'X_test.npy').astype(np.float32)
    y_train_val = np.load(data_dir / 'y_train_val.npy').astype(np.float32)
    y_test = np.load(data_dir / 'y_test.npy').astype(np.float32)
    classes = np.load(data_dir / 'classes.npy', allow_pickle=True)
    return x_train_val, x_test, y_train_val, y_test, classes


def initializer_seed(base_seed, offset):
    return int(base_seed * 100 + offset)


def add_dense_relu(model, out_features, index, seed, *, input_dim=None):
    dense_kwargs = {}
    if input_dim is not None:
        dense_kwargs['input_shape'] = (input_dim,)

    model.add(
        QDense(
            out_features,
            name=f'fc{index}',
            kernel_quantizer=quantized_bits(BITS, INT, alpha=1),
            bias_quantizer=quantized_bits(2 * BITS, 2 * INT, alpha=1),
            kernel_initializer=RandomUniform(minval=-1, maxval=1, seed=initializer_seed(seed, 2 * index)),
            bias_initializer=RandomUniform(minval=-1, maxval=1, seed=initializer_seed(seed, 2 * index + 1)),
            **dense_kwargs,
        )
    )
    model.add(QActivation(quantized_relu(BITS, INT), name=f'relu{index}'))


def build_model(input_dim, num_classes, seed):
    model = Sequential(name='jet_quantized')
    model.add(QActivation(quantized_bits(BITS, INT), name='input_quant', input_shape=(input_dim,)))
    add_dense_relu(model, 64, 0, seed)
    add_dense_relu(model, 32, 1, seed)
    add_dense_relu(model, 32, 2, seed)
    model.add(
        QDense(
            num_classes,
            name='output',
            kernel_quantizer=quantized_bits(BITS, INT, alpha=1),
            bias_quantizer=quantized_bits(2 * BITS, 2 * INT, alpha=1),
            kernel_initializer=RandomUniform(minval=-1, maxval=1, seed=initializer_seed(seed, 100)),
            bias_initializer=RandomUniform(minval=-1, maxval=1, seed=initializer_seed(seed, 101)),
            kernel_regularizer=l1(0.0001),
        )
    )
    model.add(Activation(activation='softmax', name='softmax'))
    return model


def custom_objects_for_model():
    custom_objects = {}
    _add_supported_quantized_objects(custom_objects)
    return custom_objects


def create_callbacks(run_dir):
    return [
        ModelCheckpoint(str(run_dir / 'KERAS_check_best_model.h5'), monitor='val_loss', verbose=1, save_best_only=True),
        ModelCheckpoint(
            str(run_dir / 'KERAS_check_best_model_weights.h5'),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        ),
        ModelCheckpoint(str(run_dir / 'KERAS_check_model_last.h5'), verbose=1),
        ModelCheckpoint(str(run_dir / 'KERAS_check_model_last_weights.h5'), verbose=1, save_weights_only=True),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            mode='min',
            verbose=1,
            min_delta=1e-6,
            cooldown=2,
            min_lr=1e-7,
        ),
        EarlyStopping(monitor='val_loss', patience=12, verbose=1, mode='min', restore_best_weights=True),
        CSVLogger(str(run_dir / 'training.csv')),
    ]


def save_model_summary(model, path):
    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + '\n'))
    path.write_text(buffer.getvalue())


def shell_safe_name(value):
    return ''.join(ch if ch.isalnum() else '_' for ch in value)


def format_clock_tag(clock_period):
    if float(clock_period).is_integer():
        return str(int(clock_period))
    return str(clock_period).replace('.', 'p')


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


def prepare_testbench_data(model, proj_name, x_test):
    tb_dir = BASE_DIR / 'tb_data'
    tb_dir.mkdir(parents=True, exist_ok=True)

    tb_input = np.asarray(x_test[:TB_SAMPLES], dtype=np.float32)
    tb_output = model.predict(tb_input, verbose=0)

    input_path = tb_dir / f'{proj_name}_input.npy'
    output_path = tb_dir / f'{proj_name}_output.npy'
    np.save(input_path, tb_input)
    np.save(output_path, tb_output.astype(np.float32))
    return input_path, output_path


def train_model(args):
    configure_reproducibility(args.seed)
    tf.keras.backend.clear_session()
    x_train_val, x_test, y_train_val, y_test, classes = load_dataset(DATA_DIR)
    y_train_val_labels = np.argmax(y_train_val, axis=1)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=args.validation_split,
        random_state=args.seed,
        stratify=y_train_val_labels,
    )

    permutation = np.random.default_rng(args.seed).permutation(x_train.shape[0])
    x_train = x_train[permutation]
    y_train = y_train[permutation]

    input_dim = int(x_train_val.shape[1])
    num_classes = int(y_train_val.shape[1])
    run_name = args.run_name or f'quantized_rf{args.reuse_factor}_seed{args.seed}'
    run_dir = RUN_ROOT / shell_safe_name(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(input_dim, num_classes, args.seed)
    save_model_summary(model, run_dir / 'model_summary.txt')

    optimizer = Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer, loss=['categorical_crossentropy'], metrics=['accuracy'])

    history = model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_val, y_val),
        shuffle=False,
        callbacks=create_callbacks(run_dir),
        verbose=2,
    )

    history_payload = {key: [float(value) for value in values] for key, values in history.history.items()}
    (run_dir / 'history.json').write_text(json.dumps(history_payload, indent=2) + '\n')

    best_model_path = run_dir / 'KERAS_check_best_model.h5'
    best_model = load_model(best_model_path, custom_objects=custom_objects_for_model())
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test, verbose=0)
    predictions = best_model.predict(x_test, batch_size=args.batch_size, verbose=0).astype(np.float32)
    np.save(run_dir / 'y_test_pred.npy', predictions)

    report = {
        'model_kind': 'quantized',
        'seed': args.seed,
        'initializer_seed_scheme': 'base_seed * 100 + offset',
        'dataset_name': DATASET_NAME,
        'data_split_seed': DATA_SPLIT_SEED,
        'epochs_requested': args.epochs,
        'epochs_ran': len(history.history.get('loss', [])),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'validation_split': args.validation_split,
        'classes': [str(name) for name in classes.tolist()],
        'train_shape': list(x_train.shape),
        'validation_shape': list(x_val.shape),
        'train_val_shape': list(x_train_val.shape),
        'test_shape': list(x_test.shape),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'best_model_path': str(best_model_path.resolve()),
        'data_dir': str(DATA_DIR.resolve()),
    }
    (run_dir / 'training_report.json').write_text(json.dumps(report, indent=2) + '\n')

    return {
        'run_dir': run_dir,
        'run_name': shell_safe_name(run_name),
        'model': best_model,
        'x_test': x_test,
        'report': report,
    }


def build_hls_project(model, x_test, args, run_name):
    clock_tag = format_clock_tag(args.clock_period)
    proj_name = shell_safe_name(f'{run_name}_clk{clock_tag}')
    output_dir = BASE_DIR / 'nn_prj' / proj_name
    result_dir = BASE_DIR / 'nn_result' / proj_name

    config = hls4ml.utils.config_from_keras_model(model, granularity='model', backend='Vitis')
    config['Model']['ReuseFactor'] = args.reuse_factor
    config['Model']['Strategy'] = 'Resource'
    config['Model']['Precision'] = f'ap_fixed<{BITS},{INT + 1}>'

    print('-----------------------------------')
    pprint.pprint(config, sort_dicts=False)
    print('-----------------------------------')

    tb_input_path = None
    tb_output_path = None
    if args.cosim:
        tb_input_path, tb_output_path = prepare_testbench_data(model, proj_name, x_test)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        backend='Vitis',
        output_dir=str(output_dir),
        part=args.part,
        clock_period=args.clock_period,
        io_type='io_stream',
        input_data_tb=str(tb_input_path) if tb_input_path is not None else None,
        output_data_tb=str(tb_output_path) if tb_output_path is not None else None,
    )
    hls_model.write()
    hls_model.build(
        csim=args.cosim,
        synth=True,
        cosim=args.cosim,
        validation=args.cosim,
        vsynth=args.vsynth,
        log_to_stdout=False,
    )

    copy_result_artifacts(output_dir, result_dir, proj_name, args.vsynth, args.cosim)
    return proj_name, output_dir, result_dir


def time_block(label, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f'{label} took {dt:.2f}s')
    return out, dt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--validation-split', type=float, default=0.25)
    parser.add_argument('--run-name')
    parser.add_argument('--reuse-factor', type=int, default=16)
    parser.add_argument('--part', default='xcve2802-vsvh1760-2MP-e-S')
    parser.add_argument('--clock-period', type=float, default=3.2)
    parser.add_argument('--cosim', dest='cosim', action='store_true', default=True)
    parser.add_argument('--no-cosim', dest='cosim', action='store_false')
    parser.add_argument('--vsynth', dest='vsynth', action='store_true', default=True)
    parser.add_argument('--no-vsynth', dest='vsynth', action='store_false')
    return parser.parse_args()


def main():
    args = parse_args()
    training_state, _ = time_block('train_model', train_model, args)
    (proj_name, output_dir, result_dir), _ = time_block(
        'build_hls_project',
        build_hls_project,
        training_state['model'],
        training_state['x_test'],
        args,
        training_state['run_name'],
    )

    training_report_path = training_state['run_dir'] / 'training_report.json'
    report = json.loads(training_report_path.read_text())
    report['reuse_factor'] = args.reuse_factor
    report['part'] = args.part
    report['clock_period'] = args.clock_period
    report['cosim'] = args.cosim
    report['vsynth'] = args.vsynth
    report['hls_project_name'] = proj_name
    report['hls_project_dir'] = str(output_dir.resolve())
    report['hls_result_dir'] = str(result_dir.resolve())
    training_report_path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
