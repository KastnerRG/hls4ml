import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from hls4ml.backends import VitisBackend
from hls4ml.model.flow import get_flow, register_flow
from hls4ml.report import aggregate_graph_reports, parse_vivado_report
from hls4ml.utils.simulation_utils import (
    annotate_axis_stream_widths,
    prepare_tb_inputs,
    read_testbench_log,
    write_verilog_testbench,
)

class AIEBackend(VitisBackend):
    def __init__(self, name='AIE'):
        super().__init__(name=name)


