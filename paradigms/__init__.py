"""Paradigm package - imports all paradigm subpackages to trigger registration."""

from paradigms.p01_cpu import paradigm as _p01  # noqa: F401
from paradigms.p02_gpu import paradigm as _p02  # noqa: F401
from paradigms.p03_gaas_riscv import paradigm as _p03  # noqa: F401
from paradigms.p04_cdte_turing import paradigm as _p04  # noqa: F401
from paradigms.p05_dataflow import paradigm as _p05  # noqa: F401
from paradigms.p06_tpu_systolic import paradigm as _p06  # noqa: F401
from paradigms.p01b_cpu_fused import paradigm as _p01b  # noqa: F401
from paradigms.p02b_gpu_fused import paradigm as _p02b  # noqa: F401
from paradigms.p02c_gpu_fp16 import paradigm as _p02c  # noqa: F401
from paradigms.p02d_gpu_fused_fp16 import paradigm as _p02d  # noqa: F401
from paradigms.p06b_tpu_fused import paradigm as _p06b  # noqa: F401
from paradigms.p07_optical import paradigm as _p07        # noqa: F401
from paradigms.p08_chemical import paradigm as _p08       # noqa: F401
from paradigms.p09_biological import paradigm as _p09     # noqa: F401
from paradigms.p10_quantum import paradigm as _p10        # noqa: F401
from paradigms.p11_wsn import paradigm as _p11            # noqa: F401
from paradigms.p12_iot_edge import paradigm as _p12       # noqa: F401
