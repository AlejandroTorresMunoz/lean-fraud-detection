"""Shared pytest setup.

Pin PyTorch to a single thread for the test process. The unit/e2e tests run many tiny conv forwards
in one process; on Windows the multi-threaded MKL/OpenMP path can recurse into a fatal stack overflow
(and occasional NaNs) for these small tensors. Single-threaded inference is plenty fast for the test
sizes and makes the run deterministic and stable across platforms.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force the CPU path (the Windows CUDA build of
# torch can corrupt the shared pytest process running many tiny convs); CI is CPU-only anyway

import torch  # noqa: E402  (must follow the thread-env setup above)

torch.set_num_threads(1)
