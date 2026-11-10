# ============================================================
# LOCAL PC RUN (not Colab). This is the FIRST cell of the notebook; it runs before Section 0.
# Machine: Intel i7-14700 (20 cores / 28 threads), RTX 4070 Ti SUPER 16 GB, 64 GB RAM, Windows.
# Edit only the four settings below. Everything after this cell is the unchanged v4 notebook.
# ============================================================
import json
import os
import shutil
import sys
from pathlib import Path

RUN_ROOT = Path(r"D:\user1\sepsis\sepsis_clean_run")         # NEW, EMPTY folder on the first start; reused on every resume
LEGACY_ROOT = Path(r"D:\user1\sepsis\sepsis_fresh_colab")    # the original study's folder, downloaded from Google Drive
CPU_THREADS = 20                                      # physical cores; the 8 extra hyper-threads do not speed up BLAS
RUN_OPTIONAL = False                                  # True adds the B = 200 bootstrap (II.9d) and lifts (II.9f): +3-6 h

# 1. Windows writes text files as cp1252 by default; the notebook writes UTF-8 (tau, <=, en dashes).
if os.name == "nt":
    assert sys.flags.utf8_mode, ("Python is not in UTF-8 mode. Start the run with run_clean_local.bat or "
                                 "open_jupyter.bat (both set PYTHONUTF8=1), not from a plain Jupyter shortcut.")

# 2. Thread pools are sized when NumPy / PyTorch load, so this must run before either is imported.
_already = [m for m in ("numpy", "scipy", "torch") if m in sys.modules]
if _already:
    print("WARNING: restart the kernel and run this cell first; already imported:", _already)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = str(CPU_THREADS)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # same value as Section 0; needed before CUDA starts

# 2b. Section 1 detects Colab with importlib.util.find_spec("google.colab"). That returns None only when a "google"
#     package exists (Colab always has one); without it, it raises ModuleNotFoundError. An empty placeholder
#     package makes it return None, so IN_COLAB is False and Section 1 itself stays unchanged.
import importlib.util  # noqa: E402
import types  # noqa: E402

if importlib.util.find_spec("google") is None:
    _google = types.ModuleType("google")
    _google.__path__ = []
    sys.modules["google"] = _google

# 3. Folders: the run must never write into the original study's folder, or the reverse.
RUN_ROOT, LEGACY_ROOT = RUN_ROOT.resolve(), LEGACY_ROOT.resolve()
assert RUN_ROOT != LEGACY_ROOT and LEGACY_ROOT not in RUN_ROOT.parents and RUN_ROOT not in LEGACY_ROOT.parents
for _rel in ("tables/exact_results_all.csv", "policies/Tabular_IQL_seed0.npz", "policies/Diffusion-AM-CQL_seed0.npz"):
    assert (LEGACY_ROOT / _rel).is_file(), (f"missing {LEGACY_ROOT / _rel}: copy the whole sepsis_fresh_colab folder "
                                            "from Google Drive (Section 21 would otherwise skip its replay silently)")
RUN_ROOT.mkdir(parents=True, exist_ok=True)
assert shutil.disk_usage(RUN_ROOT).free > 20e9, "less than 20 GB free on the run drive"
print("run folder:", RUN_ROOT, "| already holds", sum(1 for _ in RUN_ROOT.rglob("*")), "files (0 on the first start)")

# 4. The GPU and the package versions of the original Colab run.
import torch  # noqa: E402  (after the thread settings above)

assert torch.cuda.is_available(), "PyTorch does not see the GPU: run setup_env.bat again (CUDA build of torch)"
torch.set_num_threads(CPU_THREADS)
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| CUDA", torch.version.cuda,
      "| CPU threads", torch.get_num_threads())
ORIGINAL_RUN = {"torch": "2.11.0+cu130", "numpy": "2.1.3", "pandas": "2.2.3", "scipy": "1.16.3",
                "matplotlib": "3.10.0", "gymnasium": "1.3.0", "icu-sepsis": "2.0.1"}
from importlib.metadata import version  # noqa: E402

_diff = {k: (version(k), v) for k, v in ORIGINAL_RUN.items() if version(k) != v}
print("versions as in the original run" if not _diff else f"WARNING: versions differ from the original run: {_diff}")
print("Python", sys.version.split()[0], "(original run: 3.13)")

# 5. The notebook's own switches (read in Section 1).
os.environ["SEPSIS_ROOT"] = str(RUN_ROOT)
os.environ["SEPSIS_LEGACY_ROOT"] = str(LEGACY_ROOT)
os.environ["SEPSIS_RUN_MODE"] = "paper"
os.environ["SEPSIS_COMPUTE_PROFILE"] = "full"
os.environ["SEPSIS_EXPERIMENTS"] = json.dumps({
    "xenv_v4_bootstrap": RUN_OPTIONAL,   # II.9d, B = 200 re-estimated bootstrap
    "xenv_v4_lift": RUN_OPTIONAL,        # II.9f, physician and uniform lifts
    "xenv_v4_clean_check": False,        # II.9g is redundant: this whole run starts from an empty folder
})
print("outputs go to", RUN_ROOT / "paper")
