"""
Sanity test for serve_fp8_v100.py — verifies our wrapper imports and patches
vllm correctly without actually starting the server.

Run inside the dev container:
    ./run_docker.sh dev-test test_wrapper_imports.py
"""
import sys

# Don't pass any CLI args to the patched vllm
sys.argv = ["serve_fp8_v100.py"]

# Make the wrapper importable as a module. Since this importing path doesn't
# set __name__ == "__main__", the wrapper's CLI handoff block at the bottom
# is skipped — only the kernel-compile + patch-application module-level code
# runs.
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print("Loading serve_fp8_v100 module (this runs the patches)...")
import serve_fp8_v100   # noqa: F401  (imported for side effects only)

print()
print("=" * 70)
print("Verifying patches landed on the real vllm classes")
print("=" * 70)

# Confirm the patches are actually in place
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

min_cap = Fp8Config.get_min_capability()
print(f"  Fp8Config.get_min_capability() = {min_cap}  "
      f"(expected 70)  {'OK' if min_cap == 70 else 'FAIL'}")

# Check that __init__ was replaced (we can't easily inspect the body, but we can
# verify the method object is different from the original by checking presence of
# the closure variable that our patch creates)
init_qualname = Fp8LinearMethod.__init__.__qualname__
print(f"  Fp8LinearMethod.__init__ qualname = {init_qualname}")
init_patched = "patched" in init_qualname or "_patch_vllm_for_v100" in init_qualname
print(f"  __init__ appears patched          = {init_patched}  "
      f"{'OK' if init_patched else 'WARN'}")

apply_qualname = Fp8LinearMethod.apply.__qualname__
print(f"  Fp8LinearMethod.apply qualname    = {apply_qualname}")
apply_patched = "patched" in apply_qualname or "_patch_vllm_for_v100" in apply_qualname
print(f"  apply() appears patched           = {apply_patched}  "
      f"{'OK' if apply_patched else 'WARN'}")

pwal_qualname = Fp8LinearMethod.process_weights_after_loading.__qualname__
print(f"  process_weights_after_loading     = {pwal_qualname}")
pwal_patched = "patched" in pwal_qualname or "_patch_vllm_for_v100" in pwal_qualname
print(f"  pwal appears patched              = {pwal_patched}  "
      f"{'OK' if pwal_patched else 'WARN'}")

print()
overall = (min_cap == 70 and init_patched and apply_patched and pwal_patched)
print(f"OVERALL: {'PASS — wrapper is ready for actual serve' if overall else 'FAIL — patches did not all land'}")
sys.exit(0 if overall else 1)
