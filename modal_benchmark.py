"""
Run DeepGEMM's test_layout.py on Modal.com (GPU container)
"""

import modal

app = modal.App("deepgemm-test-layout")

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git", "build-essential", "clang")
    .pip_install("torch>=2.1", "numpy", "packaging", "wheel")
    .run_commands(
        "cd /root && git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git",
        "cd /root/DeepGEMM && cat develop.sh && ./develop.sh",
        "cd /root/DeepGEMM && pip install -e . --no-build-isolation"
    )
)

# ---------- Helper function ----------
def _run_deepgemm_test(script_name: str):
    import subprocess
    print(f"▶ Running {script_name}.py ...")
    result = subprocess.run(
        ["python", f"tests/{script_name}.py"],
        cwd="/root/DeepGEMM",
        text=True,
        capture_output=True,
    )
    print(result.stdout)
    print(result.stderr)
    return result.returncode


# ---------- Define each test function ----------
@app.function(image=image, gpu="B200:1")
def test_layout():
    return _run_deepgemm_test("test_layout")

@app.function(image=image, gpu="B200:1")
def test_attention():
    return _run_deepgemm_test("test_attention")

@app.function(image=image, gpu="B200:1")
def test_bf16():
    return _run_deepgemm_test("test_bf16")

@app.function(image=image, gpu="B200:1")
def test_fp8():
    return _run_deepgemm_test("test_fp8")

@app.function(image=image, gpu="B200:8")
def test_lazy_init():
    return _run_deepgemm_test("test_lazy_init")


# ---------- Local entrypoint ----------
@app.local_entrypoint()
def main():
    test_reqs = {
        "test_layout": test_layout,
        "test_attention": test_attention,
        "test_bf16": test_bf16,
        "test_fp8": test_fp8,
        "test_lazy_init": test_lazy_init,
    }

    futures = []
    for name, fn in test_reqs.items():
        print(f"Submitting {name} ...")
        futures.append((name, fn.spawn()))

    print(f"\n✅ Submitted {len(futures)} test jobs to Modal!\n")

    for name, f in futures:
        code = f.get()
        print(f"🧩 {name} exited with code: {code}")