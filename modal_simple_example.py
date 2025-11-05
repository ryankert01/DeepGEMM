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

@app.function(image=image, gpu="B200:1")
def run_test():
    import subprocess
    result = subprocess.run(
        ["python", "tests/test_layout.py"],
        # ["python", "tests/test_attention.py"],
        # ["python", "tests/test_bf16.py"],
        # ["python", "tests/test_fp8.py"],
        cwd="/root/DeepGEMM",
        text=True,
        capture_output=True
    )
    print(result.stdout)
    print(result.stderr)
    return result.returncode


@app.local_entrypoint()
def main():
    print("Submitting test job to Modal...")
    code = run_test.remote()
    print(f"\nTest exited with code: {code}")
