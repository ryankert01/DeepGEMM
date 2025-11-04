"""
Simple example of running a single DeepGEMM kernel on Modal.com

This is a minimal example that runs a single FP8 GEMM operation.
For full benchmarks, use modal_benchmark.py
"""

import modal

app = modal.App("deepgemm-simple-example")

# Minimal image with DeepGEMM
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install("git", "build-essential")  # build-essential includes g++, gcc, make
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
    })
    .pip_install("torch>=2.1", "numpy", "packaging", "wheel")
    .run_commands(
        "cd /root && git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git",
        "cd /root/DeepGEMM && bash install.sh",
    )
)


@app.function(image=image, gpu="H100")
def run_simple_gemm():
    """Run a simple FP8 GEMM operation"""
    import torch
    import deep_gemm
    
    print("Running simple FP8 GEMM on H100...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create random FP8 matrices
    m, n, k = 8192, 8192, 8192
    
    # Input matrices (FP8 E4M3) - create directly as FP8
    a_fp8 = (torch.randn(m, k, device='cuda') * 0.1).to(torch.float8_e4m3fn)
    b_fp8 = (torch.randn(k, n, device='cuda') * 0.1).to(torch.float8_e4m3fn)
    
    # Scaling factors (FP32)
    a_sf = torch.ones(m, 1, dtype=torch.float32, device='cuda')
    b_sf = torch.ones(1, n, dtype=torch.float32, device='cuda')
    
    # Output matrix
    d = torch.zeros(m, n, dtype=torch.float32, device='cuda')
    
    # Warm up
    for _ in range(5):
        deep_gemm.fp8_gemm_nt((a_fp8, a_sf), (b_fp8, b_sf), d)
    
    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(10):
        deep_gemm.fp8_gemm_nt((a_fp8, a_sf), (b_fp8, b_sf), d)
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end) / 10
    tflops = (2 * m * n * k) / (time_ms * 1e-3) / 1e12
    
    print(f"\nResults for {m}x{n}x{k} FP8 GEMM:")
    print(f"  Time: {time_ms:.3f} ms")
    print(f"  Performance: {tflops:.1f} TFLOPS")
    print(f"  Output shape: {d.shape}")
    print(f"  Output dtype: {d.dtype}")
    
    return {
        "time_ms": time_ms,
        "tflops": tflops,
        "shape": f"{m}x{n}x{k}"
    }


@app.local_entrypoint()
def main():
    print("Starting simple DeepGEMM example on Modal...")
    result = run_simple_gemm.remote()
    print(f"\nFinal result: {result}")
