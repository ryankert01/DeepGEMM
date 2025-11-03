"""
Modal.com script to run DeepGEMM benchmarks on H100 GPU

This script demonstrates how to run DeepGEMM on Modal.com's cloud infrastructure
using an H100 GPU. It installs dependencies, builds DeepGEMM, and runs the FP8 benchmark.

Usage:
    # Run the benchmark
    modal run modal_benchmark.py
    
    # Interactive shell for development
    modal shell modal_benchmark.py

Requirements:
    - modal (pip install modal)
    - A Modal account (https://modal.com)
    - Modal authentication: run `modal setup`

See MODAL_SETUP.md for detailed documentation.
"""

import modal
import os

# Define the Modal app
app = modal.App("deepgemm-h100-benchmark")

# DeepGEMM installation path on Modal container
DEEPGEMM_PATH = "/root/DeepGEMM"

# Create a custom image with all required dependencies
# Using CUDA 12.4 base image which is compatible with DeepGEMM
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install("git")
    # Set CUDA environment variables
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    })
    # Install PyTorch with CUDA 12.4 support
    .pip_install(
        "torch>=2.1",
        "numpy",
        "packaging",
        "wheel",
    )
    # Clone DeepGEMM repository with submodules and build it
    .run_commands(
        f"cd /root && git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git",
        f"cd {DEEPGEMM_PATH} && bash install.sh",
    )
)


@app.function(
    image=image,
    gpu="H100",  # Request H100 GPU
    timeout=3600,  # 1 hour timeout
)
def run_deepgemm_benchmark(
    benchmark_type: str = "fp8",
    num_warmups: int = 5,
    num_tests: int = 10
):
    """
    Run DeepGEMM benchmarks on H100 GPU
    
    Args:
        benchmark_type: Type of benchmark to run ('fp8', 'bf16', 'attention', 'all')
        num_warmups: Number of warmup iterations
        num_tests: Number of test iterations
    
    Returns:
        Dict with benchmark results and status
    """
    import subprocess
    import sys
    
    print("=" * 80)
    print("DeepGEMM H100 Benchmark on Modal.com")
    print("=" * 80)
    
    # Check CUDA availability and GPU info
    try:
        import torch
        print(f"\nEnvironment Information:")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU device: {torch.cuda.get_device_name(0)}")
            cap = torch.cuda.get_device_capability(0)
            print(f"  GPU compute capability: SM{cap[0]}{cap[1]}")
            
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU memory: {total_memory:.1f} GB")
        print()
    except Exception as e:
        print(f"Error checking CUDA: {e}")
        return {"error": str(e), "status": "error"}
    
    # Verify it's an H100 (SM90 architecture) or compatible GPU
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        if major < 9:
            print(f"WARNING: DeepGEMM requires SM90+ (H100) or SM100 architecture")
            print(f"         Current GPU is SM{major}{minor}, benchmark may fail")
            print()
    
    # Determine which test file to run
    test_files = {
        "fp8": "test_fp8.py",
        "bf16": "test_bf16.py", 
        "attention": "test_attention.py",
        "layout": "test_layout.py",
        "all": "test_fp8.py"  # Run FP8 as main benchmark
    }
    
    test_file = test_files.get(benchmark_type, "test_fp8.py")
    
    print(f"Running {benchmark_type.upper()} benchmark: {test_file}")
    print(f"Parameters: warmups={num_warmups}, tests={num_tests}")
    print("-" * 80)
    
    try:
        result = subprocess.run(
            ["python", f"{DEEPGEMM_PATH}/tests/{test_file}"],
            cwd=DEEPGEMM_PATH,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes
        )
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("\nSTDERR Output:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n" + "=" * 80)
            print("✓ Benchmark completed successfully!")
            print("=" * 80)
            
            # Parse TFLOPS from output (simple extraction)
            import re
            tflops_values = []
            for line in result.stdout.split('\n'):
                if 'TFLOPS' in line:
                    try:
                        # Extract TFLOPS value using regex
                        # Look for pattern: number followed by TFLOPS
                        parts = line.split('|')
                        for part in parts:
                            if 'TFLOPS' in part:
                                # Extract the number before 'TFLOPS'
                                match = re.search(r'(\d+(?:\.\d+)?)\s*TFLOPS', part)
                                if match:
                                    tflops = float(match.group(1))
                                    tflops_values.append(tflops)
                    except (ValueError, IndexError, AttributeError) as e:
                        # Skip lines that don't match expected format
                        continue
            
            avg_tflops = sum(tflops_values) / len(tflops_values) if tflops_values else 0
            max_tflops = max(tflops_values) if tflops_values else 0
            
            return {
                "status": "success",
                "benchmark_type": benchmark_type,
                "gpu": torch.cuda.get_device_name(0),
                "avg_tflops": avg_tflops,
                "max_tflops": max_tflops,
                "num_kernels": len(tflops_values),
                "output": result.stdout
            }
        else:
            print("\n" + "=" * 80)
            print(f"✗ Benchmark failed with exit code {result.returncode}")
            print("=" * 80)
            return {
                "status": "failed",
                "returncode": result.returncode,
                "stderr": result.stderr,
                "stdout": result.stdout
            }
            
    except subprocess.TimeoutExpired:
        print("✗ Benchmark timed out after 30 minutes")
        return {"status": "timeout"}
    except Exception as e:
        print(f"✗ Error running benchmark: {e}")
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.local_entrypoint()
def main(benchmark_type: str = "fp8"):
    """
    Main entry point when running with 'modal run'
    
    Args:
        benchmark_type: Type of benchmark to run ('fp8', 'bf16', 'attention', 'all')
    
    Example:
        modal run modal_benchmark.py
        modal run modal_benchmark.py --benchmark-type bf16
    """
    print("=" * 80)
    print("Starting DeepGEMM benchmark on Modal H100...")
    print(f"Benchmark type: {benchmark_type}")
    print("=" * 80)
    print()
    
    result = run_deepgemm_benchmark.remote(benchmark_type=benchmark_type)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    if result.get("status") == "success":
        print(f"✓ Status: SUCCESS")
        print(f"  GPU: {result.get('gpu', 'Unknown')}")
        print(f"  Benchmark: {result.get('benchmark_type', 'Unknown')}")
        print(f"  Kernels tested: {result.get('num_kernels', 0)}")
        if result.get('avg_tflops', 0) > 0:
            print(f"  Average TFLOPS: {result.get('avg_tflops', 0):.1f}")
            print(f"  Peak TFLOPS: {result.get('max_tflops', 0):.1f}")
    else:
        print(f"✗ Status: {result.get('status', 'UNKNOWN').upper()}")
        if 'error' in result:
            print(f"  Error: {result['error']}")
    
    print("=" * 80)
    
    return result
