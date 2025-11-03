# Running DeepGEMM on Modal.com with H100 GPU

This guide explains how to run DeepGEMM benchmarks on [Modal.com](https://modal.com) using NVIDIA H100 GPUs.

## What is Modal.com?

Modal is a serverless platform that lets you run code on cloud infrastructure, including powerful GPUs like the H100, without managing servers or infrastructure.

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install the Modal Python package:
   ```bash
   pip install modal
   ```
3. **Authentication**: Set up your Modal credentials:
   ```bash
   modal setup
   ```

## Quick Start

Run the DeepGEMM benchmark on an H100 GPU:

```bash
# Full benchmark suite
modal run modal_benchmark.py

# Run different benchmark types
modal run modal_benchmark.py --benchmark-type bf16
modal run modal_benchmark.py --benchmark-type attention

# Run simple example (single GEMM operation)
modal run modal_simple_example.py
```

This will:
- Provision an H100 GPU instance
- Install CUDA Toolkit 12.4
- Install PyTorch and dependencies
- Clone and build DeepGEMM
- Run the FP8 GEMM benchmark
- Display performance metrics

## What Gets Benchmarked

The script runs `tests/test_fp8.py`, which includes:
- Normal FP8 GEMM operations (various matrix sizes)
- M-grouped contiguous GEMM
- M-grouped masked GEMM
- K-grouped contiguous GEMM

Performance metrics include:
- Execution time (microseconds)
- TFLOPS (Tera Floating Point Operations per Second)
- Memory bandwidth (GB/s)
- Comparison with cuBLAS

## Expected Output

You should see output similar to:

```
Testing GEMM:
 > Perf (m=  8192, n=  8192, k=  8192, 1D2D, layout=NT, FP32, acc=0):  123 us | 1098 TFLOPS | 890 GB/s | 1.23x cuBLAS
...
```

The H100 GPU typically achieves:
- **1200-1500+ TFLOPS** for FP8 operations
- **High memory bandwidth** utilization
- **Competitive or better performance** vs cuBLAS

## Customization

### Running Different Benchmarks

Edit `modal_benchmark.py` to run different test files:

```python
# In run_deepgemm_benchmark function, change:
["python", "/root/DeepGEMM/tests/test_fp8.py"],
# to:
["python", "/root/DeepGEMM/tests/test_bf16.py"],  # BF16 benchmark
["python", "/root/DeepGEMM/tests/test_attention.py"],  # Attention kernels
```

### Adjusting GPU Type

Change the GPU in the `@app.function` decorator:

```python
@app.function(
    gpu="H100",  # Single H100
    # or
    gpu="A100",  # A100 (SM80)
    # or
    gpu="H100:2",  # Multi-GPU (2x H100)
)
```

Note: DeepGEMM requires SM90 (H100) or SM100 architecture for optimal performance.

### Environment Variables

Add DeepGEMM environment variables:

```python
@app.function(
    env={
        "DG_JIT_DEBUG": "1",  # Enable debug output
        "DG_PRINT_CONFIGS": "1",  # Print selected configs
    }
)
```

## Cost Considerations

- Modal charges per second of GPU usage
- H100 GPUs are premium tier
- The benchmark typically runs in 5-15 minutes
- Consider using Modal's free tier credits for initial testing

## Troubleshooting

### CUDA Version Mismatch

If you see CUDA version errors, the Modal script uses CUDA 12.4 base image, which is compatible with DeepGEMM (requires CUDA 12.3+). DeepGEMM recommends CUDA 12.9+ for best performance, but 12.4 provides good compatibility.

### Build Failures

If the build fails during image creation:
- Check that all git submodules are initialized
- Verify CUDA_HOME is set correctly
- Ensure the build has sufficient memory

### GPU Not Available

If `torch.cuda.is_available()` returns False:
- Verify you selected a GPU in the function decorator
- Check Modal's GPU availability in your region
- Ensure PyTorch is installed with CUDA support

### Performance Lower Than Expected

- Verify you're using an H100 (SM90) GPU
- The script uses CUDA 12.4; upgrading to 12.9+ may improve performance
- Ensure no CPU throttling or resource contention
- Review DeepGEMM environment variables

## Advanced Usage

### Interactive Development

Start an interactive Modal session:

```bash
modal shell modal_benchmark.py
```

Then run commands interactively:

```python
>>> import torch
>>> print(torch.cuda.get_device_name(0))
>>> import deep_gemm
>>> # Run custom benchmarks
```

### Saving Results

Modify the script to save results to Modal Volumes or export to cloud storage:

```python
volume = modal.Volume.from_name("deepgemm-results", create_if_missing=True)

@app.function(volumes={"/results": volume})
def run_benchmark():
    # ... run benchmark ...
    with open("/results/benchmark.txt", "w") as f:
        f.write(result.stdout)
    volume.commit()
```

## Additional Resources

- [Modal Documentation](https://modal.com/docs)
- [DeepGEMM Repository](https://github.com/deepseek-ai/DeepGEMM)
- [Modal GPU Guide](https://modal.com/docs/guide/gpu)

## License

The Modal integration script is provided under the same MIT License as DeepGEMM.
