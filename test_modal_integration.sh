#!/bin/bash
# Test script to validate Modal integration files

set -e

echo "=================================================="
echo "DeepGEMM Modal Integration Validation"
echo "=================================================="
echo ""

# Check if required files exist
echo "✓ Checking files..."
files=(
    "modal_benchmark.py"
    "modal_simple_example.py"
    "MODAL_SETUP.md"
    "modal_requirements.txt"
    "README.md"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file exists"
    else
        echo "  ✗ $file missing"
        exit 1
    fi
done

echo ""
echo "✓ Validating Python syntax..."
python3 -m py_compile modal_benchmark.py
python3 -m py_compile modal_simple_example.py
echo "  ✓ modal_benchmark.py syntax OK"
echo "  ✓ modal_simple_example.py syntax OK"

echo ""
echo "✓ Checking documentation..."
if grep -q "Modal.com" README.md; then
    echo "  ✓ README.md mentions Modal.com"
else
    echo "  ✗ README.md doesn't mention Modal.com"
    exit 1
fi

if grep -q "modal.gpu.H100" modal_benchmark.py; then
    echo "  ✓ modal_benchmark.py requests H100 GPU"
else
    echo "  ✗ modal_benchmark.py doesn't request H100"
    exit 1
fi

echo ""
echo "✓ Checking Modal integration..."
if grep -q "deepgemm-h100-benchmark" modal_benchmark.py; then
    echo "  ✓ App name configured correctly"
else
    echo "  ✗ App name not found"
    exit 1
fi

if grep -q "test_fp8.py" modal_benchmark.py; then
    echo "  ✓ FP8 benchmark configured"
else
    echo "  ✗ FP8 benchmark not configured"
    exit 1
fi

echo ""
echo "=================================================="
echo "All validation checks passed! ✓"
echo "=================================================="
echo ""
echo "To run on Modal.com:"
echo "  1. Install modal: pip install -r modal_requirements.txt"
echo "  2. Setup auth: modal setup"
echo "  3. Run benchmark: modal run modal_benchmark.py"
echo ""
