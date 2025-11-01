#!/bin/bash
# Setup script for EllSeg integration

echo "================================================"
echo "EllSeg Setup for Eye Pipeline"
echo "================================================"

# Create necessary directories
mkdir -p weights
mkdir -p ellseg_repo

echo ""
echo "Step 1: Clone EllSeg repository..."
if [ ! -d "ellseg_repo/.git" ]; then
    git clone https://github.com/RSKothari/EllSeg.git ellseg_repo
    echo "✅ EllSeg repository cloned"
else
    echo "⏭️  EllSeg repository already exists"
fi

echo ""
echo "Step 2: Copy model files..."
if [ -d "ellseg_repo/models" ]; then
    cp -r ellseg_repo/models ./
    cp ellseg_repo/utils.py ./
    cp ellseg_repo/helperfunctions.py ./
    echo "✅ Model files copied"
else
    echo "❌ Model files not found in ellseg_repo/"
    exit 1
fi

echo ""
echo "Step 3: Download pre-trained weights..."
echo "The weights are hosted on BitBucket or in the repo releases."
echo ""
echo "Option A: Download from releases (if available)"
echo "Option B: Use weights from cloned repo"

if [ -f "ellseg_repo/weights/all.git_ok" ]; then
    cp ellseg_repo/weights/all.git_ok weights/ellseg_all.pt
    echo "✅ Weights copied from repo"
elif [ -f "ellseg_repo/weights/all.pkl" ]; then
    cp ellseg_repo/weights/all.pkl weights/ellseg_all.pt
    echo "✅ Weights copied from repo (pkl format)"
else
    echo "⚠️  Weights not found in repository"
    echo ""
    echo "Please download manually:"
    echo "1. Check: https://github.com/RSKothari/EllSeg/releases"
    echo "2. Or: https://bitbucket.org/RSKothari/multiset-ritnet/downloads/"
    echo "3. Save as: weights/ellseg_all.pt"
fi

echo ""
echo "Step 4: Check Python dependencies..."
if python3 -c "import torch" 2>/dev/null; then
    echo "✅ PyTorch installed"
else
    echo "⚠️  PyTorch not found. Install with:"
    echo "   pip3 install torch torchvision"
fi

if python3 -c "import cv2" 2>/dev/null; then
    echo "✅ OpenCV installed"
else
    echo "⚠️  OpenCV not found. Install with:"
    echo "   pip3 install opencv-python"
fi

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To test EllSeg integration:"
echo "  python3 ellseg_integration.py"
echo ""
echo "To use in GUI:"
echo "  1. Ensure weights are at: weights/ellseg_all.pt"
echo "  2. Run: python3 pipeline_tuner_gui.py"
echo "  3. Enable EllSeg in Step 5.5"
echo ""
