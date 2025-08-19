#!/bin/bash

# LFM2-VL-1.6B Setup and Demo Script
# This script automates the setup process and runs a demo

echo "🚀 LFM2-VL-1.6B Setup and Demo"
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip not found. Please install pip first."
    exit 1
fi

echo "✅ Python and pip found"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed"

# Check if model already exists
if [ -d "./lfm2_vl_1_6b_model" ]; then
    echo "✅ Model already exists, skipping download"
else
    echo "⬇️  Downloading LFM2-VL-1.6B model..."
    python3 save_lfm2_vl_model.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to download model"
        exit 1
    fi
    
    echo "✅ Model downloaded successfully"
fi

# Test the setup
echo "🧪 Testing setup..."
python3 test_setup.py

if [ $? -ne 0 ]; then
    echo "❌ Setup test failed"
    exit 1
fi

echo "✅ Setup test passed"

# Check if image exists
if [ -f "../img1.jpg" ]; then
    echo "✅ Found img1.jpg"
    
    # Run demo
    echo "🎯 Running demo..."
    python3 demo.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 Setup and demo completed successfully!"
        echo ""
        echo "💡 You can now use the model with:"
        echo "   python3 instagram_caption_generator.py --image ../img1.jpg"
        echo "   python3 instagram_caption_generator.py --image ../img1.jpg --style creative --num-captions 5"
    else
        echo "❌ Demo failed"
        exit 1
    fi
else
    echo "⚠️  img1.jpg not found in parent directory"
    echo "💡 You can still test the model with other images:"
    echo "   python3 instagram_caption_generator.py --image /path/to/your/image.jpg"
fi

echo ""
echo "✨ Setup complete! Happy captioning! ✨"
