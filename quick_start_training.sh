#!/bin/bash

# Quick Start Script for LFM2 Instagram Training
# This script helps you get started with training the LFM2 model on Instagram captions

set -e

echo "🚀 LFM2 Instagram Training Quick Start"
echo "======================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ Python and pip are available"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📚 Installing training requirements..."
pip install -r requirements_training.txt

# Check if model exists
if [ ! -d "lfm2_vl_1_6b_model" ]; then
    echo "📥 Downloading LFM2 model (this may take a while)..."
    python save_lfm2_vl_model.py
else
    echo "✅ LFM2 model already exists"
fi

# Create sample dataset
echo "📁 Creating sample dataset structure..."
python train_lfm2_instagram.py --create-sample --data-dir ./instagram_dataset

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Add your Instagram images and captions to ./instagram_dataset/"
echo "2. Edit metadata.json with your actual data"
echo "3. Start training with:"
echo "   python train_lfm2_instagram_trainer.py --data-dir ./instagram_dataset --output-dir ./trained_model"
echo ""
echo "For more options, see README_TRAINING.md"
echo ""
echo "Happy training! 🚀"
