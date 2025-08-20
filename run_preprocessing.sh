#!/bin/bash

# Launcher script for Instagram dataset preprocessing
# This script runs the preprocessing from the preprocessing/ folder

set -e

# Colors for output
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Instagram Dataset Preprocessing Launcher${NC}"
echo "================================================"

# Check if zip file path is provided
if [ $# -eq 0 ]; then
    echo "❌ Error: Please provide the path to your Instagram dataset zip file"
    echo ""
    echo "Usage: $0 <path_to_zip_file>"
    echo ""
    echo "Example: $0 InstaDataset.zip"
    echo "Example: $0 /path/to/your/InstaDataset.zip"
    exit 1
fi

ZIP_PATH="$1"

# Check if zip file exists
if [ ! -f "$ZIP_PATH" ]; then
    echo "❌ Error: Zip file not found: $ZIP_PATH"
    exit 1
fi

echo "✅ Found zip file: $ZIP_PATH"
echo "🔄 Running preprocessing from preprocessing/ folder..."

# Run the preprocessing script from the preprocessing folder
./preprocessing/process_instagram.sh "$ZIP_PATH"
