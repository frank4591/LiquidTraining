#!/bin/bash

# Instagram Dataset Processing Script
# This script automatically processes your Instagram dataset zip file

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Instagram Dataset Preprocessing${NC}"
echo "======================================"

# Check if zip file path is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}❌ Error: Please provide the path to your Instagram dataset zip file${NC}"
    echo ""
    echo "Usage: $0 <path_to_zip_file>"
    echo ""
    echo "Example: $0 InstaDataset.zip"
    echo "Example: $0 /path/to/your/InstaDataset.zip"
    exit 1
fi

ZIP_PATH="$1"
OUTPUT_DIR="../processed_dataset"

# Check if zip file exists
if [ ! -f "$ZIP_PATH" ]; then
    echo -e "${RED}❌ Error: Zip file not found: $ZIP_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found zip file: $ZIP_PATH${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.8+ first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python 3 is available${NC}"

# Check if required packages are installed
echo -e "${YELLOW}📦 Checking required packages...${NC}"
python3 -c "import PIL, json, zipfile, shutil, pathlib" 2>/dev/null || {
    echo -e "${YELLOW}📦 Installing required packages...${NC}"
    pip3 install -r requirements.txt
}

echo -e "${GREEN}✅ Required packages are available${NC}"

# Run preprocessing
echo -e "${YELLOW}🔄 Starting preprocessing...${NC}"
echo "This may take a few minutes depending on your dataset size..."

python3 preprocess_instagram_dataset.py --zip-path "$ZIP_PATH" --output-dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 Preprocessing completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}📁 Your processed dataset is ready at: $OUTPUT_DIR/instagram_dataset${NC}"
    echo ""
    echo -e "${BLUE}🚀 Next steps:${NC}"
    echo "1. Review the generated metadata.json file"
    echo "2. Start training with:"
    echo "   python ../train_lfm2_instagram_trainer.py --data-dir $OUTPUT_DIR/instagram_dataset --output-dir ../trained_model"
    echo ""
    echo -e "${GREEN}Happy training! 🚀${NC}"
else
    echo -e "${RED}❌ Preprocessing failed. Please check the error messages above.${NC}"
    exit 1
fi
