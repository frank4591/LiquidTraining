# Instagram Dataset Preprocessing

This folder contains all the preprocessing scripts for converting your Instagram dataset zip file into a training-ready format for the LFM2 model.

## 📁 Files

- `preprocess_instagram_dataset.py` - Main Python preprocessing script
- `process_instagram.sh` - Shell script for easy preprocessing
- `README_PREPROCESSING.md` - Comprehensive preprocessing guide

## 🚀 Quick Usage

### From Main Directory (Recommended)
```bash
./run_preprocessing.sh InstaDataset.zip
```

### From This Folder
```bash
./process_instagram.sh ../InstaDataset.zip
```

### Direct Python Usage
```bash
python3 preprocess_instagram_dataset.py --zip-path ../InstaDataset.zip --output-dir ../processed_dataset
```

## 📖 Documentation

See `README_PREPROCESSING.md` for detailed usage instructions, troubleshooting, and technical details.

## 🔧 Requirements

- Python 3.8+
- PIL (Pillow) for image processing
- Standard Python libraries (json, zipfile, shutil, pathlib)

## 📊 Output

The preprocessing creates a `processed_dataset/instagram_dataset/` folder containing:
- Renamed images (post_000000.jpg, post_000001.jpg, etc.)
- metadata.json for training
- dataset_info.json with statistics
