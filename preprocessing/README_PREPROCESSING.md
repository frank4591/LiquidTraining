# Instagram Dataset Preprocessing Guide

This guide explains how to automatically preprocess your Instagram dataset zip file for training the LFM2 model.

## 🚀 Quick Start

### Option 1: One-Command Processing (Recommended)
```bash
./process_instagram.sh InstaDataset.zip
```

### Option 2: Python Script
```bash
python3 preprocess_instagram_dataset.py --zip-path InstaDataset.zip --output-dir ./processed_dataset
```

## 📁 Expected Dataset Structure

Your Instagram dataset zip file should contain:
```
InstaDataset.zip
├── media/
│   └── posts/
│       ├── 201608/          # Year 2016, Month 08
│       │   ├── image1.jpg
│       │   ├── image2.jpg
│       │   └── ...
│       ├── 201609/          # Year 2016, Month 09
│       └── ...
└── your_instagram_activity/
    ├── media.json           # Contains post metadata
    ├── posts.json           # Alternative metadata format
    └── ...
```

## 🔧 What the Preprocessor Does

### 1. **Automatic Extraction**
- Unzips your dataset zip file
- Identifies media and activity directories
- Handles various folder structures automatically

### 2. **Metadata Parsing**
- Finds all JSON metadata files
- Extracts post captions/titles
- Identifies media URIs and timestamps
- Handles multiple JSON formats

### 3. **Image Matching**
- Maps metadata to actual image files
- Uses URI patterns and timestamps for matching
- Handles different naming conventions

### 4. **Dataset Creation**
- Creates clean, training-ready dataset structure
- Renames images with consistent naming (post_000000.jpg, etc.)
- Generates metadata.json for training
- Validates image integrity

## 📊 Output Structure

After preprocessing, you'll get:
```
processed_dataset/
├── instagram_dataset/
│   ├── post_000000.jpg
│   ├── post_000001.jpg
│   ├── post_000002.jpg
│   ├── metadata.json          # Training metadata
│   └── dataset_info.json      # Dataset statistics
└── sample_metadata.json       # Reference format
```

### metadata.json Format
```json
[
    {
        "image_path": "post_000000.jpg",
        "caption": "Your Instagram caption here #hashtag",
        "timestamp": "2023-08-15"
    },
    {
        "image_path": "post_000001.jpg",
        "caption": "Another caption with emojis ✨",
        "timestamp": "2023-08-16"
    }
]
```

## 🎯 Usage Examples

### Basic Processing
```bash
# Process with default output directory
./process_instagram.sh InstaDataset.zip

# Process with custom output directory
python3 preprocess_instagram_dataset.py \
    --zip-path InstaDataset.zip \
    --output-dir ./my_custom_dataset
```

### Verbose Processing
```bash
# Enable detailed logging
python3 preprocess_instagram_dataset.py \
    --zip-path InstaDataset.zip \
    --verbose
```

### Process Different Zip Files
```bash
# Process from different location
./process_instagram.sh /path/to/other/InstagramData.zip

# Process multiple datasets
./process_instagram.sh dataset1.zip
./process_instagram.sh dataset2.zip
```

## ⚙️ Advanced Configuration

### Custom Output Directory
```bash
python3 preprocess_instagram_dataset.py \
    --zip-path InstaDataset.zip \
    --output-dir ./custom_output
```

### Verbose Logging
```bash
python3 preprocess_instagram_dataset.py \
    --zip-path InstaDataset.zip \
    --verbose
```

## 🔍 Troubleshooting

### Common Issues

1. **"Zip file not found"**
   - Check the file path
   - Ensure the zip file exists in the specified location
   - Use absolute paths if needed

2. **"No metadata files found"**
   - Verify your zip contains `your_instagram_activity` folder
   - Check that JSON files exist in the activity folder
   - Ensure JSON files are valid

3. **"No training pairs created"**
   - Check that metadata contains captions/titles
   - Verify image files exist in media folders
   - Check URI/timestamp matching logic

4. **"Permission denied"**
   - Make scripts executable: `chmod +x process_instagram.sh`
   - Check write permissions for output directory

### Debug Mode
```bash
# Enable verbose logging for debugging
python3 preprocess_instagram_dataset.py \
    --zip-path InstaDataset.zip \
    --verbose
```

## 📈 Performance Tips

### Large Datasets
- Processing time scales with dataset size
- Large datasets (>10GB) may take 10-30 minutes
- Monitor disk space during extraction

### Memory Usage
- Script uses temporary extraction directory
- Ensure sufficient disk space (2x zip size)
- Temporary files are automatically cleaned up

## 🔄 Workflow Integration

### Complete Training Pipeline
```bash
# 1. Preprocess dataset
./process_instagram.sh InstaDataset.zip

# 2. Start training
python3 train_lfm2_instagram_trainer.py \
    --data-dir ./processed_dataset/instagram_dataset \
    --output-dir ./trained_model

# 3. Test trained model
python3 instagram_caption_generator.py \
    --image test_image.jpg \
    --model-path ./trained_model/final_model
```

### Batch Processing
```bash
#!/bin/bash
# Process multiple datasets
for zip_file in *.zip; do
    echo "Processing $zip_file..."
    ./process_instagram.sh "$zip_file"
done
```

## 📚 Technical Details

### Supported Formats
- **Images**: JPG, JPEG, PNG, BMP, WebP
- **Metadata**: JSON (various structures)
- **Compression**: ZIP files

### Matching Algorithms
1. **URI Matching**: Direct filename matching
2. **Timestamp Matching**: Date-based folder matching
3. **Fallback**: Partial filename matching

### Error Handling
- Graceful degradation for corrupted files
- Detailed logging for debugging
- Automatic cleanup of temporary files

## 🤝 Support

### Getting Help
1. Check the troubleshooting section above
2. Enable verbose logging with `--verbose`
3. Review the generated log messages
4. Check file permissions and disk space

### Common Questions

**Q: Can I process multiple zip files?**
A: Yes, run the script multiple times with different zip files.

**Q: What if my metadata structure is different?**
A: The script handles various JSON formats automatically.

**Q: Can I customize the output format?**
A: Yes, modify the `create_training_dataset` method in the Python script.

**Q: How long does processing take?**
A: Depends on dataset size: small (1-5 min), medium (5-15 min), large (15-30+ min).

## 📄 License

This preprocessing code follows the same license as the base LFM2 model (LFM Open License v1.0).

---

**Ready to process your Instagram dataset?** 🚀

Just run: `./process_instagram.sh InstaDataset.zip`
