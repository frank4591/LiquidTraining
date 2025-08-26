#!/usr/bin/env python3
"""
Instagram Dataset Preprocessing Script for LFM2 Training

This script automatically:
1. Unzips the Instagram dataset
2. Parses the metadata from your_instagram_activity folder
3. Maps images to their captions/titles
4. Creates a training-ready dataset structure
5. Generates metadata.json for training

Usage:
    python preprocess_instagram_dataset.py --zip-path InstaDataset.zip --output-dir ./processed_dataset
"""

import os
import json
import zipfile
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime
import re
from PIL import Image
import hashlib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InstagramDatasetPreprocessor:
    """Preprocess Instagram dataset for LFM2 training"""
    
    def __init__(self, zip_path, output_dir):
        self.zip_path = zip_path
        self.output_dir = Path(output_dir)
        self.temp_dir = Path("./temp_instagram_extract")
        self.media_dir = None
        self.activity_dir = None
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
    def cleanup_temp(self):
        """Clean up temporary extraction directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temporary directory")
    
    def extract_zip(self):
        """Extract the Instagram dataset zip file"""
        logger.info(f"Extracting {self.zip_path}...")
        
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
            
            logger.info("Zip extraction completed successfully")
            
            # Find media and activity directories
            for item in self.temp_dir.rglob("*"):
                if item.is_dir():
                    # Look for the root media directory (contains posts folder)
                    if str(item).endswith("media") and "posts" in str(item).lower():
                        self.media_dir = item
                    # Look for the root your_instagram_activity directory
                    elif str(item).endswith("your_instagram_activity"):
                        self.activity_dir = item
            
            # If we didn't find the root directories, look for them more broadly
            if not self.media_dir:
                # Look for the media directory that contains posts subfolder
                for item in self.temp_dir.rglob("*"):
                    if item.is_dir() and "media" in str(item).lower():
                        # Check if this directory contains a posts subdirectory
                        posts_subdir = item / "posts"
                        if posts_subdir.exists():
                            self.media_dir = item
                            break
            
            if not self.activity_dir:
                for item in self.temp_dir.rglob("*"):
                    if item.is_dir() and "your_instagram_activity" in str(item).lower():
                        self.activity_dir = item
                        break
            
            if not self.media_dir:
                raise ValueError("Could not find 'media' directory in extracted files")
            if not self.activity_dir:
                raise ValueError("Could not find 'your_instagram_activity' directory in extracted files")
                
            logger.info(f"Found media directory: {self.media_dir}")
            logger.info(f"Found activity directory: {self.activity_dir}")
            
        except Exception as e:
            logger.error(f"Failed to extract zip file: {e}")
            raise
    
    def find_metadata_files(self):
        """Find all metadata JSON files in the activity directory"""
        metadata_files = []
        
        # Priority order for metadata files
        priority_files = [
            "posts_1.json",
            "posts.json", 
            "media.json",
            "saved_posts.json"
        ]
        
        # First look for priority files
        for priority_file in priority_files:
            priority_path = self.activity_dir / "media" / priority_file
            if priority_path.exists():
                metadata_files.append(priority_path)
                logger.info(f"Found priority metadata file: {priority_path}")
        
        # Then look for other JSON files
        for file_path in self.activity_dir.rglob("*.json"):
            if file_path not in metadata_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Check if this looks like Instagram metadata
                        if isinstance(data, list) or isinstance(data, dict):
                            metadata_files.append(file_path)
                            logger.info(f"Found metadata file: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not read {file_path}: {e}")
        
        return metadata_files
    
    def parse_metadata(self, metadata_files):
        """Parse metadata files to extract post information"""
        posts_data = []
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    # Look for common keys that might contain posts
                    items = []
                    for key in ['media', 'posts', 'data', 'items']:
                        if key in data and isinstance(data[key], list):
                            items = data[key]
                            break
                    if not items:
                        items = [data]  # Single post
                else:
                    continue
                
                # Process each item
                for item in items:
                    if isinstance(item, dict):
                        # Handle Instagram's nested media structure
                        if 'media' in item and isinstance(item['media'], list):
                            # Instagram posts structure: item -> media array -> individual media
                            # Check if this post has a title at the post level
                            post_title = item.get('title', '')
                            post_timestamp = item.get('creation_timestamp')
                            
                            for media_item in item['media']:
                                if isinstance(media_item, dict):
                                    # Use post-level title if available, otherwise media item title
                                    post_info = self.extract_post_info(media_item, post_title, post_timestamp)
                                    if post_info:
                                        posts_data.append(post_info)
                        else:
                            # Direct post structure
                            post_info = self.extract_post_info(item)
                            if post_info:
                                posts_data.append(post_info)
                            
            except Exception as e:
                logger.warning(f"Error parsing {metadata_file}: {e}")
                continue
        
        logger.info(f"Extracted {len(posts_data)} posts from metadata")
        return posts_data
    
    def extract_post_info(self, item, post_title=None, post_timestamp=None):
        """Extract relevant information from a single post item"""
        post_info = {}
        
        # Try to extract title/caption - prioritize post-level title over media item title
        caption = None
        
        # First check if we have a post-level title (for multi-media posts)
        if post_title and post_title.strip():
            caption = str(post_title).strip()
        else:
            # Fall back to media item title
            for key in ['title', 'caption', 'description', 'text', 'content']:
                if key in item and item[key]:
                    caption = str(item[key]).strip()
                    break
        
        # Instagram often has empty titles, so we'll create a default caption if none exists
        if not caption:
            caption = "Instagram post"  # Default caption for posts without text
        
        # Try to extract media URI
        media_uri = None
        for key in ['uri', 'media_uri', 'file_path', 'path', 'url']:
            if key in item and item[key]:
                media_uri = str(item[key])
                break
        
        # Try to extract timestamp - prioritize post-level timestamp over media item timestamp
        timestamp = None
        if post_timestamp:
            timestamp = post_timestamp
        else:
            for key in ['creation_timestamp', 'timestamp', 'created_time', 'date', 'time']:
                if key in item and item[key]:
                    timestamp = item[key]
                    break
        
        # Only return post info if we have a media URI
        if not media_uri:
            return None
        
        post_info = {
            'caption': caption,
            'media_uri': media_uri,
            'timestamp': timestamp,
            'original_data': item
        }
        
        return post_info
    
    def find_matching_images(self, posts_data):
        """Find images that match the metadata and create training pairs"""
        training_pairs = []
        
        # Get all image files from media directory
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
            image_files.extend(self.media_dir.rglob(ext))
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Create a mapping of filenames to full paths
        image_map = {}
        for img_path in image_files:
            filename = img_path.name
            image_map[filename] = img_path
        
        # Try to match posts with images
        matched_count = 0
        for post in posts_data:
            if post['media_uri']:
                # Try to match by URI
                matched_image = self.find_image_by_uri(post['media_uri'], image_map)
                if matched_image:
                    training_pairs.append({
                        'image_path': str(matched_image),
                        'caption': post['caption'],
                        'timestamp': post['timestamp']
                    })
                    matched_count += 1
                    continue
            
            # Try to match by timestamp if available
            if post['timestamp']:
                matched_image = self.find_image_by_timestamp(post['timestamp'], image_map)
                if matched_image:
                    training_pairs.append({
                        'image_path': str(matched_image),
                        'caption': post['caption'],
                        'timestamp': post['timestamp']
                    })
                    matched_count += 1
                    continue
        
        logger.info(f"Successfully matched {matched_count} posts with images")
        return training_pairs
    
    def find_image_by_uri(self, uri, image_map):
        """Find image by matching URI patterns"""
        # Extract filename from URI
        uri_parts = uri.split('/')
        if uri_parts:
            filename = uri_parts[-1]
            if filename in image_map:
                return image_map[filename]
        
        # Try to match partial filenames
        for img_filename, img_path in image_map.items():
            if any(part in img_filename for part in uri_parts if part):
                return img_path
        
        return None
    
    def find_image_by_timestamp(self, timestamp, image_map):
        """Find image by matching timestamp patterns"""
        # Convert timestamp to date format if possible
        try:
            if isinstance(timestamp, (int, float)):
                # Unix timestamp
                dt = datetime.fromtimestamp(timestamp)
                date_str = dt.strftime("%Y%m")
            elif isinstance(timestamp, str):
                # Try to parse date string
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"]:
                    try:
                        dt = datetime.strptime(timestamp, fmt)
                        date_str = dt.strftime("%Y%m")
                        break
                    except:
                        continue
                else:
                    return None
            else:
                return None
            
            # Look for images in matching date folders
            for img_filename, img_path in image_map.items():
                if date_str in str(img_path):
                    return img_path
                    
        except Exception as e:
            logger.debug(f"Error matching timestamp {timestamp}: {e}")
        
        return None
    
    def create_training_dataset(self, training_pairs):
        """Create the final training dataset structure"""
        logger.info("Creating training dataset...")
        
        # Create dataset directory
        dataset_dir = self.output_dir / "instagram_dataset"
        dataset_dir.mkdir(exist_ok=True)
        
        # Copy images and create metadata
        processed_pairs = []
        for i, pair in enumerate(training_pairs):
            try:
                # Generate a clean filename
                img_path = Path(pair['image_path'])
                file_ext = img_path.suffix
                new_filename = f"post_{i:06d}{file_ext}"
                
                # Copy image to dataset directory
                new_img_path = dataset_dir / new_filename
                shutil.copy2(img_path, new_img_path)
                
                # Verify image is valid
                try:
                    with Image.open(new_img_path) as img:
                        img.verify()
                except Exception as e:
                    logger.warning(f"Invalid image {new_filename}: {e}")
                    new_img_path.unlink()
                    continue
                
                processed_pairs.append({
                    'image_path': new_filename,
                    'caption': pair['caption'],
                    'timestamp': pair['timestamp']
                })
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1} images...")
                    
            except Exception as e:
                logger.warning(f"Error processing {pair['image_path']}: {e}")
                continue
        
        # Create metadata.json
        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(processed_pairs, f, indent=2, ensure_ascii=False)
        
        # Create dataset info
        dataset_info = {
            'total_posts': len(processed_pairs),
            'creation_date': datetime.now().isoformat(),
            'source_zip': self.zip_path,
            'processing_notes': 'Automatically processed by Instagram Dataset Preprocessor'
        }
        
        info_path = dataset_dir / "dataset_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"Training dataset created successfully!")
        logger.info(f"Total posts: {len(processed_pairs)}")
        logger.info(f"Dataset location: {dataset_dir}")
        
        return dataset_dir
    
    def create_sample_metadata(self):
        """Create a sample metadata structure for reference"""
        sample_data = [
            {
                "image_path": "post_000000.jpg",
                "caption": "Beautiful sunset vibes! 🌅 Nature never fails to amaze me. #sunset #nature #photography",
                "timestamp": "2023-08-15"
            },
            {
                "image_path": "post_000001.jpg", 
                "caption": "When the sky paints itself in golden hour magic ✨ #goldenhour #sky #photography",
                "timestamp": "2023-08-16"
            }
        ]
        
        sample_path = self.output_dir / "sample_metadata.json"
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"Sample metadata created: {sample_path}")
    
    def run_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        try:
            logger.info("Starting Instagram dataset preprocessing...")
            
            # Step 1: Extract zip file
            self.extract_zip()
            
            # Step 2: Find metadata files
            metadata_files = self.find_metadata_files()
            if not metadata_files:
                raise ValueError("No metadata files found")
            
            # Step 3: Parse metadata
            posts_data = self.parse_metadata(metadata_files)
            if not posts_data:
                raise ValueError("No posts data extracted from metadata")
            
            # Step 4: Match images with posts
            training_pairs = self.find_matching_images(posts_data)
            if not training_pairs:
                raise ValueError("No training pairs created")
            
            # Step 5: Create training dataset
            dataset_dir = self.create_training_dataset(training_pairs)
            
            # Step 6: Create sample metadata for reference
            self.create_sample_metadata()
            
            # Step 7: Cleanup
            self.cleanup_temp()
            
            logger.info("Preprocessing completed successfully!")
            logger.info(f"Dataset ready for training at: {dataset_dir}")
            
            return dataset_dir
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            self.cleanup_temp()
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Preprocess Instagram dataset for LFM2 training")
    
    parser.add_argument("--zip-path", type=str, required=True,
                       help="Path to Instagram dataset zip file")
    parser.add_argument("--output-dir", type=str, default="./processed_dataset",
                       help="Output directory for processed dataset")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate zip file exists
    if not os.path.exists(args.zip_path):
        print(f"❌ Error: Zip file not found: {args.zip_path}")
        return 1
    
    try:
        # Run preprocessing
        preprocessor = InstagramDatasetPreprocessor(args.zip_path, args.output_dir)
        dataset_dir = preprocessor.run_preprocessing()
        
        print(f"\n🎉 Preprocessing completed successfully!")
        print(f"📁 Dataset location: {dataset_dir}")
        print(f"📊 Total posts: {len(list((dataset_dir / 'metadata.json').parent.glob('*.jpg')))}")
        print(f"\n🚀 Ready for training! Use:")
        print(f"python train_lfm2_instagram_trainer.py --data-dir {dataset_dir} --output-dir ./trained_model")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
