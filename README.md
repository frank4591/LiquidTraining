# LFM2-VL-1.6B Vision Language Model

This folder contains scripts and tools for using the **LFM2-VL-1.6B** vision language model from Liquid AI. The model is designed for processing text and images with variable resolutions, optimized for low-latency and edge AI applications.

## 🚀 Features

- **2× faster inference speed** on GPUs compared to existing VLMs
- **Flexible architecture** with user-tunable speed-quality tradeoffs
- **Native resolution processing** up to 512×512 pixels
- **Lightweight**: Only 1.6B parameters
- **Multimodal**: Processes both text and images

## 📁 Files

- `save_lfm2_vl_model.py` - Downloads and saves the model locally
- `instagram_caption_generator.py` - Generates Instagram-like captions for images
- `test_setup.py` - Tests the setup and dependencies
- `requirements.txt` - Required Python packages
- `README.md` - This file

## 🛠️ Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Model

```bash
python save_lfm2_vl_model.py
```

This will download the ~1.6GB model to `./lfm2_vl_1_6b_model/`

### 3. Test the Setup

```bash
python test_setup.py
```

## 📸 Usage

### Generate Instagram Captions

```bash
# Basic usage with the provided image
python instagram_caption_generator.py --image ../img1.jpg

# Generate multiple captions with different styles
python instagram_caption_generator.py --image ../img1.jpg --style creative --num-captions 5

# Use a different output file
python instagram_caption_generator.py --image ../img1.jpg --output my_captions.txt
```

### Available Styles

- `instagram` - Trendy, relatable captions with hashtags
- `professional` - Business-appropriate descriptions
- `casual` - Friendly, conversational tone
- `creative` - Artistic, mood-capturing captions

### Command Line Options

```bash
python instagram_caption_generator.py --help
```

Options:
- `--image, -i` - Path to image file (required)
- `--style, -s` - Caption style (default: instagram)
- `--num-captions, -n` - Number of captions to generate (default: 3)
- `--output, -o` - Output file for captions (default: generated_captions.txt)
- `--model-path, -m` - Path to local model (default: ./lfm2_vl_1_6b_model)

## 🔧 Technical Details

### Model Architecture

- **Language Model**: LFM2-1.2B backbone
- **Vision Encoder**: SigLIP2 NaFlex shape-optimized (400M parameters)
- **Hybrid Backbone**: Combines convolution and attention layers
- **Context**: 32,768 text tokens
- **Image Tokens**: Dynamic, user-tunable
- **Precision**: bfloat16

### Performance

The model achieves competitive performance on various benchmarks:
- **RealWorldQA**: 65.23
- **MM-IFEval**: 37.66
- **InfoVQA**: 58.68
- **OCRBench**: 742
- **MMStar**: 49.53

### Memory Requirements

- **GPU Memory**: ~3-4GB for inference
- **Model Size**: ~1.6GB on disk
- **RAM**: ~2-3GB additional

## 💡 Example Output

For the provided `img1.jpg` image, the model might generate captions like:

```
📸 Caption 1:
Beautiful sunset vibes! 🌅 Nature never fails to amaze me. 
#sunset #nature #photography #beautiful #peaceful

📸 Caption 2:
When the sky paints itself in golden hour magic ✨ 
#goldenhour #sky #photography #nature #beauty

📸 Caption 3:
Sunset serenity - the perfect way to end the day 🌅
#serenity #sunset #peace #nature #photography
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `max_tokens` in the generation parameters
   - Use CPU if GPU memory is insufficient

2. **Model Not Found**
   - Ensure you've run `save_lfm2_vl_model.py` first
   - Check the model path in the script

3. **Dependencies Missing**
   - Install requirements: `pip install -r requirements.txt`
   - Ensure you have Python 3.8+ and PyTorch 2.0+

### Performance Tips

- Use `device_map="auto"` for automatic device placement
- Set `torch_dtype="bfloat16"` for memory efficiency
- Adjust `max_image_tokens` for speed/quality tradeoff

## 📚 References

- [LFM2-VL-1.6B Model Page](https://huggingface.co/LiquidAI/LFM2-VL-1.6B)
- [LFM2-VL Blog Post](https://huggingface.co/LiquidAI/LFM2-VL-1.6B)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

## 📄 License

This model uses the LFM Open License v1.0. Please review the license terms on the Hugging Face model page.

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the model documentation on Hugging Face
3. Test with `python test_setup.py` to verify your setup
