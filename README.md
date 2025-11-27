# SWIN Transformer for Image Captioning

An end-to-end transformer-based model for automatic image caption generation using SWIN Transformer as the backbone encoder with novel refining encoders and a transformer decoder.

## Overview

This project implements an image captioning system based on the paper **"End-to-End Transformer Based Model for Image Captioning"** by Yiyu Wang, Jungang Xu, and Yingfei Sun ([arXiv:2203.15350](https://arxiv.org/abs/2203.15350)). The model uses a SWIN Transformer as the visual encoder, followed by refining encoder layers and a transformer decoder to generate natural language descriptions of images.

## Architecture

### Model Components

1. **SWIN Transformer Encoder**: 
   - Hierarchical vision transformer with shifted window attention
   - Processes 384×384 input images
   - Extracts multi-scale visual features through patch merging

2. **Refining Encoder Layers**:
   - Novel attention mechanism that combines local patch features with global image representation
   - Cross-attention between windowed patches and global average pooled features
   - Enhances feature representation for caption generation

3. **Transformer Decoder**:
   - Multi-head attention layers with look-ahead masking
   - Cross-attention with encoder features
   - Prefusion mechanism combining decoder embeddings with global visual features

4. **Position Embedding**:
   - Fixed positional encodings for sequence modeling
   - Supports variable sequence lengths up to 39 tokens

## Features

- **Hierarchical Visual Processing**: SWIN Transformer's shifted window attention for efficient image encoding
- **Global-Local Feature Fusion**: Refining layers that combine patch-level and image-level features
- **Advanced Attention Mechanisms**: Multi-head attention with relative position bias
- **Flexible Architecture**: Configurable model parameters (heads, layers, dimensions)
- **Efficient Training**: Gradient accumulation and learning rate scheduling

## Requirements

### Dependencies

```
tensorflow>=2.8.0
tensorflow-probability
numpy
pandas
matplotlib
scikit-learn
Pillow
keras
tqdm
h5py
pickle
```

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: At least 8GB RAM, 4GB+ VRAM for training
- **Storage**: ~2GB for Flickr8k dataset

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SWIN-Transformer-for-Image-Captioning.git
cd SWIN-Transformer-for-Image-Captioning
```

2. Install dependencies:
```bash
pip install tensorflow tensorflow-probability numpy pandas matplotlib scikit-learn Pillow tqdm h5py
```

3. Download the Flickr8k dataset:
   - Images: Place in `Dataset/Flicker8k_Dataset/`
   - Captions: Place `Flickr8k.token.txt` in `Dataset/Flickr8k_text/`

## Usage

### Training

1. **Prepare the dataset**:
```python
python dataloader.py
```
This will preprocess images and captions, create tokenizers, and generate training/validation splits.

2. **Train the model**:
```python
python engine.py
```

### Model Parameters

The model uses the following default configuration:
- **Image size**: 384×384 pixels
- **Patch size**: 4×4
- **Embed dimension**: 64
- **SWIN depths**: [2, 2, 6, 2]
- **Attention heads**: [4, 8, 16, 32]
- **Window size**: 6
- **Decoder layers**: 6
- **Vocabulary size**: 8918 tokens
- **Max sequence length**: 39 tokens

### Inference

```python
from inference import Translate, load_image
import tensorflow as tf

# Load trained model
model = TransformerModel(...)  # Configure with same parameters as training
model.load_weights('./checkpoints/my_checkpoint')

# Create translator
translator = Translate(model)

# Generate caption for an image
image_path = 'path/to/your/image.jpg'
image, _ = load_image(image_path)
image = tf.expand_dims(image, axis=0)
image = tf.transpose(image, perm=[0,3,1,2])

caption = translator(image)
print("Generated caption:", ' '.join(caption))
```

## File Structure

```
├── README.md                 # Project documentation
├── LICENSE                   # MIT License
├── engine.py                # Training script with loss functions and optimization
├── transformer.py           # Main transformer model architecture
├── encoder.py               # Encoder components (AddNorm, FeedForward)
├── decoder.py               # Decoder implementation with attention layers
├── SWINblock.py            # SWIN Transformer implementation
├── multi_head_attention.py # Multi-head attention mechanisms
├── position_embedding.py   # Positional encoding layers
├── prefusion.py            # Prefusion layer for feature combination
├── dataloader.py           # Data preprocessing and loading utilities
├── inference.py            # Model inference and caption generation
├── plotting.py             # Visualization utilities
├── test.py                 # Testing and evaluation scripts
└── dec_tokenizer.pkl       # Saved tokenizer for decoding
```

## Model Architecture Details

### SWIN Transformer Encoder
- **Patch Embedding**: Converts 384×384×3 images to 96×96×64 patch tokens
- **4 Stages**: Progressive downsampling with patch merging
- **Window Attention**: 6×6 windows with shifted window mechanism
- **Relative Position Bias**: Learnable relative position encodings

### Refining Encoder
- **3 Refining Layers**: Additional processing after SWIN backbone
- **Global-Local Attention**: Combines patch features with global average pooling
- **Cross-Attention**: Enhanced feature representation for decoder

### Transformer Decoder
- **6 Decoder Layers**: Multi-head self-attention and cross-attention
- **Prefusion**: Combines word embeddings with global visual features
- **Masked Attention**: Prevents information leakage during training

## Training Details

### Loss Function
- **Sparse Categorical Cross-Entropy**: With padding mask to ignore zero tokens
- **Accuracy Metric**: Token-level accuracy excluding padding

### Optimization
- **Adam Optimizer**: With custom learning rate scheduling
- **Learning Rate Schedule**: Warmup + decay (d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5))
- **Batch Size**: 1 (configurable)
- **Epochs**: 10 (configurable)

### Data Preprocessing
- **Image Preprocessing**: Resize to 384×384, standardization, VGG16 preprocessing
- **Caption Preprocessing**: Tokenization, padding, start/end tokens
- **Vocabulary**: Top 5000 most frequent words
- **Data Split**: 80% train, 10% validation, 10% test

## Results and Performance

The model generates contextually relevant captions for input images by:
1. Extracting hierarchical visual features using SWIN Transformer
2. Refining features through global-local attention mechanisms
3. Generating sequential captions using transformer decoder with visual attention

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{wang2022end,
  title={End-to-End Transformer Based Model for Image Captioning},
  author={Wang, Yiyu and Xu, Jungang and Sun, Yingfei},
  journal={arXiv preprint arXiv:2203.15350},
  year={2022}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Original paper authors: Yiyu Wang, Jungang Xu, Yingfei Sun
- SWIN Transformer architecture by Microsoft Research
- Flickr8k dataset for training and evaluation
- TensorFlow team for the deep learning framework

## Contact

For questions or issues, please open an issue on GitHub or contact the repository maintainer.