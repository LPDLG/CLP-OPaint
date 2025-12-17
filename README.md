# Project Profile
CLP-OPaint: Progressive outpainting of Chinese landscape paintings with structure-texture decoupling.

# CLP-OPaint
Outpainting Chinese landscape paintings requires simultaneously preserving macroscopic structure and microscopic brushwork, a cross-scale challenge where existing methods often fail. To address this, we propose CLP-OPaint, a novel progressive framework tailored for traditional art restoration. We introduce a Three-Stage Progressive Training Strategy to guide the model from coarse layout to fine texture via curriculum learning. Furthermore, a Dual-Branch Decoder is designed to explicitly decouple structure and texture generation, while a U-Net Patch Smoothing Module (UPSM) eliminates boundary artifacts. Extensive experiments on our curated CLP-9k dataset and public benchmarks demonstrate that CLP-OPaint significantly outperforms state-of-the-art methods in both visual quality and quantitative metrics.

## Code
### Requirements
PyTorch >= 1.11.1;
python >= 3.8;
CUDA >= 11.6;
torchvision;

We will upload the training as well as the test code at an oppropriate time!
