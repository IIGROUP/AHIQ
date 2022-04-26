# Attention Helps CNN See Better: Hybrid Image Quality Assessment Network
[CVPRW 2022] Code for Hybrid Image Quality Assessment Network

[[Paper on arXiv]](https://arxiv.org/abs/2104.11599)

## Overview
<p align="center"> <img src="Figures/architecture.png" width="100%"> </p>

## Getting Started

### Prerequisites
- Linux
- NVIDIA GPU + CUDA CuDNN
- Python 3.7

### Dependencies:

We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). All dependencies for defining the environment are provided in `requirements.txt`.

### Pretrained Models
The pretrained models can be found at this 
[URL](https://drive.google.com/drive/folders/1-8LKOEDYt-RzmM9IDV_oW73uRBLqeRB6?usp=sharing). And you should put them into `checkpoints/ahiq_pipal/`

### Instruction
use `sh train.sh` or `sh test.sh` to train or test the model. You can also change the options in the `options/` as you like.

## Acknowledgment
The codes borrow heavily from IQT implemented by [anse3832](https://github.com/anse3832/IQT) and we really appreciate it.

## Citation
If you find our work or code helpful for your research, please consider to cite:
```bibtex
@misc{https://doi.org/10.48550/arxiv.2204.10485,
  doi = {10.48550/ARXIV.2204.10485},
  
  url = {https://arxiv.org/abs/2204.10485},
  
  author = {Lao, Shanshan and Gong, Yuan and Shi, Shuwei and Yang, Sidi and Wu, Tianhe and Wang, Jiahao and Xia, Weihao and Yang, Yujiu},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Attentions Help CNNs See Better: Attention-based Hybrid Image Quality Assessment Network},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```