# nsfw-sfw_image_sort

Project to sort nsfw and sfw photos and images in different folders, as well as photos of different people in different folders, using local (offline) image classification models.

Work in Progress

## Environment (recommended)

- Python 3.12 (compatible with both Tensorflow and onnxruntime as of 2026-10-21)
- docker build from: `nvidia/cuda:13.4.3-cudnn-devel-ubuntu26.04` for GPU
- usability dependencies for computer vision AI
    - ffmpeg
    - g++-12
    - gcc-12
    - libgl1
    - libgomp1
    - libopencv-dev
    - libprotobuf-dev protobuf-compiler
    - libsm6
    - libxext6
- dependencies for pyenv
    - https://github.com/pyenv/pyenv/wiki#troubleshooting--faq