#!/bin/bash

# Install necessary packages
pip install fvcore omegaconf fairscale timm filterpy tk  install opencv-python cloudpickle pycocotools plotly

# Clone detectron2 and set up its checkpoint directory
git clone https://github.com/facebookresearch/detectron2 detectron
mkdir -p detectron/checkpoint
cd detectron/checkpoint
wget "https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/model_final_997cc7.pkl"
cd ../..

# Clone VideoPose3D and set up its checkpoint directory
git clone "https://github.com/facebookresearch/VideoPose3D"
mkdir -p VideoPose3D/checkpoint
cd VideoPose3D/checkpoint
wget "https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin"
wget "https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_humaneva15_detectron.bin"
cd ../..
pip install opencv-python cloudpickle pycocotools plotly