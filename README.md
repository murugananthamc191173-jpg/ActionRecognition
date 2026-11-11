# An Integrated Human Action Recognition and Object Detection Framework for Intelligent Industrial Assembly Lines
Modern industrial assembly lines increasingly rely on vision-based monitoring systems to improve worker safety, process quality, and human robot collaboration. However, conventional vision systems often recognize worker motion without adequately understanding the associated tools or components involved in each activity, limiting their effectiveness in complex manufacturing environments characterized by occlusions, viewpoint variations, and background clutter. This paper addresses this limitation by proposing an integrated framework that combines human action recognition with object detection for comprehensive real-time monitoring of industrial assembly lines. The proposed framework integrates two state-of-the-art human action recognition architectures, ResNet3D-18 and SlowFast, with two variants of the YOLOv8 object detection model (YOLOv8n and YOLOv8m). The framework was evaluated using both the publicly available In-HARD benchmark dataset and a collected industrial assembly-line dataset to assess its performance under controlled and real-world manufacturing conditions. Experimental results demonstrate that the SlowFast model achieved an action recognition accuracy of 98.21\% on the In-HARD benchmark, while ResNet3D-18 exhibited superior robustness on the collected industrial dataset with an accuracy of 97.63\%. The YOLOv8 object detection framework achieved a mean Average Precision of 0.987 (mAP$_{0.5}$), accurately identifying industrial tools and components under challenging operating conditions. By simultaneously recognizing worker activities and the associated tools or components, the proposed framework provides a unified understanding of manufacturing operations that neither action recognition nor object detection can achieve independently. The integrated system enables real-time monitoring of human tool interactions, offering valuable contextual information for intelligent process supervision, safety assurance, quality inspection, and future human robot collaboration. These findings demonstrate the practical applicability of the proposed framework as a vision-based solution for intelligent industrial assembly lines and contribute toward the advancement of smart manufacturing systems.
# Project Structure: 
* **InHARD Dataset Experiments (Benchmark Dataset)**
* **`InHARD_R3D18_Train`**: Fine-tunes a 3D ResNet-18 (`R3D_18`) model using 16-frame video clips from the InHARD industrial dataset (`/nfsshare/knr/Havid/RGBSeg/`) for multi-class action classification.
* **`InHARD_R3D18_Test`**: Real-time action recognition and inference script for unsegmented test videos using the trained `r3d18` model, featuring frame buffering (`deque`) and OpenCV display overlays.
* **`InHARD_SlowFast_Train`**: Trains a PyTorchVideo SlowFast network utilizing dual pathways (Slow: 8 frames, Fast: 32 frames) on the InHARD dataset.
* **`InHARD_SlowFast_Test`**: Inference script for SlowFast models on test videos, processing real-time dual-tensor inputs (slow and fast frame samples).
* **Custom KNR Dataset Experiments**
* **`KNR_R3D18_Train`**: Trains an `R3D_18` model initialized with Kinetics-400 weights on custom KNR video dataset classes (`Checking`, `Missing`, `Picking`, `Rejected`, `Accepted`).
* **`KNR_R3D18_Test`**: Real-time action recognition testing script on unsegmented videos using the trained KNR `R3D_18` weights (`tab_r3d18.pth`).
* **`KNR_SlowFast_Train`**: Fine-tunes a PyTorchVideo SlowFast network on the KNR action recognition dataset.
* **`KNR_SlowFast_Test`**: Live video inference script for KNR action detection using the SlowFast backbone (`tab_slowfast.pth`).
* **TVS Dataset Experiments (Assembly Line Video/Frame Dataset)**
* **`TVS_R3D18_Train`**: Customized frame-based PyTorch dataset loader and `R3D_18` training pipeline for TVS assembly action classes (`Fixingbackscrew`, `Fixing frame`, `Fixingfrontscrew`, etc.) stored as image frames (`*.png`).
* **`TVS_R3D18_Test`**: Video inference script for TVS assembly line action recognition with a custom bottom-bar status overlay.
* **`TVS_SlowFast_Train`**: SlowFast dual-pathway dataset loader and trainer operating directly on TVS image sequence folders.
* **`TVS_SlowFast_Test`**: Test script evaluating TVS assembly line operations with the trained SlowFast architecture (`tvs_slowfast.pth`).
* **Live Webcam / Real-Time Deployment**
* **`Live_Webcam_Action_Recognition`**: Real-time webcam stream integration (`cv2.VideoCapture(1)`) using a 16-frame rolling buffer for live action recognition via the trained `R3D_18` model.
### Features
* Detects and classifies temporal human actions in industrial assembly lines using deep 3D networks (**R3D-18** and **SlowFast**).
* Supports multiple benchmark and custom datasets: **InHARD**, **KNR**, and **TVS Assembly Line**.
* Provides real-time unsegmented video inference with frame buffering (`deque`) and live visual overlays.
* Performs end-to-end video processing from raw `.mp4` video files to sequence-based `.png` frame folders.
* Real-time webcam integration for live action detection and cycle monitoring.
### Dataset
* **InHARD Dataset**: 14 assembly actions (e.g., *Assemble system*, *Take screwdriver*, *Consult sheets*).
* **KNR Dataset**: 5 operational classes (*Checking*, *Missing*, *Picking*, *Rejected*, *Accepted*).
* **TVS Dataset**: 6 frame-based assembly actions (*Fixingbackscrew*, *Fixing frame*, *Fixingfrontscrew*, *Fixingmiddlescrew*, *Gettingtheframe*, *Pickingscrews*).
* Configurable input pipelines handling both multi-channel video files (`.mp4`) and ordered image sequence folders (`.png`).
### Model Training
* **R3D-18 Backbone**: 3D ResNet initialized with Kinetics-400 pretrained weights, processing 16-frame temporal windows resized to $112 \times 112$.
* **SlowFast Backbone**: Dual-pathway architecture (*Slow*: 8 frames for spatial detail; *Fast*: 32 frames for motion dynamic) using $224 \times 224$ resolutions.
* Optimization with Cross-Entropy Loss and Adam optimizer across 25 epochs.
* Automatic model serialization, saving weight checkpoints (`.pth`).

### Getting Started
1. **Install dependencies**:
```bash
pip install torch torchvision pytorchvideo opencv-python numpy scikit-learn

```
2. **Train a model** (e.g., R3D-18 on KNR dataset):
Execute the corresponding training section in `Code.ipynb` or run:
```bash
python train_r3d18_knr.py

```
3. **Run unsegmented video inference**:
```bash
python test_r3d18_video.py --video_path /path/to/video.mp4 --model_path /path/to/model.pth

```
4. **Run live webcam action recognition**:
```bash
python test_live_webcam.py

```
### Usage
* Monitor assembly worker actions in real time via live webcam or pre-recorded video feeds.
* Track real-time processing performance (FPS) and predicted temporal actions on the visual overlay.
* Train, compare, and benchmark 3D-CNN (`R3D-18`) against dual-pathway architectures (`SlowFast`).
### Requirements
* **Python 3.9+**
* PyTorch & Torchvision (CUDA-enabled recommended)
* PyTorchVideo
* OpenCV (`opencv-python`)
* NumPy & Scikit-Learn
### Licence
* This project is for internal use and research purposes.
