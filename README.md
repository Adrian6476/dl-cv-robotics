# Deep Learning for Computer Vision and Robotics

This repository contains the code and resources for the Deep Learning for Computer Vision and Robotics (COMP52715) module at Durham University. The main goal of this project is to develop a sophisticated object detection system that can navigate and capture spherical targets within a 3D point cloud environment.

## Project Overview

The project comprises several essential components, including:

1. **Optimisation of PacMan Helper**: The provided `PacMan_Helper.py` code was optimised for computational efficiency, particularly the `points_to_image` function, using Numba's just-in-time compilation and parallelisation capabilities.

2. **Simple CNN Binary Model**: A Convolutional Neural Network (CNN) was implemented as a baseline approach for binary classification of image patches, distinguishing between target objects and background.

3. **Advanced Object Detection Model**: An advanced object detection model was developed using the YOLOv8 framework, demonstrating accurate and efficient detection of spherical targets within the 3D point cloud environment.

4. **Navigation and Collection**: The trained YOLOv8 model was integrated into a detection-navigation loop to capture all the target objects in the 3D point cloud environment, showcasing the system's robustness and efficiency.

5. **Robot Design**: A conceptual design for a robotic system based on the Kuka YouBot platform was proposed, incorporating the necessary mechanical configurations, sensor suite, algorithms, and control systems to undertake the navigation and capture tasks in a real-world scenario.

## Repository Structure

The repository is organised as follows:

```
dl-cv-robotics/
├── Benchmark.ipynb
├── Coursework Specification.pdf
├── lib/
│   ├── PacMan_Helper_Accelerated.py
│   ├── PacMan_Helper.py
│   └── __pycache__/
├── models/
│   ├── CNN.onnx
│   ├── CNN.pth
│   ├── YOLO.engine
│   ├── YOLO.onnx
│   ├── YOLO.pt
│   ├── YOLO.torchscript
│   ├── yolov8m.pt
│   └── yolov8n.pt
├── PacMan.ipynb
├── prof/
│   ├── profiling_results_PacMan_Helper_Accelerated.prof
│   └── profiling_results_PacMan_Helper.prof
├── README.md
├── Rubric.pdf
├── runs/
│   ├── detect/
│   │   ├── train7/
│   │   ├── val/
│   │   ├── val2/
│   │   ├── val3/
│   │   └── val4/
│   └── test/
│       └── train7/
└── training_data/
    ├── img/
    │   ├── negatives/
    │   ├── object_DNN/
    │   └── positives/
    ├── npy/
    │   ├── cloudColors.npy
    │   ├── cloudPositions.npy
    │   ├── final_world_colors.npy
    │   └── final_world_positions.npy
    └── YOLO/
        ├── PacMan.yaml
        ├── test/
        ├── train/
        ├── train.cache/
        ├── val/
        └── val.cache/
```

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/dl-cv-robotics.git`
2. Install the required dependencies (e.g., PyTorch, Numba, OpenCV, Open3D)
3. Navigate to the project directory: `cd dl-cv-robotics`
4. Explore the Jupyter Notebooks (`PacMan.ipynb`, `Benchmark.ipynb`) for code implementation and experimentation.
5. Check the `models/` directory for the trained models and their respective files.
6. Refer to the `training_data/` directory for the datasets used in training and testing.
7. Review the `Coursework Specification.pdf` and `Rubric.pdf` for project requirements and evaluation criteria.

## Usage

The main entry point for the project is the `PacMan.ipynb` Jupyter Notebook. This notebook contains the code for implementing the various components of the project, including data loading, model training, object detection, navigation, and visualisation.

To run the project, follow these steps:

1. Open the `PacMan.ipynb` notebook in a Jupyter environment.
2. Execute the cells in the notebook sequentially, following the provided instructions and comments.
3. The notebook will guide you through the optimisation of the PacMan Helper, training and evaluation of the CNN binary model, training and deployment of the advanced YOLOv8 object detection model, and the navigation and collection loop.
4. Visualisations and performance metrics will be displayed within the notebook.
