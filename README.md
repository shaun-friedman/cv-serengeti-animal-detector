# cv-serengeti-animal-detector

## Overview

**cv-serengeti-animal-detector** is a proof-of-concept / experimental project for detecting and classifying animals (and empty/blank frames) from camera-trap images, based on data from the Snapshot Serengeti dataset. The repository contains a pipeline of Jupyter notebooks and utility scripts for data preprocessing, label generation, bounding-box generation, model training (CNN / Transfer Learning / ViT), and end-to-end inference. The goal is to build a working classifier/detector that can identify wildlife species in camera-trap imagery.

## Contents

```
.
├── .gitignore
├── LICENSE
├── labels.csv               # original labels from dataset
├── labels_generated.csv     # auto-generated / processed labels
├── utils.py                 # helper functions / utilities
├── bounding_box_generator.ipynb  # notebook to generate bounding boxes for detected animals
├── Transfer_Learning_CNN.ipynb   # notebook showing transfer-learning approach using a CNN backbone
├── vit.ipynb                # notebook exploring a Vision Transformer (ViT)-based approach
├── pipeline_and_cnn.ipynb   # a combined data-preprocessing + CNN pipeline
├── final_project.ipynb      # final notebook with full workflow: preprocessing → training → evaluation → inference
└── README.md                # this file
```

## Features

- Data preprocessing of camera-trap images + label handling  
- Generation of bounding boxes (for images containing animals) to facilitate detection tasks  
- Multiple model training approaches:
  - Transfer-learning with a CNN backbone  
  - Vision Transformer (ViT) based approach  
- Modular code: helper utilities in `utils.py`, notebooks implementing different stages (preprocessing, bounding-box generation, training, inference)  
- Exportable labels and bounding boxes to support custom workflows or downstream projects  

## Getting Started

### Prerequisites  
Make sure you have a compatible Python environment. Recommended: Python 3.x, with packages such as `numpy`, `pandas`, `torch` or similar deep-learning framework, plus imaging libraries (e.g. `Pillow`, `opencv-python`) as needed depending on your workflow.

You can create a virtual environment, then install dependencies. For example:

```bash
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements.txt   # or manually install needed packages
```

### Usage

1. Prepare your dataset — place your camera-trap images and labels.csv in a directory structure as expected by the notebooks.
2. Run `bounding_box_generator.ipynb` to generate bounding boxes for animal images.
3. Use `final_project.ipynb` to execute the combined output of the modeling workflows.

## Potential Use Cases

* Wildlife monitoring, ecological research or camera-trap studies (e.g. for dataset cleaning, blank-frame removal, species classification)
* As a starting point / baseline for more advanced detectors (object detection, instance segmentation, species classification)
* Educational / learning tool for computer vision practitioners interested in real-world biodiversity / camera-trap data

## Limitations & Caveats

* This project is experimental / proof-of-concept; it may not be production-ready or robust to all camera-trap scenarios (lighting, occlusion, rare species, etc.)
* Bounding-box generation may be imperfect depending on the image quality or species detectability. Manual inspection / curation might be required for high-quality outputs.
* No rigorous validation metrics or benchmark comparison included — results depend heavily on dataset quality, training settings, and may exhibit class imbalance (common in ecological datasets).

## License

This project is distributed under the MIT License.