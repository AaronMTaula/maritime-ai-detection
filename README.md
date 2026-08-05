**# Maritime AI Detection System**

**### Auckland University of Technology | Navantia Australia Collaboration Project**



**## Overview**

**A YOLOv8-based computer vision system for detecting, classifying, and estimating the position and heading of maritime vessels from a live camera feed.**



**## Project Status**

**Currently in active development — Phase 1 of 9 complete.**



**## Current Model Performance (v2)**

**| Metric | Result |**

**|---|---|**

**| Precision | 0.894 |**

**| Recall | 0.90 |**

**| mAP50 | 0.952 |**



**## Tech Stack**

**- YOLOv8m (Ultralytics)**

**- OpenVINO INT8 (Intel inference acceleration)**

**- OpenCV (webcam and image processing)**

**- CVAT (annotation)**

**- Python 3.11**



**## Team**

**- Aaron Taula — Software Developer / Tech Lead**

**- Naz Janif — Team Lead / Software Developer**

**- Tyler Halliday — Cybersecurity and Camera Specialist**

**- Zhiyuan Zhang — AI Lead**



**## Supervisor**

**Kien Tran — Auckland University of Technology**



**## Setup**

**```bash**

**python -m venv venv**

**venv\\Scripts\\activate**

**pip install ultralytics opencv-python openvino**

**```**



**## Live Detection**

**```bash**

**python live\_detect.py**

**```**

**Press \*\*Q\*\* to quit | Press \*\*S\*\* to save screenshot**

## Improving performance
The current model is trained on a relatively small dataset, so extra training can help if you add more varied examples. The fastest path is:

1. Collect more ferry images with different lighting, weather, angles, and distances.
2. Retrain using the provided script:

```bash
python train_ferry_model.py --epochs 80 --imgsz 640 --batch 8 --device cpu
```

3. Keep the live detector using a slightly higher resolution and confidence threshold if you want fewer false positives.

