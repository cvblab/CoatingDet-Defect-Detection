# CoatingDet: High-Resolution Images for Coating Defect Detection in Wind Turbine Structures

This repository supports the research work **_“CoatingDet: High-Resolution Images for Coating Defect Detection in Wind Turbine Structures”_**, providing all tools necessary to **reproduce and extend defect detection** on coating images.

---

## 1. Installation & Dependencies

**Requirements**  
- Python **≥ 3.8**

Install required dependencies:

    pip install -r requirements.txt

## 2. Preparing the Dataset Structure

    python tile_creator.py

## 3. Training the model

To train the model, run:

    yolo task=detect mode=train model=yolo11n.pt data=data.yaml epochs=20 name=tiles_yolo11n imgsz=512 batch=8
    yolo task=detect mode=train model=rtdetr-l.pt data=data.yaml epochs=20 name=tiles_rtdetr-l imgsz=512 batch=8


## 4. Making predictions

    python evaluation.py

