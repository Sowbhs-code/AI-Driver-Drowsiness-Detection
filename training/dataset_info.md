# Dataset Information

## Datasets Used

### 1. Eye State Dataset (for EAR / CNN training)
| Dataset | Link | Classes |
|---------|------|---------|
| MRL Eye Dataset | https://mrl.cs.vsb.cz/eyedataset | Open, Closed |
| CEW Dataset | https://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html | Open, Closed |
| Drowsiness Detection Dataset | https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset | open, closed, yawn, no_yawn |

### 2. Phone Detection Dataset (for YOLOv8)
| Dataset | Link | Format |
|---------|------|--------|
| Roboflow Phone Detection | https://universe.roboflow.com/roboflow-100/cell-phones-detection | YOLOv8 |
| Mobile Phone Kaggle | https://www.kaggle.com/datasets/dataclusterlabs/mobile-phone-dataset | YOLO |

### 3. Driver Drowsiness Video Dataset
| Dataset | Link |
|---------|------|
| NTHU Drowsy Driver Dataset | http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/ |
| UTA RLDD | https://sites.google.com/view/utarldd/home |

## Directory Structure Expected
```
dataset/
├── eye_dataset/
│   ├── open/          ← eye-open images (.jpg/.png)
│   └── closed/        ← eye-closed images
├── phone_data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/     ← YOLO .txt label files
│   │   └── val/
│   └── data.yaml
└── videos/            ← optional test videos
```

## Download Instructions

### Kaggle Drowsiness Dataset (quickest)
```bash
pip install kaggle
kaggle datasets download -d dheerajperumandla/drowsiness-dataset
unzip drowsiness-dataset.zip -d dataset/eye_dataset
```

### Roboflow Phone Dataset
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("roboflow-100").project("cell-phones-detection")
dataset = project.version(1).download("yolov8", location="dataset/phone_data")
```
