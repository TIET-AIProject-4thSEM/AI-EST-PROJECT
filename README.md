# 🏭 Tata Steel — Surface Crack Detection (CNN + Explainable AI)

A deep learning–based system for detecting **micro-level surface cracks in steel plates** using high-resolution images.
This project uses **transfer learning, data augmentation, and explainable AI (Grad-CAM)** to build a robust and interpretable inspection model.

---

## 🚀 Overview

This project solves a key industrial problem:

> Automating quality inspection of steel surfaces to detect microscopic cracks.

We build a **binary image classification system**:

* **CRACK** → defect present
* **NO_CRACK** → defect absent

---

## ⚠️ Key Challenge

The dataset (**NEU-DET**) contains **only defect images**, not clean steel images.

### ❌ Problem

Naively:

* CRACK → full image
* NO_CRACK → crop from same image

👉 Model learns shortcuts (context bias), not real cracks.

---

## ✅ Solution (Core Contribution)

We redesigned dataset preparation:

### 🔬 Patch-Based Learning

* CRACK → **patch around defect bounding box**
* NO_CRACK → **patch from safe region outside defect**

### 🔒 Leakage Prevention

* Split done at **source image level**
* No patches from same image across train/val/test

👉 Forces model to learn **true defect features**

---

## 📁 Dataset Pipeline

### Input (NEU-DET)

```
data/NEU-DET/
  train/
    images/
    annotations/
  validation/
    images/
    annotations/
```

### Output

```
data/steel_binary/
  train/
    CRACK/
    NO_CRACK/
  val/
    CRACK/
    NO_CRACK/
  test/
    CRACK/
    NO_CRACK/
```

---

## ⚙️ Dataset Preparation

Run:

```bash
python prepare_data_final.py \
    --neu_dir data/NEU-DET \
    --output_dir data/steel_binary
```

### What it does:

* Parses XML annotations
* Extracts defect bounding boxes
* Generates:

  * CRACK patches
  * NO_CRACK safe-zone patches
* Splits dataset at image level
* Saves visualizations:

  * `sample_grid.png`
  * `bbox_crop_illustration.png`

---

## 🧠 Model Architecture

* Backbone: **EfficientNet-B3 (ImageNet pretrained)**
* Custom head:

  * GlobalAvgPool
  * Dropout + BatchNorm
  * FC layers → 2 classes

---

## 🏋️ Training Strategy

Three-phase transfer learning:

| Phase   | Description                             |
| ------- | --------------------------------------- |
| Phase 1 | Train classifier head (backbone frozen) |
| Phase 2 | Unfreeze top 30% of backbone            |
| Phase 3 | Full fine-tuning                        |

---

## 🔁 Data Augmentation

Using **Albumentations**:

* Flip, rotation
* Brightness/contrast
* CLAHE (enhances micro-cracks)
* Gaussian noise
* Elastic distortions
* CutOut (regularization)

---

## 📊 Training

Run:

```bash
python crack_detector.py \
    --data_dir data/steel_binary \
    --output_dir outputs
```

---

## 📈 Results

### Final Test Performance

| Metric   | Value      |
| -------- | ---------- |
| Accuracy | **96%**    |
| F1 Score | **0.96**   |
| AUC-ROC  | **0.9903** |

### Class-wise:

| Class    | Precision | Recall |
| -------- | --------- | ------ |
| NO_CRACK | 0.95      | 0.96   |
| CRACK    | 0.97      | 0.96   |

---

## 🧠 Explainability (Grad-CAM)

* Highlights regions influencing predictions
* Confirms model focuses on:

  * crack edges
  * defect textures

Output:

```
outputs/gradcam_grid.png
```

---

## 📦 Outputs

```
outputs/
  best_model.pth
  last_model.pth
  training_curves.png
  confusion_matrix.png
  roc_curve.png
  gradcam_grid.png
  metrics_history.json
  model_card.txt
```

---

## 🧪 Inference

```python
from crack_detector import CrackInferencer

infer = CrackInferencer("outputs/best_model.pth")
label, confidence, cam = infer.predict("image.jpg", return_gradcam=True)
```

---

## 🧠 Key Learnings

* Dataset design is **more important than model choice**
* Preventing leakage is critical
* Patch-based learning improves realism
* Explainability builds trust in industrial AI

---

## ⚡ Future Improvements

* Add real **clean steel images**
* Use **object detection (YOLO)** instead of classification
* Try **ConvNeXt / EfficientNet-B5**
* Deploy for real-time inspection

---

## 👨‍💻 Author

Aarav Singhal
Computer Science Engineering — Thapar University

---

## 📄 License

For academic use only.

