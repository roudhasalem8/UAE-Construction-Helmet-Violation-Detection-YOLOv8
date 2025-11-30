
# UAE Construction Helmet Violation Detection – YOLOv8s

Real-time **“NO HELMET = RED ALERT”** safety system for UAE construction sites.

Detects heads + helmets and flags uncovered heads with a **big red “NO HELMET!”** warning.

Trained on **17k+ images** (Hard Hat Workers dataset).

**mAP@50: 0.392** | **mAP@50–95: 0.247** | **F1-score: 0.400** (YOLOv8s, 30 epochs)

---

## Demo
![NO HELMET ALERT]

---

## Results (YOLOv8s – 30 epochs)

| Metric    | Value |
|-----------|--------|
| mAP@50    | 0.392 |
| mAP@50–95 | 0.247 |
| Precision | 0.351 |
| Recall    | 0.466 |
| F1-score  | 0.400 |

---

## Quick Inference

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
results = model("your_image.jpg", conf=0.4)

