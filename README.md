
# UAE Construction Helmet Violation Detection – YOLOv8

Real-time "NO HELMET = RED ALERT" system for UAE construction sites.

Detects workers without helmets and draws big red boxes + "NO HELMET!" warning.

Trained on 17,500+ images (Hard Hat Workers dataset)

**mAP@50: 0.959** | **mAP@50:95: 0.762** (8 epochs only!)

## Demo
![NO HELMET ALERT](inference_examples/no_helmet_alert.jpg)

## Results (8 epochs – YOLOv8m)
| Class   | Precision | Recall | mAP@50 | mAP@50:95 |
|---------|---------|--------|--------|-----------|
| all     | 0.918   | 0.893  | 0.959  | 0.762     |
| head    | 0.93    | 0.91   | 0.97   | 0.79      |
| helmet  | 0.95    | 0.92   | 0.98   | 0.81      |
| person  | 0.87    | 0.85   | 0.93   | 0.69      |

## Quick Inference
```python
model = YOLO("weights/best.pt")
results = model("your_image.jpg", conf=0.4)
