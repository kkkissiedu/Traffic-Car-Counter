# 🚗 Real-Time Vehicle Detection, Tracking, and Counting System

This project implements a real-time computer vision pipeline for **vehicle detection**, **tracking**, and **counting** from video streams. It uses **YOLOv8** for detection and the **SORT** algorithm for robust object tracking, providing accurate and scalable traffic analysis data.

> ✅ A powerful alternative to manual counting or expensive in-ground sensors, with applications in transportation engineering, urban planning, and infrastructure management.

---

## 🔑 Key Features

- ⚡ **Real-Time Object Detection**  
  Uses **YOLOv8** to detect multiple vehicle types such as:
  - 🚗 Car  
  - 🚚 Truck  
  - 🚌 Bus

- 🛰️ **Multi-Object Tracking**  
  Tracks each vehicle using the **SORT** algorithm, assigning unique IDs and maintaining identity even during brief occlusions.

- 🧮 **Virtual Line Counter**  
  Counts vehicles crossing a defined virtual line with **99% accuracy** in test videos.

- 🧾 **Multi-Class Counting**  
  Maintains separate counts for each vehicle class, enabling more detailed traffic analysis.

---

---

## Demonstration
![GIF](output_cars.gif)
---

## 🧰 Tech Stack & Implementation

| Component              | Technology                        |
|------------------------|------------------------------------|
| Object Detection       | YOLOv8 (Ultralytics)               |
| Object Tracking        | SORT (Simple Online Realtime Tracking) |
| Framework              | PyTorch                            |
| Core Libraries         | OpenCV, Ultralytics, cvzone        |

---

## 📈 Performance Metrics

### ✅ Counting Accuracy

- Achieved **99% accuracy** in vehicle counting on test video footage compared to manual annotation.
---

This script will:
- Detect vehicles
- Assign tracking IDs
- Count vehicles
