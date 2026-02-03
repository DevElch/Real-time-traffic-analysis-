# YOLOv8 Vehicle Tracker & Environmental Analyzer

A real-time vehicle detection and tracking project using **YOLOv8** and **OpenCV**. This software not only detects and tracks vehicles with trajectories but also performs heuristic analysis of weather conditions and location types.

## 🌟 Features

* **Real-time Detection:** High-accuracy vehicle detection using the YOLOv8 model.
* **ID-based Tracking:** Each vehicle is assigned a unique ID for consistent monitoring.
* **Interactive Trajectory:** Click on any vehicle with your mouse to visualize its path and movement direction.
* **Environment Intelligence:** * **Weather:** Automatically classifies conditions as Sunny, Cloudy, or Rainy.
* **Location:** Heuristically determines if the scene is a City or Highway based on edge density.


* **Customizable Visuals:** Color-coded bounding boxes for different classes (Car, Truck, Bus, etc.).

---

## ⚖️ Ethical Statement & Intent

This project is developed and shared with the sole intention of **contributing to civil society** and advancing research in smart city infrastructure.

* **Legal & Ethical Use:** Users are expected to utilize this software within the boundaries of local laws and ethical standards.
* **Civilian Context:** It is designed for civilian applications, such as traffic flow analysis, urban planning, and road safety research.
* **Privacy:** Please ensure compliance with data protection regulations (like GDPR) when processing public video feeds.

---

## 🚀 Optimization & Hardware Support

The current version is a base implementation designed for experimentation. You can significantly enhance performance based on your hardware:

* **CPU Optimization:** If you are using **Intel** or **AMD** processors, you can optimize the code using libraries like `OpenVINO` or `ONNX Runtime` to achieve higher frame rates.
* **GPU Acceleration:** For real-time high-resolution processing, it is highly recommended to run this on an **NVIDIA GPU** using `CUDA` and `TensorRT`.
* **Customization:** The code is open-source and modular; feel free to adjust the `CONF_THRES` and `DIST_THRESH` parameters to fit your specific use case.





