# RoadSense AI — Automated Road Damage Detection System

An intelligent computer vision system that automatically detects and classifies road damage from images and videos using a custom-trained YOLOv8 model.


## Overview

RoadSense AI analyzes road images/videos and detects **8 types of damage**:
- Potholes
- Alligator cracks
- Longitudinal cracks
- Lateral cracks
- Edge cracking
- Ravelling
- Rutting
- Striping (road markings)

The system provides **severity assessment**, **GPS location extraction**, **PDF report generation**, and **geospatial mapping**.

##  Features

-  **AI-Powered Detection**: Custom YOLOv8 model trained on real road damage data
-  **Image & Video Support**: Analyze both photos and dashcam footage
-  **GPS EXIF Extraction**: Automatically reads location data from images
-  **Severity Assessment**: Classifies damage as High/Medium/Low priority
-  **PDF Reports**: Professional inspection reports with damage counts
-  **Geospatial Mapping**: Dark-themed map visualization with GPS pins
-  **Professional UI**: Clean, modern dashboard design

## Tech Stack

- **AI/ML**: YOLOv8 (Ultralytics), PyTorch
- **Backend**: SQLite (database), OpenCV (image processing)
- **Frontend**: Streamlit, Folium (maps)
- **Deployment**: Streamlit Cloud
- **Languages**: Python

## Live Demo

**https://roaddamagedetection-6bbbuapp5oc2k34nkgugcvh.streamlit.app/**

## Installation 

### Prerequisites
- Python 3.11+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/road_damage_detection.git
   cd road_damage_detection
2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   # On Windows: venv\Scripts\activate
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
4. **Run the app**
   ```bash
   streamlit run cloud_app.py

##  Usage

Upload: Drag & drop a road image or video
Analyze: Click "Run AI Analysis"
Review: View detected damage with bounding boxes
Export: Download professional PDF report
Map: See GPS location on interactive map (if available)
