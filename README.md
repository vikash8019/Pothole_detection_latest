🚧 Pothole Detection System using YOLO & Streamlit
📌 Project Overview

This project is an AI-based Pothole Detection System designed to automatically detect potholes in road images and videos using Computer Vision and Deep Learning.
The system uses the YOLO (You Only Look Once) object detection model and provides a simple and interactive web dashboard built with Streamlit.

The goal of this project is to help in road safety, smart city planning, and automated road inspection by identifying damaged road areas efficiently.

🎯 Features

Detects potholes in Images

Detects potholes in Videos

Real-time detection support

Simple and clean Web UI

Fast and accurate detection using YOLO

Upload and analyze media easily

Bounding boxes with confidence score

🧠 Technologies Used

Python

YOLO (Ultralytics)

OpenCV

Streamlit

NumPy

Machine Learning / Deep Learning

Computer Vision

🗂 Project Structure
pothole_detection/
│
├── app.py                 # Main Streamlit Application
├── best.pt                # Trained YOLO Model
├── images/                # Sample images
├── videos/                # Sample videos
├── requirements.txt       # Dependencies
└── README.md              # Project Documentation

⚙️ Installation & Setup
1. Clone Repository
git clone https://github.com/vikash8019/pothole-detection.git
cd pothole_detection_latest

2. Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Run Application
streamlit run app.py

📷 How It Works

User uploads an image or video.

The YOLO model processes the input.

Potholes are detected using bounding boxes.

Results are displayed on the dashboard with confidence levels.

🧪 Model Training

Dataset: Custom pothole dataset

Annotation Tool: Roboflow / LabelImg

Model: YOLOv8

Training Framework: Ultralytics

🚀 Future Improvements

Live camera detection

Road damage severity analysis

GPS mapping integration

Mobile app support

Automated report generation

💡 Use Cases

Smart City Infrastructure

Government Road Monitoring

Autonomous Vehicles

Traffic Safety Systems

Civil Engineering Projects

🛠 Challenges Faced

Model accuracy tuning

Python version compatibility issues

Dependency conflicts

Streamlit deployment errors

📜 Conclusion

This project demonstrates how AI and Computer Vision can be used to solve real-world infrastructure problems.
It provides an efficient and scalable solution for pothole detection, making road inspection faster and more reliable.

## 🚀 Live Demo
You can try the application here:  
https://potholedetectionlatest-2026.streamlit.app/

👨‍💻 Author

Vikash Kumar Singh
AI & ML Enthusiast | Python Developer

⭐ If you like this project

Give it a Star on GitHub and share it
