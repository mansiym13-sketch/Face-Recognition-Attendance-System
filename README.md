# 📸 Face Recognition Attendance System

An automated, smart attendance management system that uses computer vision to recognize faces and log attendance in real-time. This project features a desktop-based tracking workflow combined with a web interface (`webapp`) for viewing and managing attendance records.

## 🚀 Features

* **📷 Face Capture & Registration:** Easily register new students by capturing multiple face samples via webcam (`takeImage.py`).
* **🧠 Local Model Training:** Train a local facial recognition model on captured student images using Haar Cascade classifiers (`trainImage.py`).
* **⏱️ Automatic Real-Time Attendance:** Instantly detect and identify faces through a live video feed and automatically log timestamps (`automaticAttedance.py`).
* **📝 Manual Backup:** A fallback system allowing manual attendance entry when required (`takemanually.py`).
* **📊 Attendance Dashboard:** A web-based application (`webapp/`) featuring a clean layout to display, verify, and track logged attendance history.

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Computer Vision:** OpenCV (utilizing `haarcascade_frontalface_default.xml`)
* **Data Management:** Pandas / CSV (for logging data in `Attendance/` and `StudentDetails/`)
* **Web Interface:** HTML5, CSS3, JavaScript (Flask or standard web server setup)

## 📂 Project Structure

```text
Face-Recognition-Attendance-System/
│
├── Attendance/                  # Stores generated attendance CSV reports
├── StudentDetails/              # Logs student profile details and CSV sheets
├── UI_Image/                    # Icons and background graphics for desktop GUI
├── webapp/                      # Web application layout (HTML/CSS views)
│
├── attendance.py                # Core attendance processing utilities
├── automaticAttedance.py       # Main script for automated live webcam tracking
├── takeImage.py                 # Script to capture and save student face samples
├── trainImage.py                # Trains the recognizer on the captured images
├── show_attendance.py           # Script to fetch and view attendance logs
├── takemanually.py              # Script for manual attendance adjustments
│
├── haarcascade_*.xml            # Pre-trained OpenCV face detection models
└── requirements.txt             # Python dependencies
⚙️ Getting Started
Follow these steps to set up and run the system locally.

Prerequisites
Python 3.8+

A functional webcam/camera connected to your machine.

Installation
1. Clone the repository:

Bash
git clone [https://github.com/mansiym13-sketch/Face-Recognition-Attendance-System.git](https://github.com/mansiym13-sketch/Face-Recognition-Attendance-System.git)
cd Face-Recognition-Attendance-System
2. Install dependencies:

Bash
pip install -r requirements.txt
🎮 How It Works (Step-by-Step)
Step 1: Register a Student
Run the image capture script to record a new student's face. Enter the student's ID and name when prompted. The script will save face snapshots inside the dataset directory.

Bash
python takeImage.py
Step 2: Train the Model
Once you have registered all students, train the recognition model using the captured face samples:

Bash
python trainImage.py
Step 3: Start Automatic Attendance Tracker
Launch the live system. The webcam will open, detect faces in the frame, match them with the trained database, and automatically log present students into the Attendance/ folder.

Bash
python automaticAttedance.py
Step 4: View Records
You can display logged attendance records via the terminal script or spin up the web dashboard inside the webapp/ directory to view the tracking metrics through your browser.

Bash
python show_attendance.py
