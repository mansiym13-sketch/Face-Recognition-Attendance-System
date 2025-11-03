import os
import time
import csv
import glob
import datetime as dt
import cv2
import pandas as pd


def _attendance_filename(root: str, subject: str) -> str:
    now = dt.datetime.now()
    d = now.strftime("%Y-%m-%d")
    t = now.strftime("%H-%M-%S")
    subject_dir = os.path.join(root, subject)
    os.makedirs(subject_dir, exist_ok=True)
    return os.path.join(subject_dir, f"{subject}_{d}_{t}.csv")


def fill_attendance_for_subject(subject: str, haarcascade_path: str, model_path: str, students_csv: str, attendance_root: str, duration_seconds: int = 20) -> str:
    if not os.path.isfile(model_path):
        raise FileNotFoundError("Model not found. Train the model first.")
    if not os.path.isfile(students_csv):
        raise FileNotFoundError("Student details CSV not found. Enroll first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    face_cascade = cv2.CascadeClassifier(haarcascade_path)

    df = pd.read_csv(students_csv)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Cannot access camera")

    col_names = ["Enrollment", "Name"]
    attendance = pd.DataFrame(columns=col_names)

    end_time = time.time() + max(5, int(duration_seconds))

    try:
        while time.time() < end_time:
            ret, frame = cam.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            for (x, y, w, h) in faces:
                Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                if conf < 70:
                    name_vals = df.loc[df["Enrollment"] == Id]["Name"].values
                    name_str = name_vals[0] if len(name_vals) else "Unknown"
                    attendance.loc[len(attendance)] = [Id, name_str]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{Id}-{name_str}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.imshow("Attendance - Press ESC to stop", frame)
            if (cv2.waitKey(30) & 0xFF) == 27:
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    # mark present for today with 1
    date_col = dt.date.today().strftime("%Y-%m-%d")
    if not attendance.empty:
        attendance = attendance.drop_duplicates(["Enrollment"], keep="first")
        attendance[date_col] = 1

    out_csv = _attendance_filename(attendance_root, subject)
    attendance.to_csv(out_csv, index=False)
    return out_csv


def build_attendance_summary(subject: str, attendance_root: str) -> str:
    subject_dir = os.path.join(attendance_root, subject)
    pattern = os.path.join(subject_dir, f"{subject}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("No attendance files for this subject yet.")

    dfs = [pd.read_csv(f) for f in files]
    merged = dfs[0]
    for i in range(1, len(dfs)):
        merged = merged.merge(dfs[i], how="outer")
    merged.fillna(0, inplace=True)
    if merged.shape[1] > 2:
        merged["Attendance"] = (merged.iloc[:, 2:].mean(axis=1) * 100).round().astype(int).astype(str) + "%"
    else:
        merged["Attendance"] = "0%"

    out_csv = os.path.join(subject_dir, "attendance_summary.csv")
    merged.to_csv(out_csv, index=False)
    return out_csv
