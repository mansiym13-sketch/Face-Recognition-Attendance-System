import os
import csv
import cv2
from typing import Optional

def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def append_student(students_csv: str, enrollment: str, name: str):
    ensure_parent(students_csv)
    file_exists = os.path.exists(students_csv)
    write_header = not file_exists or os.path.getsize(students_csv) == 0
    with open(students_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["Enrollment", "Name"]) 
        w.writerow([enrollment, name])


def capture_user_images(enrollment: str, name: str, haarcascade_path: str, output_root: str, samples: int = 50, students_csv: Optional[str] = None) -> int:
    if not enrollment.isdigit():
        raise ValueError("Enrollment must be numeric")
    person_dir = os.path.join(output_root, f"{enrollment}_{name}")
    os.makedirs(person_dir, exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Cannot access camera")

    detector = cv2.CascadeClassifier(haarcascade_path)
    count = 0
    try:
        while True:
            ret, img = cam.read()
            if not ret:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                count += 1
                face_img = gray[y:y+h, x:x+w]
                out_path = os.path.join(person_dir, f"{name}_{enrollment}_{count}.jpg")
                cv2.imwrite(out_path, face_img)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow("Capture - Press q to stop", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if count >= samples:
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
    # record student in CSV once we successfully captured
    if students_csv:
        append_student(students_csv, enrollment, name)
    return count
