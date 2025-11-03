import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from datetime import datetime

from core.capture import capture_user_images
from core.train import train_model
from core.recognize import fill_attendance_for_subject, build_attendance_summary

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIRS = {
    "haar": os.path.join(ROOT_DIR, "haarcascade_frontalface_default.xml"),
    "train_images": os.path.join(ROOT_DIR, "TrainingImage"),
    "model": os.path.join(ROOT_DIR, "TrainingImageLabel", "Trainner.yml"),
    "students": os.path.join(ROOT_DIR, "StudentDetails", "studentdetails.csv"),
    "attendance": os.path.join(ROOT_DIR, "Attendance"),
}

os.makedirs(os.path.dirname(DATA_DIRS["model"]), exist_ok=True)
os.makedirs(DATA_DIRS["train_images"], exist_ok=True)
os.makedirs(os.path.dirname(DATA_DIRS["students"]), exist_ok=True)
os.makedirs(DATA_DIRS["attendance"], exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/enroll", methods=["GET", "POST"])
def enroll():
    if request.method == "POST":
        enrollment = request.form.get("enrollment", "").strip()
        name = request.form.get("name", "").strip()
        if not enrollment or not name or not enrollment.isdigit():
            flash("Enter a numeric enrollment and a name.", "error")
            return redirect(url_for("enroll"))
        try:
            count = capture_user_images(
                enrollment=enrollment,
                name=name,
                haarcascade_path=DATA_DIRS["haar"],
                output_root=DATA_DIRS["train_images"],
                samples=50,
                students_csv=DATA_DIRS["students"],
            )
            flash(f"Captured {count} images for {name} ({enrollment}).", "success")
            return redirect(url_for("train"))
        except Exception as e:
            flash(f"Failed to capture images: {e}", "error")
            return redirect(url_for("enroll"))
    return render_template("enroll.html")


@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "POST":
        try:
            trained = train_model(
                haarcascade_path=DATA_DIRS["haar"],
                images_root=DATA_DIRS["train_images"],
                model_path=DATA_DIRS["model"],
            )
            flash(f"Training completed. {trained} face images processed.", "success")
        except Exception as e:
            flash(f"Training failed: {e}", "error")
        return redirect(url_for("train"))
    return render_template("train.html")


@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        duration = int(request.form.get("duration", 20))
        if not subject:
            flash("Enter a subject name.", "error")
            return redirect(url_for("attendance"))
        try:
            csv_path = fill_attendance_for_subject(
                subject=subject,
                haarcascade_path=DATA_DIRS["haar"],
                model_path=DATA_DIRS["model"],
                students_csv=DATA_DIRS["students"],
                attendance_root=DATA_DIRS["attendance"],
                duration_seconds=duration,
            )
            flash(f"Attendance captured. File: {os.path.relpath(csv_path, ROOT_DIR)}", "success")
            return redirect(url_for("attendance"))
        except Exception as e:
            flash(f"Attendance failed: {e}", "error")
            return redirect(url_for("attendance"))
    return render_template("attendance.html")


@app.route("/reports", methods=["GET", "POST"])
def reports():
    summary = None
    subject = None
    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        if not subject:
            flash("Enter a subject name.", "error")
            return redirect(url_for("reports"))
        try:
            summary_path = build_attendance_summary(subject=subject, attendance_root=DATA_DIRS["attendance"]) 
            summary = os.path.relpath(summary_path, ROOT_DIR)
            flash("Summary generated.", "success")
        except Exception as e:
            flash(f"Failed to build summary: {e}", "error")
    return render_template("reports.html", summary=summary, subject=subject)


@app.route("/download")
def download():
    rel = request.args.get("path")
    if not rel:
        flash("No file specified.", "error")
        return redirect(url_for("index"))
    abs_path = os.path.join(ROOT_DIR, rel)
    if not os.path.isfile(abs_path):
        flash("File not found.", "error")
        return redirect(url_for("index"))
    return send_file(abs_path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
