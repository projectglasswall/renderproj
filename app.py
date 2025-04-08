from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import io

app = Flask(__name__)
model = YOLO("yolov10n.pt")  # make sure this file is in your project root

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return "No image uploaded", 400

    file = request.files["image"]
    in_memory = io.BytesIO()
    file.save(in_memory)
    np_img = np.frombuffer(in_memory.getvalue(), np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    _, buffer = cv2.imencode(".jpg", frame)
    return send_file(io.BytesIO(buffer.tobytes()), mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
