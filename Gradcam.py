import os
import time
from datetime import datetime
import json
import threading
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from models import GiMeFive
import dlib

# ---------------- CONFIG ----------------
app = Flask(__name__)
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# โหลดโมเดล
model_path = "best_GiMeFive.pth"
model = GiMeFive().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------------- Grad-CAM Hook ----------------
class Hook():
    def __init__(self):
        self.forward_out = None
        self.backward_out = None

    def hook_fn_forward(self, module, input, output):
        self.forward_out = output

    def hook_fn_backward(self, module, grad_input, grad_output):
        self.backward_out = grad_output[0]

    def register_hook(self, module):
        module.register_forward_hook(self.hook_fn_forward)
        module.register_full_backward_hook(self.hook_fn_backward)

# hook Grad-CAM ที่ conv5
final_layer = model.conv5
hook = Hook()
hook.register_hook(final_layer)

# ---------- Hook สำหรับ Layer Activation Visualization ----------
class ActivationHook():
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.save_output)

    def save_output(self, module, input, output):
        self.features = output.detach().cpu()

    def remove(self):
        self.hook.remove()

activation_layer = model.conv3
activation_hook = ActivationHook(activation_layer)

# class labels
class_labels = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

# transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# display settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_color = (154, 1, 254)
thickness = 2
line_type = cv2.LINE_AA
transparency = 0.4
evaluation_frequency = 5  # calc emotion every 5 frame

# ---------- Recording Globals ----------
recording = False
writer = None
metadata_fp = None
record_lock = threading.Lock()
output_dir = "recordings"
os.makedirs(output_dir, exist_ok=True)

# ---------- Clean old files ----------
def clean_old_files():
    now = time.time()
    for f in os.listdir(output_dir):
        if f.endswith(".mp4") or f.endswith(".jsonl"):
            full_path = os.path.join(output_dir, f)
            file_age = now - os.path.getmtime(full_path)
            if file_age > 1800:  # 30 นาที
                try:
                    os.remove(full_path)
                    print("Removed old:", f)
                except:
                    pass

# ---------- Activation Visualization ----------
def visualize_activations(activations, size=64, num_maps=6):
    activations = activations[0]  
    activations = activations[:num_maps]

    grid_images = []
    for i in range(num_maps):
        act = activations[i].numpy()
        act = act - act.min()
        if act.max() != 0:
            act = act / act.max()
        act = (act * 255).astype(np.uint8)
        act_resized = cv2.resize(act, (size, size))
        act_resized = cv2.applyColorMap(act_resized, cv2.COLORMAP_JET)
        grid_images.append(act_resized)

    return np.hstack(grid_images)

# ---------- Grad-CAM ----------
def detect_emotion_with_heatmap(face_img):
    image = Image.fromarray(face_img)
    image_tensor = transform(image).unsqueeze(0).to(device)

    outputs = model(image_tensor)
    probs = F.softmax(outputs, dim=1)
    scores = probs.detach().cpu().numpy().flatten()
    max_index = np.argmax(scores)

    # backward
    model.zero_grad()
    one_hot = torch.zeros_like(outputs)
    one_hot[0, max_index] = 1
    outputs.backward(gradient=one_hot, retain_graph=True)

    gradients = hook.backward_out
    activations = hook.forward_out

    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * activations, dim=1).squeeze()
    cam = F.relu(cam).cpu().detach().numpy()

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)

    return scores, cam

def overlay_heatmap_on_face(frame, x, y, w, h, cam):
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float32) / 255

    roi = frame[y:y+h, x:x+w, :].astype(np.float32) / 255
    overlay = heatmap * transparency + roi * (1 - transparency)
    frame[y:y+h, x:x+w, :] = np.uint8(255 * overlay)

# ---------- Init Writer ----------
def init_writer(frame_shape):
    ts = time.strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(output_dir, f"record_{ts}.mp4")
    meta_path = os.path.join(output_dir, f"record_{ts}.jsonl")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frame_shape[0], frame_shape[1]
    writer_local = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))

    return writer_local, video_path, open(meta_path, "w", encoding="utf-8")

# ---------- Streaming + Recording ----------
def generate_frames():
    global recording, writer, metadata_fp

    cap = cv2.VideoCapture(0)
    counter = 0
    save_counter = 0

    scores = [0] * len(class_labels)
    max_emotion = ''
    cam = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        orig_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40,40))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            crop = orig_frame[y:y+h, x:x+w]

            if counter == 0 or cam is None:
                try:
                    scores, cam = detect_emotion_with_heatmap(crop)
                    max_emotion = class_labels[np.argmax(scores)]
                except:
                    pass

            if cam is not None:
                overlay_heatmap_on_face(frame, x, y, w, h, cam)

            cv2.putText(frame, max_emotion, (x, y-10),
                        font, font_scale, font_color, thickness, line_type)

            # landmarks
            try:
                drect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
                shape = predictor(gray, drect)
                for i in range(68):
                    lx, ly = shape.part(i).x, shape.part(i).y
                    cv2.circle(frame, (lx, ly), 1, (0,0,255), -1)
            except:
                pass

            # activation map
            if activation_hook.features is not None:
                act_img = visualize_activations(activation_hook.features, size=50, num_maps=6)
                ah, aw, _ = act_img.shape
                ys = frame.shape[0] - ah
                xs = frame.shape[1]//2 - aw//2
                if ys >= 0 and xs >= 0:
                    frame[ys:ys+ah, xs:xs+aw] = act_img

        # update counters
        counter = (counter + 1) % evaluation_frequency
        save_counter += 1

        # ----- Recording -----
        with record_lock:
            if recording:
                if writer is None:
                    clean_old_files()
                    writer, video_path, metadata_fp = init_writer(frame.shape)
                    print("Start recording:", video_path)

                writer.write(frame)

                # บันทึกทุก 10 เฟรม
                if save_counter % 10 == 0 and metadata_fp is not None:
                    frame_metadata = {
                        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        "emotion": max_emotion,
                        "confidence": round(float(max(scores)), 4)
                    }
                    metadata_fp.write(json.dumps(frame_metadata, ensure_ascii=False) + "\n")

        # stream
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    with record_lock:
        if writer:
            writer.release()
            writer = None
        if metadata_fp:
            metadata_fp.close()
            metadata_fp = None

# ---------- Flask endpoints ----------
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_record', methods=['POST'])
def start_record():
    global recording
    with record_lock:
        if recording:
            return jsonify({"ok": False, "msg": "Already recording"}), 400
        recording = True
    return jsonify({"ok": True})

@app.route('/stop_record', methods=['POST'])
def stop_record():
    global recording, writer, metadata_fp
    with record_lock:
        recording = False
        if writer:
            writer.release()
            writer = None
        if metadata_fp:
            metadata_fp.close()
            metadata_fp = None
    return jsonify({"ok": True})

@app.route('/recording_status')
def recording_status():
    return jsonify({"recording": recording})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)