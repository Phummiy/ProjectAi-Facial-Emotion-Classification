import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from flask import Flask, Response
from flask_cors import CORS
from models import GiMeFive
import dlib  # เพิ่มตรงนี้

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

# ---------- Hook Class สำหรับ Grad-CAM ----------
class Hook():
    def __init__(self):
        self.hook_forward = None
        self.hook_backward = None
        self.forward_out = None
        self.backward_out = None

    def hook_fn_forward(self, module, input, output):
        self.forward_out = output

    def hook_fn_backward(self, module, grad_input, grad_output):
        self.backward_out = grad_output[0] 

    def register_hook(self, module):
        self.hook_forward = module.register_forward_hook(self.hook_fn_forward)
        self.hook_backward = module.register_full_backward_hook(self.hook_fn_backward)

    def unregister_hook(self):
        self.hook_forward.remove()
        self.hook_backward.remove()

# ผูก hook เข้ากับ conv layer สุดท้าย
final_layer = model.conv5
hook = Hook()
hook.register_hook(final_layer)

# class labels ของโมเดล GiMeFive
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

# dlib predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# display settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (154, 1, 254)
thickness = 2
line_type = cv2.LINE_AA
transparency = 0.4
evaluation_frequency = 5

# ---------- ฟังก์ชันหลัก ----------
def detect_emotion_with_heatmap(face_img):
    """Crop ใบหน้าแล้ว return emotion scores และ heatmap Grad-CAM"""
    image = Image.fromarray(face_img)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Forward pass
    outputs = model(image_tensor)
    probs = F.softmax(outputs, dim=1)
    scores = probs.detach().cpu().numpy().flatten()
    max_index = np.argmax(scores)

    # ทำ one-hot และ backward เพื่อหา gradient
    one_hot = torch.zeros_like(outputs)
    one_hot[0, max_index] = 1
    outputs.backward(gradient=one_hot, retain_graph=True)

    # คำนวณ Grad-CAM
    gradients = hook.backward_out
    activations = hook.forward_out
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * activations, dim=1).squeeze()
    cam = F.relu(cam)
    cam = cam.cpu().detach().numpy()
    cam = cam - np.min(cam)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)
    else:
        cam = np.zeros_like(cam)
    return scores, cam

def overlay_heatmap_on_face(frame, x, y, w, h, cam):
    """Overlay heatmap บนกรอบหน้า"""
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    roi = frame[y:y+h, x:x+w, :]
    overlay = heatmap * transparency + roi / 255 * (1 - transparency)
    frame[y:y+h, x:x+w, :] = np.uint8(255 * overlay)

# ---------- สตรีมกล้อง ----------
def generate_frames():
    cap = cv2.VideoCapture(0)
    counter = 0
    max_emotion = ''
    scores = [0] * len(class_labels)
    cam = None  # กำหนดค่าเริ่มต้นให้ cam

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40,40))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            crop_img = frame[y:y+h, x:x+w]

            # Update scores & cam ทุก evaluation_frequency หรือถ้า cam ยังไม่มีค่า
            if counter == 0 or cam is None:
                scores, cam = detect_emotion_with_heatmap(crop_img)
                max_emotion = class_labels[np.argmax(scores)]

            # overlay heatmap เฉพาะเมื่อ cam มีค่า
            if cam is not None:
                overlay_heatmap_on_face(frame, x, y, w, h, cam)

            # แสดง emotion สูงสุด
            cv2.putText(frame, max_emotion, (x, y-10), font, font_scale, font_color, thickness, line_type)

            # แสดงคะแนน emotion ทั้งหมด
            for idx, label in enumerate(class_labels):
                text = f"{label}: {scores[idx]:.2f}"
                cv2.putText(frame, text, (x+w+10, y + idx*25), font, 0.6, (0,255,0), 1, line_type)

            # DLIB landmarks
            dlib_rect = dlib.rectangle(x, y, x+w, y+h)
            landmarks = predictor(gray, dlib_rect)
            for i in range(68):
                lx = landmarks.part(i).x
                ly = landmarks.part(i).y
                cv2.circle(frame, (lx, ly), 1, (0,0,255), -1)

        counter += 1
        if counter >= evaluation_frequency:
            counter = 0

        # encode frame และ yield
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
