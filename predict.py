import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from flask import Flask, Response
from flask_cors import CORS
from models import GiMeFive  # โมเดลของคุณ

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# โหลดโมเดล
model_path = "best_GiMeFive.pth"
model = GiMeFive()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

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

# settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0,255,0)
thickness = 2
line_type = cv2.LINE_AA
evaluation_frequency = 5

def detect_emotion(face_img):
    """รับ crop face image แล้ว return scores 6 อารมณ์"""
    image = Image.fromarray(face_img)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        scores = probs.cpu().numpy().flatten()
    return scores

def generate_frames():
    cap = cv2.VideoCapture(0)
    counter = 0
    max_emotion = ''
    scores = [0]*len(class_labels)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40,40))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            
            if counter == 0:
                crop_img = frame[y:y+h, x:x+w]
                scores = detect_emotion(crop_img)
                max_index = np.argmax(scores)
                max_emotion = class_labels[max_index]
            
            # แสดง emotion สูงสุด
            cv2.putText(frame, max_emotion, (x, y-10), font, font_scale, font_color, thickness, line_type)
            
            # แสดงคะแนน emotion ทั้งหมด (6 ค่า)
            for idx, label in enumerate(class_labels):
                if idx >= len(scores):
                    continue
                text = f"{label}: {scores[idx]:.2f}"
                cv2.putText(frame, text, (x+w+10, y + idx*25), font, 0.6, (0,255,0), 1, line_type)
        
        counter += 1
        if counter == evaluation_frequency:
            counter = 0
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
