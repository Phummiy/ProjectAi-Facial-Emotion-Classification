import sys, json, os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
from models import GiMeFive

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "best_GiMeFive.pth"
EMOTIONS = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

model = GiMeFive().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


# โหลด Haar Cascade สำหรับ detect หน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(img_path):
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        raise ValueError("Cannot read image file")
    
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        raise ValueError("No face detected")
    
    # crop หน้าแรก
    x, y, w, h = faces[0]
    face_img = img_cv[y:y+h, x:x+w]
    
    # แปลงเป็น PIL
    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    tensor = transform(face_pil).unsqueeze(0).to(DEVICE)
    return tensor

def predict_image(img_path):
    img_tensor = preprocess_face(img_path)
    with torch.no_grad():
        out = model(img_tensor)
        probs = F.softmax(out, dim=1).cpu().numpy().flatten()
    return probs

if __name__ == "__main__":
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        print(json.dumps({"error":"No image file provided"}))
        sys.exit(1)

    try:
        probs = predict_image(sys.argv[1])
        idx = probs.argmax()
        result = {
            "prediction": {
                "emotion": EMOTIONS[idx],
                "confidence": float(probs[idx]),
                "all": [{"label": EMOTIONS[i], "score": float(probs[i])} for i in range(len(EMOTIONS))]
            }
        }
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)