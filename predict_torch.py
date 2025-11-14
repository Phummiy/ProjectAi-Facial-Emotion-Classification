import sys
import json
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
from models import GiMeFive

# ========== CONFIG ==========
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "best_GiMeFive.pth"
EMOTIONS = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
OUTPUT_DIR = r"C:\ProjectAI\SEP-CVDL-main\SEP-CVDL-main\static\heatmaps"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ============================

# โหลดโมเดล
model = GiMeFive().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# transform สำหรับ crop หน้า
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Haar Cascade สำหรับหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def preprocess_face(img_path):
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        raise ValueError("Cannot read image file")

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        raise ValueError("No face detected")

    x, y, w, h = faces[0]
    face_img = img_cv[y:y + h, x:x + w].copy()  # BGR
    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    tensor = transform(face_pil).unsqueeze(0).to(DEVICE)
    return tensor, face_img


# ========== Grad-CAM Helpers ==========
def find_last_conv_layer(model):
    for layer in reversed(list(model.modules())):
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    raise ValueError("No Conv2d layer found in the model")


def generate_gradcam(model, img_tensor, target_layer):
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    try:
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)
    except AttributeError:
        bwd_handle = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    out = model(img_tensor)
    class_idx = out.argmax(dim=1).item()
    score = out[0, class_idx]
    score.backward(retain_graph=False)

    if len(gradients) == 0 or len(activations) == 0:
        fwd_handle.remove()
        bwd_handle.remove()
        raise RuntimeError("Failed to capture activations or gradients for Grad-CAM.")

    grads = gradients[0].detach()
    acts = activations[0].detach()
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)

    cam_np = cam.squeeze().cpu().numpy()
    if cam_np.max() - cam_np.min() <= 1e-6:
        cam_norm = np.zeros_like(cam_np)
    else:
        cam_norm = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())

    fwd_handle.remove()
    bwd_handle.remove()
    return cam_norm, class_idx


def overlay_heatmap_on_image(face_img, heatmap):
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (face_img.shape[1], face_img.shape[0]))
    overlay = cv2.addWeighted(face_img, 0.6, heatmap_color, 0.4, 0)
    return overlay


def predict_image(img_path):
    img_tensor, face_img = preprocess_face(img_path)

    # รอบแรก: ทำนาย probs
    with torch.no_grad():
        out = model(img_tensor)
        probs = F.softmax(out, dim=1).cpu().numpy().flatten()

    target_layer = find_last_conv_layer(model)

    # รอบสอง: Grad-CAM
    heatmap, _ = generate_gradcam(model, img_tensor.requires_grad_(True), target_layer)
    overlay = overlay_heatmap_on_image(face_img, heatmap)

    # บันทึกไฟล์ heatmap
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    heatmap_fname = f"heatmap_{base_name}.jpg"
    heatmap_path_abs = os.path.join(OUTPUT_DIR, heatmap_fname)
    cv2.imwrite(heatmap_path_abs, overlay)

    # ส่ง path แบบ relative สำหรับ frontend
    heatmap_path_rel = f"heatmaps/{heatmap_fname}"
    return probs, heatmap_path_rel


if __name__ == "__main__":
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        print(json.dumps({"error": "No image file provided"}))
        sys.exit(1)

    try:
        probs, heatmap_path = predict_image(sys.argv[1])
        idx = probs.argmax()
        result = {
            "prediction": {
                "emotion": EMOTIONS[idx],
                "confidence": float(probs[idx]),
                "all": [{"label": EMOTIONS[i], "score": float(probs[i])} for i in range(len(EMOTIONS))],
                "heatmap_path": heatmap_path
            }
        }
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)