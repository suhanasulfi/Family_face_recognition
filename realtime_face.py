import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Relations dictionary
relation_map = {
    "yaseen": "brother of suhana",
    "nisa": "mother of suhana",
    "sulu": "sister of suhana",
    "sulfi": "father of suhana",
    "fariyal": "niece of suhana",
}

# Load model
checkpoint = torch.load("face_model.pth", map_location="cpu")
classes = checkpoint["classes"]

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["model"])
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Relation → text color (BGR format)
color_map = {
    "Mother of suhana": (255, 0, 255),   # Pink
    "Father of suhana": (0, 255, 255),   # Yellow
    "sister of suhana": (255, 255, 0),   # Cyan
    "brother of suhana": (0, 255, 0),     
    "niece of suhana"  : (0,255,0)      # Green
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0)

        with torch.no_grad():
            out = model(face_tensor)
            pred = out.argmax(1).item()
            name = classes[pred]
            relation = relation_map.get(name, "")
            label = f"{name} ({relation})"

        # Pick text color based on relation (default = white)
        text_color = color_map.get(relation, (255, 255, 255))

        # Draw box + label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, text_color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
