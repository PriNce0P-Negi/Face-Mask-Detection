import torch
import cv2
import numpy as np
from torchvision import models
from utils import get_transforms
from PIL import Image
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()

classes = ["With Mask", "Without Mask"]

transform = get_transforms()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

pred_queue = deque(maxlen=5)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        pred_queue.append(pred.item())

        final_pred = max(set(pred_queue), key=pred_queue.count)

        label = classes[final_pred]
        conf = confidence.item() * 100

        color = (0, 255, 0) if final_pred == 0 else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        text = f"{label} ({conf:.1f}%)"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Mask Detection (Improved)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
