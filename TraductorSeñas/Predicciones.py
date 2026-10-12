import os
import cv2
import time
import json
import torch
import numpy as np
import torchvision.transforms as T
from torchvision import models
from collections import deque
from torchvision.models.video import r3d_18
from torchvision.models.video import R3D_18_Weights

DISP = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELO_ESTATICO_DIR = r"C:\Users\mario\Downloads\IASeñas\RevistaIA-master\EstaticoPrueba_18_6\Mejor_Modelo.pth"
MODELO_ESTATICO_ID = MODELO_ESTATICO_DIR.replace('Mejor_Modelo.pth', 'CLASES_ID.json')

MODELO_DINAMICO_DIR = r"C:\Users\mario\Downloads\IASeñas\RevistaIA-master\Mejor_Modelo.pth"
MODELO_DINAMICO_ID = MODELO_DINAMICO_DIR.replace('Mejor_Modelo.pth', 'CLASES_ID.json')

SEQ_LENGTH = 20
ROI = (100, 100, 300, 300)
PARAMETRO_MOVIMIENTO = 500
CONFIANZA = 0.70
CAMBIO_FRAME = 5

transformacion_stat = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def cargar_modelo_estatico():
    with open(MODELO_ESTATICO_ID) as f:
        CLASE_ID = json.load(f)
    ID_CLASE = {v: k for k, v in CLASE_ID.items()}

    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASE_ID))
    model.load_state_dict(torch.load(MODELO_ESTATICO_DIR, map_location=DISP, weights_only=True))
    model.eval().to(DISP)
    return model, ID_CLASE

def cargar_modelo_dinamico():
    with open(MODELO_DINAMICO_ID, encoding="utf8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))
    model.load_state_dict(torch.load(MODELO_DINAMICO_DIR, map_location=DISP, weights_only=True))
    model.eval().to(DISP)
    return model, idx_to_class

modelo_estatico, labels_estaticos = cargar_modelo_estatico()
modelo_dinamico, labels_dinamicos = cargar_modelo_dinamico()

def prediccion_estatico(frame):
    img = transformacion_stat(frame).unsqueeze(0).to(DISP)
    with torch.no_grad():
        output = modelo_estatico(img)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, pred = torch.max(probs, dim=1)
    return labels_estaticos[pred.item()], confidence.item()

def prediccion_dinamico(frames):
    clip = []
    for f in frames:
        f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(f_rgb).permute(2, 0, 1).float() / 255.0
        clip.append(tensor)
    
    clip = torch.stack(clip)  
    
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1)
    clip = (clip - mean) / std
    
    clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(DISP) 

    with torch.no_grad():
        output = modelo_dinamico(clip)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, pred = torch.max(probs, dim=1)
    return labels_dinamicos[pred.item()], confidence.item()

def Deteccion_Movimiento(act_frame, prev_frame):
    if prev_frame is None:
        return False
        
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gris2 = cv2.cvtColor(act_frame, cv2.COLOR_BGR2GRAY)
    difer = cv2.absdiff(gray1, gris2)
    _, umbral = cv2.threshold(difer, 25, 255, cv2.THRESH_BINARY)
    nota_movimiento = np.sum(umbral) / 255
    return nota_movimiento > PARAMETRO_MOVIMIENTO

cap = cv2.VideoCapture(0)
frame_buffer = deque(maxlen=SEQ_LENGTH)
prev_roi = None
modo = "Estatico"
prediccion_actual = "Iniciando..."
ultima_prediccion_con_confianza = "Ninguna sena detectada"
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    x1, y1, w, h = ROI
    x2, y2 = x1 + w, y1 + h
    roi = frame[y1:y2, x1:x2]
    
    roi_arr = cv2.resize(roi, (112, 112))
    frame_buffer.append(roi_arr.copy())

    if len(frame_buffer) == SEQ_LENGTH:
        if frame_count % CAMBIO_FRAME == 0:  
            if Deteccion_Movimiento(roi, prev_roi):
                prediction, confidence = prediccion_dinamico(list(frame_buffer))
                modo = "Dinamico"
            else:
                prediction, confidence = prediccion_estatico(roi)
                modo = "Estatico"

            if confidence >= CONFIANZA:
                prediccion_actual = f"Confianza: {prediction} ({confidence:.2f})"
                ultima_prediccion_con_confianza = prediction
            else:
                prediccion_actual = f"Confianza baja: {prediction} ({confidence:.2f})"
            
            prev_roi = roi.copy()

    display_text = f"{modo}: {ultima_prediccion_con_confianza}"
    cv2.putText(frame, display_text, (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)

    cv2.putText(frame, prediccion_actual, (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Reconocimiento de lenguaje de senas", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()