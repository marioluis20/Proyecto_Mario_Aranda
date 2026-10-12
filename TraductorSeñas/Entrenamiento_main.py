import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.io import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torch.cuda.amp import autocast, GradScaler

def main():
    DATA_DIR = r'C:\Users\GyroOH\Documents\RevistaIA\Muestras\SeñasV2'  
    MODEL_DIR = r'C:\Users\GyroOH\Documents\RevistaIA\Nuevo_Modelo\2025298'
    os.makedirs(MODEL_DIR, exist_ok=True)

    EPOCHS = 100
    BATCH_SIZE = 8
    DISP = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    PACIENCIA = 5
    VIDEO_DURACION = 20
    ACCUMULATION_STEPS = 2  # Gradient accumulation steps
    WARMUP_EPOCHS = 5      # Learning rate warmup epochs
    print('Usando', DISP)

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    MEAN = [0.43216, 0.394666, 0.37645]
    STD = [0.22803, 0.22145, 0.216989]

    class Video_recorte_rand:
        def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)):
            self.size = size
            self.scale = scale
            self.ratio = ratio

        def __call__(self, video):
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                video[0], scale=self.scale, ratio=self.ratio
            )
            
            video_recorte = []
            for t in range(video.size(0)):
                frame = video[t]
                frame = F.resized_crop(
                    frame, i, j, h, w, self.size, interpolation=transforms.InterpolationMode.BILINEAR
                )
                video_recorte.append(frame)
            
            return torch.stack(video_recorte)

    class Video_horizontalflip_rand:
        def __init__(self, p=0.5):
            self.p = p

        def __call__(self, video):
            if random.random() < self.p:
                return video.flip(-1)  
            return video
        
    def frames_arreglo_tmo(video):
        return torch.stack([transforms.functional.resize(frame, 128) for frame in video])

    def centro_frames(video):
        return torch.stack([transforms.functional.center_crop(frame, 112) for frame in video])

    def trans_entrenamiento():
        return transforms.Compose([
            Video_recorte_rand((112, 112)), 
            Video_horizontalflip_rand(p=0.5),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

    def trans_validacion():
        return transforms.Compose([
            transforms.Lambda(frames_arreglo_tmo),
            transforms.Lambda(centro_frames),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

    class Video_Dataset(Dataset):
        def __init__(self, pares_label, clase_id, transform=None, clip_dur=VIDEO_DURACION):
            self.pares_label = pares_label
            self.clase_id = clase_id
            self.transform = transform
            self.clip_dur = clip_dur

        def __len__(self):
            return len(self.pares_label)

        def __getitem__(self, idx):
            video_dir, label = self.pares_label[idx]

            try:
                video, _, info = read_video(video_dir, pts_unit='sec')
                video = video.float() / 255.0
                video = video.permute(0, 3, 1, 2)

                # Ajustar duración
                if video.shape[0] > self.clip_dur:
                    start_idx = random.randint(0, video.shape[0] - self.clip_dur)
                    video = video[start_idx:start_idx+self.clip_dur]
                elif video.shape[0] < self.clip_dur:
                    pad_len = self.clip_dur - video.shape[0]
                    pad_tensor = video[-1:].repeat(pad_len, 1, 1, 1)
                    video = torch.cat([video, pad_tensor], dim=0)

                if self.transform:
                    video = self.transform(video)

                return video, label

            except Exception as e:
                print(f"Error al cargar {video_dir}: {e}")
                # Video dummy con forma correcta
                dummy_video = torch.zeros((self.clip_dur, 3, 112, 112), dtype=torch.float)
                return dummy_video, label  # usar la misma etiqueta o una etiqueta neutra


    def Video_labels(DATASET_DIR):
        class_to_id = {}
        video_label_pairs = []
        
        for class_id, class_name in enumerate(sorted(os.listdir(DATASET_DIR))):
            class_path = os.path.join(DATASET_DIR, class_name)
            if not os.path.isdir(class_path):
                continue
                
            class_to_id[class_name] = class_id
            
            for video_file in os.listdir(class_path):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(class_path, video_file)
                    video_label_pairs.append((video_path, class_id))
        
        return video_label_pairs, class_to_id

    video_label_pares, clase_id = Video_labels(DATA_DIR)
    NUM_CLASES = len(clase_id)

    random.shuffle(video_label_pares)
    split_id = int(0.8 * len(video_label_pares))
    PARES_ENTRENAMIENTO = video_label_pares[:split_id]
    PARES_VALIDACION = video_label_pares[split_id:]

    print(f"Muestras de entrenamiento: {len(PARES_ENTRENAMIENTO)}, muestras de validación: {len(PARES_VALIDACION)}")
    print(f"Clases: {list(clase_id.keys())}")

    DATASET_ENTRENAMIENTO = Video_Dataset(PARES_ENTRENAMIENTO, clase_id, transform=trans_entrenamiento())
    DATASET_VALIDACION = Video_Dataset(PARES_VALIDACION, clase_id, transform=trans_validacion())

    # Usar más workers si está disponible
    num_workers = 0
    # min(4, os.cpu_count() - 1) if os.cpu_count() > 1 else 0

    ENTRENAMIENTO_LOADER = DataLoader(
        DATASET_ENTRENAMIENTO, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DISP.type == 'cuda')
    )

    VALIDACION_LOADER = DataLoader(
        DATASET_VALIDACION, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DISP.type == 'cuda')
    )

    with open(os.path.join(MODEL_DIR, 'CLASES_ID.json'), 'w') as f:
        json.dump(clase_id, f)

    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASES)
    model = model.to(DISP)

    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(model.parameters(), lr=1e-4)
    rutina = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, 'min', patience=3)

    # Mixed precision training
    scaler = GradScaler()

    MJR_LOSS = float('inf')
    ENTRENAMIENTO_LOSSES = []   
    VALIDACION_LOSSES = []      
    ENTRENAMIENTO_ACC_TOT = []      
    VALIDACION_ACC_TOT = []         
    SIN_MEJORA = 0

    # Learning rate warmup function
    def warmup_lr(epoch, warmup_epochs=WARMUP_EPOCHS, base_lr=1e-4):
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        return base_lr

    for epoch in range(EPOCHS):
        # Aplicar warmup de learning rate
        current_lr = warmup_lr(epoch)
        for param_group in optimizador.param_groups:
            param_group['lr'] = current_lr
        
        model.train()
        epoch_LOSS, CORRECT, TOTAL = 0, 0, 0
        
        loop = tqdm(ENTRENAMIENTO_LOADER, desc=f'Epoch {epoch + 1} / {EPOCHS}')
        
        # Gradient accumulation
        optimizador.zero_grad()
        
        for i, (videos, labels) in enumerate(loop):
            videos = videos.permute(0, 2, 1, 3, 4).to(DISP)
            labels = labels.to(DISP)
            
            # Mixed precision training
            with autocast():
                outputs = model(videos)
                loss = criterio(outputs, labels)
                loss = loss / ACCUMULATION_STEPS  # Normalize loss for accumulation
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            
            # Step optimizer every accumulation steps
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizador)
                scaler.update()
                optimizador.zero_grad()
            
            epoch_LOSS += loss.item() * ACCUMULATION_STEPS  # Unnormalize for logging
            pred = outputs.argmax(dim=1)
            CORRECT += (pred == labels).sum().item()
            TOTAL += labels.size(0)
            
            loop.set_postfix(
                loss=loss.item() * ACCUMULATION_STEPS, 
                acc=100 * CORRECT / TOTAL,
                lr=current_lr
            )
            
            # Limpiar memoria
            del videos, labels, outputs, loss
            if DISP.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Step for any remaining gradients
        if (i + 1) % ACCUMULATION_STEPS != 0:
            scaler.step(optimizador)
            scaler.update()
            optimizador.zero_grad()
        
        ENTRENAMIENTO_LOSS = epoch_LOSS / len(ENTRENAMIENTO_LOADER)
        ENTRENAMIENTO_PCN = 100 * CORRECT / TOTAL
        ENTRENAMIENTO_LOSSES.append(ENTRENAMIENTO_LOSS)
        ENTRENAMIENTO_ACC_TOT.append(ENTRENAMIENTO_PCN)

        model.eval()
        VALIDACION_LOSS, VALIDACION_CORRECT, VALIDACION_TOTAL = 0, 0, 0
        with torch.no_grad():
            for videos, labels in VALIDACION_LOADER:
                videos = videos.permute(0, 2, 1, 3, 4).to(DISP)
                labels = labels.to(DISP)
                
                outputs = model(videos)
                loss_val = criterio(outputs, labels)  
                VALIDACION_LOSS += loss_val.item()
                
                pred = outputs.argmax(dim=1)
                VALIDACION_CORRECT += (pred == labels).sum().item()
                VALIDACION_TOTAL += labels.size(0)
                
                # Limpiar memoria
                del videos, labels, outputs, loss_val
                if DISP.type == 'cuda':
                    torch.cuda.empty_cache()

        VALIDACION_LOSS /= len(VALIDACION_LOADER)
        VALIDACION_PCN = 100 * VALIDACION_CORRECT / VALIDACION_TOTAL
        VALIDACION_LOSSES.append(VALIDACION_LOSS)
        VALIDACION_ACC_TOT.append(VALIDACION_PCN)
        
        print(f"Epoch {epoch + 1} | "
                f"Train Loss: {ENTRENAMIENTO_LOSS:.4f} | Train Acc: {ENTRENAMIENTO_PCN:.2f}% | "
                f"Val Loss: {VALIDACION_LOSS:.4f} | Val Acc: {VALIDACION_PCN:.2f}% | "
                f"LR: {current_lr:.2e}")
        
        rutina.step(VALIDACION_LOSS)
        
        # Guardar checkpoint periódico
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(MODEL_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizador.state_dict(),
                'loss': VALIDACION_LOSS,
            }, checkpoint_path)
            print(f'Checkpoint guardado: {checkpoint_path}')
        
        if VALIDACION_LOSS < MJR_LOSS:
            MJR_LOSS = VALIDACION_LOSS
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'Mejor_Modelo.pth'))
            print('Modelo guardado')
            SIN_MEJORA = 0
        else:
            SIN_MEJORA += 1
        
        if SIN_MEJORA >= PACIENCIA:
            print(f'Early stopping en epoch {epoch + 1} - Sin mejora en {PACIENCIA} épocas')
            break

    # Cargar el mejor modelo para evaluación final
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'Mejor_Modelo.pth')))
    model.eval()

    # Evaluación final con TTA (Test Time Augmentation)
    def evaluate_with_tta(model, dataloader, num_augmentations=3):
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for videos, labels in tqdm(dataloader, desc="Evaluating with TTA"):
                videos = videos.permute(0, 2, 1, 3, 4).to(DISP)
                labels = labels.to(DISP)
                
                # Promediar predicciones sobre múltiples aumentaciones
                outputs_avg = None
                for _ in range(num_augmentations):
                    # Aplicar aumentación de entrenamiento
                    augmented_videos = trans_entrenamiento()(videos.permute(0, 2, 1, 3, 4).cpu())
                    augmented_videos = augmented_videos.permute(0, 2, 1, 3, 4).to(DISP)
                    
                    with autocast():
                        outputs = model(augmented_videos)
                    
                    if outputs_avg is None:
                        outputs_avg = outputs
                    else:
                        outputs_avg += outputs
                
                outputs_avg /= num_augmentations
                pred = outputs_avg.argmax(dim=1).cpu()
                
                all_preds.extend(pred.numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Limpiar memoria
                del videos, labels, outputs_avg, augmented_videos
                if DISP.type == 'cuda':
                    torch.cuda.empty_cache()
        
        return all_preds, all_labels

    # Evaluación estándar
    PRED_TODOS = []
    LABEL_TODOS = []

    with torch.no_grad():
        for videos, labels in VALIDACION_LOADER:
            videos = videos.permute(0, 2, 1, 3, 4).to(DISP)
            with autocast():
                outputs = model(videos)
            pred = outputs.argmax(dim=1).cpu()
            PRED_TODOS.extend(pred.numpy())
            LABEL_TODOS.extend(labels.numpy())

    # Evaluación con TTA (opcional, comentar si no se necesita)
    # PRED_TODOS, LABEL_TODOS = evaluate_with_tta(model, VALIDACION_LOADER)

    NOMBRES_CLASE = list(clase_id.keys())

    conf_matriz = confusion_matrix(LABEL_TODOS, PRED_TODOS)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matriz, display_labels=NOMBRES_CLASE)

    precision, recall, f1, _ = precision_recall_fscore_support(
        LABEL_TODOS, PRED_TODOS, labels=range(len(clase_id)))

    x = np.arange(len(clase_id))
    width = 0.25

    plt.figure(figsize=(20, 15))

    plt.subplot(2, 2, 1)
    plt.plot(ENTRENAMIENTO_LOSSES, 'b-', label='Entrenamiento')
    plt.plot(VALIDACION_LOSSES, 'r-', label='Validación')
    plt.title('Pérdida vs Épocas')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(ENTRENAMIENTO_ACC_TOT, 'b-', label='Entrenamiento')
    plt.plot(VALIDACION_ACC_TOT, 'r-', label='Validación')
    plt.title('Precisión vs Épocas')
    plt.xlabel('Época')
    plt.ylabel('Precisión (%)')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), values_format='d')  
    plt.title("Matriz de Confusión")

    plt.subplot(2, 2, 4)
    plt.bar(x - width, precision, width, label='Precision', color='skyblue')
    plt.bar(x, recall, width, label='Recall', color='lightgreen')
    plt.bar(x + width, f1, width, label='F1-score', color='salmon')
    plt.xticks(x, NOMBRES_CLASE, rotation=45)  
    plt.ylim(0, 1.1)
    plt.title('Métricas por clase')
    plt.xlabel('Clase')
    plt.ylabel('Valor')
    plt.legend()

    plt.tight_layout(pad=3.0)  
    plt.savefig(os.path.join(MODEL_DIR, 'resumen_completo.png'))
    plt.show()

    plt.figure(figsize=(10, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión")
    plt.savefig(os.path.join(MODEL_DIR, 'matriz_de_confusion.png'))
    plt.close()  

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width, label='Precision', color='skyblue')
    plt.bar(x, recall, width, label='Recall', color='lightgreen')
    plt.bar(x + width, f1, width, label='F1-score', color='salmon')
    plt.xticks(x, NOMBRES_CLASE, rotation=45)
    plt.ylim(0, 1.1)
    plt.title('Métricas por clase')
    plt.xlabel('Clase')
    plt.ylabel('Valor')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'precision_recall_f1.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(ENTRENAMIENTO_LOSSES, 'b-', label='Entrenamiento')
    plt.plot(VALIDACION_LOSSES, 'r-', label='Validación')
    plt.title('Péridda vs Épocas')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(ENTRENAMIENTO_ACC_TOT, 'b-', label='Entrenamiento')
    plt.plot(VALIDACION_ACC_TOT, 'r-', label='Validación')
    plt.title('Precisión vs Épocas')
    plt.xlabel('Época')
    plt.ylabel('Precisión (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'curvas_entrenamiento.png'))
    plt.close()

if __name__ == '__main__':
    main()