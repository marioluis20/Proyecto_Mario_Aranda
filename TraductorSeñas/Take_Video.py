import cv2 as cv
import random as rd
import uuid as id
import time
import os

labels = ['Personas', 'Amigo', 'Casado']
fourcc = cv.VideoWriter_fourcc(*'MJPG')
FRAMES = 20
PATH = r"C:\Users\GyroOH\Documents\RevistaIA\Muestras"
TiempoToma = 2
NumeroVideos = 15


for label in labels:
    try:
        os.mkdir(PATH + f"/{label}")
        print("Directorio creado con éxito")
    except FileExistsError:
        pass
    except Exception:
        print("Problemon")

video_camara = cv.VideoCapture(0)
out = None

while video_camara.isOpened():

    ret, frame = video_camara.read()
    frame = cv.flip(frame, 1)
    cv.imshow("Tomar videos", frame)

    if ret == True:
        key = cv.waitKey(1) & 0xFF
        if key == ord('s'):
            for label in labels:
                for ImgNum in range(NumeroVideos):
                    if ImgNum == 0:
                        print("Grabando video en...")
                        for Num in range(3):
                            print(Num + 1)
                            time.sleep(1)                        
                    else:
                        print("Siguiente video en...")
                        for Num in range(3):
                            print(Num + 1)
                            time.sleep(1)
                    cv.imshow("Captura de Video", frame)
                    r = rd.Random()
                    id_unico = id.UUID(int=r.getrandbits(128), version=4)
                    Video_Path = PATH + f"/{label}"
                    TiempoInicio = time.time()
                    out = cv.VideoWriter(Video_Path + f'/v_{label}' + "." + str(id_unico) + '.avi', fourcc, FRAMES, (640, 480))
                    
                    while time.time() - TiempoInicio < TiempoToma:
                        ret, frame = video_camara.read()
                        frame = cv.flip(frame, 1)
                        out.write(frame)
                        cv.imshow("Captura de Video", frame)
                        key = cv.waitKey(1) & 0xFF
                        if key == 27:
                            break
                    out.release()
                    key = cv.waitKey(1) & 0xFF
                    if key == 27:
                        break
                key = cv.waitKey(1)
                input("Preciones ENTER par ir al siguiente Label...")
        elif key == 27:
            break
    else:
        print("No se ha podidio tomar video.")

video_camara.release()
out.release()
cv.destroyAllWindows()

def holamundo():
    ...