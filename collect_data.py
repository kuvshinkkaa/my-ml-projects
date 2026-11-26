"""Сбор датасета с веб-камеры.
Клавиши:  G - писать класс "glasses" (в очках)
          N - писать класс "noglasses" (без очков)
          S - пауза записи
          ESC - выход
Кроп лица делается ТЕМ ЖЕ каскадом и с теми же параметрами, что и в C++.
"""
import os
import cv2

CLASSES = {"glasses": 0, "noglasses": 0}
for c in CLASSES:
    os.makedirs(f"dataset/{c}", exist_ok=True)
    CLASSES[c] = len(os.listdir(f"dataset/{c}"))  # продолжить нумерацию

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
current = None
tick = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    faces = cascade.detectMultiScale(gray, 1.1, 6, minSize=(80, 80))
    if len(faces):
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # защита границ
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        tick += 1
        if current and tick % 2 == 0:  # сохраняем каждый 2-й кадр
            roi = frame[y0:y1, x0:x1]
            cv2.imwrite(f"dataset/{current}/{CLASSES[current]:05d}.jpg", roi)
            CLASSES[current] += 1

    status = f"REC: {current}" if current else "pause"
    cv2.putText(frame, f"{status} | glasses={CLASSES['glasses']} "
                f"noglasses={CLASSES['noglasses']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("collect", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k in (ord("g"), ord("G")):
        current = "glasses"
    elif k in (ord("n"), ord("N")):
        current = "noglasses"
    elif k in (ord("s"), ord("S")):
        current = None

cap.release()
cv2.destroyAllWindows()
