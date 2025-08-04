import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import mediapipe as mp

# Tenta abrir a webcam no índice 0 (pode mudar para 1, 2 ou -1 se necessário)
cap = cv2.VideoCapture(0)
print("🟢 Webcam aberta?", cap.isOpened())

if not cap.isOpened():
    print("❌ Não foi possível acessar a webcam.")
    exit()

# Janela com nome fixo para evitar erro no Windows
cv2.namedWindow("Camera Segurança IA", cv2.WINDOW_NORMAL)

# Carrega modelo MoveNet do TensorFlow Hub
print("🔄 Carregando modelo MoveNet...")
model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = model.signatures['serving_default']
print("✅ Modelo MoveNet carregado!")

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# Controles de visualização
show_body = True
show_hands = True
show_fingers = True

def draw_body_keypoints(frame, keypoints):
    h, w, _ = frame.shape
    for kp in keypoints:
        y, x, conf = kp
        if conf > 0.3:
            cv2.rectangle(frame, (int(x * w) - 5, int(y * h) - 5), (int(x * w) + 5, int(y * h) + 5), (0, 255, 0), 2)

def detect_body(frame):
    img = cv2.resize(frame, (192, 192))
    img = tf.cast(img, dtype=tf.int32)
    img = tf.expand_dims(img, axis=0)
    outputs = movenet(img)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]
    draw_body_keypoints(frame, keypoints)

def detect_hands(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for i, lm in enumerate(hand_landmarks.landmark):
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                if show_fingers:
                    cv2.rectangle(frame, (x - 4, y - 4), (x + 4, y + 4), (255, 0, 0), 2)
            if show_hands:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

def toggle_option(key):
    global show_body, show_hands, show_fingers
    if key == ord('1'):
        show_body = not show_body
    elif key == ord('2'):
        show_hands = not show_hands
    elif key == ord('3'):
        show_fingers = not show_fingers

# Loop principal da câmera
while True:
    ret, frame = cap.read()
    print("🎥 Frame capturado?", ret)

    if not ret:
        print("❌ Frame não pôde ser lido.")
        break

    frame = cv2.flip(frame, 1)  # Corrige espelhamento

    if show_body:
        detect_body(frame)
    if show_hands or show_fingers:
        detect_hands(frame)

    # Overlay de status
    cv2.putText(frame, "[1] Corpo  [{}]".format("ON" if show_body else "OFF"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, "[2] Mãos   [{}]".format("ON" if show_hands else "OFF"), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.putText(frame, "[3] Dedos  [{}]".format("ON" if show_fingers else "OFF"), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Camera Segurança IA", frame)

    key = cv2.waitKey(1) & 0xFF
    toggle_option(key)
    if key == 27:  # ESC para sair
        print("👋 Encerrando pelo ESC.")
        break

# Limpeza final
cap.release()
cv2.destroyAllWindows()
