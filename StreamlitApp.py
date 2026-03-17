import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Charger le modèle YOLO
model = YOLO("best.pt")  # ton modèle entraîné casque

st.title("🪖 Détection Casque EPI - YOLO + Streamlit")

run = st.checkbox('Démarrer la caméra')

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Erreur caméra")
        break

    # Prédiction YOLO
    results = model(frame)

    # Dessiner les résultats
    annotated_frame = results[0].plot()

    # Affichage Streamlit
    FRAME_WINDOW.image(annotated_frame, channels="BGR")

camera.release()
