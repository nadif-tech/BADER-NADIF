import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Détection Casque EPI", layout="wide")
st.title("🪖 Détection Casque EPI avec YOLO")

# Charger le modèle YOLO (remplacer 'best.pt' par ton modèle)
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt', force_reload=True)
    return model

model = load_model()

# Upload image ou utiliser webcam
option = st.radio("Choisir le mode :", ["Image Upload", "Webcam (Live)"])

if option == "Image Upload":
    uploaded_file = st.file_uploader("Choisir une image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Image originale', use_column_width=True)

        results = model(np.array(img))
        # Affichage des résultats
        results_img = np.squeeze(results.render())
        st.image(results_img, caption='Image détectée', use_column_width=True)

elif option == "Webcam (Live)":
    import streamlit_webrtc as webrtc
    st.warning("⚠️ Fonction Webcam nécessite streamlit-webrtc. Installé via pip install streamlit-webrtc")
    
    from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.model = model

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = self.model(img)
            img = np.squeeze(results.render())
            return img

    webrtc_streamer(key="epi-detection", video_transformer_factory=VideoTransformer)
