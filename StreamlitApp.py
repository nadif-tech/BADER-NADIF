# app.py
import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import time
import pandas as pd
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Détection EPI Multi-Couleurs",
    page_icon="🛡️",
    layout="wide"
)

# Initialisation des sessions
if 'detection_active' not in st.session_state:
    st.session_state.detection_active = False
if 'statistiques' not in st.session_state:
    st.session_state.statistiques = defaultdict(int)
if 'historique_detections' not in st.session_state:
    st.session_state.historique_detections = []

# Classes d'EPI avec leurs couleurs associées
EPI_CLASSES = {
    'casque': {
        'couleurs': ['blanc', 'jaune', 'bleu', 'rouge', 'vert', 'orange', 'gris'],
        'importance': 'haute',
        'description': 'Protection de la tête'
    },
    'gants': {
        'couleurs': ['bleu', 'vert', 'rouge', 'jaune', 'blanc', 'noir'],
        'importance': 'haute',
        'description': 'Protection des mains'
    },
    'lunettes': {
        'couleurs': ['transparent', 'fumé', 'jaune', 'bleu', 'clair'],
        'importance': 'moyenne',
        'description': 'Protection des yeux'
    },
    'masque': {
        'couleurs': ['blanc', 'bleu', 'vert', 'noir'],
        'importance': 'haute',
        'description': 'Protection respiratoire'
    },
    'gilet': {
        'couleurs': ['jaune', 'orange', 'vert fluo', 'rouge', 'bleu'],
        'importance': 'moyenne',
        'description': 'Haute visibilité'
    },
    'bottes': {
        'couleurs': ['noir', 'marron', 'vert', 'bleu'],
        'importance': 'moyenne',
        'description': 'Protection des pieds'
    },
    'combinaison': {
        'couleurs': ['blanc', 'bleu', 'vert', 'jaune'],
        'importance': 'haute',
        'description': 'Protection du corps'
    }
}

# Dictionnaire de couleurs BGR pour l'affichage
COULEURS_BGR = {
    'blanc': (255, 255, 255),
    'jaune': (0, 255, 255),
    'bleu': (255, 0, 0),
    'rouge': (0, 0, 255),
    'vert': (0, 255, 0),
    'orange': (0, 165, 255),
    'gris': (128, 128, 128),
    'noir': (0, 0, 0),
    'marron': (42, 42, 165),
    'transparent': (200, 200, 200),
    'fumé': (100, 100, 100),
    'clair': (240, 240, 240),
    'vert fluo': (0, 255, 128)
}

# Couleurs pour les boîtes de détection
COULEURS_BOX = {
    'casque': (255, 0, 255),      # Magenta
    'gants': (255, 165, 0),       # Orange
    'lunettes': (0, 255, 255),    # Jaune
    'masque': (0, 255, 0),        # Vert
    'gilet': (255, 255, 0),       # Cyan
    'bottes': (128, 0, 128),      # Violet
    'combinaison': (255, 192, 203) # Rose
}

class PPEDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """Initialisation du détecteur YOLO"""
        self.model = YOLO(model_path)
        self.confidence_threshold = 0.5
        
    def detect_ppe(self, frame):
        """Détection des EPI dans l'image"""
        results = self.model(frame, conf=self.confidence_threshold)
        return results[0]
    
    def analyze_color(self, roi, detected_class):
        """Analyse de la couleur dominante dans la région détectée"""
        if roi.size == 0:
            return 'non_determine'
        
        # Convertir en HSV pour meilleure analyse des couleurs
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Définir les plages de couleurs en HSV
        color_ranges = {
            'rouge': [(0, 50, 50), (10, 255, 255)],
            'rouge2': [(160, 50, 50), (180, 255, 255)],
            'bleu': [(100, 50, 50), (130, 255, 255)],
            'vert': [(40, 40, 40), (80, 255, 255)],
            'jaune': [(20, 50, 50), (35, 255, 255)],
            'orange': [(5, 50, 50), (15, 255, 255)],
            'violet': [(130, 50, 50), (160, 255, 255)],
            'blanc': [(0, 0, 200), (180, 30, 255)],
            'noir': [(0, 0, 0), (180, 255, 50)],
            'gris': [(0, 0, 50), (180, 30, 200)],
            'marron': [(0, 50, 20), (20, 255, 150)]
        }
        
        # Compter les pixels pour chaque couleur
        color_counts = {}
        hsv_reshaped = hsv.reshape(-1, 3)
        
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            color_counts[color_name] = np.sum(mask > 0)
        
        # Trouver la couleur dominante
        if max(color_counts.values()) > 100:  # Seuil minimum de pixels
            dominant_color = max(color_counts, key=color_counts.get)
            return dominant_color
        
        return 'couleur_non_standard'

def afficher_statistiques():
    """Affiche les statistiques de détection"""
    st.sidebar.header("📊 Statistiques en temps réel")
    
    if st.session_state.statistiques:
        # Créer un DataFrame pour les statistiques
        stats_df = pd.DataFrame([
            {"EPI": epi.replace('_', ' ').title(), 
             "Détections": count}
            for epi, count in st.session_state.statistiques.items()
        ])
        
        # Graphique en barres
        fig = px.bar(stats_df, x="EPI", y="Détections", 
                     title="Détections par type d'EPI",
                     color="EPI")
        st.sidebar.plotly_chart(fig, use_container_width=True)
        
        # Métriques résumées
        total_detections = sum(st.session_state.statistiques.values())
        st.sidebar.metric("Total détections", total_detections)
        
        # Dernières détections
        if st.session_state.historique_detections:
            st.sidebar.subheader("🕐 Dernières détections")
            for det in st.session_state.historique_detections[-5:]:
                st.sidebar.info(f"🟢 {det}")

def interface_principale():
    """Interface principale de l'application"""
    st.title("🛡️ Détection d'Équipements de Protection Individuelle (EPI)")
    st.markdown("### Analyse multi-couleurs en temps réel avec YOLO")
    
    # Barre latérale pour les contrôles
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Sélection du modèle
        model_option = st.selectbox(
            "Modèle YOLO",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            index=0
        )
        
        # Seuil de confiance
        confidence = st.slider(
            "Seuil de confiance",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        # Types d'EPI à détecter
        st.subheader("🔍 EPI à détecter")
        epi_selection = {}
        for epi in EPI_CLASSES.keys():
            epi_selection[epi] = st.checkbox(
                f"{epi.replace('_', ' ').title()}",
                value=True
            )
        
        # Filtre par couleur
        st.subheader("🎨 Filtre par couleur")
        couleur_filtre = st.multiselect(
            "Couleurs à afficher",
            options=list(COULEURS_BGR.keys()),
            default=list(COULEURS_BGR.keys())
        )
        
        st.divider()
        
        # Boutons de contrôle
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button(
                "▶️ Démarrer",
                type="primary",
                use_container_width=True
            )
        with col2:
            stop_button = st.button(
                "⏹️ Arrêter",
                type="secondary",
                use_container_width=True
            )
        
        if start_button:
            st.session_state.detection_active = True
        if stop_button:
            st.session_state.detection_active = False
        
        # Bouton de réinitialisation
        if st.button("🔄 Réinitialiser les stats", use_container_width=True):
            st.session_state.statistiques.clear()
            st.session_state.historique_detections.clear()
            st.rerun()
        
        st.divider()
        
        # Informations sur les EPI
        with st.expander("ℹ️ Types d'EPI"):
            for epi, info in EPI_CLASSES.items():
                st.markdown(f"**{epi.title()}**")
                st.markdown(f"- Couleurs: {', '.join(info['couleurs'])}")
                st.markdown(f"- Importance: {info['importance']}")
                st.markdown("---")
    
    # Zone principale - Flux vidéo
    col_video, col_info = st.columns([2, 1])
    
    with col_video:
        st.subheader("📹 Flux Webcam")
        frame_placeholder = st.empty()
        video_placeholder = st.empty()
        
        # Options d'affichage
        display_options = st.multiselect(
            "Options d'affichage",
            ["Afficher les boîtes", "Afficher les couleurs", "Afficher la confiance"],
            default=["Afficher les boîtes", "Afficher les couleurs"]
        )
    
    with col_info:
        st.subheader("📋 Détections en direct")
        detection_placeholder = st.empty()
        alerte_placeholder = st.empty()
    
    # Affichage des statistiques dans la sidebar
    afficher_statistiques()
    
    # Démarrer la détection
    if st.session_state.detection_active:
        try:
            # Initialisation de la webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Impossible d'ouvrir la webcam")
                return
            
            # Initialisation du détecteur
            detector = PPEDetector(model_path=model_option)
            detector.confidence_threshold = confidence
            
            # Création du placeholder pour les stats en direct
            stats_placeholder = st.empty()
            
            while st.session_state.detection_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Erreur de capture")
                    break
                
                # Redimensionnement pour meilleures performances
                frame = cv2.resize(frame, (640, 480))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Détection
                results = detector.detect_ppe(frame)
                
                # Traitement des détections
                detections_actuelles = []
                
                if results.boxes is not None:
                    for box in results.boxes:
                        # Coordonnées de la boîte
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        
                        # Extraire la région d'intérêt pour l'analyse couleur
                        roi = frame[y1:y2, x1:x2]
                        
                        # Analyse de la couleur
                        couleur_detected = detector.analyze_color(roi, "default")
                        
                        # Vérifier si la couleur est dans le filtre
                        if couleur_detected not in couleur_filtre and couleur_filtre:
                            continue
                        
                        # Type d'EPI détecté (simulation - à adapter selon votre modèle)
                        class_names = ['casque', 'gants', 'lunettes', 'masque', 'gilet', 'bottes', 'combinaison']
                        class_id = int(box.cls[0].cpu().numpy()) if len(box.cls) > 0 else 0
                        epi_type = class_names[class_id % len(class_names)]
                        
                        # Vérifier si ce type d'EPI est sélectionné
                        if not epi_selection.get(epi_type, True):
                            continue
                        
                        # Mise à jour des statistiques
                        st.session_state.statistiques[epi_type] += 1
                        
                        # Enregistrement dans l'historique
                        detection_time = time.strftime("%H:%M:%S")
                        detection_info = f"{detection_time} - {epi_type} ({couleur_detected}) - {conf:.2f}"
                        st.session_state.historique_detections.append(detection_info)
                        
                        detections_actuelles.append({
                            'type': epi_type,
                            'couleur': couleur_detected,
                            'confiance': conf,
                            'position': (x1, y1, x2, y2)
                        })
                        
                        # Dessin sur l'image selon les options
                        if "Afficher les boîtes" in display_options:
                            # Couleur de la boîte selon le type d'EPI
                            box_color = COULEURS_BOX.get(epi_type, (0, 255, 0))
                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                            
                            # Texte d'information
                            text = epi_type.replace('_', ' ').upper()
                            if "Afficher les couleurs" in display_options:
                                text += f" - {couleur_detected}"
                            if "Afficher la confiance" in display_options:
                                text += f" ({conf:.2f})"
                            
                            # Arrière-plan pour le texte
                            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), box_color, -1)
                            
                            # Texte
                            cv2.putText(frame, text, (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Affichage des informations en direct
                with detection_placeholder.container():
                    if detections_actuelles:
                        st.success(f"✅ {len(detections_actuelles)} EPI détectés")
                        
                        # Tableau des détections actuelles
                        df_detections = pd.DataFrame(detections_actuelles)
                        st.dataframe(
                            df_detections[['type', 'couleur', 'confiance']],
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Alertes pour les EPI manquants
                        epi_manquants = []
                        for epi in epi_selection:
                            if epi_selection[epi] and epi not in [d['type'] for d in detections_actuelles]:
                                epi_manquants.append(epi)
                        
                        if epi_manquants:
                            with alerte_placeholder.container():
                                st.warning(f"⚠️ EPI manquants: {', '.join(epi_manquants)}")
                        else:
                            with alerte_placeholder.container():
                                st.success("✅ Tous les EPI requis sont présents!")
                    else:
                        st.info("👀 Aucun EPI détecté")
                
                # Affichage du flux vidéo
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Petite pause pour limiter l'utilisation CPU
                time.sleep(0.03)
            
            cap.release()
            
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")
            st.session_state.detection_active = False

def main():
    """Fonction principale"""
    interface_principale()
    
    # Pied de page
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🛡️ Détection EPI multi-couleurs avec YOLO et Streamlit</p>
            <p style='font-size: 0.8em; color: gray;'>
                Utilisez votre webcam pour détecter les équipements de protection en temps réel
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
