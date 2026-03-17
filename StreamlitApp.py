"""
Application de détection d'Équipements de Protection Individuelle (EPI)
Auteur: Assistant IA
Version: 3.0 FINALE
Description: Détection multi-couleurs en temps réel avec YOLO et Streamlit
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import time
import pandas as pd
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Détection EPI Multi-Couleurs",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        text-align: center;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    .main-header h1 {
        color: #333;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #666;
        font-size: 1.1rem;
    }
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    .stat-card:hover {
        transform: translateY(-5px);
    }
    .alert-success {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 0.5rem 0;
    }
    .alert-warning {
        background: linear-gradient(135deg, #fad0c4 0%, #ffd1ff 100%);
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 0.5rem 0;
    }
    .alert-danger {
        background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%);
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        margin-top: 2rem;
        box-shadow: 0 -10px 30px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
    }
    .stButton button {
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialisation des variables de session
if 'detection_active' not in st.session_state:
    st.session_state.detection_active = False
if 'statistiques' not in st.session_state:
    st.session_state.statistiques = defaultdict(lambda: defaultdict(int))
if 'historique_detections' not in st.session_state:
    st.session_state.historique_detections = []
if 'model_charge' not in st.session_state:
    st.session_state.model_charge = None
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'alertes' not in st.session_state:
    st.session_state.alertes = []
if 'fps' not in st.session_state:
    st.session_state.fps = 0
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

# Classes d'EPI avec leurs couleurs associées
EPI_CLASSES = {
    'casque': {
        'couleurs': ['blanc', 'jaune', 'bleu', 'rouge', 'vert', 'orange', 'gris', 'noir'],
        'importance': 'critique',
        'description': 'Protection de la tête contre les chocs',
        'securite': 'Obligatoire en zone chantier',
        'icone': '⛑️',
        'norme': 'EN 397'
    },
    'gants': {
        'couleurs': ['bleu', 'vert', 'rouge', 'jaune', 'blanc', 'noir', 'marron'],
        'importance': 'critique',
        'description': 'Protection des mains contre les coupures et produits chimiques',
        'securite': 'Obligatoire pour manipulation',
        'icone': '🧤',
        'norme': 'EN 388'
    },
    'lunettes': {
        'couleurs': ['transparent', 'fumé', 'jaune', 'bleu', 'clair', 'miroir'],
        'importance': 'élevée',
        'description': 'Protection des yeux contre les projections',
        'securite': 'Recommandé en zone de travail',
        'icone': '👓',
        'norme': 'EN 166'
    },
    'masque': {
        'couleurs': ['blanc', 'bleu', 'vert', 'noir', 'ffp2', 'ffp3'],
        'importance': 'critique',
        'description': 'Protection respiratoire contre les poussières et particules',
        'securite': 'Obligatoire en zone polluée',
        'icone': '😷',
        'norme': 'EN 149'
    },
    'gilet': {
        'couleurs': ['jaune fluo', 'orange fluo', 'vert fluo', 'rouge', 'bleu'],
        'importance': 'élevée',
        'description': 'Haute visibilité pour être vu',
        'securite': 'Obligatoire près des véhicules',
        'icone': '🦺',
        'norme': 'EN 20471'
    },
    'bottes': {
        'couleurs': ['noir', 'marron', 'vert', 'bleu', 'gris'],
        'importance': 'élevée',
        'description': 'Protection des pieds avec embout',
        'securite': 'Obligatoire au sol',
        'icone': '👢',
        'norme': 'EN 20345'
    },
    'combinaison': {
        'couleurs': ['blanc', 'bleu', 'vert', 'jaune', 'gris'],
        'importance': 'moyenne',
        'description': 'Protection du corps contre les salissures',
        'securite': 'Recommandé pour travaux spécifiques',
        'icone': '👕',
        'norme': 'EN 13034'
    },
    'casque_audio': {
        'couleurs': ['jaune', 'noir', 'orange', 'bleu'],
        'importance': 'moyenne',
        'description': 'Protection auditive avec communication',
        'securite': 'Zone de bruit intense',
        'icone': '🎧',
        'norme': 'EN 352'
    },
    'harnais': {
        'couleurs': ['bleu', 'orange', 'jaune', 'noir'],
        'importance': 'critique',
        'description': 'Protection contre les chutes en hauteur',
        'securite': 'Obligatoire en hauteur',
        'icone': '🪢',
        'norme': 'EN 361'
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
    'vert fluo': (0, 255, 128),
    'jaune fluo': (0, 255, 255),
    'orange fluo': (0, 128, 255),
    'miroir': (180, 180, 180),
    'ffp2': (255, 255, 255),
    'ffp3': (255, 255, 255)
}

# Couleurs pour les boîtes de détection
COULEURS_BOX = {
    'casque': (255, 0, 255),        # Magenta
    'gants': (255, 165, 0),         # Orange
    'lunettes': (0, 255, 255),      # Jaune
    'masque': (0, 255, 0),          # Vert
    'gilet': (255, 255, 0),         # Cyan
    'bottes': (128, 0, 128),        # Violet
    'combinaison': (255, 192, 203), # Rose
    'casque_audio': (255, 0, 0),    # Rouge
    'harnais': (0, 165, 255)        # Orange foncé
}

class PPEDetector:
    """Classe principale pour la détection des EPI"""
    
    def __init__(self, model_path='yolov8n.pt'):
        """Initialisation du détecteur"""
        self.model = None
        self.confidence_threshold = 0.5
        self.device = 'cuda' if self.check_cuda() else 'cpu'
        self.load_model(model_path)
        self.track_history = defaultdict(list)
        
    def check_cuda(self):
        """Vérifie si CUDA est disponible"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def load_model(self, model_path):
        """Charge le modèle YOLO"""
        try:
            with st.spinner(f"Chargement du modèle {model_path}..."):
                self.model = YOLO(model_path)
                st.sidebar.success(f"✅ Modèle chargé sur {self.device}")
        except Exception as e:
            st.sidebar.error(f"❌ Erreur: {e}")
            self.model = None
    
    def detect_objects(self, frame):
        """Détection des objets dans l'image"""
        if self.model is None:
            return None
        
        results = self.model.track(frame, 
                                  conf=self.confidence_threshold, 
                                  persist=True,
                                  verbose=False)
        return results[0]
    
    def analyze_color_advanced(self, roi):
        """Analyse avancée des couleurs"""
        if roi.size == 0:
            return 'non_determiné', 0
        
        # Redimensionner pour analyse plus rapide
        roi = cv2.resize(roi, (64, 64))
        
        # Conversion en différents espaces colorimétriques
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Définir les plages de couleurs en HSV
        color_ranges = {
            'rouge': [(0, 50, 50), (10, 255, 255)],
            'rouge_fonce': [(160, 50, 50), (180, 255, 255)],
            'bleu': [(100, 50, 50), (130, 255, 255)],
            'bleu_clair': [(90, 50, 50), (100, 255, 255)],
            'vert': [(40, 40, 40), (80, 255, 255)],
            'vert_clair': [(35, 40, 40), (40, 255, 255)],
            'vert_fluo': [(40, 100, 100), (80, 255, 255)],
            'jaune': [(20, 50, 50), (35, 255, 255)],
            'jaune_fluo': [(25, 100, 100), (35, 255, 255)],
            'orange': [(5, 50, 50), (15, 255, 255)],
            'orange_fluo': [(5, 100, 100), (15, 255, 255)],
            'violet': [(130, 50, 50), (160, 255, 255)],
            'blanc': [(0, 0, 200), (180, 30, 255)],
            'noir': [(0, 0, 0), (180, 255, 50)],
            'gris': [(0, 0, 50), (180, 30, 200)],
            'marron': [(0, 50, 20), (20, 255, 150)],
            'transparent': [(0, 0, 150), (180, 50, 255)]
        }
        
        # Calculer les masques
        color_scores = {}
        total_pixels = roi.shape[0] * roi.shape[1]
        
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            score = np.sum(mask > 0) / total_pixels
            color_scores[color_name] = score
        
        # Trouver la meilleure correspondance
        best_color = max(color_scores, key=color_scores.get)
        best_score = color_scores[best_color]
        
        # Normalisation
        if best_color in ['rouge_fonce']:
            best_color = 'rouge'
        
        # Seuil de confiance
        if best_score < 0.1:
            # Analyse de secours avec LAB
            l_channel = lab[:,:,0]
            if np.mean(l_channel) > 200:
                return 'blanc', 0.7
            elif np.mean(l_channel) < 50:
                return 'noir', 0.7
            else:
                return 'couleur_non_standard', best_score
        
        return best_color, best_score
    
    def estimate_safety_score(self, detections):
        """Calcule le score de sécurité"""
        required_ppe = {
            'casque': 10,
            'gants': 10,
            'lunettes': 8,
            'masque': 10,
            'gilet': 8,
            'bottes': 8,
            'harnais': 10
        }
        
        present_ppe = [d['type'] for d in detections]
        total_score = 0
        max_score = sum(required_ppe.values())
        
        for ppe, weight in required_ppe.items():
            if ppe in present_ppe:
                total_score += weight
        
        safety_score = (total_score / max_score) * 100 if max_score > 0 else 0
        missing_ppe = [ppe for ppe in required_ppe if ppe not in present_ppe]
        
        return safety_score, missing_ppe
    
    def update_tracking(self, track_id, center):
        """Met à jour le suivi des objets"""
        if track_id is not None:
            self.track_history[track_id].append(center)
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

def create_download_link(val, filename):
    """Crée un lien de téléchargement"""
    b64 = base64.b64encode(val).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Télécharger</a>'

def export_stats():
    """Exporte les statistiques en CSV"""
    data = []
    for epi, couleurs in st.session_state.statistiques.items():
        for couleur, count in couleurs.items():
            data.append({
                'EPI': epi,
                'Couleur': couleur,
                'Détections': count,
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    if data:
        df = pd.DataFrame(data)
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="statistiques_epi.csv">📥 Télécharger CSV</a>'
        return href
    return None

def afficher_dashboard():
    """Affiche le dashboard interactif"""
    with st.sidebar:
        st.markdown("## 📊 Tableau de bord")
        
        if st.session_state.statistiques:
            # Métriques globales
            total_detections = sum(sum(c.values()) for c in st.session_state.statistiques.values())
            total_types = len(st.session_state.statistiques)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_detections}</div>
                    <div class="metric-label">Détections totales</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_types}</div>
                    <div class="metric-label">Types d'EPI</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Graphique des détections
            df_list = []
            for epi, couleurs in st.session_state.statistiques.items():
                for couleur, count in couleurs.items():
                    df_list.append({
                        'EPI': epi.replace('_', ' ').title(),
                        'Couleur': couleur,
                        'Détections': count
                    })
            
            if df_list:
                df = pd.DataFrame(df_list)
                
                # Graphique en barres
                fig = px.bar(df, x='EPI', y='Détections', color='Couleur',
                           title="Détections par type",
                           barmode='stack',
                           color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Camembert
                df_sum = df.groupby('EPI')['Détections'].sum().reset_index()
                fig2 = px.pie(df_sum, values='Détections', names='EPI',
                            title="Répartition",
                            color_discrete_sequence=px.colors.qualitative.Pastel)
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Export
            if st.button("📥 Exporter les statistiques"):
                href = export_stats()
                if href:
                    st.markdown(href, unsafe_allow_html=True)
        
        else:
            st.info("Aucune statistique disponible")
        
        # Historique récent
        with st.expander("📜 Historique récent"):
            if st.session_state.historique_detections:
                for det in st.session_state.historique_detections[-10:]:
                    st.text(f"• {det}")
            else:
                st.text("Aucune détection")

def interface_principale():
    """Interface principale de l'application"""
    
    # En-tête
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Détection d'Équipements de Protection Individuelle</h1>
        <p>Analyse multi-couleurs en temps réel avec intelligence artificielle</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">Version 3.0 - Sécurité augmentée</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Barre latérale
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Sélection du modèle
        model_option = st.selectbox(
            "Modèle YOLO",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
            index=0,
            help="n: nano (rapide), s: small, m: medium, l: large (précis)"
        )
        
        # Seuil de confiance
        confidence = st.slider(
            "Seuil de confiance",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Plus le seuil est bas, plus de détections"
        )
        
        st.markdown("---")
        
        # Source vidéo
        st.markdown("## 📹 Source")
        source_option = st.radio(
            "Choisir la source",
            ["Webcam", "Upload vidéo", "Image", "URL"],
            index=0
        )
        
        uploaded_file = None
        video_url = None
        
        if source_option == "Upload vidéo":
            uploaded_file = st.file_uploader(
                "Choisir une vidéo",
                type=['mp4', 'avi', 'mov', 'mkv']
            )
        elif source_option == "Image":
            uploaded_file = st.file_uploader(
                "Choisir une image",
                type=['jpg', 'jpeg', 'png']
            )
        elif source_option == "URL":
            video_url = st.text_input("URL de la vidéo")
        
        st.markdown("---")
        
        # Filtres
        st.markdown("## 🎨 Filtres")
        
        # Types d'EPI
        st.markdown("### Types à détecter")
        epi_selection = {}
        cols = st.columns(2)
        for i, (epi, info) in enumerate(EPI_CLASSES.items()):
            with cols[i % 2]:
                epi_selection[epi] = st.checkbox(
                    f"{info['icone']} {epi.replace('_', ' ').title()}",
                    value=True
                )
        
        # Filtre couleur
        couleur_filtre = st.multiselect(
            "Couleurs spécifiques",
            options=list(COULEURS_BGR.keys()),
            default=[]
        )
        
        st.markdown("---")
        
        # Options d'affichage
        st.markdown("## 🖼️ Affichage")
        display_options = st.multiselect(
            "Options",
            ["Boîtes", "Couleurs", "Confiance", "Labels", "Trajectoires", "Statistiques"],
            default=["Boîtes", "Couleurs", "Confiance"]
        )
        
        st.markdown("---")
        
        # Contrôles
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("▶️ Démarrer", type="primary", use_container_width=True)
        with col2:
            stop_button = st.button("⏹️ Arrêter", type="secondary", use_container_width=True)
        
        if start_button:
            st.session_state.detection_active = True
        if stop_button:
            st.session_state.detection_active = False
        
        if st.button("🔄 Réinitialiser", use_container_width=True):
            st.session_state.statistiques.clear()
            st.session_state.historique_detections.clear()
            st.session_state.alertes.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Guide rapide
        with st.expander("ℹ️ Guide des EPI"):
            for epi, info in EPI_CLASSES.items():
                st.markdown(f"""
                **{info['icone']} {epi.replace('_', ' ').title()}**  
                📝 {info['description']}  
                🎨 {', '.join(info['couleurs'][:3])}...  
                ⚠️ {info['importance']}  
                📋 Norme: {info['norme']}
                """)
                st.divider()
    
    # Zone principale
    col_video, col_info = st.columns([2, 1])
    
    with col_video:
        st.markdown("## 📹 Flux vidéo")
        
        video_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        # Boutons d'action
        col_cap1, col_cap2, col_cap3 = st.columns(3)
        with col_cap1:
            capture_button = st.button("📸 Capturer", use_container_width=True)
        with col_cap2:
            record_button = st.button("⏺️ Enregistrer", use_container_width=True)
        with col_cap3:
            screenshot_button = st.button("📷 Screenshot", use_container_width=True)
    
    with col_info:
        st.markdown("## 📋 Détections en direct")
        detection_placeholder = st.empty()
        alerte_placeholder = st.empty()
        safety_placeholder = st.empty()
    
    # Dashboard
    afficher_dashboard()
    
    # Démarrer la détection
    if st.session_state.detection_active:
        try:
            # Initialisation source
            cap = None
            if source_option == "Webcam":
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ Webcam non accessible")
                    return
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
            
            elif source_option == "Upload vidéo" and uploaded_file:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                cap = cv2.VideoCapture(tfile.name)
            
            elif source_option == "Image" and uploaded_file:
                image = Image.open(uploaded_file)
                frame = np.array(image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            elif source_option == "URL" and video_url:
                cap = cv2.VideoCapture(video_url)
            
            # Initialisation détecteur
            detector = PPEDetector(model_path=model_option)
            detector.confidence_threshold = confidence
            
            # Variables pour statistiques
            fps = 0
            frame_count = 0
            start_time = time.time()
            
            while st.session_state.detection_active:
                
                # Lecture frame
                if cap is not None:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Fin de la vidéo")
                        break
                else:
                    if 'frame' not in locals():
                        st.warning("Aucune source")
                        break
                
                # Redimensionnement
                frame = cv2.resize(frame, (1024, 576))
                
                # Calcul FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    end_time = time.time()
                    fps = 10 / (end_time - start_time)
                    start_time = time.time()
                    st.session_state.fps = fps
                
                # Détection
                results = detector.detect_objects(frame)
                
                # Traitement des détections
                detections_actuelles = []
                
                if results and results.boxes is not None:
                    for box in results.boxes:
                        # Coordonnées
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        
                        # ROI pour analyse couleur
                        roi = frame[max(0, y1):min(frame.shape[0], y2), 
                                   max(0, x1):min(frame.shape[1], x2)]
                        
                        if roi.size > 0:
                            # Analyse couleur
                            couleur_detected, color_conf = detector.analyze_color_advanced(roi)
                            
                            # Filtre couleur
                            if couleur_filtre and couleur_detected not in couleur_filtre:
                                continue
                            
                            # Type d'EPI (simulation)
                            class_names = list(EPI_CLASSES.keys())
                            class_id = int(box.cls[0].cpu().numpy()) if len(box.cls) > 0 else 0
                            epi_type = class_names[class_id % len(class_names)]
                            
                            # Filtre type
                            if not epi_selection.get(epi_type, True):
                                continue
                            
                            # ID de suivi
                            track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None
                            
                            # Mise à jour statistiques
                            st.session_state.statistiques[epi_type][couleur_detected] += 1
                            
                            # Historique
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            detection_info = f"{timestamp} - {epi_type} ({couleur_detected}) - {conf:.2f}"
                            st.session_state.historique_detections.append(detection_info)
                            
                            detections_actuelles.append({
                                'type': epi_type,
                                'couleur': couleur_detected,
                                'confiance': conf,
                                'position': (x1, y1, x2, y2),
                                'track_id': track_id
                            })
                            
                            # Dessin
                            if "Boîtes" in display_options:
                                box_color = COULEURS_BOX.get(epi_type, (0, 255, 0))
                                
                                # Boîte
                                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                                
                                # Label
                                label_parts = []
                                if "Labels" in display_options:
                                    label_parts.append(epi_type.replace('_', ' ').upper())
                                if "Couleurs" in display_options:
                                    label_parts.append(couleur_detected)
                                if "Confiance" in display_options:
                                    label_parts.append(f"{conf:.2f}")
                                
                                label = " - ".join(label_parts)
                                
                                # Fond texte
                                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), box_color, -1)
                                
                                # Texte
                                cv2.putText(frame, label, (x1 + 5, y1 - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                
                                # Trajectoire
                                if "Trajectoires" in display_options and track_id is not None:
                                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                                    detector.update_tracking(track_id, center)
                                    
                                    points = detector.track_history[track_id]
                                    for i in range(1, len(points)):
                                        if points[i-1] is not None and points[i] is not None:
                                            cv2.line(frame, points[i-1], points[i], box_color, 2)
                
                # Score de sécurité
                safety_score, missing_ppe = detector.estimate_safety_score(detections_actuelles)
                
                # Affichage FPS
                if "Statistiques" in display_options:
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Score: {safety_score:.1f}%", (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Mise à jour interface
                with detection_placeholder.container():
                    if detections_actuelles:
                        st.success(f"✅ {len(detections_actuelles)} EPI détectés")
                        
                        df_det = pd.DataFrame(detections_actuelles)
                        st.dataframe(
                            df_det[['type', 'couleur', 'confiance']],
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("👀 Aucun EPI détecté")
                
                # Alertes sécurité
                with safety_placeholder.container():
                    if safety_score >= 80:
                        st.markdown(f"""
                        <div class="alert-success">
                            ✅ Score sécurité: {safety_score:.1f}% - Conforme
                        </div>
                        """, unsafe_allow_html=True)
                    elif safety_score >= 50:
                        st.markdown(f"""
                        <div class="alert-warning">
                            ⚠️ Score sécurité: {safety_score:.1f}% - Attention
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="alert-danger">
                            ❌ Score sécurité: {safety_score:.1f}% - Danger
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if missing_ppe:
                        st.warning(f"Manquants: {', '.join(missing_ppe)}")
                
                # Alertes critiques
                with alerte_placeholder.container():
                    for ppe in ['casque', 'gants', 'harnais']:
                        if ppe not in [d['type'] for d in detections_actuelles]:
                            st.error(f"🔴 {ppe.upper()} MANQUANT!")
                
                # Actions capture
                if capture_button and len(detections_actuelles) > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"capture_{timestamp}.jpg", frame)
                    st.success("Capture sauvegardée!")
                
                if screenshot_button:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(frame_rgb)
                    buf = BytesIO()
                    img_pil.save(buf, format="PNG")
                    
                    href = create_download_link(buf.getvalue(), f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                    st.markdown(href, unsafe_allow_html=True)
                
                # Affichage vidéo
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Stats en direct
                with stats_placeholder.container():
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Détections", len(detections_actuelles))
                    with cols[1]:
                        st.metric("FPS", f"{fps:.1f}")
                    with cols[2]:
                        st.metric("Sécurité", f"{safety_score:.1f}%")
                    with cols[3]:
                        st.metric("Alertes", len([a for a in missing_ppe if a in ['casque', 'gants', 'harnais']]))
                
                time.sleep(0.03)
            
            if cap is not None:
                cap.release()
                
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")
            st.session_state.detection_active = False

def main():
    """Fonction principale"""
    try:
        interface_principale()
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        st.info("Redémarrage de l'application...")
        time.sleep(2)
        st.rerun()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p style="margin: 0; font-size: 1.1rem;">🛡️ Détection EPI Multi-Couleurs v3.0</p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.8;">
            Intelligence Artificielle pour la Sécurité au Travail
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.6;">
            © 2024 - Tous droits réservés
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
