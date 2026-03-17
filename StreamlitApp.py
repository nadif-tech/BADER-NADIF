"""
Application de détection d'Équipements de Protection Individuelle (EPI)
Auteur: Assistant IA
Version: 2.0
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
        background-color: #f0f2f6;
    }
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .alert-success {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .alert-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .alert-danger {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
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

# Classes d'EPI avec leurs couleurs associées
EPI_CLASSES = {
    'casque': {
        'couleurs': ['blanc', 'jaune', 'bleu', 'rouge', 'vert', 'orange', 'gris', 'noir'],
        'importance': 'critique',
        'description': 'Protection de la tête contre les chocs',
        'securite': 'Obligatoire en zone chantier',
        'icone': '⛑️'
    },
    'gants': {
        'couleurs': ['bleu', 'vert', 'rouge', 'jaune', 'blanc', 'noir', 'marron'],
        'importance': 'critique',
        'description': 'Protection des mains contre les coupures et produits chimiques',
        'securite': 'Obligatoire pour manipulation',
        'icone': '🧤'
    },
    'lunettes': {
        'couleurs': ['transparent', 'fumé', 'jaune', 'bleu', 'clair', 'miroir'],
        'importance': 'élevée',
        'description': 'Protection des yeux contre les projections',
        'securite': 'Recommandé en zone de travail',
        'icone': '👓'
    },
    'masque': {
        'couleurs': ['blanc', 'bleu', 'vert', 'noir', 'FFP2', 'FFP3'],
        'importance': 'critique',
        'description': 'Protection respiratoire contre les poussières et particules',
        'securite': 'Obligatoire en zone polluée',
        'icone': '😷'
    },
    'gilet': {
        'couleurs': ['jaune fluo', 'orange fluo', 'vert fluo', 'rouge', 'bleu'],
        'importance': 'élevée',
        'description': 'Haute visibilité pour être vu',
        'securite': 'Obligatoire près des véhicules',
        'icone': '🦺'
    },
    'bottes': {
        'couleurs': ['noir', 'marron', 'vert', 'bleu', 'gris'],
        'importance': 'élevée',
        'description': 'Protection des pieds avec embout',
        'securite': 'Obligatoire au sol',
        'icone': '👢'
    },
    'combinaison': {
        'couleurs': ['blanc', 'bleu', 'vert', 'jaune', 'gris'],
        'importance': 'moyenne',
        'description': 'Protection du corps contre les salissures',
        'securite': 'Recommandé pour travaux spécifiques',
        'icone': '👕'
    },
    'casque_audio': {
        'couleurs': ['jaune', 'noir', 'orange', 'bleu'],
        'importance': 'moyenne',
        'description': 'Protection auditive avec communication',
        'securite': 'Zone de bruit intense',
        'icone': '🎧'
    },
    'harnais': {
        'couleurs': ['bleu', 'orange', 'jaune', 'noir'],
        'importance': 'critique',
        'description': 'Protection contre les chutes en hauteur',
        'securite': 'Obligatoire en hauteur',
        'icone': '🪢'
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
    'FFP2': (255, 255, 255),
    'FFP3': (255, 255, 255)
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Charge le modèle YOLO"""
        try:
            self.model = YOLO(model_path)
            st.success(f"✅ Modèle chargé avec succès sur {self.device}")
        except Exception as e:
            st.error(f"❌ Erreur de chargement du modèle: {e}")
            self.model = None
    
    def detect_objects(self, frame):
        """Détection des objets dans l'image"""
        if self.model is None:
            return None
        
        results = self.model(frame, conf=self.confidence_threshold)
        return results[0]
    
    def analyze_color_hsv(self, roi):
        """Analyse avancée des couleurs en espace HSV"""
        if roi.size == 0:
            return 'non_determiné'
        
        # Conversion en HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Définir les plages de couleurs
        color_ranges = {
            'rouge': [(0, 50, 50), (10, 255, 255)],
            'rouge_fonce': [(160, 50, 50), (180, 255, 255)],
            'bleu': [(100, 50, 50), (130, 255, 255)],
            'bleu_clair': [(90, 50, 50), (100, 255, 255)],
            'vert': [(40, 40, 40), (80, 255, 255)],
            'vert_clair': [(35, 40, 40), (40, 255, 255)],
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
        
        # Calculer les masques et compter les pixels
        color_counts = {}
        total_pixels = roi.shape[0] * roi.shape[1]
        
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            color_counts[color_name] = np.sum(mask > 0)
        
        # Trouver la couleur dominante (min 10% de l'image)
        if max(color_counts.values()) > total_pixels * 0.1:
            dominant_color = max(color_counts, key=color_counts.get)
            
            # Fusionner les variantes de rouge
            if dominant_color in ['rouge_fonce']:
                dominant_color = 'rouge'
            
            return dominant_color
        
        return 'couleur_non_standard'
    
    def estimate_ppe_compliance(self, detections):
        """Estime la conformité des EPI"""
        required_ppe = ['casque', 'gants', 'lunettes', 'masque', 'gilet', 'bottes']
        present_ppe = [d['type'] for d in detections]
        
        missing_ppe = [ppe for ppe in required_ppe if ppe not in present_ppe]
        compliance_score = ((len(required_ppe) - len(missing_ppe)) / len(required_ppe)) * 100
        
        return compliance_score, missing_ppe

def get_image_download_link(img, filename, text):
    """Génère un lien de téléchargement pour l'image"""
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def afficher_dashboard():
    """Affiche le dashboard des statistiques"""
    st.sidebar.header("📊 Tableau de bord")
    
    if st.session_state.statistiques:
        # Métriques globales
        total_detections = sum(sum(couleurs.values()) for couleurs in st.session_state.statistiques.values())
        st.sidebar.metric("Total détections", total_detections)
        
        # Graphique des détections par type
        df_stats = []
        for epi, couleurs in st.session_state.statistiques.items():
            for couleur, count in couleurs.items():
                df_stats.append({
                    'EPI': epi,
                    'Couleur': couleur,
                    'Détections': count
                })
        
        if df_stats:
            df = pd.DataFrame(df_stats)
            
            # Graphique en barres
            fig = px.bar(df, x='EPI', y='Détections', color='Couleur',
                        title="Détections par type et couleur",
                        barmode='group')
            st.sidebar.plotly_chart(fig, use_container_width=True)
            
            # Camembert des proportions
            fig2 = px.pie(df, values='Détections', names='EPI',
                         title="Répartition des détections")
            st.sidebar.plotly_chart(fig2, use_container_width=True)
    
    # Historique
    with st.sidebar.expander("📜 Historique"):
        for det in st.session_state.historique_detections[-10:]:
            st.text(det)

def interface_principale():
    """Interface principale de l'application"""
    
    # En-tête
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Détection d'Équipements de Protection Individuelle (EPI)</h1>
        <p>Analyse multi-couleurs en temps réel avec intelligence artificielle</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Barre latérale - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Modèle
        model_option = st.selectbox(
            "Modèle YOLO",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
            index=0,
            help="Choisissez le modèle (n=nanos, s=small, m=medium, l=large)"
        )
        
        # Seuil de confiance
        confidence = st.slider(
            "Seuil de confiance",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Plus le seuil est bas, plus de détections mais plus de faux positifs"
        )
        
        # Source vidéo
        st.subheader("📹 Source vidéo")
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
            video_url = st.text_input("Entrez l'URL de la vidéo")
        
        # Filtres
        st.subheader("🎨 Filtres")
        
        # Types d'EPI à détecter
        epi_selection = {}
        cols = st.columns(2)
        for i, (epi, info) in enumerate(EPI_CLASSES.items()):
            with cols[i % 2]:
                epi_selection[epi] = st.checkbox(
                    f"{info['icone']} {epi.replace('_', ' ').title()}",
                    value=True
                )
        
        # Filtre par couleur
        couleur_filtre = st.multiselect(
            "Couleurs à afficher",
            options=list(COULEURS_BGR.keys()),
            default=[]
        )
        
        # Options d'affichage
        st.subheader("🖼️ Affichage")
        display_options = st.multiselect(
            "Options",
            ["Boîtes", "Couleurs", "Confiance", "Labels", "Trajectoires"],
            default=["Boîtes", "Couleurs", "Confiance"]
        )
        
        # Contrôles
        st.divider()
        
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
        
        # Réinitialisation
        if st.button("🔄 Réinitialiser", use_container_width=True):
            st.session_state.statistiques.clear()
            st.session_state.historique_detections.clear()
            st.session_state.alertes.clear()
            st.rerun()
        
        st.divider()
        
        # Information sur les EPI
        with st.expander("ℹ️ Guide des EPI"):
            for epi, info in EPI_CLASSES.items():
                st.markdown(f"""
                **{info['icone']} {epi.replace('_', ' ').title()}**  
                - 📝 {info['description']}  
                - 🎨 Couleurs: {', '.join(info['couleurs'])}  
                - ⚠️ Importance: {info['importance']}  
                - 🔒 {info['securite']}
                """)
                st.divider()
    
    # Zone principale
    col_video, col_info = st.columns([2, 1])
    
    with col_video:
        st.subheader("📹 Flux vidéo")
        
        # Placeholders pour la vidéo
        video_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        # Bouton de capture
        col_cap1, col_cap2, col_cap3 = st.columns(3)
        with col_cap1:
            capture_button = st.button("📸 Capturer")
        with col_cap2:
            record_button = st.button("⏺️ Enregistrer")
        with col_cap3:
            screenshot_button = st.button("📷 Screenshot")
    
    with col_info:
        st.subheader("📋 Détections en direct")
        detection_placeholder = st.empty()
        alerte_placeholder = st.empty()
        compliance_placeholder = st.empty()
    
    # DASHBOARD
    afficher_dashboard()
    
    # Démarrer la détection
    if st.session_state.detection_active:
        try:
            # Initialisation de la source vidéo
            cap = None
            if source_option == "Webcam":
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ Impossible d'ouvrir la webcam")
                    return
            elif source_option == "Upload vidéo" and uploaded_file:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                cap = cv2.VideoCapture(tfile.name)
            elif source_option == "Image" and uploaded_file:
                image = Image.open(uploaded_file)
                frame = np.array(image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif source_option == "URL" and video_url:
                cap = cv2.VideoCapture(video_url)
            
            # Initialisation du détecteur
            detector = PPEDetector(model_path=model_option)
            detector.confidence_threshold = confidence
            
            # Variables pour le suivi
            fps = 0
            frame_count = 0
            start_time = time.time()
            
            while st.session_state.detection_active:
                
                # Lecture du frame
                if cap is not None:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Fin de la vidéo")
                        break
                else:
                    # Pour l'image statique
                    if 'frame' in locals():
                        pass
                    else:
                        st.warning("Aucune source vidéo")
                        break
                
                # Redimensionnement
                frame = cv2.resize(frame, (854, 480))
                
                # Calcul FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    end_time = time.time()
                    fps = 30 / (end_time - start_time)
                    start_time = time.time()
                
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
                        roi = frame[y1:y2, x1:x2]
                        roi = cv2.resize(roi, (100, 100)) if roi.size > 0 else roi
                        
                        # Analyse couleur
                        couleur_detected = detector.analyze_color_hsv(roi)
                        
                        # Filtre par couleur
                        if couleur_filtre and couleur_detected not in couleur_filtre:
                            continue
                        
                        # Type d'EPI (simulation - à adapter selon votre modèle entraîné)
                        class_names = list(EPI_CLASSES.keys())
                        class_id = int(box.cls[0].cpu().numpy()) if len(box.cls) > 0 else 0
                        epi_type = class_names[class_id % len(class_names)]
                        
                        # Filtre par type
                        if not epi_selection.get(epi_type, True):
                            continue
                        
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
                            'position': (x1, y1, x2, y2)
                        })
                        
                        # Dessin sur l'image
                        if "Boîtes" in display_options:
                            box_color = COULEURS_BOX.get(epi_type, (0, 255, 0))
                            
                            # Boîte
                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                            
                            # Label
                            label_parts = []
                            if "Labels" in display_options:
                                label_parts.append(epi_type.replace('_', ' ').upper())
                            if "Couleurs" in display_options:
                                label_parts.append(couleur_detected)
                            if "Confiance" in display_options:
                                label_parts.append(f"{conf:.2f}")
                            
                            label = " - ".join(label_parts)
                            
                            # Fond pour le texte
                            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), box_color, -1)
                            
                            # Texte
                            cv2.putText(frame, label, (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        # Trajectoires (simplifié)
                        if "Trajectoires" in display_options:
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            cv2.circle(frame, center, 3, (0, 255, 0), -1)
                
                # Calcul du score de conformité
                compliance_score, missing_ppe = detector.estimate_ppe_compliance(detections_actuelles)
                
                # Ajout FPS sur l'image
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Mise à jour des informations
                with detection_placeholder.container():
                    if detections_actuelles:
                        st.success(f"✅ {len(detections_actuelles)} EPI détectés")
                        
                        # Tableau
                        df_det = pd.DataFrame(detections_actuelles)
                        st.dataframe(
                            df_det[['type', 'couleur', 'confiance']],
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("👀 Aucun EPI détecté")
                
                # Alertes de conformité
                with compliance_placeholder.container():
                    if compliance_score >= 80:
                        st.markdown(f"""
                        <div class="alert-success">
                            ✅ Score de conformité: {compliance_score:.1f}% - Excellent!
                        </div>
                        """, unsafe_allow_html=True)
                    elif compliance_score >= 50:
                        st.markdown(f"""
                        <div class="alert-warning">
                            ⚠️ Score de conformité: {compliance_score:.1f}% - Améliorable
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="alert-danger">
                            ❌ Score de conformité: {compliance_score:.1f}% - Critique!
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if missing_ppe:
                        st.warning(f"EPI manquants: {', '.join(missing_ppe)}")
                
                # Alertes importantes
                with alerte_placeholder.container():
                    alertes = []
                    for ppe in ['casque', 'gants', 'bottes']:
                        if ppe not in [d['type'] for d in detections_actuelles]:
                            if EPI_CLASSES[ppe]['importance'] == 'critique':
                                alertes.append(f"🔴 {ppe.upper()} MANQUANT!")
                    
                    if alertes:
                        for alerte in alertes:
                            st.error(alerte)
                
                # Actions de capture
                if capture_button:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"capture_epi_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    st.success(f"Capture sauvegardée: {filename}")
                
                if screenshot_button:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(frame_rgb)
                    
                    # Lien de téléchargement
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    href = get_image_download_link(img_pil, f"screenshot_{timestamp}.jpg", "📥 Télécharger screenshot")
                    st.markdown(href, unsafe_allow_html=True)
                
                # Affichage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Stats en temps réel
                with stats_placeholder.container():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Détections", len(detections_actuelles))
                    with col2:
                        st.metric("FPS", f"{fps:.1f}")
                    with col3:
                        st.metric("Conformité", f"{compliance_score:.1f}%")
                    with col4:
                        st.metric("Alertes", len(alertes))
                
                # Petite pause
                time.sleep(0.03)
            
            # Libération des ressources
            if cap is not None:
                cap.release()
                
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")
            st.session_state.detection_active = False

def main():
    """Fonction principale"""
    
    try:
        import torch
        interface_principale()
    except ImportError as e:
        st.error(f"Erreur d'importation: {e}")
        st.info("Vérifiez que toutes les dépendances sont installées")
    
    # Pied de page
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px;'>
        <p style='margin: 0;'>🛡️ Détection EPI Multi-Couleurs v2.0 - Intelligence Artificielle pour la Sécurité au Travail</p>
        <p style='margin: 0; font-size: 0.8em; opacity: 0.8;'>© 2024 - Tous droits réservés</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
