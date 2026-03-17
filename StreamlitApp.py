"""
Application simple de détection EPI - Version légère
Sans OpenCV complexe, avec simulation pour Streamlit Cloud
"""

import streamlit as st
import numpy as np
import time
import pandas as pd
from datetime import datetime
import random
from collections import defaultdict
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Détection EPI Simple",
    page_icon="🛡️",
    layout="wide"
)

# Style CSS simple
st.markdown("""
<style>
    .main-title {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .epi-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .danger-box {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation session
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'historique' not in st.session_state:
    st.session_state.historique = []
if 'en_cours' not in st.session_state:
    st.session_state.en_cours = False

# Liste des EPI
EPIS = [
    {"nom": "Casque", "icone": "⛑️", "couleurs": ["Blanc", "Jaune", "Bleu", "Rouge"], "critique": True},
    {"nom": "Gants", "icone": "🧤", "couleurs": ["Bleu", "Vert", "Rouge", "Noir"], "critique": True},
    {"nom": "Lunettes", "icone": "👓", "couleurs": ["Transparent", "Fumé", "Jaune"], "critique": False},
    {"nom": "Masque", "icone": "😷", "couleurs": ["Blanc", "Bleu", "FFP2"], "critique": True},
    {"nom": "Gilet", "icone": "🦺", "couleurs": ["Jaune fluo", "Orange fluo"], "critique": False},
    {"nom": "Bottes", "icone": "👢", "couleurs": ["Noir", "Marron"], "critique": False}
]

def simuler_detection():
    """Simule une détection aléatoire"""
    epi = random.choice(EPIS)
    couleur = random.choice(epi["couleurs"])
    confiance = random.uniform(0.7, 0.99)
    
    return {
        "nom": epi["nom"],
        "icone": epi["icone"],
        "couleur": couleur,
        "confiance": confiance,
        "critique": epi["critique"],
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

# Titre principal
st.markdown("""
<div class="main-title">
    <h1>🛡️ Détection EPI - Version Simple</h1>
    <p>Application légère pour Streamlit Cloud</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Contrôles")
    
    # Boutons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Démarrer", type="primary", use_container_width=True):
            st.session_state.en_cours = True
    with col2:
        if st.button("⏹️ Arrêter", use_container_width=True):
            st.session_state.en_cours = False
    
    # Vitesse de simulation
    vitesse = st.slider("Vitesse simulation", 1, 10, 5)
    
    # Réinitialisation
    if st.button("🔄 Réinitialiser", use_container_width=True):
        st.session_state.detections = []
        st.session_state.historique = []
        st.rerun()
    
    st.divider()
    
    # Filtres
    st.subheader("🎨 Filtres")
    types_filter = st.multiselect(
        "Types d'EPI",
        options=[epi["nom"] for epi in EPIS],
        default=[epi["nom"] for epi in EPIS]
    )
    
    couleurs_filter = st.multiselect(
        "Couleurs",
        options=["Blanc", "Jaune", "Bleu", "Rouge", "Vert", "Noir", "Transparent"],
        default=[]
    )

# Zone principale
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Simulation Webcam")
    
    # Placeholder pour la "vidéo"
    video_placeholder = st.empty()
    
    # Image simulée
    frame_placeholder = st.empty()
    
    # Zone de simulation
    if st.session_state.en_cours:
        # Afficher une "caméra" simulée
        with video_placeholder.container():
            st.markdown("""
            <div style="background: #2d2d2d; padding: 2rem; border-radius: 10px; text-align: center;">
                <h3 style="color: #00ff00;">📹 CAMERA ACTIVE</h3>
                <p style="color: white;">Simulation en cours...</p>
                <div style="font-size: 3rem;">🎥</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Simulation de détections
        if random.random() < 0.3:  # 30% de chance de détection
            detection = simuler_detection()
            
            # Appliquer les filtres
            if detection["nom"] in types_filter:
                if not couleurs_filter or detection["couleur"] in couleurs_filter:
                    st.session_state.detections.append(detection)
                    st.session_state.historique.append(f"{detection['timestamp']} - {detection['icone']} {detection['nom']} ({detection['couleur']})")
        
        # Rafraîchissement
        time.sleep(1/vitesse)
        st.rerun()

with col2:
    st.subheader("📊 Détections actuelles")
    
    # Dernières détections
    if st.session_state.detections:
        dernieres = st.session_state.detections[-5:]
        
        for det in reversed(dernieres):
            if det["critique"]:
                st.markdown(f"""
                <div class="danger-box">
                    {det['icone']} <strong>{det['nom']}</strong> - {det['couleur']}<br>
                    <small>{det['timestamp']} | Confiance: {det['confiance']:.1%}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                    {det['icone']} <strong>{det['nom']}</strong> - {det['couleur']}<br>
                    <small>{det['timestamp']} | Confiance: {det['confiance']:.1%}</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Aucune détection")

# Statistiques
st.divider()
st.subheader("📈 Statistiques")

col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)

with col_stats1:
    st.metric("Total détections", len(st.session_state.detections))

with col_stats2:
    epis_detectes = len(set([d["nom"] for d in st.session_state.detections]))
    st.metric("Types d'EPI", epis_detectes)

with col_stats3:
    critiques = len([d for d in st.session_state.detections if d["critique"]])
    st.metric("EPI critiques", critiques)

with col_stats4:
    if st.session_state.detections:
        conf_moyenne = sum([d["confiance"] for d in st.session_state.detections]) / len(st.session_state.detections)
        st.metric("Confiance moyenne", f"{conf_moyenne:.1%}")
    else:
        st.metric("Confiance moyenne", "0%")

# Graphiques
if st.session_state.detections:
    col_graph1, col_graph2 = st.columns(2)
    
    with col_graph1:
        # Compter par type
        types_count = {}
        for d in st.session_state.detections:
            types_count[d["nom"]] = types_count.get(d["nom"], 0) + 1
        
        df_types = pd.DataFrame({
            "EPI": list(types_count.keys()),
            "Nombre": list(types_count.values())
        })
        
        fig = px.bar(df_types, x="EPI", y="Nombre", title="Détections par type",
                    color="EPI", color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_graph2:
        # Compter par couleur
        couleurs_count = {}
        for d in st.session_state.detections:
            couleurs_count[d["couleur"]] = couleurs_count.get(d["couleur"], 0) + 1
        
        df_couleurs = pd.DataFrame({
            "Couleur": list(couleurs_count.keys()),
            "Nombre": list(couleurs_count.values())
        })
        
        fig2 = px.pie(df_couleurs, values="Nombre", names="Couleur", 
                     title="Répartition par couleur")
        st.plotly_chart(fig2, use_container_width=True)

# Historique
with st.expander("📜 Historique complet"):
    if st.session_state.historique:
        for item in reversed(st.session_state.historique[-50:]):
            st.text(item)
    else:
        st.text("Aucun historique")

# Liste des EPI
with st.expander("ℹ️ Liste des EPI"):
    for epi in EPIS:
        st.markdown(f"""
        **{epi['icone']} {epi['nom']}**  
        - Couleurs: {', '.join(epi['couleurs'])}  
        - Critique: {'Oui' if epi['critique'] else 'Non'}
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
    <p>🛡️ Détection EPI - Version Simulée pour Streamlit Cloud</p>
    <p style="font-size: 0.8rem; color: gray;">Sans OpenCV - 100% compatible</p>
</div>
""", unsafe_allow_html=True)
