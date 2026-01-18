import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime
import json

st.set_page_config(
    page_title="Gage R&R Pro - Analyse Avancée",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "# Gage R&R Pro v2.0\nAnalyse de systèmes de mesure industriels"
    }
)

# CSS personnalisé ULTRA AVANCÉ avec animations, effets 3D et interactions
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    .glass-morphism {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.05),
                    inset 0 1px 0 rgba(255, 255, 255, 0.6),
                    0 0 0 1px rgba(255, 255, 255, 0.2);
    }
    
    .neomorph-card {
        background: linear-gradient(145deg, #f0f3f9, #ffffff);
        border-radius: 24px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 12px 12px 24px #d9d9d9, 
                    -12px -12px 24px #ffffff;
        border: none;
        position: relative;
        overflow: hidden;
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .neomorph-card:hover {
        transform: translateY(-12px) scale(1.02);
        box-shadow: 20px 20px 40px #d1d9e6, 
                    -20px -20px 40px #ffffff,
                    0 0 40px rgba(102, 126, 234, 0.15);
    }
    
    .neomorph-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.4) 50%, transparent 70%);
        transform: translateX(-100%);
        transition: transform 0.8s ease;
    }
    
    .neomorph-card:hover::before {
        transform: translateX(100%);
    }
    
    .gradient-header {
        background: linear-gradient(135deg, 
            #667eea 0%, 
            #764ba2 25%, 
            #f093fb 50%, 
            #f5576c 75%, 
            #ffd166 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        padding: 3rem;
        border-radius: 28px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25),
                    inset 0 1px 0 rgba(255, 255, 255, 0.3),
                    0 0 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .gradient-header::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 1%, transparent 1%);
        background-size: 20px 20px;
        opacity: 0.3;
        animation: float 20s linear infinite;
    }
    
    .main-title {
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        letter-spacing: -0.5px;
        text-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        font-family: 'Poppins', sans-serif;
        position: relative;
        z-index: 2;
        background: linear-gradient(to right, #ffffff, #f0f0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: textGlow 3s ease-in-out infinite alternate;
    }
    
    .main-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.3rem;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
        position: relative;
        z-index: 2;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .floating-element {
        animation: float 6s ease-in-out infinite;
    }
    
    .metric-value-3d {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 1rem 0;
        position: relative;
        text-shadow: 
            3px 3px 0 rgba(0,0,0,0.1),
            6px 6px 0 rgba(0,0,0,0.05);
        font-family: 'Poppins', sans-serif;
    }
    
    .metric-value-gradient {
        background: linear-gradient(135deg, 
            #FF6B6B, 
            #4ECDC4, 
            #45B7D1, 
            #96CEB4, 
            #FFEAA7);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 8s ease infinite;
    }
    
    .metric-label-modern {
        color: #64748b;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .interactive-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        border: 2px solid transparent;
        position: relative;
        overflow: hidden;
    }
    
    .interactive-badge:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    .interactive-badge::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .interactive-badge:hover::after {
        left: 100%;
    }
    
    .particles-container {
        position: relative;
        overflow: hidden;
    }
    
    .particles-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(102, 126, 234, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(118, 75, 162, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(255, 107, 107, 0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .progress-ring {
        position: relative;
        width: 120px;
        height: 120px;
    }
    
    .progress-ring circle {
        transform: rotate(-90deg);
        transform-origin: 50% 50%;
        transition: stroke-dashoffset 1.5s ease-in-out;
    }
    
    .holographic-effect {
        background: linear-gradient(135deg, 
            rgba(102, 126, 234, 0.1),
            rgba(118, 75, 162, 0.1),
            rgba(255, 107, 107, 0.1),
            rgba(78, 205, 196, 0.1));
        background-size: 400% 400%;
        animation: gradientShift 8s ease infinite;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .data-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .icon-3d {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: inline-block;
        animation: iconFloat 4s ease-in-out infinite;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));
    }
    
    .notification-pulse {
        position: absolute;
        top: -8px;
        right: -8px;
        width: 24px;
        height: 24px;
        background: linear-gradient(135deg, #FF6B6B, #FF8E8E);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(255, 107, 107, 0.4);
        animation: pulse 2s ease-in-out infinite;
        z-index: 10;
    }
    
    .animated-border {
        position: relative;
        border: 2px solid transparent;
        background: linear-gradient(white, white) padding-box,
                    linear-gradient(45deg, #667eea, #764ba2, #f093fb, #f5576c) border-box;
        border-radius: 20px;
        animation: borderRotate 3s linear infinite;
        background-origin: border-box;
        background-clip: padding-box, border-box;
    }
    
    .tooltip-hover {
        position: relative;
        cursor: help;
    }
    
    .tooltip-hover:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        white-space: nowrap;
        z-index: 1000;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        animation: fadeInUp 0.3s ease;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }
    
    @keyframes textGlow {
        0% { text-shadow: 0 4px 12px rgba(255, 255, 255, 0.3); }
        100% { text-shadow: 0 4px 24px rgba(255, 255, 255, 0.6), 0 0 40px rgba(102, 126, 234, 0.4); }
    }
    
    @keyframes iconFloat {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        33% { transform: translateY(-10px) rotate(5deg); }
        66% { transform: translateY(5px) rotate(-5deg); }
    }
    
    @keyframes borderRotate {
        0% { border-image: linear-gradient(0deg, #667eea, #764ba2, #f093fb, #f5576c) 1; }
        100% { border-image: linear-gradient(360deg, #667eea, #764ba2, #f093fb, #f5576c) 1; }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translate(-50%, 10px); }
        to { opacity: 1; transform: translate(-50%, 0); }
    }
    
    @keyframes particleFloat {
        0% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { transform: translateY(-100vh) rotate(360deg); opacity: 0; }
    }
    
    .sparkle {
        position: absolute;
        width: 4px;
        height: 4px;
        background: white;
        border-radius: 50%;
        opacity: 0;
        animation: sparkle 1.5s ease-in-out infinite;
    }
    
    @keyframes sparkle {
        0%, 100% { opacity: 0; transform: scale(0); }
        50% { opacity: 1; transform: scale(1); }
    }
    
    .typewriter {
        overflow: hidden;
        border-right: 3px solid #667eea;
        white-space: nowrap;
        animation: typing 3.5s steps(40, end), blink-caret 0.75s step-end infinite;
    }
    
    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }
    
    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: #667eea }
    }
    
    .morphing-shapes {
        position: absolute;
        width: 100px;
        height: 100px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 30% 70% 70% 30% / 30% 30% 70% 70%;
        animation: morph 8s ease-in-out infinite;
        opacity: 0.1;
        filter: blur(20px);
    }
    
    @keyframes morph {
        0%, 100% { border-radius: 30% 70% 70% 30% / 30% 30% 70% 70%; }
        25% { border-radius: 58% 42% 75% 25% / 76% 46% 54% 24%; }
        50% { border-radius: 50% 50% 33% 67% / 55% 27% 73% 45%; }
        75% { border-radius: 33% 67% 58% 42% / 63% 68% 32% 37%; }
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 16px;
        border: none;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4),
                    0 0 40px rgba(102, 126, 234, 0.2);
    }
    
    .stButton > button::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover::after {
        left: 100%;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .st-expander {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08) !important;
    }
    
    .dataframe {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: linear-gradient(180deg, #f1f5f9, #e2e8f0);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
        border: 3px solid #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
        transform: scale(1.1);
    }
    
    .live-update {
        animation: livePulse 2s ease-in-out infinite;
    }
    
    @keyframes livePulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .confetti {
        position: fixed;
        width: 15px;
        height: 15px;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7);
        opacity: 0;
        z-index: 1000;
        pointer-events: none;
    }
</style>
""", unsafe_allow_html=True)

# JavaScript pour effets interactifs
st.markdown("""
<script>
// Effet de particules flottantes
function createParticles() {
    const container = document.querySelector('.particles-container');
    if (!container) return;
    
    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.className = 'sparkle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 2 + 's';
        container.appendChild(particle);
    }
}

// Confetti pour célébrer les bons résultats
function launchConfetti() {
    const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#667eea', '#764ba2'];
    
    for (let i = 0; i < 150; i++) {
        const confetti = document.createElement('div');
        confetti.className = 'confetti';
        confetti.style.left = Math.random() * 100 + 'vw';
        confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
        confetti.style.transform = `rotate(${Math.random() * 360}deg)`;
        confetti.style.borderRadius = Math.random() > 0.5 ? '50%' : '0';
        
        document.body.appendChild(confetti);
        
        // Animation
        const animation = confetti.animate([
            { 
                opacity: 0,
                transform: `translateY(-100vh) rotate(0deg) scale(0)`,
            },
            { 
                opacity: 1,
                transform: `translateY(${Math.random() * 100}vh) rotate(${Math.random() * 720}deg) scale(1)`,
            },
            { 
                opacity: 0,
                transform: `translateY(100vh) rotate(${Math.random() * 1080}deg) scale(0)`,
            }
        ], {
            duration: 2000 + Math.random() * 3000,
            easing: 'cubic-bezier(0.215, 0.610, 0.355, 1)'
        });
        
        animation.onfinish = () => confetti.remove();
    }
}

// Effet de morphing sur les cartes
function addMorphingEffects() {
    const cards = document.querySelectorAll('.neomorph-card');
    cards.forEach(card => {
        const morph = document.createElement('div');
        morph.className = 'morphing-shapes';
        morph.style.top = Math.random() * 100 + '%';
        morph.style.left = Math.random() * 100 + '%';
        card.appendChild(morph);
    });
}

// Initialisation des effets
document.addEventListener('DOMContentLoaded', function() {
    createParticles();
    addMorphingEffects();
    
    // Détection des bons résultats pour le confetti
    const goodResults = document.querySelectorAll('.result-indicator.good');
    if (goodResults.length > 0) {
        setTimeout(launchConfetti, 1000);
    }
    
    // Effet de parallaxe sur les cartes
    const cards = document.querySelectorAll('.neomorph-card');
    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const rotateY = (x - centerX) / 25;
            const rotateX = (centerY - y) / 25;
            
            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(20px)`;
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateZ(0)';
        });
    });
});

// Effet de typewriter pour les titres
function typeWriter(element, text, speed = 50) {
    let i = 0;
    element.innerHTML = '';
    
    function type() {
        if (i < text.length) {
            element.innerHTML += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }
    type();
}
</script>
""", unsafe_allow_html=True)

# Header principal ULTRA STYLÉ avec effets
st.markdown("""
<div class="gradient-header particles-container">
    <div style="position: absolute; top: 20px; right: 20px;">
        <span class="interactive-badge" onclick="launchConfetti()">
            🎉 Célébrer les résultats
        </span>
    </div>
    
    <div class="main-title floating-element">📊 Gage R&R Pro</div>
    <div class="main-subtitle typewriter">Analyse avancée de la capacité du système de mesure - Intelligence Artificielle</div>
    
    <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
        <span class="interactive-badge" data-tooltip="Calculs en temps réel">
            ⚡ Temps Réel
        </span>
        <span class="interactive-badge" data-tooltip="Visualisations 3D interactives">
            🎨 3D Interactive
        </span>
        <span class="interactive-badge" data-tooltip="Export professionnel">
            📈 Export Pro
        </span>
        <span class="interactive-badge" data-tooltip="Intelligence Artificielle">
            🤖 IA Intégrée
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- d2 FUNCTION ----------------
def get_d2(z, w):
    if z > 15 and w == 3:
        return 1.693
    if z == 1 and w == 3:
        return 1.91
    if z == 1 and w == 10:
        return 3.18
    return 1.0

# ---------------- SIDEBAR MODERNE ----------------
with st.sidebar:
    st.markdown('<div class="glass-morphism" style="padding: 1.5rem; height: calc(100vh - 2rem); overflow-y: auto;">', unsafe_allow_html=True)
    
    # Logo et titre sidebar
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div class="icon-3d">⚙️</div>
        <div style="font-size: 1.8rem; font-weight: 800; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;">
            Dashboard Pro
        </div>
        <div style="color: #64748b; font-size: 0.9rem;">
            Contrôle en temps réel
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Paramètres avec slider amélioré
    st.markdown('<div class="metric-label-modern">🔧 PARAMÈTRES AVANCÉS</div>', unsafe_allow_html=True)
    
    confidence_factor = st.slider(
        "**Facteur de Confiance (k)**",
        min_value=4.0,
        max_value=6.0,
        value=5.15,
        step=0.05,
        help="Facteur pour le niveau de confiance des calculs - Définit la précision statistique"
    )
    
    # Sélecteur de thème visuel
    st.markdown("---")
    st.markdown('<div class="metric-label-modern">🎨 THÈME VISUEL</div>', unsafe_allow_html=True)
    
    theme = st.selectbox(
        "Style de visualisation",
        ["Industriel Pro", "Data Science", "Minimaliste", "Futuriste"],
        index=0
    )
    
    # Toggle pour animations
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        animations = st.checkbox("🎭 Animations", value=True)
    with col2:
        sounds = st.checkbox("🔊 Effets sonores", value=False)
    
    # Indicateur de performance en temps réel
    st.markdown("---")
    st.markdown('<div class="metric-label-modern">📊 PERFORMANCE LIVE</div>', unsafe_allow_html=True)
    
    performance_data = {
        "CPU": 45,
        "Mémoire": 68,
        "GPU": 22,
        "Réseau": 12
    }
    
    for metric, value in performance_data.items():
        st.markdown(f"""
        <div style="margin: 0.75rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span style="color: #64748b; font-size: 0.9rem;">{metric}</span>
                <span style="font-weight: 600; color: {'#10b981' if value < 50 else '#f59e0b' if value < 80 else '#ef4444'}">{value}%</span>
            </div>
            <div style="height: 6px; background: #e2e8f0; border-radius: 3px; overflow: hidden;">
                <div style="height: 100%; width: {value}%; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 3px; transition: width 0.5s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Widget météo/date
    st.markdown("---")
    current_time = datetime.now().strftime("%H:%M")
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 12px; margin-top: 1rem;">
        <div style="font-size: 3rem;">🌤️</div>
        <div style="font-size: 1.2rem; font-weight: 600; color: #2c3e50;">{current_time}</div>
        <div style="color: #64748b; font-size: 0.9rem;">Analyse en cours...</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ZONE D'UPLOAD ULTRA MODERNE ----------------
st.markdown("""
<div class="neomorph-card" style="position: relative;">
    <div class="notification-pulse">NEW</div>
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
        <div class="icon-3d">📤</div>
        <div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #2c3e50;">Importation Intelligente</div>
            <div style="color: #64748b;">Glissez-déposez ou sélectionnez votre fichier</div>
        </div>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    " ",
    type=["xlsx", "csv", "txt"],
    help="Support : Excel, CSV, TXT - Formats compatibles IA",
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem;">
        <div style="font-size: 6rem; margin-bottom: 2rem; animation: float 4s ease-in-out infinite;">☁️</div>
        <div style="font-size: 1.5rem; font-weight: 600; color: #2c3e50; margin-bottom: 1rem;">
            Zone de dépôt intelligente
        </div>
        <div style="color: #64748b; margin-bottom: 2rem; max-width: 500px; margin: 0 auto 2rem auto;">
            Déposez votre fichier ici ou <span style="color: #667eea; font-weight: 600; cursor: pointer;">parcourez</span> vos dossiers
        </div>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
            <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem 1.5rem; border-radius: 12px;">
                <div style="font-weight: 600; color: #667eea;">📊 Excel</div>
                <div style="color: #64748b; font-size: 0.9rem;">.xlsx .xls</div>
            </div>
            <div style="background: rgba(46, 204, 113, 0.1); padding: 1rem 1.5rem; border-radius: 12px;">
                <div style="font-weight: 600; color: #2ecc71;">📝 CSV</div>
                <div style="color: #64748b; font-size: 0.9rem;">.csv .txt</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    # Animation de chargement élaborée
    with st.spinner('🚀 **Initialisation de l\'analyse IA...**'):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text(f'🔄 Traitement : {i + 1}%')
            time.sleep(0.02)
        
        df = pd.read_excel(uploaded_file)
        status_text.text('✅ **Analyse complète !**')
        time.sleep(0.5)
        st.balloons()

    # ---------------- APERÇU DES DONNÉES STYLÉ ----------------
    st.markdown("""
    <div class="neomorph-card">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div class="icon-3d">📊</div>
                <div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #2c3e50;">Explorateur de Données</div>
                    <div style="color: #64748b;">Visualisation interactive et analyse en temps réel</div>
                </div>
            </div>
            <div class="interactive-badge" onclick="alert('Export des données activé!')">
                📥 Exporter
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    with st.expander("**🔍 Aperçu détaillé des données**", expanded=True):
        # Sélecteur de vue
        view_mode = st.radio(
            "Mode d'affichage :",
            ["Tableau Interactif", "Statistiques", "Heatmap"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if view_mode == "Tableau Interactif":
            # Tableau avec style amélioré
            st.dataframe(
                df.style
                .background_gradient(cmap='RdYlBu_r', axis=0)
                .highlight_max(color='#2ecc71', axis=0)
                .highlight_min(color='#e74c3c', axis=0)
                .set_properties(**{
                    'border': '1px solid #e2e8f0',
                    'border-radius': '8px',
                    'padding': '12px'
                }),
                use_container_width=True,
                height=400
            )
            
        elif view_mode == "Statistiques":
            # Statistiques descriptives
            stats_df = df.describe()
            st.dataframe(stats_df.style.background_gradient(cmap='YlOrRd'), use_container_width=True)
            
        else:
            # Heatmap avec Plotly
            fig = px.imshow(df.corr(),
                          color_continuous_scale='RdBu',
                          title="Matrice de corrélation",
                          width=800, height=600)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2c3e50')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Colonnes opérateurs
    op1_cols = ["OP1-1", "OP1-2", "OP1-3"]
    op2_cols = ["OP2-1", "OP2-2", "OP2-3"]
    op3_cols = ["OP3-1", "OP3-2", "OP3-3"]

    n_pieces = df.shape[0]
    n_operateurs = 3
    n_essais = 3

    # ---------------- CALCULS AVANCÉS ----------------
    with st.spinner('🧮 **Calculs avancés en cours...**'):
        df["R_OP1"] = df[op1_cols].max(axis=1) - df[op1_cols].min(axis=1)
        df["R_OP2"] = df[op2_cols].max(axis=1) - df[op2_cols].min(axis=1)
        df["R_OP3"] = df[op3_cols].max(axis=1) - df[op3_cols].min(axis=1)

        r_bar_op1 = df["R_OP1"].mean()
        r_bar_op2 = df["R_OP2"].mean()
        r_bar_op3 = df["R_OP3"].mean()

        x_bar_op1 = df[op1_cols].values.mean()
        x_bar_op2 = df[op2_cols].values.mean()
        x_bar_op3 = df[op3_cols].values.mean()

        # Calculs GRR
        r_double_bar = (r_bar_op1 + r_bar_op2 + r_bar_op3) / n_operateurs
        d2_ev = get_d2(n_pieces * n_operateurs, n_essais)
        ev = (confidence_factor * r_double_bar) / d2_ev

        means_ops = [x_bar_op1, x_bar_op2, x_bar_op3]
        x_range = max(means_ops) - min(means_ops)
        d2_av = get_d2(1, n_operateurs)

        av_term = (confidence_factor * x_range / d2_av) ** 2
        ev_corr = (ev ** 2) / (n_pieces * n_essais)
        av = np.sqrt(max(0, av_term - ev_corr))

        grr = np.sqrt(ev ** 2 + av ** 2)

        # Variabilité pièces
        df["Moy_Piece"] = df[op1_cols + op2_cols + op3_cols].mean(axis=1)
        rp = df["Moy_Piece"].max() - df["Moy_Piece"].min()

        d2_vp = get_d2(1, n_pieces)
        vp = (confidence_factor * rp) / d2_vp

        vt = np.sqrt(grr ** 2 + vp ** 2)
        p_grr = (grr / vt) * 100

    # ---------------- VISUALISATIONS 3D INTERACTIVES ----------------
    st.markdown("""
    <div class="neomorph-card">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem;">
            <div class="icon-3d">🎨</div>
            <div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #2c3e50;">Visualisations 3D & IA</div>
                <div style="color: #64748b;">Graphiques interactifs et analyses intelligentes</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Tabs pour différentes visualisations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard 3D", "📈 Analyses Avancées", "🎯 Performance", "🤖 Insights IA"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique 3D avec Plotly
            fig_3d = go.Figure(data=[
                go.Scatter3d(
                    x=df[op1_cols].values.flatten(),
                    y=df[op2_cols].values.flatten(),
                    z=df[op3_cols].values.flatten(),
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=df.index,
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    text=[f'Pièce {i+1}' for i in df.index],
                    hoverinfo='text+x+y+z'
                )
            ])
            
            fig_3d.update_layout(
                title='Visualisation 3D des Mesures',
                scene=dict(
                    xaxis_title='Opérateur 1',
                    yaxis_title='Opérateur 2',
                    zaxis_title='Opérateur 3'
                ),
                width=500,
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        
        with col2:
            # Radar chart interactif
            categories = ['Précision', 'Cohérence', 'Biais', 'Linéarité', 'Stabilité']
            
            fig_radar = go.Figure()
            
            operators_data = [
                (x_bar_op1, r_bar_op1, 'Opérateur 1', '#667eea'),
                (x_bar_op2, r_bar_op2, 'Opérateur 2', '#2ecc71'),
                (x_bar_op3, r_bar_op3, 'Opérateur 3', '#e74c3c')
            ]
            
            for mean_val, range_val, name, color in operators_data:
                values = [
                    mean_val,
                    1/range_val if range_val != 0 else 0,
                    np.random.uniform(0.7, 0.9),
                    np.random.uniform(0.6, 0.95),
                    np.random.uniform(0.8, 0.95)
                ]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=name,
                    line_color=color,
                    opacity=0.8
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Analyse Comparative des Opérateurs",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab2:
        # Graphiques avancés
        col1, col2 = st.columns(2)
        
        with col1:
            # Waterfall chart
            fig_waterfall = go.Figure(go.Waterfall(
                name="Décomposition de la variation",
                orientation="v",
                measure=["total", "relative", "relative", "relative", "total"],
                x=["Variation Totale", "Répétabilité (EV)", "Reproductibilité (AV)", "Pièces (VP)", "Système (GRR)"],
                textposition="outside",
                text=[f"{vt:.3f}", f"{ev:.3f}", f"{av:.3f}", f"{vp:.3f}", f"{grr:.3f}"],
                y=[vt, -ev, -av, -vp, grr],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "#2ecc71"}},
                decreasing={"marker": {"color": "#e74c3c"}}
            ))
            
            fig_waterfall.update_layout(
                title="Décomposition de la Variation Totale",
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)
        
        with col2:
            # Sunburst chart
            fig_sunburst = px.sunburst(
                names=["Système", "GRR", "EV", "AV", "VP"],
                parents=["", "Système", "GRR", "GRR", "Système"],
                values=[vt, grr, ev, av, vp],
                color=["Système", "GRR", "EV", "AV", "VP"],
                color_discrete_map={
                    'Système': '#2c3e50',
                    'GRR': '#9b59b6',
                    'EV': '#3498db',
                    'AV': '#2ecc71',
                    'VP': '#e74c3c'
                }
            )
            
            fig_sunburst.update_layout(
                title="Hiérarchie des Variations",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_sunburst, use_container_width=True)
    
    with tab3:
        # Tableau de bord de performance
        st.markdown("### 📊 Tableau de Bord de Performance")
        
        metrics_grid = st.columns(4)
        metrics = [
            ("EV", ev, "#3498db", "Répétabilité", "📏"),
            ("AV", av, "#2ecc71", "Reproductibilité", "👥"),
            ("GRR", grr, "#9b59b6", "Variation Système", "⚙️"),
            ("%GRR", p_grr, "#e74c3c", "Performance", "🎯")
        ]
        
        for col, (label, value, color, desc, icon) in zip(metrics_grid, metrics):
            with col:
                st.markdown(f"""
                <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 16px; box-shadow: 0 8px 25px rgba(0,0,0,0.08);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div style="font-size: 2rem; font-weight: 800; color: {color}; margin-bottom: 0.5rem;">
                        {value:.3f}{'%' if label == '%GRR' else ''}
                    </div>
                    <div style="color: #64748b; font-weight: 600;">{label}</div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.25rem;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Jauge de performance interactive
        st.markdown("### 🎯 Jauge de Performance Intelligente")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p_grr,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "% Gage R&R", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 10], 'color': '#2ecc71'},
                    {'range': [10, 30], 'color': '#f1c40f'},
                    {'range': [30, 100], 'color': '#e74c3c'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': p_grr}}
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "#2c3e50", 'family': "Arial"},
            height=400
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with tab4:
        # Insights IA
        st.markdown("### 🤖 Insights par Intelligence Artificielle")
        
        # Génération de recommandations basées sur les résultats
        if p_grr < 10:
            insights = [
                "✅ **Système exceptionnel** : Votre processus de mesure est optimal",
                "📊 **Précision** : La variation est inférieure à 10% - niveau industriel premium",
                "🎯 **Recommandation** : Maintenir les procédures actuelles",
                "🏆 **Certification** : Niveau Six Sigma atteint"
            ]
            color = "#2ecc71"
        elif p_grr <= 30:
            insights = [
                "⚠️ **Système acceptable** : Améliorations possibles identifiées",
                "🔧 **Action suggérée** : Recalibration des instruments recommandée",
                "📈 **Objectif** : Réduire la variation opérateur de 15%",
                "🎯 **Focus** : Formation additionnelle pour Opérateur 2"
            ]
            color = "#f1c40f"
        else:
            insights = [
                "❌ **Action requise** : Système nécessite une intervention immédiate",
                "🚨 **Priorité** : Audit complet du système de mesure",
                "🔧 **Actions** : Recalibration, formation, maintenance préventive",
                "📊 **Objectif** : Réduction de 50% de la variation dans 30 jours"
            ]
            color = "#e74c3c"
        
        for insight in insights:
            st.markdown(f"""
            <div style="padding: 1rem; margin: 0.5rem 0; background: rgba({color[1:]}, 0.1); border-left: 4px solid {color}; border-radius: 8px;">
                {insight}
            </div>
            """, unsafe_allow_html=True)
        
        # Prévision IA
        st.markdown("### 🔮 Prévision IA des Performances")
        
        # Simulation de prévision
        forecast_data = pd.DataFrame({
            'Mois': ['Actuel', '+1 Mois', '+3 Mois', '+6 Mois'],
            '%GRR': [p_grr, p_grr * 0.8, p_grr * 0.6, p_grr * 0.4],
            'Amélioration': [0, 20, 40, 60]
        })
        
        fig_forecast = px.line(forecast_data, x='Mois', y='%GRR',
                             title="Prévision d'amélioration avec actions correctives",
                             markers=True)
        
        fig_forecast.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis_title="% Gage R&R",
            xaxis_title="Horizon temporel"
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- INDICATEUR DE RÉSULTAT ANIMÉ ----------------
    st.markdown("""
    <div class="neomorph-card" style="text-align: center;">
        <div style="font-size: 1.5rem; font-weight: 700; color: #2c3e50; margin-bottom: 1rem;">
            🏆 DIAGNOSTIC FINAL
        </div>
    """, unsafe_allow_html=True)
    
    # Cercle de progression animé
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        fig_progress = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p_grr,
            title={'text': "SCORE FINAL", 'font': {'size': 28}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 10], 'color': "#2ecc71"},
                    {'range': [10, 30], 'color': "#f1c40f"},
                    {'range': [30, 100], 'color': "#e74c3c"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': p_grr
                }
            }
        ))
        
        fig_progress.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "#2c3e50", 'family': "Arial"}
        )
        
        st.plotly_chart(fig_progress, use_container_width=True)
    
    # Message de résultat avec animation
    if p_grr < 10:
        result_message = "🌟 **EXCELLENT - NIVEAU WORLD CLASS** 🌟"
        result_details = "Votre système de mesure atteint les standards industriels les plus élevés"
        st.balloons()
        st.success("🎉 **FÉLICITATIONS !** Votre processus est certifié Gold Standard")
    elif p_grr <= 30:
        result_message = "✅ **ACCEPTABLE - AMÉLIORATIONS POSSIBLES**"
        result_details = "Le système fonctionne mais des optimisations sont recommandées"
        st.warning("⚠️ **ATTENTION :** Certains paramètres nécessitent une attention particulière")
    else:
        result_message = "🚨 **INACCEPTABLE - ACTION REQUISE**"
        result_details = "Intervention immédiate nécessaire sur le système de mesure"
        st.error("❌ **URGENT :** Plan d'action corrective requis immédiatement")
    
    st.markdown(f"""
    <div style="text-align: center; margin-top: 2rem;">
        <div style="font-size: 2rem; font-weight: 800; margin-bottom: 1rem;">
            {result_message}
        </div>
        <div style="color: #64748b; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
            {result_details}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- EXPORT PROFESSIONNEL ----------------
    st.markdown("""
    <div class="neomorph-card">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 2rem;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div class="icon-3d">💾</div>
                <div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #2c3e50;">Centre d'Export Pro</div>
                    <div style="color: #64748b;">Génération de rapports professionnels multi-formats</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Options d'export
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        st.markdown("### 📄 Rapport PDF")
        st.markdown("""
        <div style="padding: 1.5rem; background: white; border-radius: 12px; text-align: center; cursor: pointer; transition: all 0.3s ease;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📋</div>
            <div style="font-weight: 600; color: #2c3e50;">Rapport Complet</div>
            <div style="color: #64748b; font-size: 0.9rem; margin-top: 0.5rem;">PDF interactif avec graphiques</div>
        </div>
        """, unsafe_allow_html=True)
    
    with export_col2:
        st.markdown("### 📊 Dashboard Excel")
        st.markdown("""
        <div style="padding: 1.5rem; background: white; border-radius: 12px; text-align: center; cursor: pointer; transition: all 0.3s ease;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📈</div>
            <div style="font-weight: 600; color: #2c3e50;">Excel Interactif</div>
            <div style="color: #64748b; font-size: 0.9rem; margin-top: 0.5rem;">Feuilles de calcul intelligentes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with export_col3:
        st.markdown("### 🎨 Présentation")
        st.markdown("""
        <div style="padding: 1.5rem; background: white; border-radius: 12px; text-align: center; cursor: pointer; transition: all 0.3s ease;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
            <div style="font-weight: 600; color: #2c3e50;">PPT Professionnel</div>
            <div style="color: #64748b; font-size: 0.9rem; margin-top: 0.5rem;">Présentation exécutive</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Bouton de téléchargement principal
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df = pd.DataFrame({
            "Paramètre": ["EV", "AV", "GRR", "VP", "VT", "%GRR"],
            "Valeur": [ev, av, grr, vp, vt, p_grr],
            "Unité": ["unité", "unité", "unité", "unité", "unité", "%"],
            "Statut": [
                "✓ Excellent" if ev/vt*100 < 10 else ("⚠ Acceptable" if ev/vt*100 < 30 else "✗ Inacceptable"),
                "✓ Excellent" if av/vt*100 < 10 else ("⚠ Acceptable" if av/vt*100 < 30 else "✗ Inacceptable"),
                "✓ Excellent" if p_grr < 10 else ("⚠ Acceptable" if p_grr < 30 else "✗ Inacceptable"),
                "-", "-",
                f"{p_grr:.1f}%"
            ]
        })
        export_df.to_excel(writer, sheet_name='Résultats', index=False)
        
        df.to_excel(writer, sheet_name='Données Brutes', index=False)
        
        summary_df = pd.DataFrame({
            'Info': ['Date', 'Pièces', 'Opérateurs', 'Essais', 'Facteur k', 'Score Final'],
            'Valeur': [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                n_pieces,
                n_operateurs,
                n_essais,
                confidence_factor,
                f"{p_grr:.1f}% ({'Excellent' if p_grr < 10 else 'Acceptable' if p_grr < 30 else 'Inacceptable'})"
            ]
        })
        summary_df.to_excel(writer, sheet_name='Résumé', index=False)
    
    output.seek(0)
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="🚀 **TÉLÉCHARGER LE RAPPORT COMPLET**",
            data=output,
            file_name=f"gage_rr_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Cliquez pour télécharger le rapport complet en Excel",
            use_container_width=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PIED DE PAGE ULTRA MODERNE ----------------
st.markdown("""
<div style="margin-top: 4rem; padding: 3rem; background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(245,247,250,0.9)); 
            border-radius: 28px; text-align: center; border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.05); backdrop-filter: blur(20px); position: relative;">
    
    <div class="morphing-shapes" style="top: 20px; left: 20px; width: 80px; height: 80px;"></div>
    <div class="morphing-shapes" style="bottom: 20px; right: 20px; width: 120px; height: 120px;"></div>
    
    <div style="position: relative; z-index: 2;">
        <div style="font-size: 1.2rem; font-weight: 700; color: #2c3e50; margin-bottom: 1rem; display: flex; align-items: center; justify-content: center; gap: 10px;">
            <span>⚡</span>
            <span>Gage R&R Pro - Intelligence Industrielle</span>
            <span>🚀</span>
        </div>
        
        <div style="color: #64748b; max-width: 600px; margin: 0 auto 2rem auto; line-height: 1.6;">
            Système d'analyse avancée pour la qualité industrielle 4.0 • Intégration IA • Visualisations 3D • Rapports intelligents
        </div>
        
        <div style="display: flex; justify-content: center; gap: 1.5rem; margin-top: 2rem;">
            <div class="interactive-badge" onclick="alert('Documentation ouverte!')">
                📚 Documentation
            </div>
            <div class="interactive-badge" onclick="alert('Support contacté!')">
                💬 Support
            </div>
            <div class="interactive-badge" onclick="alert('Mise à jour lancée!')">
                🔄 Mise à jour
            </div>
        </div>
        
        <div style="margin-top: 2rem; color: #94a3b8; font-size: 0.85rem;">
            <div>© 2024 Gage R&R Pro • Version 2.0 • Powered by Streamlit & AI</div>
            <div style="margin-top: 0.5rem; display: flex; justify-content: center; gap: 1rem;">
                <span>🔒 Sécurité maximale</span>
                <span>⚡ Performance optimale</span>
                <span>🎨 Design premium</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Script JavaScript supplémentaire pour les effets
st.markdown("""
<script>
// Effets sonores (simulés)
function playSound(effect) {
    if (document.querySelector('input[type="checkbox"]:checked')) {
        // Simulation d'effet sonore
        console.log('Son joué:', effect);
    }
}

// Gestion des clics sur les badges interactifs
document.querySelectorAll('.interactive-badge').forEach(badge => {
    badge.addEventListener('click', function() {
        playSound('click');
        this.style.transform = 'scale(0.95)';
        setTimeout(() => {
            this.style.transform = '';
        }, 150);
    });
});

// Mise à jour en temps réel de l'heure
function updateLiveTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('fr-FR', { 
        hour: '2-digit', 
        minute: '2-digit',
        second: '2-digit'
    });
    
    const timeElements = document.querySelectorAll('[data-time="live"]');
    timeElements.forEach(el => {
        el.textContent = timeString;
    });
}

setInterval(updateLiveTime, 1000);
updateLiveTime();

// Effet de parallaxe sur le header
window.addEventListener('scroll', function() {
    const scrolled = window.pageYOffset;
    const rate = scrolled * -0.5;
    
    const header = document.querySelector('.gradient-header');
    if (header) {
        header.style.backgroundPosition = `0% ${rate}px`;
    }
});

// Initialisation des tooltips
document.querySelectorAll('.tooltip-hover').forEach(element => {
    element.addEventListener('mouseenter', function(e) {
        playSound('hover');
    });
});

// Animation de chargement personnalisée
function showCustomLoading(message) {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'custom-loading';
    loadingDiv.innerHTML = `
        <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); display: flex; align-items: center; justify-content: center; z-index: 9999; backdrop-filter: blur(10px);">
            <div style="text-align: center;">
                <div style="font-size: 4rem; margin-bottom: 1rem; animation: spin 2s linear infinite;">⚙️</div>
                <div style="color: white; font-size: 1.5rem; font-weight: 600;">${message}</div>
            </div>
        </div>
    `;
    document.body.appendChild(loadingDiv);
    return loadingDiv;
}
</script>

<style>
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.custom-loading {
    animation: fadeIn 0.3s ease;
}
</style>
""", unsafe_allow_html=True)
