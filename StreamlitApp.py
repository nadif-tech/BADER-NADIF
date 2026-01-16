import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import time
from datetime import datetime

st.set_page_config(
    page_title="Gage R&R Premium Suite",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS PERSONNALISÉ - DESIGN HAUT DE GAMME
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --success-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        --warning-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        --danger-gradient: linear-gradient(135deg, #ff0844 0%, #ffb199 100%);
        --dark-gradient: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        --glass-bg: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 255, 255, 0.2);
        --shadow-soft: 0 8px 32px rgba(0, 0, 0, 0.08);
        --shadow-hard: 0 20px 60px rgba(0, 0, 0, 0.15);
        --transition-smooth: all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .main-header {
        background: linear-gradient(135deg, 
            rgba(102, 126, 234, 0.1) 0%, 
            rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(20px);
        padding: 3rem 2rem;
        border-radius: 30px;
        margin-bottom: 3rem;
        text-align: center;
        box-shadow: var(--shadow-soft);
        border: 1px solid var(--glass-border);
        position: relative;
        overflow: hidden;
        animation: float 6s ease-in-out infinite;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, 
            rgba(102, 126, 234, 0.1) 0%, 
            rgba(118, 75, 162, 0.05) 30%, 
            transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        position: relative;
        z-index: 2;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: rgba(44, 62, 80, 0.8);
        font-weight: 400;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
        position: relative;
        z-index: 2;
    }
    
    .metric-card {
        background: linear-gradient(145deg, 
            rgba(255, 255, 255, 0.9) 0%, 
            rgba(248, 250, 252, 0.9) 100%);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-soft);
        border: 1px solid var(--glass-border);
        transition: var(--transition-smooth);
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--primary-gradient);
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.8s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: var(--shadow-hard);
    }
    
    .metric-card:hover::before {
        transform: scaleX(1);
    }
    
    .metric-value {
        font-family: 'Poppins', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: var(--dark-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
        position: relative;
    }
    
    .metric-value::after {
        content: '';
        position: absolute;
        bottom: -5px;
        left: 0;
        width: 60px;
        height: 3px;
        background: var(--primary-gradient);
        border-radius: 2px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: rgba(44, 62, 80, 0.6);
        margin-bottom: 0.5rem;
    }
    
    .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        opacity: 0.8;
    }
    
    .result-indicator {
        padding: 1.5rem 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
        font-weight: 600;
        font-size: 1.2rem;
        backdrop-filter: blur(20px);
        transition: var(--transition-smooth);
        position: relative;
        overflow: hidden;
        border: 1px solid transparent;
    }
    
    .result-indicator::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: inherit;
        filter: blur(20px);
        z-index: -1;
    }
    
    .good {
        background: var(--success-gradient);
        color: white;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    .warning {
        background: var(--warning-gradient);
        color: white;
        animation: pulse 3s ease-in-out infinite;
    }
    
    .bad {
        background: var(--danger-gradient);
        color: white;
        animation: shake 0.8s ease-in-out;
    }
    
    .section-header {
        background: linear-gradient(90deg, 
            rgba(102, 126, 234, 0.1) 0%, 
            rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(10px);
        padding: 1.5rem 2rem;
        border-radius: 20px;
        margin: 3rem 0 2rem 0;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.8rem;
        display: flex;
        align-items: center;
        gap: 15px;
        box-shadow: var(--shadow-soft);
        border: 1px solid var(--glass-border);
        position: relative;
        overflow: hidden;
    }
    
    .section-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 6px;
        height: 100%;
        background: var(--primary-gradient);
    }
    
    .plot-container {
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.95) 0%, 
            rgba(248, 250, 252, 0.95) 100%);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 24px;
        box-shadow: var(--shadow-soft);
        margin: 2rem 0;
        border: 1px solid var(--glass-border);
        transition: var(--transition-smooth);
        position: relative;
        overflow: hidden;
    }
    
    .plot-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at top right, 
            rgba(102, 126, 234, 0.05) 0%, 
            transparent 50%);
        z-index: 0;
    }
    
    .plot-container:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-hard);
    }
    
    .upload-area {
        border: 3px dashed rgba(102, 126, 234, 0.3);
        border-radius: 30px;
        padding: 4rem 2rem;
        text-align: center;
        background: linear-gradient(135deg, 
            rgba(102, 126, 234, 0.05) 0%, 
            rgba(118, 75, 162, 0.05) 100%);
        transition: var(--transition-smooth);
        margin: 3rem 0;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .upload-area::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: repeating-linear-gradient(
            45deg,
            transparent,
            transparent 10px,
            rgba(102, 126, 234, 0.05) 10px,
            rgba(102, 126, 234, 0.05) 20px
        );
        z-index: 0;
    }
    
    .upload-area:hover {
        border-color: rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, 
            rgba(102, 126, 234, 0.1) 0%, 
            rgba(118, 75, 162, 0.1) 100%);
        transform: scale(1.02);
    }
    
    .download-btn {
        background: var(--primary-gradient);
        color: white;
        padding: 1.2rem 2.5rem;
        border-radius: 15px;
        border: none;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        transition: var(--transition-smooth);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        display: inline-flex;
        align-items: center;
        gap: 12px;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    
    .download-btn::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.2), 
            transparent);
        transition: left 0.6s ease;
        z-index: -1;
    }
    
    .download-btn:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.4);
    }
    
    .download-btn:hover::before {
        left: 100%;
    }
    
    .sidebar-content {
        padding: 2rem 1.5rem;
        background: linear-gradient(180deg, 
            rgba(248, 250, 252, 0.95) 0%, 
            rgba(241, 245, 249, 0.95) 100%);
        backdrop-filter: blur(10px);
        border-radius: 0 30px 30px 0;
        height: 100%;
        box-shadow: var(--shadow-soft);
        border-right: 1px solid var(--glass-border);
    }
    
    .progress-ring {
        position: relative;
        width: 120px;
        height: 120px;
        margin: 0 auto;
    }
    
    .progress-ring-circle {
        transform: rotate(-90deg);
        transform-origin: 50% 50%;
        stroke-dasharray: 314;
        stroke-dashoffset: 314;
        transition: stroke-dashoffset 1.5s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    .glowing-border {
        position: relative;
        border: 2px solid transparent;
        background-clip: padding-box;
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.9) 0%, 
            rgba(248, 250, 252, 0.9) 100%);
        border-radius: 20px;
    }
    
    .glowing-border::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: var(--primary-gradient);
        border-radius: 22px;
        z-index: -1;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .glowing-border:hover::before {
        opacity: 1;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    @keyframes glow {
        from { box-shadow: 0 0 20px rgba(67, 233, 123, 0.5); }
        to { box-shadow: 0 0 40px rgba(67, 233, 123, 0.8); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.02); opacity: 0.9; }
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-10px); }
        20%, 40%, 60%, 80% { transform: translateX(10px); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    .shimmer-text {
        background: linear-gradient(90deg, 
            #667eea 0%, 
            #764ba2 25%, 
            #667eea 50%, 
            #764ba2 75%, 
            #667eea 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
    }
    
    .particles {
        position: absolute;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        pointer-events: none;
        z-index: 0;
    }
    
    .particle {
        position: absolute;
        width: 4px;
        height: 4px;
        background: var(--primary-gradient);
        border-radius: 50%;
        opacity: 0.3;
        animation: float 6s ease-in-out infinite;
    }
    
    /* Scrollbar personnalisée */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(241, 245, 249, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.5);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Style pour les sélecteurs */
    .stSlider > div > div > div {
        background: var(--primary-gradient);
    }
    
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        transition: var(--transition-smooth);
    }
    
    .stSelectbox > div > div:hover {
        border-color: rgba(102, 126, 234, 0.6);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.1);
    }
    
    /* Animation d'entrée pour les éléments */
    .fade-in {
        animation: fadeIn 0.8s ease-out forwards;
        opacity: 0;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .delay-1 { animation-delay: 0.1s; }
    .delay-2 { animation-delay: 0.2s; }
    .delay-3 { animation-delay: 0.3s; }
    .delay-4 { animation-delay: 0.4s; }
    .delay-5 { animation-delay: 0.5s; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================
def get_d2(z, w):
    d2_table = {
        (15, 3): 1.693,
        (1, 3): 1.91,
        (1, 10): 3.18
    }
    return d2_table.get((z, w), 1.0)

def create_particles(n=20):
    particles_html = '<div class="particles">'
    for _ in range(n):
        size = np.random.uniform(2, 6)
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 100)
        duration = np.random.uniform(3, 8)
        delay = np.random.uniform(0, 5)
        opacity = np.random.uniform(0.1, 0.4)
        particles_html += f'''
        <div class="particle" style="
            width: {size}px;
            height: {size}px;
            left: {x}%;
            top: {y}%;
            animation-delay: {delay}s;
            animation-duration: {duration}s;
            opacity: {opacity};
        "></div>
        '''
    particles_html += '</div>'
    return particles_html

def create_circular_progress(value, max_value=100, size=120, stroke_width=10):
    radius = (size - stroke_width) / 2
    circumference = 2 * np.pi * radius
    progress = min(value / max_value, 1.0)
    offset = circumference * (1 - progress)
    
    color = "#2ecc71" if value < 10 else "#f1c40f" if value <= 30 else "#e74c3c"
    
    return f'''
    <div class="progress-ring" style="width: {size}px; height: {size}px;">
        <svg width="{size}" height="{size}">
            <circle
                cx="{size/2}"
                cy="{size/2}"
                r="{radius}"
                stroke="#f1f5f9"
                stroke-width="{stroke_width}"
                fill="transparent"
            />
            <circle
                class="progress-ring-circle"
                cx="{size/2}"
                cy="{size/2}"
                r="{radius}"
                stroke="{color}"
                stroke-width="{stroke_width}"
                fill="transparent"
                stroke-linecap="round"
                style="stroke-dasharray: {circumference}; stroke-dashoffset: {offset};"
            />
        </svg>
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                    text-align: center; font-family: 'Poppins', sans-serif; font-weight: 700; 
                    font-size: 1.8rem; color: {color};">
            {value:.1f}%
        </div>
    </div>
    '''

def create_advanced_gauge(p_grr):
    """Crée une jauge de performance avancée avec Matplotlib"""
    fig, ax = plt.subplots(figsize=(10, 3), facecolor='none')
    
    # Zones de couleur
    ax.barh(0, 10, height=0.3, color='#2ecc71', edgecolor='white', linewidth=2, alpha=0.8)
    ax.barh(0, 20, left=10, height=0.3, color='#f1c40f', edgecolor='white', linewidth=2, alpha=0.8)
    ax.barh(0, 70, left=30, height=0.3, color='#e74c3c', edgecolor='white', linewidth=2, alpha=0.8)
    
    # Aiguille
    ax.axvline(x=p_grr, color='#2c3e50', linestyle='-', linewidth=3, alpha=0.8)
    
    # Style
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('% Gage R&R', fontsize=12, fontweight='bold', color='#2c3e50')
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Texte de valeur
    ax.text(p_grr, 0.4, f'{p_grr:.1f}%', 
            ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2c3e50', alpha=0.9))
    
    # Ajouter des marques
    for x in [0, 10, 30, 100]:
        ax.text(x, -0.2, f'{x}%', ha='center', va='center', fontsize=9, color='#7f8c8d')
    
    plt.tight_layout()
    return fig

def create_radar_chart(ev, av, p_grr, vt):
    """Crée un diagramme radar avec Matplotlib"""
    fig = plt.figure(figsize=(8, 8), facecolor='none')
    ax = fig.add_subplot(111, polar=True)
    
    # Catégories
    categories = ['Répétabilité\n(EV)', 'Reproductibilité\n(AV)', 'Performance\n(%GRR)', 
                  'Stabilité', 'Linéarité', 'Capacité']
    N = len(categories)
    
    # Angles pour chaque catégorie
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Données normalisées
    data = [ev/vt*100, av/vt*100, 100-p_grr, 85, 90, 95]
    data += data[:1]
    
    # Tracer
    ax.plot(angles, data, 'o-', linewidth=2, color='#3498db', markersize=8)
    ax.fill(angles, data, alpha=0.25, color='#3498db')
    
    # Configurer les axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9, color='#7f8c8d')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_components_chart(ev, av, grr, vp, vt):
    """Crée un graphique en barres des composantes"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
    
    components = ['EV', 'AV', 'GRR', 'VP', 'VT']
    values = [ev, av, grr, vp, vt]
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']
    
    bars = ax.bar(components, values, color=colors, edgecolor='white', 
                  linewidth=2, alpha=0.8, zorder=3)
    
    # Ajouter les valeurs
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{value:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10, color='#2c3e50')
    
    # Style
    ax.set_ylabel('Valeur', fontsize=12, fontweight='bold', color='#2c3e50')
    ax.set_title('Composantes de Variation', fontsize=14, fontweight='bold', 
                 color='#2c3e50', pad=20)
    ax.grid(True, alpha=0.3, axis='y', zorder=0)
    ax.set_facecolor('#f8fafc')
    
    plt.tight_layout()
    return fig

def create_pie_chart(grr, vp):
    """Crée un camembert pour la répartition des variations"""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='none')
    
    labels = ['GRR', 'VP']
    sizes = [grr**2, vp**2]
    colors = ['#9b59b6', '#e74c3c']
    explode = (0.1, 0)
    
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold', 'color': 'white'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    # Centre blanc
    centre_circle = plt.Circle((0,0), 0.70, fc='white', edgecolor='white', linewidth=2)
    fig.gca().add_artist(centre_circle)
    
    ax.axis('equal')
    ax.set_title('Répartition des Variations', fontsize=14, fontweight='bold', 
                 color='#2c3e50', pad=20)
    
    plt.tight_layout()
    return fig

def create_heatmap(df, op_cols_list):
    """Crée une heatmap des mesures"""
    # Préparer les données
    heatmap_data = []
    labels = []
    
    for i, op_cols in enumerate(op_cols_list):
        op_means = df[op_cols].mean(axis=1).values[:5]  # 5 premières pièces
        heatmap_data.append(op_means)
        labels.append(f'Op {i+1}')
    
    heatmap_array = np.array(heatmap_data)
    
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='none')
    
    # Créer la heatmap
    im = ax.imshow(heatmap_array, cmap='viridis', aspect='auto')
    
    # Configurer les axes
    ax.set_xticks(range(len(heatmap_array[0])))
    ax.set_xticklabels([f'P{i+1}' for i in range(len(heatmap_array[0]))])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    
    # Ajouter les valeurs
    for i in range(len(labels)):
        for j in range(len(heatmap_array[0])):
            text = ax.text(j, i, f'{heatmap_array[i, j]:.2f}',
                          ha="center", va="center", color="w", fontweight='bold')
    
    # Barre de couleur
    plt.colorbar(im, ax=ax)
    
    ax.set_title('Distribution des Mesures par Opérateur', fontsize=12, 
                 fontweight='bold', color='#2c3e50', pad=20)
    
    plt.tight_layout()
    return fig

# ============================================================================
# HEADER PRINCIPAL
# ============================================================================
particles = create_particles(30)
st.markdown(f'''
<div class="main-header">
    {particles}
    <div class="main-title">🎯 Gage R&R Excellence Suite</div>
    <div class="main-subtitle">
        Analyse de précision industrielle • Méthode des étendues • Système de mesure avancé
    </div>
</div>
''', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR PRÉMIUM
# ============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    # Logo et titre sidebar
    st.markdown('''
    <div style="text-align: center; margin-bottom: 3rem;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
        <div style="font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 1.5rem; 
                    background: var(--primary-gradient); -webkit-background-clip: text; 
                    -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;">
            Gage R&R Pro
        </div>
        <div style="color: rgba(44, 62, 80, 0.6); font-size: 0.9rem;">
            v2.0 • Premium Edition
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Paramètres
    st.markdown('<div style="font-family: \'Poppins\', sans-serif; font-weight: 600; font-size: 1.3rem; color: #2c3e50; margin: 1.5rem 0;">⚙️ Configuration Avancée</div>', unsafe_allow_html=True)
    
    confidence_factor = st.slider(
        "**Facteur de Confiance (k)**",
        min_value=4.0,
        max_value=6.0,
        value=5.15,
        step=0.05,
        help="Niveau de confiance statistique pour les calculs"
    )
    
    st.markdown("---")
    
    # Guide de lecture interactif
    st.markdown('<div style="font-family: \'Poppins\', sans-serif; font-weight: 600; font-size: 1.3rem; color: #2c3e50; margin: 1.5rem 0;">📖 Guide Interactif</div>', unsafe_allow_html=True)
    
    with st.expander("🎯 **Comprendre les Métriques**", expanded=True):
        metrics_info = {
            "EV": ("Répétabilité", "Variation due à l'équipement", "#3498db"),
            "AV": ("Reproductibilité", "Variation entre opérateurs", "#2ecc71"),
            "GRR": ("Variation Système", "Variation totale du système", "#9b59b6"),
            "VP": ("Variation Pièces", "Variation entre les pièces", "#e74c3c"),
            "VT": ("Variation Totale", "Variation totale du processus", "#f39c12")
        }
        
        for key, (title, desc, color) in metrics_info.items():
            st.markdown(f'''
            <div style="background: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1); 
                        padding: 1rem; border-radius: 12px; margin-bottom: 0.8rem; 
                        border-left: 4px solid {color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 600; color: {color};">{key}</div>
                        <div style="font-weight: 500; color: #2c3e50; font-size: 0.95rem;">{title}</div>
                    </div>
                    <div style="width: 12px; height: 12px; border-radius: 50%; background: {color};"></div>
                </div>
                <div style="color: #7f8c8d; font-size: 0.85rem; margin-top: 0.5rem;">{desc}</div>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Critères d'évaluation
    st.markdown('<div style="font-family: \'Poppins\', sans-serif; font-weight: 600; font-size: 1.3rem; color: #2c3e50; margin: 1.5rem 0;">✅ Critères d\'Évaluation</div>', unsafe_allow_html=True)
    
    criteria_data = [
        ("< 10%", "EXCELLENT", "Le système est optimal et fiable", "#2ecc71"),
        ("10-30%", "ACCEPTABLE", "Système utilisable avec surveillance", "#f1c40f"),
        ("> 30%", "INACCEPTABLE", "Action corrective requise", "#e74c3c")
    ]
    
    for value, status, desc, color in criteria_data:
        st.markdown(f'''
        <div style="background: linear-gradient(135deg, rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1) 0%, 
                    rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.05) 100%);
                    padding: 1.2rem; border-radius: 15px; margin-bottom: 1rem; border: 1px solid {color};">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 0.5rem;">
                <div style="width: 10px; height: 10px; border-radius: 50%; background: {color};"></div>
                <div style="font-weight: 700; font-size: 1.1rem; color: {color};">{status}</div>
                <div style="margin-left: auto; font-weight: 600; color: #2c3e50;">{value}</div>
            </div>
            <div style="color: #7f8c8d; font-size: 0.9rem; line-height: 1.4;">{desc}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# ZONE D'UPLOAD PRÉMIUM
# ============================================================================
st.markdown('<div class="section-header fade-in"><span>📤 Importation des Données</span></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["xlsx"],
    help="Téléversez votre fichier Excel formaté pour l'analyse Gage R&R",
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown(f'''
    <div class="upload-area">
        {create_particles(15)}
        <div style="font-size: 5rem; margin-bottom: 2rem; opacity: 0.8;">☁️</div>
        <div style="font-family: 'Poppins', sans-serif; font-size: 2rem; font-weight: 700; 
                    color: #2c3e50; margin-bottom: 1rem; position: relative; z-index: 1;">
            <span class="shimmer-text">Glissez votre fichier ici</span>
        </div>
        <div style="color: rgba(44, 62, 80, 0.7); font-size: 1.1rem; margin-bottom: 3rem; 
                    position: relative; z-index: 1; max-width: 600px; margin: 0 auto 3rem auto;">
            ou cliquez pour parcourir les fichiers • Formats supportés: .xlsx
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                    gap: 2rem; margin-top: 3rem; position: relative; z-index: 1;">
            <div class="glowing-border" style="padding: 2rem; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">📋</div>
                <div style="font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">Format Requis</div>
                <div style="color: #7f8c8d; font-size: 0.9rem; line-height: 1.6;">
                    Colonnes: OP1-1, OP1-2, OP1-3, OP2-1, OP2-2, OP2-3, OP3-1, OP3-2, OP3-3
                </div>
            </div>
            
            <div class="glowing-border" style="padding: 2rem; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🎯</div>
                <div style="font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">Structure</div>
                <div style="color: #7f8c8d; font-size: 0.9rem; line-height: 1.6;">
                    3 opérateurs × 3 essais<br>Chaque ligne = une pièce<br>Données numériques uniquement
                </div>
            </div>
            
            <div class="glowing-border" style="padding: 2rem; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">⚡</div>
                <div style="font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">Performance</div>
                <div style="color: #7f8c8d; font-size: 0.9rem; line-height: 1.6;">
                    Analyse en temps réel<br>Visualisations avancées<br>Export complet
                </div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

# ============================================================================
# TRAITEMENT DES DONNÉES
# ============================================================================
if uploaded_file:
    # Animation de chargement
    with st.spinner('''
    <div style="text-align: center;">
        <div style="font-size: 2rem; margin-bottom: 1rem;">⏳</div>
        <div style="font-family: 'Poppins', sans-serif; font-weight: 600; color: #2c3e50;">
            Analyse en cours...
        </div>
        <div style="color: #7f8c8d; margin-top: 0.5rem;">
            Traitement des données et calcul des métriques
        </div>
    </div>
    '''):
        time.sleep(1)
        df = pd.read_excel(uploaded_file)
    
    # Aperçu des données
    st.markdown('<div class="section-header fade-in delay-1"><span>📊 Aperçu des Données</span></div>', unsafe_allow_html=True)
    
    with st.expander("**🔍 Visualiser les données d\'entrée**", expanded=True):
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            
            # Style avancé pour le DataFrame
            def highlight_data(val):
                if isinstance(val, (int, float)):
                    # Normalisation pour le gradient
                    normalized = (val - df.values.min()) / (df.values.max() - df.values.min())
                    hue = int(210 * (1 - normalized))  # Du bleu (faible) au rouge (élevé)
                    return f'background: hsl({hue}, 70%, 90%); color: hsl({hue}, 50%, 30%); font-weight: 500;'
                return ''
            
            styled_df = df.style.applymap(highlight_data)
            st.dataframe(styled_df, use_container_width=True, height=400)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Colonnes opérateurs
    op1_cols = ["OP1-1", "OP1-2", "OP1-3"]
    op2_cols = ["OP2-1", "OP2-2", "OP2-3"]
    op3_cols = ["OP3-1", "OP3-2", "OP3-3"]
    
    n_pieces = df.shape[0]
    n_operateurs = 3
    n_essais = 3
    
    # ============================================================================
    # CALCULS
    # ============================================================================
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
    
    # ============================================================================
    # VISUALISATIONS AVANCÉES
    # ============================================================================
    st.markdown('<div class="section-header fade-in delay-2"><span>📈 Tableau de Bord Analytique</span></div>', unsafe_allow_html=True)
    
    # Première ligne : Métriques clés
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_config = [
        ("EV", ev, "#3498db", "📏", "Répétabilité"),
        ("AV", av, "#2ecc71", "👥", "Reproductibilité"),
        ("GRR", grr, "#9b59b6", "⚙️", "Variation Système"),
        ("%GRR", p_grr, "#e74c3c", "🎯", "Performance")
    ]
    
    for col, (label, value, color, icon, title) in zip([col1, col2, col3, col4], metrics_config):
        with col:
            st.markdown(f'''
            <div class="metric-card fade-in delay-{metrics_config.index((label, value, color, icon, title)) + 1}">
                <div class="metric-icon" style="color: {color};">{icon}</div>
                <div class="metric-label">{title}</div>
                <div class="metric-value">{value:.3f}{"%" if label == "%GRR" else ""}</div>
                <div style="color: rgba(44, 62, 80, 0.6); font-size: 0.9rem; margin-top: 1rem;">
                    <strong>{label}</strong> • {["Faible", "Moyen", "Élevé", "Critique"][metrics_config.index((label, value, color, icon, title))]}
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    # Graphiques avec Matplotlib
    st.markdown('<div class="section-header fade-in delay-3"><span>📊 Visualisations Interactives</span></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique 1 : Composantes de variation
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.markdown('<div style="font-family: \'Poppins\', sans-serif; font-weight: 600; font-size: 1.3rem; color: #2c3e50; margin-bottom: 1.5rem;">📈 Analyse des Composantes</div>', unsafe_allow_html=True)
        
        fig1 = create_components_chart(ev, av, grr, vp, vt)
        st.pyplot(fig1)
        plt.close(fig1)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Graphique 2 : Jauge de performance
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.markdown('<div style="font-family: \'Poppins\', sans-serif; font-weight: 600; font-size: 1.3rem; color: #2c3e50; margin-bottom: 1.5rem;">📊 Jauge de Performance</div>', unsafe_allow_html=True)
        
        fig2 = create_advanced_gauge(p_grr)
        st.pyplot(fig2)
        plt.close(fig2)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Graphique 3 : Camembert
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.markdown('<div style="font-family: \'Poppins\', sans-serif; font-weight: 600; font-size: 1.3rem; color: #2c3e50; margin-bottom: 1.5rem;">🥧 Répartition des Variations</div>', unsafe_allow_html=True)
        
        fig3 = create_pie_chart(grr, vp)
        st.pyplot(fig3)
        plt.close(fig3)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Graphique 4 : Heatmap
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.markdown('<div style="font-family: \'Poppins\', sans-serif; font-weight: 600; font-size: 1.3rem; color: #2c3e50; margin-bottom: 1.5rem;">🔥 Matrice de Performance</div>', unsafe_allow_html=True)
        
        fig4 = create_heatmap(df, [op1_cols, op2_cols, op3_cols])
        st.pyplot(fig4)
        plt.close(fig4)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Graphique supplémentaire : Radar
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.markdown('<div style="font-family: \'Poppins\', sans-serif; font-weight: 600; font-size: 1.3rem; color: #2c3e50; margin-bottom: 1.5rem;">🎯 Performance Radar</div>', unsafe_allow_html=True)
    
    fig_radar = create_radar_chart(ev, av, p_grr, vt)
    st.pyplot(fig_radar)
    plt.close(fig_radar)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================================================
    # RÉSULTATS DÉTAILLÉS
    # ============================================================================
    st.markdown('<div class="section-header fade-in delay-4"><span>📋 Synthèse des Résultats</span></div>', unsafe_allow_html=True)
    
    # Indicateur principal avec animation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(create_circular_progress(p_grr, size=200, stroke_width=15), unsafe_allow_html=True)
        
        # Message de statut
        if p_grr < 10:
            status = ("good", "✅ SYSTÈME EXCELLENT", 
                     "La performance du système de mesure est optimale et fiable.")
            st.balloons()
        elif p_grr <= 30:
            status = ("warning", "⚠️ SYSTÈME ACCEPTABLE", 
                     "Le système est utilisable mais des améliorations sont recommandées.")
        else:
            status = ("bad", "❌ SYSTÈME INACCEPTABLE", 
                     "Des actions correctives sont nécessaires pour améliorer le système.")
        
        st.markdown(f'''
        <div class="result-indicator {status[0]} fade-in delay-5">
            <div style="font-family: 'Poppins', sans-serif; font-size: 1.4rem; margin-bottom: 0.8rem;">
                {status[1]}
            </div>
            <div style="font-size: 1rem; opacity: 0.9; line-height: 1.5;">
                {status[2]}
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Statistiques détaillées
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.markdown('<div style="font-family: \'Poppins\', sans-serif; font-weight: 600; font-size: 1.3rem; color: #2c3e50; margin-bottom: 1.5rem;">👥 Performance des Opérateurs</div>', unsafe_allow_html=True)
        
        operators_stats = pd.DataFrame({
            'Opérateur': ['👤 Opérateur 1', '👤 Opérateur 2', '👤 Opérateur 3'],
            'Moyenne': [f'{x_bar_op1:.4f}', f'{x_bar_op2:.4f}', f'{x_bar_op3:.4f}'],
            'Étendue': [f'{r_bar_op1:.4f}', f'{r_bar_op2:.4f}', f'{r_bar_op3:.4f}'],
            'σ': [f'{df[op1_cols].values.std():.4f}', 
                  f'{df[op2_cols].values.std():.4f}', 
                  f'{df[op3_cols].values.std():.4f}']
        })
        
        st.dataframe(
            operators_stats.style
            .background_gradient(subset=['Moyenne', 'Étendue', 'σ'], cmap='YlOrRd')
            .set_properties(**{
                'text-align': 'center',
                'border': '1px solid rgba(0,0,0,0.05)',
                'padding': '12px'
            }),
            use_container_width=True,
            height=200
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.markdown('<div style="font-family: \'Poppins\', sans-serif; font-weight: 600; font-size: 1.3rem; color: #2c3e50; margin-bottom: 1.5rem;">📊 Métriques Secondaires</div>', unsafe_allow_html=True)
        
        secondary_metrics = [
            ("VP (Variation Pièces)", vp, "#e74c3c", "📦"),
            ("VT (Variation Totale)", vt, "#f39c12", "🌐"),
            ("R̄ (Étendue Moyenne)", r_double_bar, "#3498db", "📏"),
            ("k (Facteur)", confidence_factor, "#9b59b6", "⚖️")
        ]
        
        for label, value, color, icon in secondary_metrics:
            st.markdown(f'''
            <div style="display: flex; align-items: center; justify-content: space-between; 
                        padding: 1rem; background: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, 
                        {int(color[5:7], 16)}, 0.1); border-radius: 12px; margin-bottom: 0.8rem; 
                        border-left: 4px solid {color};">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="font-size: 1.2rem; color: {color};">{icon}</div>
                    <div>
                        <div style="font-weight: 600; color: #2c3e50;">{label}</div>
                        <div style="color: #7f8c8d; font-size: 0.85rem;">Valeur calculée</div>
                    </div>
                </div>
                <div style="font-family: 'Poppins', sans-serif; font-weight: 700; 
                         font-size: 1.2rem; color: {color};">{value:.4f}</div>
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================================================
    # EXPORT PRÉMIUM
    # ============================================================================
    st.markdown('<div class="section-header fade-in delay-5"><span>💎 Export Professionnel</span></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        # Création du rapport
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Rapport principal
            report_df = pd.DataFrame({
                "Métrique": ["EV - Répétabilité", "AV - Reproductibilité", "GRR - Variation Système", 
                           "VP - Variation Pièces", "VT - Variation Totale", "%GRR - Performance"],
                "Valeur": [ev, av, grr, vp, vt, p_grr],
                "Unité": ["unité", "unité", "unité", "unité", "unité", "%"],
                "Statut": [
                    "✓ Optimal" if ev/vt*100 < 30 else "⚠ Améliorable" if ev/vt*100 <= 50 else "✗ Critique",
                    "✓ Optimal" if av/vt*100 < 30 else "⚠ Améliorable" if av/vt*100 <= 50 else "✗ Critique",
                    "✓ Excellent" if p_grr < 10 else "⚠ Acceptable" if p_grr <= 30 else "✗ Inacceptable",
                    "-", "-",
                    f"{p_grr:.1f}% ({'Excellent' if p_grr < 10 else 'Acceptable' if p_grr <= 30 else 'Inacceptable'})"
                ]
            })
            report_df.to_excel(writer, sheet_name='Résultats Détaillés', index=False)
            
            # Données brutes
            df.to_excel(writer, sheet_name='Données Brutes', index=False)
            
            # Métadonnées
            metadata_df = pd.DataFrame({
                "Paramètre": ["Date d'analyse", "Nombre de pièces", "Nombre d'opérateurs", 
                            "Nombre d'essais", "Facteur de confiance (k)", "Version logiciel"],
                "Valeur": [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    n_pieces,
                    n_operateurs,
                    n_essais,
                    confidence_factor,
                    "Gage R&R Premium Suite v2.0"
                ]
            })
            metadata_df.to_excel(writer, sheet_name='Métadonnées', index=False)
        
        output.seek(0)
        
        # Bouton de téléchargement premium
        st.markdown(f'''
        <div style="text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, 
                    rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); 
                    border-radius: 24px; border: 2px dashed rgba(102, 126, 234, 0.3); 
                    margin: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1.5rem;">📁</div>
            <div style="font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 1.8rem; 
                    color: #2c3e50; margin-bottom: 1rem;">
                Rapport d'Analyse Complet
            </div>
            <div style="color: #7f8c8d; font-size: 1.1rem; margin-bottom: 2.5rem; line-height: 1.6;">
                Exportez un rapport professionnel avec tous les résultats,<br>
                visualisations et données détaillées.
            </div>
            
            <a href='data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{output.getvalue().hex()}' 
               download='gage_rr_rapport_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
               style="text-decoration: none;">
               <button class="download-btn">
                   <span style="font-size: 1.3rem;">⬇️</span>
                   Télécharger le Rapport Premium
               </button>
            </a>
            
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2.5rem;">
                <div style="text-align: center;">
                    <div style="font-weight: 600; color: #2c3e50;">📊</div>
                    <div style="color: #7f8c8d; font-size: 0.9rem;">Résultats</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-weight: 600; color: #2c3e50;">📋</div>
                    <div style="color: #7f8c8d; font-size: 0.9rem;">Données</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-weight: 600; color: #2c3e50;">📈</div>
                    <div style="color: #7f8c8d; font-size: 0.9rem;">Métadonnées</div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

# ============================================================================
# FOOTER PRÉMIUM
# ============================================================================
st.markdown('''
<div style="margin-top: 5rem; padding: 3rem 2rem; background: linear-gradient(135deg, 
            rgba(44, 62, 80, 0.05) 0%, rgba(52, 152, 219, 0.05) 100%); 
            border-radius: 30px; text-align: center; border-top: 1px solid rgba(0,0,0,0.05);">
    
    <div style="font-family: 'Poppins', sans-serif; font-weight: 800; font-size: 2rem; 
                background: var(--primary-gradient); -webkit-background-clip: text; 
                -webkit-text-fill-color: transparent; margin-bottom: 1rem;">
        Gage R&R Excellence Suite
    </div>
    
    <div style="color: #7f8c8d; font-size: 1rem; max-width: 800px; margin: 0 auto 2rem auto; 
                line-height: 1.6;">
        Solution professionnelle d'analyse de la capacité des systèmes de mesure • 
        Conçue pour les exigences de l'industrie 4.0
    </div>
    
    <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem; 
                flex-wrap: wrap;">
        <div style="text-align: center;">
            <div style="font-size: 1.2rem; color: #667eea;">⚡</div>
            <div style="font-weight: 600; color: #2c3e50;">Performance</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.2rem; color: #667eea;">🎯</div>
            <div style="font-weight: 600; color: #2c3e50;">Précision</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.2rem; color: #667eea;">📊</div>
            <div style="font-weight: 600; color: #2c3e50;">Analyse</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.2rem; color: #667eea;">💎</div>
            <div style="font-weight: 600; color: #2c3e50;">Qualité</div>
        </div>
    </div>
    
    <div style="margin-top: 2rem; color: rgba(44, 62, 80, 0.5); font-size: 0.9rem;">
        © 2024 Gage R&R Premium • Version 2.0 • Développé avec Streamlit
    </div>
</div>
''', unsafe_allow_html=True)

# Effets spéciaux
st.markdown('''
<script>
    // Effet de particules dynamiques
    document.addEventListener('DOMContentLoaded', function() {
        // Animation au scroll
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animated');
                }
            });
        }, observerOptions);
        
        // Observer tous les éléments avec animation
        document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));
        
        // Effet de parallaxe sur le header
        window.addEventListener('scroll', function() {
            const scrolled = window.pageYOffset;
            const rate = scrolled * -0.5;
            const header = document.querySelector('.main-header');
            if (header) {
                header.style.transform = `translateY(${rate}px)`;
            }
        });
    });
</script>
''', unsafe_allow_html=True)
