import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import base64
from datetime import datetime
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# Configuration Matplotlib
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.facecolor'] = '#f8fafc'
plt.rcParams['figure.facecolor'] = 'white'

st.set_page_config(
    page_title="Gage R&R - Étendues",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé avec animations et effets visuels
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.1);
        animation: fadeIn 1s ease-out;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-title {
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .main-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f5f7fa);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
        box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.05), 
                    -5px -5px 15px rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 15px 15px 30px rgba(0, 0, 0, 0.1), 
                    -15px -15px 30px rgba(255, 255, 255, 0.9);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.6s ease;
    }
    
    .metric-card:hover::before {
        transform: scaleX(1);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2c3e50, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #7f8c8d;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .result-indicator {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        backdrop-filter: blur(10px);
        transition: all 0.4s ease;
        border: 2px solid transparent;
        animation: pulse 2s infinite;
    }
    
    .result-indicator:hover {
        transform: scale(1.03);
    }
    
    .good {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.15), rgba(39, 174, 96, 0.25));
        color: #27ae60;
        border-color: #2ecc71;
    }
    
    .warning {
        background: linear-gradient(135deg, rgba(241, 196, 15, 0.15), rgba(243, 156, 18, 0.25));
        color: #f39c12;
        border-color: #f1c40f;
    }
    
    .bad {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.15), rgba(192, 57, 43, 0.25));
        color: #c0392b;
        border-color: #e74c3c;
        animation: shake 0.5s ease-in-out;
    }
    
    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .plot-container {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
        margin: 1.5rem 0;
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }
    
    .plot-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
    }
    
    .dataframe-container {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #e0e6ed;
    }
    
    .upload-area {
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
        margin: 2rem 0;
    }
    
    .upload-area:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: #764ba2;
    }
    
    .download-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        display: inline-flex;
        align-items: center;
        gap: 10px;
        margin: 1rem 0;
    }
    
    .download-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .download-btn-pdf {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
        display: inline-flex;
        align-items: center;
        gap: 10px;
        margin: 1rem 0;
    }
    
    .download-btn-pdf:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(231, 76, 60, 0.4);
    }
    
    .sidebar-content {
        padding: 1.5rem;
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 0 20px 20px 0;
        height: 100%;
    }
    
    .report-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-left: 5px solid;
        transition: transform 0.3s ease;
    }
    
    .report-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }
    
    .floating-badge {
        position: absolute;
        top: -10px;
        right: -10px;
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(255, 107, 107, 0.3);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
        20%, 40%, 60%, 80% { transform: translateX(5px); }
    }
    
    .progress-container {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 3px;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 10px;
        border-radius: 8px;
        background: linear-gradient(90deg, #2ecc71, #f1c40f, #e74c3c);
        transition: width 1.5s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #3498db;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        border-left-color: #667eea;
        transform: translateX(5px);
    }
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
</style>
""", unsafe_allow_html=True)

# Header principal avec animation
st.markdown("""
<div class="main-header">
    <div class="main-title">📊 Gage R&R - Méthode des Étendues</div>
    <div class="main-subtitle">Analyse avancée avec visualisations détaillées du système de mesure</div>
</div>
""", unsafe_allow_html=True)

# ---------------- FONCTIONS UTILITAIRES ----------------
def get_d2(z, w):
    """Retourne la valeur d2 pour les calculs Gage R&R"""
    if z > 15 and w == 3:
        return 1.693
    if z == 1 and w == 3:
        return 1.91
    if z == 1 and w == 10:
        return 3.18
    return 1.0

def create_gauge_chart(value, title, min_val=0, max_val=100, thresholds=[10, 30], colors=['#2ecc71', '#f1c40f', '#e74c3c']):
    """Crée un graphique de jauge (gauge chart)"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Créer la jauge
    angles = np.linspace(0, 180, 300)
    angles_rad = np.deg2rad(angles)
    
    # Zones colorées
    for i in range(len(thresholds)+1):
        if i == 0:
            start_angle = 0
            end_angle = thresholds[0]
        elif i == len(thresholds):
            start_angle = thresholds[-1]
            end_angle = 100
        else:
            start_angle = thresholds[i-1]
            end_angle = thresholds[i]
        
        start_idx = int(start_angle * 3)
        end_idx = int(end_angle * 3)
        
        ax.plot(angles_rad[start_idx:end_idx], [1]*len(angles_rad[start_idx:end_idx]), 
                color=colors[i], linewidth=30, solid_capstyle='round')
    
    # Aiguille
    needle_angle = (value / 100) * 180
    needle_rad = np.deg2rad(needle_angle)
    ax.plot([0, needle_rad], [0, 0.8], color='#2c3e50', linewidth=3, solid_capstyle='round')
    ax.plot(needle_rad, 0.8, 'o', color='#2c3e50', markersize=12)
    
    # Style
    ax.set_ylim(0, 1.2)
    ax.set_xlim(-0.2, np.pi + 0.2)
    ax.axis('off')
    
    # Valeur
    ax.text(np.pi/2, 1.1, f'{value:.1f}%', ha='center', va='center', 
            fontsize=24, fontweight='bold', color='#2c3e50')
    
    # Titre
    ax.text(np.pi/2, -0.1, title, ha='center', va='center', 
            fontsize=14, fontweight='bold', color='#2c3e50')
    
    # Seuils
    for threshold in thresholds:
        angle = np.deg2rad((threshold / 100) * 180)
        ax.text(angle, 1.15, f'{threshold}%', ha='center', va='center',
                fontsize=10, color='#7f8c8d')
    
    plt.tight_layout()
    return fig

def create_variation_breakdown_chart(ev, av, vp, grr, vt):
    """Crée un graphique de décomposition de la variation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Graphique 1: Barres empilées
    components = ['EV', 'AV', 'VP']
    values = [ev, av, vp]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    percentages = [(ev/vt)*100, (av/vt)*100, (vp/vt)*100] if vt > 0 else [0, 0, 0]
    
    bars = ax1.bar(components, values, color=colors, edgecolor='white', linewidth=2)
    
    # Ajouter les valeurs
    for bar, value, percent in zip(bars, values, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{value:.3f}\n({percent:.1f}%)', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    ax1.set_title('📊 Décomposition de la Variation', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Valeur', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)
    
    # Légende
    legend_elements = [
        mpatches.Patch(color='#3498db', label='Répétabilité (EV)'),
        mpatches.Patch(color='#2ecc71', label='Reproductibilité (AV)'),
        mpatches.Patch(color='#e74c3c', label='Variation Pièces (VP)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Graphique 2: Diagramme en barres groupées
    categories = ['Variation', 'Pourcentage']
    x = np.arange(len(categories))
    width = 0.25
    
    ev_data = [ev, (ev/vt)*100] if vt > 0 else [ev, 0]
    av_data = [av, (av/vt)*100] if vt > 0 else [av, 0]
    vp_data = [vp, (vp/vt)*100] if vt > 0 else [vp, 0]
    
    ax2.bar(x - width, ev_data, width, label='EV', color='#3498db', edgecolor='white')
    ax2.bar(x, av_data, width, label='AV', color='#2ecc71', edgecolor='white')
    ax2.bar(x + width, vp_data, width, label='VP', color='#e74c3c', edgecolor='white')
    
    ax2.set_title('📈 Comparaison par Composante', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)
    
    # Ajouter les valeurs sur les barres
    for i, (ev_val, av_val, vp_val) in enumerate(zip(ev_data, av_data, vp_data)):
        ax2.text(i - width, ev_val + max(ev_data+av_data+vp_data)*0.01, f'{ev_val:.2f}', 
                ha='center', va='bottom', fontsize=9)
        ax2.text(i, av_val + max(ev_data+av_data+vp_data)*0.01, f'{av_val:.2f}', 
                ha='center', va='bottom', fontsize=9)
        ax2.text(i + width, vp_val + max(ev_data+av_data+vp_data)*0.01, f'{vp_val:.2f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_operators_comparison_chart(df, operators_data):
    """Crée un graphique de comparaison des opérateurs"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Données pour les opérateurs
    op_names = [f'OP{i+1}' for i in range(3)]
    means = [op['moyenne'] for op in operators_data]
    ranges = [op['etendue'] for op in operators_data]
    stds = [op['ecart_type'] for op in operators_data]
    
    # Graphique 1: Moyennes des opérateurs
    bars1 = axes[0].bar(op_names, means, color=['#3498db', '#2ecc71', '#9b59b6'], 
                       edgecolor='white', linewidth=2)
    axes[0].set_title('📊 Moyennes par Opérateur', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Valeur Moyenne', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar, mean in zip(bars1, means):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.01,
                    f'{mean:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Graphique 2: Étendues par opérateur
    bars2 = axes[1].bar(op_names, ranges, color=['#e74c3c', '#f39c12', '#1abc9c'], 
                       edgecolor='white', linewidth=2)
    axes[1].set_title('📏 Étendues par Opérateur', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Étendue Moyenne', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, range_val in zip(bars2, ranges):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ranges)*0.01,
                    f'{range_val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Graphique 3: Écart-type par opérateur
    bars3 = axes[2].bar(op_names, stds, color=['#34495e', '#7f8c8d', '#95a5a6'], 
                       edgecolor='white', linewidth=2)
    axes[2].set_title('📈 Écart-Type par Opérateur', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Écart-Type', fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for bar, std in zip(bars3, stds):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds)*0.01,
                    f'{std:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Graphique 4: Diagramme en boîte (boxplot) des opérateurs
    box_data = []
    for i in range(3):
        op_cols = [f'OP{i+1}-1', f'OP{i+1}-2', f'OP{i+1}-3']
        op_values = df[op_cols].values.flatten()
        box_data.append(op_values)
    
    bp = axes[3].boxplot(box_data, labels=op_names, patch_artist=True)
    
    # Couleurs des boîtes
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for whisker in bp['whiskers']:
        whisker.set(color='#7f8c8d', linewidth=1.5)
    
    for cap in bp['caps']:
        cap.set(color='#7f8c8d', linewidth=1.5)
    
    for median in bp['medians']:
        median.set(color='#2c3e50', linewidth=2)
    
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e74c3c', alpha=0.5)
    
    axes[3].set_title('📦 Distribution des Mesures par Opérateur', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('Valeurs', fontweight='bold')
    axes[3].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def create_parts_variation_chart(df):
    """Crée un graphique de variation par pièce"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculer la moyenne par pièce
    all_op_cols = []
    for i in range(1, 4):
        for j in range(1, 4):
            all_op_cols.append(f'OP{i}-{j}')
    
    df['Piece_Mean'] = df[all_op_cols].mean(axis=1)
    df['Piece_Range'] = df[all_op_cols].max(axis=1) - df[all_op_cols].min(axis=1)
    
    pieces = range(1, len(df) + 1)
    
    # Graphique 1: Moyennes par pièce
    ax1.plot(pieces, df['Piece_Mean'], 'o-', color='#3498db', linewidth=2, 
             markersize=8, markerfacecolor='white', markeredgewidth=2)
    
    # Ligne de moyenne générale
    overall_mean = df['Piece_Mean'].mean()
    ax1.axhline(y=overall_mean, color='#e74c3c', linestyle='--', linewidth=2, 
                label=f'Moyenne générale: {overall_mean:.3f}')
    
    ax1.fill_between(pieces, df['Piece_Mean'], overall_mean, 
                     where=(df['Piece_Mean'] > overall_mean), 
                     color='#2ecc71', alpha=0.2)
    ax1.fill_between(pieces, df['Piece_Mean'], overall_mean, 
                     where=(df['Piece_Mean'] <= overall_mean), 
                     color='#e74c3c', alpha=0.2)
    
    ax1.set_title('📈 Moyennes par Pièce', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Numéro de Pièce', fontweight='bold')
    ax1.set_ylabel('Valeur Moyenne', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Ajouter les valeurs
    for i, (piece, mean) in enumerate(zip(pieces, df['Piece_Mean'])):
        if i % max(1, len(df)//10) == 0:  # Afficher environ 10 valeurs
            ax1.text(piece, mean + (df['Piece_Mean'].max() - df['Piece_Mean'].min())*0.02, 
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Graphique 2: Étendues par pièce
    bars = ax2.bar(pieces, df['Piece_Range'], color='#9b59b6', edgecolor='white', linewidth=1)
    ax2.set_title('📏 Étendues par Pièce', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Numéro de Pièce', fontweight='bold')
    ax2.set_ylabel('Étendue (Max - Min)', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Ajouter la ligne de moyenne des étendues
    range_mean = df['Piece_Range'].mean()
    ax2.axhline(y=range_mean, color='#e74c3c', linestyle='--', linewidth=2, 
                label=f'Moyenne: {range_mean:.3f}')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def create_radar_chart(operators_data):
    """Crée un graphique radar pour comparer les opérateurs"""
    fig = plt.figure(figsize=(10, 8))
    
    # Catégories pour le radar
    categories = ['Précision\n(Moyenne)', 'Stabilité\n(1/Étendue)', 'Cohérence\n(1/Écart-Type)']
    N = len(categories)
    
    # Angles pour chaque catégorie
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Fermer le polygone
    
    # Valeurs normalisées pour chaque opérateur
    op_colors = ['#3498db', '#2ecc71', '#9b59b6']
    op_names = [f'Opérateur {i+1}' for i in range(3)]
    
    ax = plt.subplot(111, polar=True)
    
    for i, op in enumerate(operators_data):
        # Normaliser les valeurs entre 0 et 1
        values = [
            op['moyenne'] / max([op['moyenne'] for op in operators_data]),
            1 / (op['etendue'] * max([1/op['etendue'] for op in operators_data])),
            1 / (op['ecart_type'] * max([1/op['ecart_type'] for op in operators_data]))
        ]
        values += values[:1]  # Fermer le polygone
        
        ax.plot(angles, values, 'o-', linewidth=2, label=op_names[i], color=op_colors[i])
        ax.fill(angles, values, alpha=0.1, color=op_colors[i])
    
    # Configuration du graphique radar
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Titre
    plt.title('🎯 Performance Comparative des Opérateurs', fontsize=16, fontweight='bold', pad=20)
    
    # Légende
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    return fig

def create_contribution_chart(ev, av, vp, grr):
    """Crée un graphique de contribution à la variation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Données pour les graphiques
    components = ['EV', 'AV', 'VP']
    values_squared = [ev**2, av**2, vp**2]
    percentages = [(ev**2/grr**2)*100 if grr > 0 else 0, 
                   (av**2/grr**2)*100 if grr > 0 else 0,
                   (vp**2/grr**2)*100 if grr > 0 else 0]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # Graphique 1: Diagramme en cascade
    cumulative = 0
    for i, (comp, val, color) in enumerate(zip(components, values_squared, colors)):
        ax1.bar(i, val, bottom=cumulative, color=color, edgecolor='white', linewidth=2)
        cumulative += val
    
    ax1.set_title('📊 Cascade des Variations (Carrés)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Composantes', fontweight='bold')
    ax1.set_ylabel('Variation²', fontweight='bold')
    ax1.set_xticks(range(len(components)))
    ax1.set_xticklabels(components)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Graphique 2: Diagramme à barres empilées
    bottom = np.zeros(len(components))
    for i, (val, color) in enumerate(zip(percentages, colors)):
        ax2.bar('Contribution', val, bottom=bottom[i], color=color, edgecolor='white', linewidth=2)
        bottom[i] = val
    
    ax2.set_title('📈 Contribution Relative (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Pourcentage', fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les pourcentages
    cumulative = 0
    for i, percent in enumerate(percentages):
        if percent > 5:  # N'afficher que si significatif
            ax2.text(0, cumulative + percent/2, f'{percent:.1f}%', 
                    ha='center', va='center', fontweight='bold', color='white')
        cumulative += percent
    
    plt.tight_layout()
    return fig

def create_statistical_summary_chart(results):
    """Crée un tableau de bord statistique"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Données statistiques
    stats_data = [
        ('%GRR', results['p_grr'], '%', 'Rouge' if results['p_grr'] > 30 else 'Orange' if results['p_grr'] > 10 else 'Vert'),
        ('EV', results['ev'], '', 'Rouge' if results['ev_percent'] > 20 else 'Orange' if results['ev_percent'] > 10 else 'Vert'),
        ('AV', results['av'], '', 'Rouge' if results['av_percent'] > 20 else 'Orange' if results['av_percent'] > 10 else 'Vert'),
        ('VP', results['vp'], '', 'Vert' if results['vp_percent'] > 50 else 'Orange' if results['vp_percent'] > 30 else 'Rouge'),
        ('Ratio VP/GRR', results['ratio_vp_grr'], 'x', 'Vert' if results['ratio_vp_grr'] > 4 else 'Orange' if results['ratio_vp_grr'] > 2 else 'Rouge'),
        ('Étendue Moy', results['r_double_bar'], '', 'Info')
    ]
    
    colors_map = {
        'Vert': '#2ecc71',
        'Orange': '#f39c12',
        'Rouge': '#e74c3c',
        'Info': '#3498db'
    }
    
    for idx, (title, value, unit, color_key) in enumerate(stats_data):
        ax = axes[idx]
        
        # Créer un cadran
        ax.add_patch(plt.Circle((0.5, 0.5), 0.45, color=colors_map[color_key], alpha=0.2))
        ax.add_patch(plt.Circle((0.5, 0.5), 0.45, fill=False, edgecolor=colors_map[color_key], linewidth=2))
        
        # Afficher la valeur
        ax.text(0.5, 0.6, f'{value:.2f}{unit}', ha='center', va='center', 
                fontsize=24, fontweight='bold', color='#2c3e50')
        
        # Afficher le titre
        ax.text(0.5, 0.4, title, ha='center', va='center', 
                fontsize=12, fontweight='bold', color='#7f8c8d')
        
        # Ajouter une icône selon le type
        icons = {
            '%GRR': '🎯',
            'EV': '📏',
            'AV': '👥',
            'VP': '⚙️',
            'Ratio VP/GRR': '📊',
            'Étendue Moy': '📈'
        }
        
        if title in icons:
            ax.text(0.5, 0.8, icons[title], ha='center', va='center', 
                    fontsize=20, color=colors_map[color_key])
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.suptitle('📋 Tableau de Bord Statistique', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def create_improvement_potential_chart(p_grr, ev_percent, av_percent, vp_percent):
    """Crée un graphique de potentiel d'amélioration"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculer le potentiel d'amélioration
    current_total = p_grr
    target_total = 10  # Objectif < 10%
    
    # Répartition actuelle
    current_ev = ev_percent * (p_grr/100)
    current_av = av_percent * (p_grr/100)
    current_vp = vp_percent * (p_grr/100)
    
    # Objectif (réduire EV et AV de 50%)
    target_ev = current_ev * 0.5
    target_av = current_av * 0.5
    target_vp = current_vp  # VP reste identique
    
    # Données pour le graphique
    categories = ['Actuel', 'Objectif']
    ev_values = [current_ev, target_ev]
    av_values = [current_av, target_av]
    vp_values = [current_vp, target_vp]
    
    x = np.arange(len(categories))
    width = 0.25
    
    # Barres empilées
    bottom = np.zeros(len(categories))
    
    # Barre EV
    bars_ev = ax.bar(x - width, ev_values, width, label='EV', 
                     color='#3498db', edgecolor='white')
    # Barre AV
    bars_av = ax.bar(x, av_values, width, label='AV', 
                     color='#2ecc71', edgecolor='white', bottom=ev_values)
    # Barre VP
    bars_vp = ax.bar(x + width, vp_values, width, label='VP', 
                     color='#e74c3c', edgecolor='white')
    
    # Configuration
    ax.set_title('🎯 Potentiel d\'Amélioration du Système', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Scénario', fontweight='bold')
    ax.set_ylabel('Contribution au %GRR', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs
    for bars, values in zip([bars_ev, bars_av, bars_vp], [ev_values, av_values, vp_values]):
        for bar, value in zip(bars, values):
            if value > 0.5:  # N'afficher que si suffisamment grand
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                       f'{value:.1f}%', ha='center', va='center', 
                       fontweight='bold', color='white')
    
    # Ajouter les totaux
    totals = [current_ev + current_av + current_vp, target_ev + target_av + target_vp]
    for i, total in enumerate(totals):
        ax.text(i, total + 0.5, f'Total: {total:.1f}%', 
               ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Ligne de référence 10%
    ax.axhline(y=10, color='#27ae60', linestyle='--', linewidth=2, 
               label='Seuil Excellent (10%)')
    
    # Ligne de référence 30%
    ax.axhline(y=30, color='#f39c12', linestyle='--', linewidth=2, 
               label='Seuil Acceptable (30%)')
    
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

def create_timeline_chart(df):
    """Crée un graphique chronologique des mesures"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Préparer les données
    all_data = []
    colors = []
    labels = []
    
    for i in range(1, 4):  # Opérateurs
        for j in range(1, 4):  # Essais
            col_name = f'OP{i}-{j}'
            data = df[col_name].values
            all_data.append(data)
            
            # Couleur par opérateur
            if i == 1:
                colors.append('#3498db')
            elif i == 2:
                colors.append('#2ecc71')
            else:
                colors.append('#9b59b6')
            
            labels.append(f'OP{i}-T{j}')
    
    # Créer le graphique
    x_positions = np.arange(len(df))
    width = 0.08
    
    for idx, (data, color, label) in enumerate(zip(all_data, colors, labels)):
        positions = x_positions + (idx - len(all_data)/2) * width
        ax.bar(positions, data, width, color=color, alpha=0.7, 
               edgecolor='white', linewidth=0.5, label=label if idx < 3 else "")
    
    # Ajouter la moyenne par pièce
    piece_means = df[[f'OP{i}-{j}' for i in range(1, 4) for j in range(1, 4)]].mean(axis=1)
    ax.plot(x_positions, piece_means, 'o-', color='#e74c3c', linewidth=3, 
            markersize=8, label='Moyenne par pièce', zorder=5)
    
    # Configuration
    ax.set_title('🕒 Chronologie des Mesures par Pièce', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Numéro de Pièce', fontweight='bold')
    ax.set_ylabel('Valeur Mesurée', fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'P{i+1}' for i in range(len(df))])
    ax.grid(True, alpha=0.3)
    
    # Légende simplifiée
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#3498db', lw=4, label='Opérateur 1'),
        Line2D([0], [0], color='#2ecc71', lw=4, label='Opérateur 2'),
        Line2D([0], [0], color='#9b59b6', lw=4, label='Opérateur 3'),
        Line2D([0], [0], color='#e74c3c', lw=3, marker='o', label='Moyenne par pièce')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig

# ---------------- SIDEBAR STYLÉE ----------------
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    st.markdown('<div style="font-size: 1.5rem; font-weight: 700; color: #2c3e50; margin-bottom: 2rem;">⚙️ Configuration</div>', unsafe_allow_html=True)
    
    confidence_factor = st.slider(
        "**Facteur de Confiance (k)**",
        min_value=4.0,
        max_value=6.0,
        value=5.15,
        step=0.05,
        help="Facteur pour le niveau de confiance des calculs"
    )
    
    # Options de visualisation
    st.markdown("---")
    st.markdown('<div style="font-size: 1.2rem; font-weight: 600; color: #2c3e50; margin: 1.5rem 0 1rem 0;">📊 Options Graphiques</div>', unsafe_allow_html=True)
    
    show_all_charts = st.checkbox("Afficher tous les graphiques", value=True)
    
    if not show_all_charts:
        selected_charts = st.multiselect(
            "Graphiques à afficher",
            ["Décomposition de la variation", "Comparaison opérateurs", 
             "Variation par pièce", "Graphique radar", "Jauge %GRR",
             "Contribution relative", "Tableau de bord", "Potentiel d'amélioration",
             "Chronologie des mesures"],
            default=["Décomposition de la variation", "Jauge %GRR", "Comparaison opérateurs"]
        )
    
    st.markdown("---")
    st.markdown('<div style="font-size: 1.2rem; font-weight: 600; color: #2c3e50; margin: 1.5rem 0 1rem 0;">🎯 Guide des Graphiques</div>', unsafe_allow_html=True)
    
    with st.expander("📖 Légende des Visualisations"):
        st.markdown("""
        **🎯 Jauge %GRR**: Indicateur principal de performance
        **📊 Décomposition**: Répartition EV/AV/VP
        **👥 Opérateurs**: Comparaison des performances
        **📈 Par pièce**: Analyse de la variabilité
        **🔄 Radar**: Performance comparative
        **📋 Tableau de bord**: Vue d'ensemble statistique
        **🎯 Potentiel**: Objectifs d'amélioration
        **🕒 Chronologie**: Évolution des mesures
        """)
    
    st.markdown("---")
    st.markdown('<div style="font-size: 1.2rem; font-weight: 600; color: #2c3e50; margin: 1.5rem 0 1rem 0;">📈 Critères d\'Acceptation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(46, 204, 113, 0.1), rgba(39, 174, 96, 0.2)); 
                padding: 1rem; border-radius: 10px; border-left: 4px solid #2ecc71; margin-bottom: 0.5rem;">
        <div style="font-weight: 600; color: #27ae60;">✓ EXCELLENT</div>
        <div style="color: #7f8c8d; font-size: 0.9rem;">&lt; 10% - Système optimal</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(241, 196, 15, 0.1), rgba(243, 156, 18, 0.2)); 
                padding: 1rem; border-radius: 10px; border-left: 4px solid #f1c40f; margin-bottom: 0.5rem;">
        <div style="font-weight: 600; color: #f39c12;">⚠ ACCEPTABLE</div>
        <div style="color: #7f8c8d; font-size: 0.9rem;">10-30% - Amélioration souhaitée</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(231, 76, 60, 0.1), rgba(192, 57, 43, 0.2)); 
                padding: 1rem; border-radius: 10px; border-left: 4px solid #e74c3c;">
        <div style="font-weight: 600; color: #c0392b;">✗ INACCEPTABLE</div>
        <div style="color: #7f8c8d; font-size: 0.9rem;">&gt; 30% - Action corrective requise</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ZONE D'UPLOAD ----------------
st.markdown('<div class="section-header"><span>📥 Importation des Données</span></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["xlsx"],
    help="Téléversez votre fichier Excel contenant les mesures",
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown("""
    <div class="upload-area">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📁</div>
        <div style="font-size: 1.5rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">
            Glissez-déposez votre fichier Excel
        </div>
        <div style="color: #7f8c8d; margin-bottom: 2rem;">
            ou cliquez pour parcourir
        </div>
        <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; display: inline-block;">
            <div style="font-weight: 600; color: #667eea;">Format requis :</div>
            <div style="color: #7f8c8d; font-size: 0.9rem; text-align: left; margin-top: 0.5rem;">
                • Colonnes : OP1-1, OP1-2, OP1-3, OP2-1, OP2-2, OP2-3, OP3-1, OP3-2, OP3-3<br>
                • Lignes : Pièces mesurées<br>
                • 3 opérateurs × 3 essais
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file:
    # Animation de chargement
    with st.spinner('🔄 Traitement des données en cours...'):
        time.sleep(0.5)
        df = pd.read_excel(uploaded_file)

    # ---------------- APERÇU DES DONNÉES ----------------
    st.markdown('<div class="section-header"><span>📄 Aperçu des Données</span></div>', unsafe_allow_html=True)
    
    with st.expander("Voir les données détaillées", expanded=True):
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        
        def color_gradient(val):
            if isinstance(val, (int, float)):
                intensity = min(0.8, abs(val - df.values.mean()) / df.values.std() * 0.3)
                if val > df.values.mean():
                    return f'background: linear-gradient(90deg, rgba(46, 204, 113, {intensity}), rgba(39, 174, 96, {intensity/2}))'
                else:
                    return f'background: linear-gradient(90deg, rgba(52, 152, 219, {intensity}), rgba(41, 128, 185, {intensity/2}))'
            return ''
        
        styled_df = df.style.applymap(color_gradient)
        st.dataframe(styled_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Colonnes opérateurs
    op1_cols = ["OP1-1", "OP1-2", "OP1-3"]
    op2_cols = ["OP2-1", "OP2-2", "OP2-3"]
    op3_cols = ["OP3-1", "OP3-2", "OP3-3"]

    n_pieces = df.shape[0]
    n_operateurs = 3
    n_essais = 3

    # ---------------- CALCULS ----------------
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

    # Calculs pour les graphiques
    ev_percent = (ev / vt) * 100 if vt > 0 else 0
    av_percent = (av / vt) * 100 if vt > 0 else 0
    vp_percent = (vp / vt) * 100 if vt > 0 else 0
    ratio_vp_grr = vp / grr if grr > 0 else 0

    # Données des opérateurs
    operators_data = []
    for i in range(3):
        op_cols = [f"OP{i+1}-1", f"OP{i+1}-2", f"OP{i+1}-3"]
        op_values = df[op_cols].values.flatten()
        operators_data.append({
            'name': f'Opérateur {i+1}',
            'moyenne': np.mean(op_values),
            'etendue': [r_bar_op1, r_bar_op2, r_bar_op3][i],
            'ecart_type': np.std(op_values),
            'valeurs': op_values
        })

    # Données pour les résultats
    results = {
        'p_grr': p_grr,
        'ev': ev,
        'av': av,
        'grr': grr,
        'vp': vp,
        'vt': vt,
        'ev_percent': ev_percent,
        'av_percent': av_percent,
        'vp_percent': vp_percent,
        'ratio_vp_grr': ratio_vp_grr,
        'r_double_bar': r_double_bar,
        'n_pieces': n_pieces,
        'n_operateurs': n_operateurs,
        'n_essais': n_essais
    }

    # ---------------- MÉTRIQUES PRINCIPALES ----------------
    st.markdown('<div class="section-header"><span>📊 Métriques Principales</span></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_data = [
        ("EV", ev, "#3498db", f"Répétabilité\n({ev_percent:.1f}%)"),
        ("AV", av, "#2ecc71", f"Reproductibilité\n({av_percent:.1f}%)"),
        ("GRR", grr, "#9b59b6", f"Variation Système\n({p_grr:.1f}%)"),
        ("VP", vp, "#e74c3c", f"Variation Pièces\n({vp_percent:.1f}%)")
    ]
    
    for col, (label, value, color, desc) in zip([col1, col2, col3, col4], metrics_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{desc}</div>
                <div class="metric-value" style="background: linear-gradient(135deg, {color}, #2c3e50); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {value:.3f}
                </div>
                <div style="color: #95a5a6; font-size: 0.9rem; margin-top: 0.5rem;">
                    <strong>{label}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Indicateur de résultat
    if p_grr < 10:
        status = ("good", "✅", "SYSTÈME EXCELLENT", "Le système de mesure est optimal")
        st.balloons()
    elif p_grr <= 30:
        status = ("warning", "⚠️", "SYSTÈME ACCEPTABLE", "Améliorations possibles")
    else:
        status = ("bad", "❌", "SYSTÈME INACCEPTABLE", "Action corrective requise")
    
    st.markdown(f"""
    <div class="result-indicator {status[0]}">
        <div style="font-size: 1.3rem; margin-bottom: 0.5rem;">{status[1]} {status[2]}</div>
        <div style="font-size: 0.95rem; opacity: 0.9;">{status[3]}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- VISUALISATIONS DÉTAILLÉES ----------------
    st.markdown('<div class="section-header"><span>📈 Visualisations Détaillées</span></div>', unsafe_allow_html=True)
    
    # Définir quels graphiques afficher
    if show_all_charts:
        selected_charts = [
            "Décomposition de la variation", "Comparaison opérateurs", 
            "Variation par pièce", "Graphique radar", "Jauge %GRR",
            "Contribution relative", "Tableau de bord", "Potentiel d'amélioration",
            "Chronologie des mesures"
        ]
    
    # Afficher les graphiques
    chart_functions = {
        "Jauge %GRR": lambda: create_gauge_chart(p_grr, "Indicateur %GRR"),
        "Décomposition de la variation": lambda: create_variation_breakdown_chart(ev, av, vp, grr, vt),
        "Comparaison opérateurs": lambda: create_operators_comparison_chart(df, operators_data),
        "Variation par pièce": lambda: create_parts_variation_chart(df),
        "Graphique radar": lambda: create_radar_chart(operators_data),
        "Contribution relative": lambda: create_contribution_chart(ev, av, vp, grr),
        "Tableau de bord": lambda: create_statistical_summary_chart(results),
        "Potentiel d'amélioration": lambda: create_improvement_potential_chart(p_grr, ev_percent, av_percent, vp_percent),
        "Chronologie des mesures": lambda: create_timeline_chart(df)
    }
    
    # Afficher les graphiques par ligne de 2
    charts_list = [chart for chart in selected_charts if chart in chart_functions]
    
    for i in range(0, len(charts_list), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(charts_list):
                chart_name = charts_list[i + j]
                with cols[j]:
                    st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-weight: 600; color: #2c3e50; margin-bottom: 1rem; text-align: center;">{chart_name}</div>', unsafe_allow_html=True)
                    try:
                        fig = chart_functions[chart_name]()
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Erreur lors de la génération du graphique {chart_name}: {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # ---------------- INTERPRÉTATION DES GRAPHIQUES ----------------
    st.markdown('<div class="section-header"><span>🔍 Interprétation des Graphiques</span></div>', unsafe_allow_html=True)
    
    with st.expander("📖 Guide d'interprétation des visualisations", expanded=False):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 **Jauge %GRR**
            - **<10% (Vert)**: Système excellent
            - **10-30% (Orange)**: Système acceptable
            - **>30% (Rouge)**: Système inacceptable
            
            ### 📊 **Décomposition de la Variation**
            - **EV (Bleu)**: Répétabilité (variation intra-opérateur)
            - **AV (Vert)**: Reproductibilité (variation inter-opérateur)
            - **VP (Rouge)**: Variation naturelle des pièces
            
            ### 👥 **Comparaison Opérateurs**
            - **Moyennes**: Cohérence des mesures
            - **Étendues**: Stabilité des mesures
            - **Écart-type**: Dispersion des résultats
            - **Boxplot**: Distribution complète
            
            ### 📈 **Variation par Pièce**
            - **Tendance**: Détection de motifs
            - **Points extrêmes**: Pièces problématiques
            - **Étendues**: Consistance des mesures
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 **Potentiel d'Amélioration**
            - **Actuel vs Objectif**: Gains potentiels
            - **Répartition**: Sources d'amélioration
            - **Seuils**: Objectifs à atteindre
            
            ### 📋 **Tableau de Bord**
            - **Vue d'ensemble**: Tous les indicateurs
            - **Couleurs**: Statut instantané
            - **Priorisation**: Points d'attention
            
            ### 🕒 **Chronologie des Mesures**
            - **Séquence**: Ordre des mesures
            - **Cohérence**: Évolution dans le temps
            - **Anomalies**: Mesures aberrantes
            
            ### 🔄 **Graphique Radar**
            - **Performance comparative**: Opérateurs vs opérateurs
            - **Points forts/faibles**: Par opérateur
            - **Équilibre**: Compétences globales
            """)
        
        st.markdown("""
        ### 📊 **Comment Analyser les Résultats**
        1. **Commencez par la jauge %GRR** pour le statut global
        2. **Analysez la décomposition** pour identifier la source principale
        3. **Comparez les opérateurs** pour détecter les divergences
        4. **Examinez la variation par pièce** pour la discrimination
        5. **Utilisez le tableau de bord** pour une vue synthétique
        6. **Étudiez le potentiel d'amélioration** pour les actions
        """)

    # ---------------- STATISTIQUES DÉTAILLÉES ----------------
    st.markdown('<div class="section-header"><span>📋 Statistiques Détaillées</span></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Indicateurs de Performance")
        
        performance_metrics = [
            ("%GRR Total", f"{p_grr:.2f}%", p_grr),
            ("Ratio VP/GRR", f"{ratio_vp_grr:.2f}:1", ratio_vp_grr * 10),  # Normalisé pour l'affichage
            ("Répétabilité (EV)", f"{ev_percent:.1f}%", ev_percent),
            ("Reproductibilité (AV)", f"{av_percent:.1f}%", av_percent),
            ("Discrimination Pièces", f"{vp_percent:.1f}%", vp_percent)
        ]
        
        for label, value, raw_value in performance_metrics:
            color = "#27ae60" if raw_value <= 10 else "#f39c12" if raw_value <= 30 else "#c0392b"
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; 
                        border-left: 4px solid {color}; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 600; color: #2c3e50;">{label}</div>
                    <div style="font-weight: 700; color: {color}; font-size: 1.1rem;">{value}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📊 Données de l'Étude")
        
        study_data = [
            ("Nombre de pièces", str(n_pieces), "#3498db"),
            ("Nombre d'opérateurs", str(n_operateurs), "#2ecc71"),
            ("Nombre d'essais", str(n_essais), "#9b59b6"),
            ("Facteur k", f"{confidence_factor:.2f}", "#e74c3c"),
            ("Étendue moyenne (R̄)", f"{r_double_bar:.4f}", "#f39c12"),
            ("Variation totale (VT)", f"{vt:.4f}", "#34495e")
        ]
        
        for label, value, color in study_data:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; 
                        box-shadow: 0 2px 8px rgba(0,0,0,0.05); border-top: 3px solid {color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 600; color: #2c3e50;">{label}</div>
                    <div style="font-weight: 700; color: {color}; font-size: 1.1rem;">{value}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---------------- RECOMMANDATIONS ----------------
    st.markdown('<div class="section-header"><span>🎯 Recommandations Basées sur l\'Analyse</span></div>', unsafe_allow_html=True)
    
    if p_grr < 10:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #d4edda, #c3e6cb); 
                    padding: 2rem; border-radius: 15px; border-left: 6px solid #28a745;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #155724; margin-bottom: 1rem;">
                ✅ SYSTÈME OPTIMAL - MAINTENIR L'EXCELLENCE
            </div>
            <div style="color: #155724;">
                <p><strong>Actions recommandées :</strong></p>
                <ul style="margin-left: 1.5rem;">
                    <li>Continuer les procédures actuelles</li>
                    <li>Maintenir le programme d'étalonnage</li>
                    <li>Documenter les bonnes pratiques</li>
                    <li>Surveiller régulièrement les performances</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif p_grr <= 30:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fff3cd, #ffeaa7); 
                    padding: 2rem; border-radius: 15px; border-left: 6px solid #ffc107;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #856404; margin-bottom: 1rem;">
                ⚠ SYSTÈME ACCEPTABLE - AMÉLIORATIONS RECOMMANDÉES
            </div>
            <div style="color: #856404;">
                <p><strong>Source principale :</strong> {"Répétabilité (EV)" if ev_percent > av_percent else "Reproductibilité (AV)"}</p>
                <p><strong>Actions prioritaires :</strong></p>
                <ul style="margin-left: 1.5rem;">
                    <li>{"Vérifier l'étalonnage des instruments" if ev_percent > av_percent else "Harmoniser les méthodes entre opérateurs"}</li>
                    <li>{"Standardiser les procédures de mesure" if ev_percent > av_percent else "Organiser une formation commune"}</li>
                    <li>{"Contrôler les conditions environnementales" if ev_percent > av_percent else "Créer des aides visuelles"}</li>
                    <li>Planifier une réévaluation dans 3 mois</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f8d7da, #f5c6cb); 
                    padding: 2rem; border-radius: 15px; border-left: 6px solid #dc3545;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #721c24; margin-bottom: 1rem;">
                ❌ SYSTÈME INACCEPTABLE - ACTIONS CORRECTIVES REQUISES
            </div>
            <div style="color: #721c24;">
                <p><strong>Urgence :</strong> Suspension recommandée pour les mesures critiques</p>
                <p><strong>Actions immédiates :</strong></p>
                <ul style="margin-left: 1.5rem;">
                    <li>{"Réétalonner tous les instruments de mesure" if ev_percent > av_percent else "Former/reformer tous les opérateurs"}</li>
                    <li>{"Revoir les procédures de mesure" if ev_percent > av_percent else "Implémenter des gabarits de mesure"}</li>
                    <li>{"Contrôler la stabilité environnementale" if ev_percent > av_percent else "Organiser des audits croisés"}</li>
                    <li>Refaire l'étude après corrections</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Pied de page
st.markdown("""
<div style="margin-top: 4rem; padding: 2rem; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); 
            border-radius: 20px; text-align: center; border-top: 1px solid #e0e6ed;">
    <div style="font-size: 0.9rem; color: #7f8c8d;">
        <div style="display: flex; justify-content: center; align-items: center; gap: 10px; margin-bottom: 0.5rem;">
            <div>📊</div>
            <div><strong>Gage R&R - Visualisations Avancées</strong></div>
            <div>⚡</div>
        </div>
        <div>Analyse complète avec graphiques professionnels • Conforme aux normes AIAG MSA</div>
        <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.7;">
            Outil d'analyse statistique • Développé avec Streamlit, Matplotlib et NumPy
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
