import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime

st.set_page_config(
    page_title="Gage R&R Analytics Pro",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS modernisé avec design système
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --success-gradient: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        --warning-gradient: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        --danger-gradient: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        --dark-bg: #0f172a;
        --card-bg: rgba(255, 255, 255, 0.95);
        --shadow-lg: 0 20px 60px rgba(0, 0, 0, 0.1);
        --shadow-sm: 0 4px 20px rgba(0, 0, 0, 0.05);
        --border-radius: 20px;
        --transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 3rem 2rem;
        border-radius: var(--border-radius);
        margin-bottom: 3rem;
        text-align: center;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: slideUp 0.8s ease-out;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--primary-gradient);
    }
    
    .main-title {
        color: white;
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 1rem;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-subtitle {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.2rem;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: var(--border-radius);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card:hover {
        transform: translateY(-8px);
        box-shadow: var(--shadow-lg);
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: var(--primary-gradient);
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.6s ease;
    }
    
    .glass-card:hover::before {
        transform: scaleX(1);
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2c3e50, #4a5568);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .status-indicator {
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        text-align: center;
        font-weight: 700;
        font-size: 1.2rem;
        backdrop-filter: blur(20px);
        transition: var(--transition);
        border: 2px solid transparent;
        position: relative;
        overflow: hidden;
    }
    
    .status-excellent {
        background: linear-gradient(135deg, rgba(0, 176, 155, 0.15), rgba(150, 201, 61, 0.25));
        color: #00b09b;
        border-color: rgba(0, 176, 155, 0.3);
    }
    
    .status-acceptable {
        background: linear-gradient(135deg, rgba(247, 151, 30, 0.15), rgba(255, 210, 0, 0.25));
        color: #f7971e;
        border-color: rgba(247, 151, 30, 0.3);
    }
    
    .status-unacceptable {
        background: linear-gradient(135deg, rgba(255, 65, 108, 0.15), rgba(255, 75, 43, 0.25));
        color: #ff416c;
        border-color: rgba(255, 65, 108, 0.3);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid;
        border-image: var(--primary-gradient) 1;
    }
    
    .upload-zone {
        border: 3px dashed;
        border-image: var(--primary-gradient) 1;
        border-radius: var(--border-radius);
        padding: 4rem 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.03);
        transition: var(--transition);
        cursor: pointer;
        margin: 2rem 0;
    }
    
    .upload-zone:hover {
        background: rgba(102, 126, 234, 0.08);
        transform: scale(1.01);
    }
    
    .download-button {
        background: var(--primary-gradient);
        color: white;
        padding: 1rem 2.5rem;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: var(--transition);
        display: inline-flex;
        align-items: center;
        gap: 12px;
        text-decoration: none;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    .download-button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar-content {
        padding: 2rem;
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        height: 100%;
    }
    
    .badge {
        background: var(--primary-gradient);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .progress-track {
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 4px;
        background: var(--primary-gradient);
        transition: width 1s ease-out;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    
    .stPlotlyChart {
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.2rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header principal avec design premium
st.markdown("""
<div class="main-header">
    <div style="margin-bottom: 1.5rem;">
        <div style="font-size: 5rem; margin-bottom: -1.5rem;">📐</div>
    </div>
    <div class="main-title">Gage R&R Analytics Pro</div>
    <div class="main-subtitle">
        Analyse avancée de la capacité des systèmes de mesure • Méthode des étendues
    </div>
</div>
""", unsafe_allow_html=True)

# Fonction d2 optimisée
def get_d2(z, w):
    d2_table = {
        (1, 3): 1.91,
        (1, 10): 3.18,
        (15, 3): 1.693,
        (10, 3): 1.72,
        (5, 3): 1.74
    }
    return d2_table.get((z, w), 1.0)

# ------------------------- SIDEBAR PREMIUM -------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚙️</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b;">Configuration</div>
        <div style="color: #64748b; font-size: 0.9rem;">Paramètres d'analyse</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Paramètres avec design amélioré
    confidence_factor = st.slider(
        "**Facteur de Confiance (k)**",
        min_value=4.0,
        max_value=6.0,
        value=5.15,
        step=0.05,
        help="Facteur k pour le niveau de confiance des calculs"
    )
    
    st.markdown("---")
    
    # Guide interactif
    st.markdown("""
    <div style="margin: 1.5rem 0;">
        <div style="font-size: 1.2rem; font-weight: 600; color: #1e293b; margin-bottom: 1rem;">🎯 Critères d'acceptation</div>
        <div style="display: grid; gap: 0.8rem;">
            <div style="display: flex; align-items: center; gap: 10px; padding: 0.8rem; background: rgba(0, 176, 155, 0.1); border-radius: 10px; border-left: 4px solid #00b09b;">
                <div style="font-weight: 600; color: #00b09b;">✓ Excellent</div>
                <div style="color: #64748b; font-size: 0.9rem; margin-left: auto;">&lt; 10%</div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; padding: 0.8rem; background: rgba(247, 151, 30, 0.1); border-radius: 10px; border-left: 4px solid #f7971e;">
                <div style="font-weight: 600; color: #f7971e;">⚠ Acceptable</div>
                <div style="color: #64748b; font-size: 0.9rem; margin-left: auto;">10-30%</div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; padding: 0.8rem; background: rgba(255, 65, 108, 0.1); border-radius: 10px; border-left: 4px solid #ff416c;">
                <div style="font-weight: 600; color: #ff416c;">✗ Inacceptable</div>
                <div style="color: #64748b; font-size: 0.9rem; margin-left: auto;">&gt; 30%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Statistiques rapides
    st.markdown("""
    <div style="margin-top: 1.5rem;">
        <div style="font-size: 1.1rem; font-weight: 600; color: #1e293b; margin-bottom: 1rem;">📊 Indicateurs clés</div>
        <div style="display: grid; gap: 0.8rem;">
            <div style="display: flex; justify-content: space-between; padding: 0.6rem; background: white; border-radius: 8px;">
                <span style="color: #64748b;">EV</span>
                <span style="font-weight: 600; color: #3498db;">Répétabilité</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 0.6rem; background: white; border-radius: 8px;">
                <span style="color: #64748b;">AV</span>
                <span style="font-weight: 600; color: #2ecc71;">Reproductibilité</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 0.6rem; background: white; border-radius: 8px;">
                <span style="color: #64748b;">%GRR</span>
                <span style="font-weight: 600; color: #e74c3c;">Score final</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------- ZONE D'UPLOAD -------------------------
st.markdown('<div class="section-title"><span>📥 Importation des données</span></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["xlsx", "csv", "xls"],
    help="Importez votre fichier de données au format Excel ou CSV",
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown("""
    <div class="upload-zone">
        <div style="font-size: 4rem; margin-bottom: 1.5rem; color: #667eea;">📁</div>
        <div style="font-size: 1.6rem; font-weight: 700; color: #1e293b; margin-bottom: 0.5rem;">
            Glissez-déposez votre fichier
        </div>
        <div style="color: #64748b; margin-bottom: 2rem; font-size: 1.1rem;">
            Formats supportés : .xlsx, .csv, .xls
        </div>
        <div style="display: inline-flex; align-items: center; gap: 8px; padding: 0.8rem 1.5rem; background: rgba(102, 126, 234, 0.1); border-radius: 10px;">
            <div style="font-weight: 600; color: #667eea;">Format recommandé</div>
        </div>
        <div style="margin-top: 2rem; text-align: left; max-width: 600px; margin-left: auto; margin-right: auto;">
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">👥</div>
                    <div style="font-weight: 600; color: #1e293b;">3 Opérateurs</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">🔄</div>
                    <div style="font-weight: 600; color: #1e293b;">3 Essais</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">⚙️</div>
                    <div style="font-weight: 600; color: #1e293b;">10+ Pièces</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file:
    # Animation de chargement
    with st.spinner('🔄 Traitement des données en cours...'):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    
    # ------------------------- APERÇU DES DONNÉES -------------------------
    st.markdown('<div class="section-title"><span>📄 Aperçu des données</span></div>', unsafe_allow_html=True)
    
    with st.expander("Explorer les données", expanded=True):
        # Metrics cards pour statistiques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">📊 Dimensions</div>
                <div class="metric-value">{df.shape[0]}×{df.shape[1]}</div>
                <div style="color: #64748b; font-size: 0.9rem;">Lignes × Colonnes</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">🎯 Valeurs</div>
                <div class="metric-value">{df.size}</div>
                <div style="color: #64748b; font-size: 0.9rem;">Mesures totales</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">📈 Gamme</div>
                <div class="metric-value">{df.values.flatten().std():.3f}</div>
                <div style="color: #64748b; font-size: 0.9rem;">Écart-type</div>
            </div>
            """, unsafe_allow_html=True)
        
        # DataFrame stylisé
        st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
        
        # Création d'un DataFrame avec mise en forme conditionnelle
        def highlight_extremes(val):
            if isinstance(val, (int, float)):
                mean_val = df.values.mean()
                std_val = df.values.std()
                if abs(val - mean_val) > 1.5 * std_val:
                    return 'background: linear-gradient(90deg, rgba(255, 65, 108, 0.1), rgba(255, 75, 43, 0.2)); color: #ff416c; font-weight: 600;'
                elif val > mean_val:
                    return 'background: linear-gradient(90deg, rgba(0, 176, 155, 0.1), rgba(150, 201, 61, 0.2)); color: #00b09b;'
                else:
                    return 'background: linear-gradient(90deg, rgba(52, 152, 219, 0.1), rgba(41, 128, 185, 0.2)); color: #3498db;'
            return ''
        
        styled_df = df.style.applymap(highlight_extremes)
        st.dataframe(styled_df, use_container_width=True, height=400)

    # ------------------------- CONFIGURATION DES COLONNES -------------------------
    if 'OP1-1' not in df.columns:
        st.warning("⚠️ Les colonnes ne suivent pas le format standard. Veuillez sélectionner manuellement.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            op1_cols = st.multiselect("Opérateur 1", df.columns, default=df.columns[:3])
        with col2:
            op2_cols = st.multiselect("Opérateur 2", df.columns, default=df.columns[3:6])
        with col3:
            op3_cols = st.multiselect("Opérateur 3", df.columns, default=df.columns[6:9])
    else:
        op1_cols = ["OP1-1", "OP1-2", "OP1-3"]
        op2_cols = ["OP2-1", "OP2-2", "OP2-3"]
        op3_cols = ["OP3-1", "OP3-2", "OP3-3"]

    n_pieces = df.shape[0]
    n_operateurs = 3
    n_essais = 3

    # ------------------------- CALCULS -------------------------
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

    # ------------------------- VISUALISATIONS PLOTLY -------------------------
    st.markdown('<div class="section-title"><span>📈 Tableau de bord analytique</span></div>', unsafe_allow_html=True)
    
    # Layout avec onglets
    tab1, tab2, tab3 = st.tabs(["📊 Vue d'ensemble", "📈 Graphiques avancés", "📋 Analyses détaillées"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique en radar (amélioré)
            fig_radar = go.Figure()
            
            categories = ['Répétabilité', 'Reproductibilité', 'Consistance', 'Précision']
            values = [
                (1 - ev/vt) * 100 if vt > 0 else 0,
                (1 - av/vt) * 100 if vt > 0 else 0,
                (1 - grr/vt) * 100 if vt > 0 else 0,
                100 - p_grr
            ]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values + values[:1],
                theta=categories + categories[:1],
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.3)',
                line=dict(color='#667eea', width=2),
                name='Performance'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        tickfont=dict(size=10),
                        gridcolor='rgba(0,0,0,0.1)'
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=11),
                        rotation=90
                    ),
                    bgcolor='rgba(255,255,255,0.1)'
                ),
                showlegend=False,
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title=dict(
                    text="🎯 Performance du système",
                    font=dict(size=16, color='#1e293b')
                )
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            # Jauge %GRR interactive
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=p_grr,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "% Gage R&R", 'font': {'size': 20}},
                delta={'reference': 30, 'increasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 10], 'color': 'rgba(0, 176, 155, 0.3)'},
                        {'range': [10, 30], 'color': 'rgba(247, 151, 30, 0.3)'},
                        {'range': [30, 100], 'color': 'rgba(255, 65, 108, 0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 30
                    }
                }
            ))
            
            fig_gauge.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "#1e293b"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot par opérateur
            data_op1 = df[op1_cols].values.flatten()
            data_op2 = df[op2_cols].values.flatten()
            data_op3 = df[op3_cols].values.flatten()
            
            fig_box = go.Figure()
            
            fig_box.add_trace(go.Box(
                y=data_op1,
                name='Opérateur 1',
                marker_color='#3498db',
                boxmean='sd'
            ))
            
            fig_box.add_trace(go.Box(
                y=data_op2,
                name='Opérateur 2',
                marker_color='#2ecc71',
                boxmean='sd'
            ))
            
            fig_box.add_trace(go.Box(
                y=data_op3,
                name='Opérateur 3',
                marker_color='#9b59b6',
                boxmean='sd'
            ))
            
            fig_box.update_layout(
                title="📦 Distribution par opérateur",
                yaxis_title="Valeurs mesurées",
                showlegend=True,
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Graphique des composantes
            labels = ['Répétabilité (EV)', 'Reproductibilité (AV)', 'Variation pièces (PV)']
            values = [ev**2, av**2, vp**2]
            colors = ['#3498db', '#2ecc71', '#e74c3c']
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.5,
                marker_colors=colors,
                textinfo='label+percent',
                textposition='outside',
                textfont=dict(size=12)
            )])
            
            fig_pie.update_layout(
                title="🥧 Répartition des variations",
                showlegend=False,
                height=400,
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab3:
        # Heatmap des corrélations
        all_data = pd.concat([df[op1_cols], df[op2_cols], df[op3_cols]], axis=1)
        correlation_matrix = all_data.corr()
        
        fig_heatmap = px.imshow(
            correlation_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="🔥 Heatmap des corrélations"
        )
        
        fig_heatmap.update_layout(
            height=500,
            xaxis_title="Mesures",
            yaxis_title="Mesures"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # ------------------------- RÉSULTATS PRINCIPAUX -------------------------
    st.markdown('<div class="section-title"><span>📊 Résultats synthétiques</span></div>', unsafe_allow_html=True)
    
    # Métriques principales avec design premium
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("EV", ev, "#3498db", "Répétabilité", "repeatability"),
        ("AV", av, "#2ecc71", "Reproductibilité", "reproducibility"),
        ("GRR", grr, "#9b59b6", "Variation système", "system_variation"),
        ("%GRR", p_grr, "#ff416c", "Score final", "final_score")
    ]
    
    for col, (label, value, color, desc, icon) in zip([col1, col2, col3, col4], metrics):
        with col:
            percentage = (value/vt)*100 if label != "%GRR" else value
            
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label" style="color: {color};">{icon} {desc}</div>
                <div class="metric-value" style="background: linear-gradient(135deg, {color}, #2c3e50); -webkit-background-clip: text;">
                    {value:.4f}{'%' if label == '%GRR' else ''}
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="width: {min(percentage, 100)}%; 
                    background: linear-gradient(90deg, {color}, {color}88);"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #64748b;">
                    <span>{label}</span>
                    <span>{percentage:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Indicateur de statut avec animation
    if p_grr < 10:
        status_class = "status-excellent"
        status_icon = "✨"
        status_text = "EXCELLENT"
        status_desc = "Le système de mesure est optimal"
        st.balloons()
    elif p_grr <= 30:
        status_class = "status-acceptable"
        status_icon = "✅"
        status_text = "ACCEPTABLE"
        status_desc = "Améliorations possibles recommandées"
    else:
        status_class = "status-unacceptable"
        status_icon = "⚠️"
        status_text = "INACCEPTABLE"
        status_desc = "Action corrective immédiate requise"
    
    st.markdown(f"""
    <div class="status-indicator {status_class}">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{status_icon} {status_text}</div>
        <div style="font-size: 1rem; opacity: 0.9;">{status_desc}</div>
        <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.7;">
            Score : {p_grr:.2f}% • Seuil : {"10%" if p_grr < 10 else "30%" if p_grr <= 30 else "30%"}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ------------------------- ANALYSE DÉTAILLÉE -------------------------
    st.markdown('<div class="section-title"><span>📋 Analyse détaillée</span></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Tableau détaillé des résultats
        results_data = {
            "Paramètre": ["EV", "AV", "GRR", "PV", "TV", "%GRR", "%EV", "%AV", "nDC"],
            "Valeur": [
                f"{ev:.6f}", f"{av:.6f}", f"{grr:.6f}", 
                f"{vp:.6f}", f"{vt:.6f}", f"{p_grr:.2f}%",
                f"{(ev/vt*100):.2f}%", f"{(av/vt*100):.2f}%",
                f"{(vp/grr):.2f}"
            ],
            "Statut": [
                "✅" if (ev/vt*100) < 30 else "⚠️",
                "✅" if (av/vt*100) < 30 else "⚠️",
                "✅" if p_grr < 10 else "⚠️" if p_grr <= 30 else "❌",
                "-", "-",
                "✅" if p_grr < 10 else "⚠️" if p_grr <= 30 else "❌",
                "✅" if (ev/vt*100) < 30 else "⚠️",
                "✅" if (av/vt*100) < 30 else "⚠️",
                "✅" if (vp/grr) >= 4 else "⚠️"
            ]
        }
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(
            results_df.style
            .applymap(lambda x: 'color: #00b09b; font-weight: bold' if x == "✅" else 
                     ('color: #f7971e; font-weight: bold' if x == "⚠️" else 
                     ('color: #ff416c; font-weight: bold' if x == "❌" else '')),
                     subset=['Statut'])
            .background_gradient(subset=['Valeur'], cmap='YlOrRd'),
            use_container_width=True
        )
    
    with col2:
        # Recommandations
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea15, #764ba215); 
                    padding: 1.5rem; border-radius: 16px; margin-top: 1rem;">
            <div style="font-size: 1.2rem; font-weight: 600; color: #1e293b; margin-bottom: 1rem;">
                💡 Recommandations
            </div>
        """, unsafe_allow_html=True)
        
        if p_grr < 10:
            st.success("**Système optimal** - Aucune action requise")
        elif p_grr < 30:
            st.warning("**Améliorations recommandées** :")
            st.markdown("""
            - Vérifier la procédure de mesure
            - Former les opérateurs
            - Calibrer l'équipement
            """)
        else:
            st.error("**Actions correctives requises** :")
            st.markdown("""
            - Revoir le système de mesure
            - Formation intensive
            - Nouvelle étude après corrections
            """)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Info-bulles techniques
        st.markdown("""
        <div style="margin-top: 2rem; padding: 1rem; background: #f8fafc; border-radius: 12px;">
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">📈 Paramètres calculés</div>
            <div style="font-size: 0.85rem; color: #64748b;">
                <div>• k = {:.2f} (facteur de confiance)</div>
                <div>• d₂ = {:.3f} (facteur statistique)</div>
                <div>• Échantillon : {} pièces</div>
                <div>• Opérateurs : {} × {} essais</div>
            </div>
        </div>
        """.format(confidence_factor, d2_ev, n_pieces, n_operateurs, n_essais), unsafe_allow_html=True)

    # ------------------------- EXPORT PROFESSIONNEL -------------------------
    st.markdown('<div class="section-title"><span>💾 Rapport d'analyse</span></div>', unsafe_allow_html=True)
    
    # Création du rapport Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Feuille principale
        summary_data = {
            "Date d'analyse": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Fichier source": [uploaded_file.name],
            "Nombre de pièces": [n_pieces],
            "Nombre d'opérateurs": [n_operateurs],
            "Nombre d'essais": [n_essais],
            "Facteur k": [confidence_factor],
            "% Gage R&R": [f"{p_grr:.2f}%"],
            "Statut": [status_text],
            "Répétabilité (EV)": [f"{ev:.6f}"],
            "Reproductibilité (AV)": [f"{av:.6f}"],
            "Variation système (GRR)": [f"{grr:.6f}"],
            "Variation pièces (PV)": [f"{vp:.6f}"],
            "Variation totale (TV)": [f"{vt:.6f}"]
        }
        
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Résumé', index=False)
        
        # Données brutes
        df.to_excel(writer, sheet_name='Données brutes', index=False)
        
        # Calculs intermédiaires
        calc_data = pd.DataFrame({
            "Opérateur": ["OP1", "OP2", "OP3"],
            "Moyenne": [x_bar_op1, x_bar_op2, x_bar_op3],
            "Étendue moyenne": [r_bar_op1, r_bar_op2, r_bar_op3]
        })
        calc_data.to_excel(writer, sheet_name='Calculs', index=False)
        
        # Recommandations
        recommendations = pd.DataFrame({
            "Priorité": ["Haute" if p_grr > 30 else "Moyenne" if p_grr > 10 else "Basse"],
            "Action": [status_desc],
            "Date de suivi": [(datetime.now() + pd.Timedelta(days=30)).strftime("%Y-%m-%d")]
        })
        recommendations.to_excel(writer, sheet_name='Plan d\'action', index=False)
    
    output.seek(0)
    
    # Bouton de téléchargement élégant
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.markdown(f"""
        <div style="text-align: center; margin: 2rem 0;">
            <a href='data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{output.getvalue().hex()}' 
               download='gage_rr_rapport_{timestamp}.xlsx'
               class='download-button'>
               📥 Télécharger le rapport complet
            </a>
            <div style="color: #64748b; font-size: 0.9rem; margin-top: 1rem;">
                Inclut : Résumé • Données brutes • Calculs • Plan d'action
            </div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------- PIED DE PAGE PROFESSIONNEL -------------------------
st.markdown("""
<div style="margin-top: 4rem; padding: 3rem 2rem; 
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border-radius: var(--border-radius); 
            text-align: center; 
            color: white;">
    
    <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 2rem;">
        <div style="text-align: center;">
            <div style="font-size: 2.5rem;">📐</div>
            <div style="font-weight: 600; font-size: 1.1rem;">Précision</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 2.5rem;">⚡</div>
            <div style="font-weight: 600; font-size: 1.1rem;">Performance</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 2.5rem;">🔬</div>
            <div style="font-weight: 600; font-size: 1.1rem;">Analyse</div>
        </div>
    </div>
    
    <div style="font-size: 1.8rem; font-weight: 700; margin-bottom: 1rem; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;">
        Gage R&R Analytics Pro
    </div>
    
    <div style="color: rgba(255, 255, 255, 0.7); max-width: 600px; margin: 0 auto; line-height: 1.6;">
        Outil d'analyse avancée pour l'évaluation des systèmes de mesure selon les normes industrielles
    </div>
    
    <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.1);">
        <div style="display: flex; justify-content: center; gap: 2rem; color: rgba(255, 255, 255, 0.5); font-size: 0.9rem;">
            <div>© 2024 Quality Analytics Suite</div>
            <div>•</div>
            <div>Version 2.0.0</div>
            <div>•</div>
            <div>AI-Powered Insights</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
