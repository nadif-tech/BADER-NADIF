import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import time
from scipy import stats

st.set_page_config(
    page_title="Gage R&R - Étendues | Dashboard Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé amélioré avec design system moderne
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&display=swap');
    
    :root {
        --primary: #4361ee;
        --primary-dark: #3a56d4;
        --secondary: #7209b7;
        --success: #4cc9f0;
        --warning: #f8961e;
        --danger: #f72585;
        --dark: #1a1a2e;
        --light: #f8f9fa;
        --gray: #6c757d;
        --border-radius: 16px;
        --shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 2.5rem;
        border-radius: var(--border-radius);
        margin-bottom: 2.5rem;
        text-align: center;
        box-shadow: var(--shadow);
        animation: slideDown 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #4cc9f0, #f72585, #4361ee);
        animation: shimmer 3s infinite;
    }
    
    .main-title {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .main-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        font-weight: 400;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    .metric-card-pro {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        border-radius: var(--border-radius);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.8);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .metric-card-pro:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(67, 97, 238, 0.15);
    }
    
    .metric-card-pro::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        transform-origin: left;
        transition: transform 0.6s ease;
    }
    
    .metric-value-pro {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -1px;
    }
    
    .section-header-pro {
        background: linear-gradient(90deg, rgba(67, 97, 238, 0.1), rgba(114, 9, 183, 0.1));
        color: var(--dark);
        padding: 1.2rem 2rem;
        border-radius: var(--border-radius);
        margin: 2.5rem 0 1.5rem 0;
        font-weight: 700;
        font-size: 1.4rem;
        border-left: 5px solid var(--primary);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .plot-container-pro {
        background: white;
        padding: 2rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        margin: 2rem 0;
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: var(--transition);
    }
    
    .plot-container-pro:hover {
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.1);
    }
    
    .status-indicator {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.2rem;
        transition: var(--transition);
    }
    
    .status-excellent {
        background: linear-gradient(135deg, #4cc9f0, #4895ef);
        color: white;
        box-shadow: 0 4px 15px rgba(76, 201, 240, 0.3);
    }
    
    .status-acceptable {
        background: linear-gradient(135deg, #f8961e, #f9c74f);
        color: white;
        box-shadow: 0 4px 15px rgba(248, 150, 30, 0.3);
    }
    
    .status-unacceptable {
        background: linear-gradient(135deg, #f72585, #b5179e);
        color: white;
        box-shadow: 0 4px 15px rgba(247, 37, 133, 0.3);
    }
    
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .analytics-panel {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: white;
        padding: 2rem;
        border-radius: var(--border-radius);
        margin: 2rem 0;
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .floating-icon {
        animation: float 3s ease-in-out infinite;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4cc9f0, #f72585, #4361ee);
    }
    
    .data-table {
        background: white;
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: var(--shadow);
    }
    
    .data-table table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
    }
    
    .data-table th {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        padding: 1rem;
        font-weight: 600;
        text-align: left;
    }
    
    .data-table td {
        padding: 1rem;
        border-bottom: 1px solid #f1f3f4;
        transition: var(--transition);
    }
    
    .data-table tr:hover td {
        background: rgba(67, 97, 238, 0.05);
    }
    
    .glassmorphism {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header principal avec animation
st.markdown("""
<div class="main-header">
    <div class="main-title">📊 Gage R&R Analytics Pro</div>
    <div class="main-subtitle">Analyse avancée de la capacité du système de mesure avec visualisations interactives et rapports détaillés</div>
</div>
""", unsafe_allow_html=True)

# ---------------- d2 FUNCTION ----------------
def get_d2(z, w):
    d2_table = {
        (1, 3): 1.91,
        (1, 10): 3.18,
        (2, 3): 1.81,
        (2, 10): 2.52,
        (3, 3): 1.77,
        (3, 10): 2.26,
        (4, 3): 1.75,
        (4, 10): 2.09,
        (5, 3): 1.74,
        (5, 10): 1.96,
        (6, 3): 1.73,
        (6, 10): 1.87,
        (7, 3): 1.72,
        (7, 10): 1.81,
        (8, 3): 1.72,
        (8, 10): 1.77,
        (9, 3): 1.71,
        (9, 10): 1.74,
        (10, 3): 1.71,
        (10, 10): 1.72,
        (15, 3): 1.693,
        (15, 10): 1.67,
        (20, 3): 1.68,
        (20, 10): 1.64
    }
    return d2_table.get((z, w), 1.693)

# ---------------- SIDEBAR AVANCÉE ----------------
with st.sidebar:
    st.markdown('<div class="glassmorphism" style="padding: 2rem; border-radius: var(--border-radius);">', unsafe_allow_html=True)
    
    st.markdown('<div style="font-size: 1.8rem; font-weight: 800; color: var(--primary); margin-bottom: 2rem; display: flex; align-items: center; gap: 10px;">⚙️ <span>Configuration Pro</span></div>', unsafe_allow_html=True)
    
    # Paramètres avancés
    col1, col2 = st.columns(2)
    with col1:
        confidence_factor = st.number_input(
            "**Facteur K**",
            min_value=4.0,
            max_value=6.0,
            value=5.15,
            step=0.05,
            help="Facteur de confiance statistique"
        )
    
    with col2:
        tolerance = st.number_input(
            "**Tolérance**",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.1,
            help="Tolérance du processus"
        )
    
    st.markdown("---")
    
    # Options d'affichage
    st.markdown('<div style="font-size: 1.2rem; font-weight: 600; color: var(--dark); margin: 1.5rem 0 1rem 0;">🎨 Options Graphiques</div>', unsafe_allow_html=True)
    
    show_3d = st.checkbox("📊 Graphiques 3D", value=True)
    show_interactive = st.checkbox("🔄 Graphiques interactifs", value=True)
    theme = st.selectbox(
        "🎭 Thème",
        ["Light", "Dark", "Corporate"],
        index=0
    )
    
    st.markdown("---")
    
    # Métriques rapides
    st.markdown('<div style="font-size: 1.2rem; font-weight: 600; color: var(--dark); margin: 1.5rem 0 1rem 0;">📈 KPI Standards</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 1rem;">
        <div class="status-indicator status-excellent">&lt; 10%</div>
        <div style="color: var(--gray); font-size: 0.9rem;">Excellent</div>
        <div class="status-indicator status-acceptable">10-30%</div>
        <div style="color: var(--gray); font-size: 0.9rem;">Acceptable</div>
        <div class="status-indicator status-unacceptable">&gt; 30%</div>
        <div style="color: var(--gray); font-size: 0.9rem;">Inacceptable</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ZONE D'UPLOAD STYLÉE ----------------
st.markdown('<div class="section-header-pro">📥 Importation des Données</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["xlsx", "csv"],
    help="Téléversez votre fichier de données (Excel ou CSV)",
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown("""
    <div style="text-align: center; padding: 4rem; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); 
                border-radius: var(--border-radius); border: 2px dashed var(--primary); margin: 2rem 0;">
        <div class="floating-icon" style="font-size: 5rem; margin-bottom: 1rem;">📁</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: var(--dark); margin-bottom: 0.5rem;">
            Glissez-déposez votre fichier
        </div>
        <div style="color: var(--gray); margin-bottom: 2rem; max-width: 600px; margin: 0 auto;">
            Formats supportés : Excel (.xlsx) • CSV (.csv)
        </div>
        <div style="background: rgba(67, 97, 238, 0.1); padding: 1.5rem; border-radius: var(--border-radius); 
                    display: inline-block; text-align: left; max-width: 500px;">
            <div style="font-weight: 700; color: var(--primary); margin-bottom: 0.5rem;">📋 Structure recommandée :</div>
            <div style="color: var(--gray); font-size: 0.9rem; line-height: 1.6;">
                • 3 opérateurs × 3 essais chacun<br>
                • Colonnes : OP1-1, OP1-2, OP1-3, OP2-1, OP2-2, OP2-3, OP3-1, OP3-2, OP3-3<br>
                • 10-30 pièces recommandées<br>
                • Données numériques uniquement
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file:
    # Animation de chargement avec progression
    with st.spinner('🔄 Traitement des données en cours...'):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    
    # ---------------- APERÇU DES DONNÉES AVANCÉ ----------------
    st.markdown('<div class="section-header-pro">📊 Explorateur de Données</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📈 Pièces", df.shape[0])
    with col2:
        st.metric("👥 Opérateurs", 3)
    with col3:
        st.metric("🎯 Essais", 3)
    
    with st.expander("📋 Données Détailées", expanded=True):
        # Statistiques descriptives
        st.markdown("**📊 Statistiques Descriptives**")
        desc_stats = df.describe().T
        desc_stats['CV%'] = (desc_stats['std'] / desc_stats['mean'] * 100).round(2)
        st.dataframe(desc_stats.style.format("{:.4f}"), use_container_width=True)
        
        # Visualisation de la distribution des données
        st.markdown("**📈 Distribution des Mesures**")
        fig_dist = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Opérateur 1', 'Opérateur 2', 'Opérateur 3']
        )
        
        op_cols_groups = [["OP1-1", "OP1-2", "OP1-3"], 
                         ["OP2-1", "OP2-2", "OP2-3"], 
                         ["OP3-1", "OP3-2", "OP3-3"]]
        
        for i, op_cols in enumerate(op_cols_groups, 1):
            data = df[op_cols].values.flatten()
            fig_dist.add_trace(
                go.Violin(y=data, name=f'Op {i}', box_visible=True, 
                         line_color=f'rgba({i*80}, {150-i*40}, 238, 0.8)'),
                row=1, col=i
            )
        
        fig_dist.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

    # Colonnes opérateurs
    op1_cols = ["OP1-1", "OP1-2", "OP1-3"]
    op2_cols = ["OP2-1", "OP2-2", "OP2-3"]
    op3_cols = ["OP3-1", "OP3-2", "OP3-3"]

    n_pieces = df.shape[0]
    n_operateurs = 3
    n_essais = 3

    # ---------------- CALCULS AVANCÉS ----------------
    with st.spinner('🧮 Calculs statistiques en cours...'):
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
        
        # Calculs supplémentaires
        ndc = 1.41 * (vp / grr)  # Discrimination
        p_tv = (grr / tolerance) * 100 if tolerance > 0 else 0
        
        # Tests statistiques
        f_stat, p_value = stats.f_oneway(
            df[op1_cols].values.flatten(),
            df[op2_cols].values.flatten(),
            df[op3_cols].values.flatten()
        )

    # ---------------- DASHBOARD INTERACTIF ----------------
    st.markdown('<div class="section-header-pro">🎯 Dashboard Analytics</div>', unsafe_allow_html=True)
    
    # KPI Principaux
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card-pro">
            <div style="color: var(--gray); font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">% Gage R&R</div>
            <div class="metric-value-pro">{p_grr:.1f}%</div>
            <div style="margin-top: 1rem;">
                {"<span class='status-indicator status-excellent'>EXCELLENT</span>" if p_grr < 10 else 
                 "<span class='status-indicator status-acceptable'>ACCEPTABLE</span>" if p_grr <= 30 else 
                 "<span class='status-indicator status-unacceptable'>INACCEPTABLE</span>"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card-pro">
            <div style="color: var(--gray); font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Discrimination</div>
            <div class="metric-value-pro">{ndc:.1f}</div>
            <div style="color: var(--gray); font-size: 0.85rem; margin-top: 0.5rem;">
                {"✅ > 5" if ndc > 5 else "⚠️ ≤ 5"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card-pro">
            <div style="color: var(--gray); font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Répétabilité (EV)</div>
            <div class="metric-value-pro">{ev:.4f}</div>
            <div style="color: var(--gray); font-size: 0.85rem; margin-top: 0.5rem;">
                Contribution: {(ev/vt*100):.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card-pro">
            <div style="color: var(--gray); font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Reproductibilité (AV)</div>
            <div class="metric-value-pro">{av:.4f}</div>
            <div style="color: var(--gray); font-size: 0.85rem; margin-top: 0.5rem;">
                Contribution: {(av/vt*100):.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ---------------- VISUALISATIONS PROFESSIONNELLES ----------------
    st.markdown('<div class="section-header-pro">📈 Visualisations Avancées</div>', unsafe_allow_html=True)
    
    # Graphique 1: Sunburst Chart interactif
    st.markdown('<div class="plot-container-pro">', unsafe_allow_html=True)
    st.markdown("**🌳 Carte des Variations - Sunburst Chart**")
    
    labels = ["Variation Totale", "Variation Système", "Répétabilité (EV)", 
              "Reproductibilité (AV)", "Variation Pièces (VP)"]
    parents = ["", "Variation Totale", "Variation Système", "Variation Système", "Variation Totale"]
    values = [vt**2, grr**2, ev**2, av**2, vp**2]
    
    fig_sunburst = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        textinfo="label+percent entry",
        hoverinfo="label+value+percent parent",
        marker=dict(
            colors=['#1a1a2e', '#4361ee', '#4cc9f0', '#4895ef', '#7209b7'],
            line=dict(color='white', width=2)
        ),
        textfont=dict(size=14)
    ))
    
    fig_sunburst.update_layout(
        height=600,
        margin=dict(t=0, l=0, r=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_sunburst, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Graphiques 2 et 3: Ligne et Radar
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="plot-container-pro">', unsafe_allow_html=True)
        st.markdown("**📊 Performance par Pièce**")
        
        fig_line = go.Figure()
        
        for i, (op_cols, color, name) in enumerate(zip(
            [op1_cols, op2_cols, op3_cols],
            ['#4cc9f0', '#4895ef', '#4361ee'],
            ['Opérateur 1', 'Opérateur 2', 'Opérateur 3']
        ), 1):
            means_by_piece = df[op_cols].mean(axis=1)
            fig_line.add_trace(go.Scatter(
                x=list(range(1, len(means_by_piece) + 1)),
                y=means_by_piece,
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=3),
                marker=dict(size=8),
                hovertemplate=f'{name}<br>Pièce %{{x}}<br>Valeur: %{{y:.4f}}<extra></extra>'
            ))
        
        fig_line.update_layout(
            height=400,
            xaxis_title="Numéro de Pièce",
            yaxis_title="Valeur Mesurée",
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="plot-container-pro">', unsafe_allow_html=True)
        st.markdown("**🎯 Radar des Opérateurs**")
        
        categories = ['Précision', 'Répétabilité', 'Biais', 'Linéarité', 'Stabilité']
        
        fig_radar = go.Figure()
        
        op_stats = []
        for op_cols in [op1_cols, op2_cols, op3_cols]:
            data = df[op_cols].values.flatten()
            precision = 1 / np.std(data) * 100
            repeatability = 1 / (df[[f"R_OP{i+1}" for i in range(3)][op_cols == op1_cols]].mean()[0]) * 100
            bias = abs(np.mean(data) - df[op1_cols + op2_cols + op3_cols].values.flatten().mean())
            linearity = 1 / (np.polyfit(range(len(data)), data, 1)[0] + 0.001)
            stability = 1 / (np.std([data[:len(data)//3].mean(), 
                                   data[len(data)//3:2*len(data)//3].mean(), 
                                   data[2*len(data)//3:].mean()]) + 0.001)
            
            op_stats.append([precision, repeatability, bias, linearity, stability])
        
        # Normalisation
        op_stats_norm = []
        for stats in op_stats:
            max_val = max(stats)
            op_stats_norm.append([s/max_val*100 for s in stats])
        
        for i, (stats_norm, color, name) in enumerate(zip(
            op_stats_norm,
            ['rgba(76, 201, 240, 0.8)', 'rgba(72, 149, 239, 0.8)', 'rgba(67, 97, 238, 0.8)'],
            ['Opérateur 1', 'Opérateur 2', 'Opérateur 3']
        )):
            fig_radar.add_trace(go.Scatterpolar(
                r=stats_norm + [stats_norm[0]],
                theta=categories + [categories[0]],
                name=name,
                fill='toself',
                fillcolor=color.replace('0.8', '0.3'),
                line=dict(color=color, width=3)
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Graphique 4: Heatmap 3D
    if show_3d:
        st.markdown('<div class="plot-container-pro">', unsafe_allow_html=True)
        st.markdown("**🔥 Matrice de Corrélation 3D**")
        
        all_data = []
        for i in range(n_pieces):
            for j, op_cols in enumerate([op1_cols, op2_cols, op3_cols], 1):
                for k, col in enumerate(op_cols, 1):
                    all_data.append({
                        'Pièce': i + 1,
                        'Opérateur': j,
                        'Essai': k,
                        'Valeur': df.iloc[i][col]
                    })
        
        heatmap_df = pd.DataFrame(all_data)
        
        fig_3d = go.Figure(data=go.Volume(
            x=heatmap_df['Pièce'],
            y=heatmap_df['Opérateur'],
            z=heatmap_df['Essai'],
            value=heatmap_df['Valeur'],
            isomin=heatmap_df['Valeur'].min(),
            isomax=heatmap_df['Valeur'].max(),
            opacity=0.1,
            surface_count=20,
            colorscale='Viridis',
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))
        
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='Pièce',
                yaxis_title='Opérateur',
                zaxis_title='Essai',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=500,
            margin=dict(l=0, r=0, b=0, t=0)
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ---------------- ANALYTICS DÉTAILLÉS ----------------
    st.markdown('<div class="section-header-pro">🔍 Analyse Statistique Approfondie</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📋 Résultats Complets", "📊 ANOVA", "🎯 Recommandations"])
    
    with tab1:
        results_data = {
            'Paramètre': ['EV (Répétabilité)', 'AV (Reproductibilité)', 'GRR (Système)', 
                         'VP (Pièces)', 'VT (Totale)', '%GRR', 'Ndc', '%Tolérance'],
            'Valeur': [f'{ev:.6f}', f'{av:.6f}', f'{grr:.6f}', 
                      f'{vp:.6f}', f'{vt:.6f}', f'{p_grr:.2f}%', 
                      f'{ndc:.2f}', f'{p_tv:.2f}%'],
            'Contribution': [f'{ev/vt*100:.1f}%', f'{av/vt*100:.1f}%', f'{grr/vt*100:.1f}%',
                           f'{vp/vt*100:.1f}%', '100%', '-', '-', '-'],
            'Statut': [
                '✅' if ev/vt*100 < 30 else '⚠️',
                '✅' if av/vt*100 < 30 else '⚠️',
                '✅' if p_grr < 10 else '⚠️' if p_grr <= 30 else '❌',
                '📊',
                '📊',
                '✅' if p_grr < 10 else '⚠️' if p_grr <= 30 else '❌',
                '✅' if ndc > 5 else '⚠️',
                '✅' if p_tv < 10 else '⚠️' if p_tv <= 30 else '❌'
            ]
        }
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📊 Test d'ANOVA**")
            st.write(f"**F-statistique:** {f_stat:.4f}")
            st.write(f"**P-value:** {p_value:.6f}")
            st.write(f"**Significatif à 5%:** {'✅ Oui' if p_value < 0.05 else '❌ Non'}")
            
            # Diagramme de Pareto des variations
            fig_pareto = go.Figure()
            
            sources = ['EV', 'AV', 'VP']
            values = [ev**2, av**2, vp**2]
            cum_sum = np.cumsum(values)
            
            fig_pareto.add_trace(go.Bar(
                x=sources,
                y=values,
                name='Variation',
                marker_color=['#4cc9f0', '#4895ef', '#4361ee']
            ))
            
            fig_pareto.add_trace(go.Scatter(
                x=sources,
                y=cum_sum,
                name='Cumulé',
                yaxis='y2',
                marker_color='#f72585',
                mode='lines+markers'
            ))
            
            fig_pareto.update_layout(
                height=300,
                yaxis=dict(title='Variation'),
                yaxis2=dict(title='Cumulé (%)', overlaying='y', side='right',
                           range=[0, max(cum_sum)*1.1]),
                showlegend=True
            )
            
            st.plotly_chart(fig_pareto, use_container_width=True)
        
        with col2:
            st.markdown("**📈 Contrôle Statistique**")
            
            # Carte de contrôle X-bar R
            fig_control = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Carte X-bar', 'Carte R'),
                vertical_spacing=0.15
            )
            
            # X-bar chart
            x_bar = df[op1_cols + op2_cols + op3_cols].mean(axis=1)
            x_bar_mean = x_bar.mean()
            x_bar_ucl = x_bar_mean + 3 * x_bar.std() / np.sqrt(n_essais)
            x_bar_lcl = x_bar_mean - 3 * x_bar.std() / np.sqrt(n_essais)
            
            fig_control.add_trace(go.Scatter(
                y=x_bar,
                mode='lines+markers',
                name='Moyenne',
                line=dict(color='#4361ee', width=2)
            ), row=1, col=1)
            
            fig_control.add_hline(y=x_bar_mean, line_dash="dash", 
                                 line_color="green", row=1, col=1)
            fig_control.add_hline(y=x_bar_ucl, line_dash="dot", 
                                 line_color="red", row=1, col=1)
            fig_control.add_hline(y=x_bar_lcl, line_dash="dot", 
                                 line_color="red", row=1, col=1)
            
            # R chart
            ranges = df[["R_OP1", "R_OP2", "R_OP3"]].mean(axis=1)
            r_mean = ranges.mean()
            r_ucl = r_mean * 2.574  # D4 pour n=3
            
            fig_control.add_trace(go.Scatter(
                y=ranges,
                mode='lines+markers',
                name='Étendue',
                line=dict(color='#f72585', width=2)
            ), row=2, col=1)
            
            fig_control.add_hline(y=r_mean, line_dash="dash", 
                                 line_color="green", row=2, col=1)
            fig_control.add_hline(y=r_ucl, line_dash="dot", 
                                 line_color="red", row=2, col=1)
            
            fig_control.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_control, use_container_width=True)
    
    with tab3:
        if p_grr < 10:
            st.success("""
            ## 🎉 **EXCELLENT - Système Optimal**
            
            ### ✅ **Actions recommandées :**
            - Maintenir les procédures actuelles
            - Documenter les bonnes pratiques
            - Effectuer un suivi périodique
            
            ### 📊 **Statistiques favorables :**
            - Le système discrimine bien les pièces (ndc > 5)
            - Variation système sous contrôle
            - Processus mesurable avec confiance
            """)
        elif p_grr <= 30:
            st.warning("""
            ## ⚠️ **ACCEPTABLE - Améliorations Possibles**
            
            ### 🔧 **Actions recommandées :**
            1. **Formation des opérateurs**
               - Standardiser les méthodes de mesure
               - Vérifier la compréhension des procédures
            2. **Amélioration de l'équipement**
               - Calibration plus fréquente
               - Maintenance préventive
            3. **Optimisation du processus**
               - Améliorer les fixations
               - Standardiser les conditions de mesure
            """)
        else:
            st.error("""
            ## ❌ **INACCEPTABLE - Action Corrective Requise**
            
            ### 🚨 **Actions prioritaires :**
            1. **Équipement**
               - Recalibrer l'équipement
               - Vérifier l'usure
               - Considérer un remplacement
            2. **Méthodes**
               - Redéfinir les procédures de mesure
               - Améliorer les fixations
               - Standardiser les conditions
            3. **Personnel**
               - Formation intensive
               - Certification des opérateurs
               - Supervision renforcée
            4. **Analyse approfondie**
               - Identifier la source principale de variation
               - Mener une étude complémentaire
               - Établir un plan d'action détaillé
            """)
        
        # Matrice de décision
        st.markdown("### 📋 Matrice de Décision")
        
        decision_data = {
            'Critère': ['%GRR', 'Discrimination (ndc)', '% Tolérance', 'Statistiques'],
            'Valeur': [f'{p_grr:.1f}%', f'{ndc:.1f}', f'{p_tv:.1f}%', 
                      'ANOVA ' + ('✅' if p_value < 0.05 else '❌')],
            'Seuil': ['< 10%', '> 5', '< 30%', 'p < 0.05'],
            'Statut': [
                '✅' if p_grr < 10 else '⚠️' if p_grr <= 30 else '❌',
                '✅' if ndc > 5 else '❌',
                '✅' if p_tv < 30 else '❌',
                '✅' if p_value < 0.05 else '❌'
            ]
        }
        
        st.dataframe(pd.DataFrame(decision_data), use_container_width=True)
    
    # ---------------- EXPORT PROFESSIONNEL ----------------
    st.markdown('<div class="section-header-pro">💾 Rapport Complet</div>', unsafe_allow_html=True)
    
    # Création du rapport
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Feuille de résultats
        results_export = pd.DataFrame({
            'Paramètre': ['EV', 'AV', 'GRR', 'VP', 'VT', '%GRR', 'Ndc', '%Tolérance'],
            'Valeur': [ev, av, grr, vp, vt, p_grr, ndc, p_tv],
            'Unité': ['unité', 'unité', 'unité', 'unité', 'unité', '%', 'sans', '%'],
            'Statut': [
                'Excellent' if ev/vt*100 < 10 else 'Acceptable' if ev/vt*100 < 30 else 'Inacceptable',
                'Excellent' if av/vt*100 < 10 else 'Acceptable' if av/vt*100 < 30 else 'Inacceptable',
                'Excellent' if p_grr < 10 else 'Acceptable' if p_grr <= 30 else 'Inacceptable',
                '-', '-',
                'Excellent' if p_grr < 10 else 'Acceptable' if p_grr <= 30 else 'Inacceptable',
                'Suffisant' if ndc > 5 else 'Insuffisant',
                'Acceptable' if p_tv < 30 else 'Inacceptable'
            ]
        })
        results_export.to_excel(writer, sheet_name='Résultats', index=False)
        
        # Données brutes
        df.to_excel(writer, sheet_name='Données_Brutes', index=False)
        
        # Statistiques opérateurs
        operators_stats = pd.DataFrame({
            'Opérateur': ['Opérateur 1', 'Opérateur 2', 'Opérateur 3'],
            'Moyenne': [x_bar_op1, x_bar_op2, x_bar_op3],
            'Étendue Moyenne': [r_bar_op1, r_bar_op2, r_bar_op3],
            'Écart-type': [
                df[op1_cols].values.flatten().std(),
                df[op2_cols].values.flatten().std(),
                df[op3_cols].values.flatten().std()
            ],
            'CV%': [
                df[op1_cols].values.flatten().std() / x_bar_op1 * 100,
                df[op2_cols].values.flatten().std() / x_bar_op2 * 100,
                df[op3_cols].values.flatten().std() / x_bar_op3 * 100
            ]
        })
        operators_stats.to_excel(writer, sheet_name='Stats_Opérateurs', index=False)
        
        # Métadonnées
        metadata = pd.DataFrame({
            'Information': ['Date', 'Pièces', 'Opérateurs', 'Essais', 'Facteur K', 
                          'Tolérance', 'Version', 'Statut Final'],
            'Valeur': [
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                n_pieces,
                n_operateurs,
                n_essais,
                confidence_factor,
                tolerance,
                'Gage R&R Pro v2.0',
                'Excellent' if p_grr < 10 else 'Acceptable' if p_grr <= 30 else 'Inacceptable'
            ]
        })
        metadata.to_excel(writer, sheet_name='Métadonnées', index=False)
    
    output.seek(0)
    
    # Bouton de téléchargement
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📥 Télécharger le Rapport Complet (Excel)",
            data=output,
            file_name=f"gage_rr_rapport_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        st.markdown("""
        <div style="text-align: center; color: var(--gray); font-size: 0.9rem; margin-top: 1rem;">
            <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 0.5rem;">
                <div>📊 Résultats détaillés</div>
                <div>📈 Graphiques statistiques</div>
                <div>📋 Recommandations</div>
            </div>
            <div>Rapport professionnel conforme aux normes industrielles</div>
        </div>
        """, unsafe_allow_html=True)

# Pied de page professionnel
st.markdown("""
<div style="margin-top: 4rem; padding: 3rem; background: linear-gradient(135deg, var(--dark), #16213e); 
            border-radius: var(--border-radius); color: white;">
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem;">
        <div>
            <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 1rem; color: var(--success);">📊 Gage R&R Analytics Pro</div>
            <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; line-height: 1.6;">
                Solution avancée d'analyse de la capacité des systèmes de mesure. 
                Conforme aux normes industrielles internationales.
            </div>
        </div>
        <div>
            <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 1rem; color: var(--success);">⚡ Fonctionnalités</div>
            <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem;">
                • Analyses statistiques avancées<br>
                • Visualisations interactives 3D<br>
                • Rapports automatisés<br>
                • Suivi des performances
            </div>
        </div>
        <div>
            <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 1rem; color: var(--success);">🔧 Technologies</div>
            <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem;">
                • Python • Streamlit • Plotly<br>
                • Pandas • NumPy • SciPy<br>
                • Conforme AIAG MSA
            </div>
        </div>
    </div>
    <div style="margin-top: 2rem; padding-top: 2rem; border-top: 1px solid rgba(255, 255, 255, 0.1); 
                text-align: center; color: rgba(255, 255, 255, 0.6); font-size: 0.8rem;">
        © 2024 Gage R&R Analytics Pro | Version 2.0 | Pour usage professionnel
    </div>
</div>
""", unsafe_allow_html=True)
