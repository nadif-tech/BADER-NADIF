import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyBboxPatch
import matplotlib.patheffects as path_effects
import time
from scipy import stats

st.set_page_config(
    page_title="Gage R&R Pro - Méthode des Étendues",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé avec design moderne
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
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
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4cc9f0, #f72585, #4361ee);
    }
    
    .data-table {
        background: white;
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: var(--shadow);
    }
    
    .glassmorphism {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
    }
    
    .floating-icon {
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* Amélioration des boutons Streamlit */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: var(--transition);
        box-shadow: 0 4px 15px rgba(67, 97, 238, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(67, 97, 238, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <div class="main-title">📊 Gage R&R Analytics Pro</div>
    <div class="main-subtitle">Analyse avancée de la capacité du système de mesure avec visualisations professionnelles</div>
</div>
""", unsafe_allow_html=True)

# ---------------- d2 FUNCTION ----------------
def get_d2(z, w):
    d2_table = {
        (1, 3): 1.91, (1, 10): 3.18,
        (2, 3): 1.81, (2, 10): 2.52,
        (3, 3): 1.77, (3, 10): 2.26,
        (4, 3): 1.75, (4, 10): 2.09,
        (5, 3): 1.74, (5, 10): 1.96,
        (6, 3): 1.73, (6, 10): 1.87,
        (7, 3): 1.72, (7, 10): 1.81,
        (8, 3): 1.72, (8, 10): 1.77,
        (9, 3): 1.71, (9, 10): 1.74,
        (10, 3): 1.71, (10, 10): 1.72,
        (15, 3): 1.693, (15, 10): 1.67,
        (20, 3): 1.68, (20, 10): 1.64
    }
    return d2_table.get((z, w), 1.693)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown('<div class="glassmorphism" style="padding: 2rem; border-radius: var(--border-radius);">', unsafe_allow_html=True)
    
    st.markdown('<div style="font-size: 1.8rem; font-weight: 800; color: var(--primary); margin-bottom: 2rem;">⚙️ Configuration</div>', unsafe_allow_html=True)
    
    confidence_factor = st.slider(
        "**Facteur de Confiance (k)**",
        min_value=4.0,
        max_value=6.0,
        value=5.15,
        step=0.05,
        help="Facteur pour le niveau de confiance des calculs"
    )
    
    tolerance = st.number_input(
        "**Tolérance du Processus**",
        min_value=0.1,
        max_value=100.0,
        value=10.0,
        step=0.1,
        help="Valeur de tolérance pour le calcul %Tolérance"
    )
    
    st.markdown("---")
    
    st.markdown('<div style="font-size: 1.2rem; font-weight: 600; color: var(--dark); margin: 1.5rem 0 1rem 0;">🎯 Critères d'Évaluation</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="status-indicator status-excellent">Excellent</div>', unsafe_allow_html=True)
        st.caption("< 10%")
    with col2:
        st.markdown('<div class="status-indicator status-acceptable">Acceptable</div>', unsafe_allow_html=True)
        st.caption("10-30%")
    with col3:
        st.markdown('<div class="status-indicator status-unacceptable">Inacceptable</div>', unsafe_allow_html=True)
        st.caption("> 30%")
    
    st.markdown("---")
    
    st.markdown('<div style="font-size: 1.2rem; font-weight: 600; color: var(--dark); margin: 1.5rem 0 1rem 0;">📊 Paramètres Graphiques</div>', unsafe_allow_html=True)
    
    theme = st.selectbox(
        "Thème des Graphiques",
        ["Modern", "Corporate", "Technical"],
        index=0
    )
    
    show_advanced = st.checkbox("Afficher les graphiques avancés", value=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ZONE D'UPLOAD ----------------
st.markdown('<div class="section-header-pro">📥 Importation des Données</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["xlsx"],
    help="Téléversez votre fichier Excel contenant les mesures",
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown("""
    <div style="text-align: center; padding: 4rem; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); 
                border-radius: var(--border-radius); border: 2px dashed var(--primary); margin: 2rem 0;">
        <div class="floating-icon" style="font-size: 5rem; margin-bottom: 1rem;">📁</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: var(--dark); margin-bottom: 0.5rem;">
            Glissez-déposez votre fichier Excel
        </div>
        <div style="color: var(--gray); margin-bottom: 2rem;">
            Format requis : 3 opérateurs × 3 essais
        </div>
        <div style="background: rgba(67, 97, 238, 0.1); padding: 1.5rem; border-radius: var(--border-radius); 
                    display: inline-block; text-align: left;">
            <div style="font-weight: 700; color: var(--primary); margin-bottom: 0.5rem;">📋 Structure :</div>
            <div style="color: var(--gray); font-size: 0.9rem;">
                OP1-1, OP1-2, OP1-3<br>
                OP2-1, OP2-2, OP2-3<br>
                OP3-1, OP3-2, OP3-3
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
        
        df = pd.read_excel(uploaded_file)

    # ---------------- APERÇU DES DONNÉES ----------------
    st.markdown('<div class="section-header-pro">📄 Aperçu des Données</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📈 Nombre de Pièces", df.shape[0])
    with col2:
        st.metric("👥 Opérateurs", 3)
    with col3:
        st.metric("🎯 Essais par Opérateur", 3)
    
    with st.expander("📋 Données Détailées", expanded=True):
        st.markdown('<div class="data-table">', unsafe_allow_html=True)
        st.dataframe(df.style.format("{:.4f}"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Colonnes opérateurs
    op1_cols = ["OP1-1", "OP1-2", "OP1-3"]
    op2_cols = ["OP2-1", "OP2-2", "OP2-3"]
    op3_cols = ["OP3-1", "OP3-2", "OP3-3"]

    n_pieces = df.shape[0]
    n_operateurs = 3
    n_essais = 3

    # ---------------- CALCULS ----------------
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
        ndc = 1.41 * (vp / grr) if grr > 0 else 0
        p_tv = (grr / tolerance) * 100 if tolerance > 0 else 0
        
        # Test ANOVA
        f_stat, p_value = stats.f_oneway(
            df[op1_cols].values.flatten(),
            df[op2_cols].values.flatten(),
            df[op3_cols].values.flatten()
        )

    # ---------------- DASHBOARD DES KPIs ----------------
    st.markdown('<div class="section-header-pro">📊 Tableau de Bord</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card-pro">
            <div style="color: var(--gray); font-size: 0.9rem; font-weight: 600;">% Gage R&R</div>
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
            <div style="color: var(--gray); font-size: 0.9rem; font-weight: 600;">Discrimination (ndc)</div>
            <div class="metric-value-pro">{ndc:.1f}</div>
            <div style="color: var(--gray); font-size: 0.85rem; margin-top: 0.5rem;">
                {"✅ > 5" if ndc > 5 else "⚠️ ≤ 5"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card-pro">
            <div style="color: var(--gray); font-size: 0.9rem; font-weight: 600;">Répétabilité (EV)</div>
            <div class="metric-value-pro">{ev:.4f}</div>
            <div style="color: var(--gray); font-size: 0.85rem; margin-top: 0.5rem;">
                {(ev/vt*100):.1f}% du total
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card-pro">
            <div style="color: var(--gray); font-size: 0.9rem; font-weight: 600;">Reproductibilité (AV)</div>
            <div class="metric-value-pro">{av:.4f}</div>
            <div style="color: var(--gray); font-size: 0.85rem; margin-top: 0.5rem;">
                {(av/vt*100):.1f}% du total
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ---------------- VISUALISATIONS PROFESSIONNELLES ----------------
    st.markdown('<div class="section-header-pro">📈 Visualisations Avancées</div>', unsafe_allow_html=True)
    
    # Configuration du style des graphiques
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#4361ee', '#4cc9f0', '#7209b7', '#f8961e', '#f72585']
    
    # Graphique 1: Composantes de Variation (barres 3D)
    st.markdown('<div class="plot-container-pro">', unsafe_allow_html=True)
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})
    
    # Graphique à barres avec effet 3D
    components = ['EV\nRépétabilité', 'AV\nReproductibilité', 'GRR\nSystème', 'VP\nPièces', 'VT\nTotale']
    values = [ev, av, grr, vp, vt]
    contributions = [v/vt*100 for v in values[:4]] + [100]
    
    bars = ax1.bar(components, values, color=colors, edgecolor='white', linewidth=2, alpha=0.9, zorder=3)
    
    # Ajout d'un effet d'ombre et de texture
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # Effet de dégradé
        gradient = np.linspace(0.8, 0.3, 100).reshape(100, 1)
        gradient = np.repeat(gradient, 3, axis=1)
        gradient[:, 0] *= int(colors[i][1:3], 16) / 255
        gradient[:, 1] *= int(colors[i][3:5], 16) / 255
        gradient[:, 2] *= int(colors[i][5:7], 16) / 255
        
        # Annotation avec valeur
        ax1.text(bar.get_x() + bar.get_width()/2, height + max(values)*0.01,
                f'{values[i]:.3f}\n({contributions[i]:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=10,
                color='#2c3e50')
    
    ax1.set_title('📊 Composantes de Variation', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Valeur', fontweight='bold')
    ax1.grid(True, alpha=0.3, zorder=0)
    ax1.set_facecolor('#f8fafc')
    
    # Camembert des contributions
    pie_colors = ['#4361ee', '#4cc9f0', '#7209b7']
    pie_sizes = [ev**2, av**2, vp**2]
    pie_labels = [f'EV\n{ev**2/vt**2*100:.1f}%', 
                  f'AV\n{av**2/vt**2*100:.1f}%', 
                  f'VP\n{vp**2/vt**2*100:.1f}%']
    
    wedges, texts, autotexts = ax2.pie(pie_sizes, labels=pie_labels, colors=pie_colors,
                                       autopct='', startangle=90,
                                       wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'width': 0.4})
    
    # Effet de relief sur le camembert
    for wedge in wedges:
        wedge.set_alpha(0.8)
    
    centre_circle = plt.Circle((0,0), 0.30, fc='white', edgecolor='white', linewidth=2)
    ax2.add_artist(centre_circle)
    ax2.text(0, 0, f'VT\n{vt:.3f}', ha='center', va='center', 
             fontweight='bold', fontsize=12)
    
    ax2.set_title('🥧 Répartition des Variations', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    st.pyplot(fig1)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close()
    
    # Graphique 2: Performance des Opérateurs
    st.markdown('<div class="plot-container-pro">', unsafe_allow_html=True)
    fig2 = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    
    # Sous-graphique 1: Radar Chart
    ax3 = fig2.add_subplot(gs[0, 0], projection='polar')
    
    categories = ['Précision', 'Répétabilité', 'Stabilité', 'Exactitude', 'Capacité']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Données normalisées pour chaque opérateur
    op_metrics = []
    for op_cols, color in zip([op1_cols, op2_cols, op3_cols], colors[:3]):
        data = df[op_cols].values.flatten()
        metrics = [
            1 / np.std(data) * 100,  # Précision
            1 / df[[f"R_OP{i+1}" for i in range(3)][op_cols == op1_cols]].mean()[0] * 100,  # Répétabilité
            1 / np.std([data[:len(data)//3].mean(), 
                       data[len(data)//3:2*len(data)//3].mean(),
                       data[2*len(data)//3:].mean()]) * 100,  # Stabilité
            abs(np.mean(data) - df[op1_cols + op2_cols + op3_cols].values.flatten().mean()),  # Exactitude
            1 / (np.std(data) / np.mean(data)) * 100 if np.mean(data) != 0 else 0  # Capacité
        ]
        # Normalisation
        metrics = [m/max(metrics) * 100 for m in metrics]
        metrics += metrics[:1]
        op_metrics.append(metrics)
    
    for i, (metrics, color, label) in enumerate(zip(op_metrics, colors[:3], ['Opérateur 1', 'Opérateur 2', 'Opérateur 3'])):
        ax3.plot(angles, metrics, 'o-', linewidth=2, label=label, color=color)
        ax3.fill(angles, metrics, alpha=0.1, color=color)
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, fontsize=9)
    ax3.set_ylim(0, 100)
    ax3.set_title('🎯 Performance par Opérateur', fontsize=14, fontweight='bold', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax3.grid(True, alpha=0.3)
    
    # Sous-graphique 2: Jauge de Performance
    ax4 = fig2.add_subplot(gs[0, 1])
    
    # Création d'une jauge moderne
    gauge_colors = ['#4cc9f0', '#f8961e', '#f72585']
    gauge_ranges = [(0, 10), (10, 30), (30, 100)]
    
    for (start, end), color in zip(gauge_ranges, gauge_colors):
        ax4.barh(0, end-start, left=start, height=0.5, color=color, 
                edgecolor='white', linewidth=2, alpha=0.8)
    
    # Aiguille
    ax4.axvline(x=p_grr, color='#1a1a2e', linestyle='-', linewidth=4, alpha=0.8, 
                path_effects=[path_effects.withStroke(linewidth=6, foreground="white")])
    
    # Style
    ax4.set_xlim(0, 100)
    ax4.set_ylim(-0.5, 0.5)
    ax4.set_yticks([])
    ax4.set_xlabel('% Gage R&R', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.set_facecolor('#f8fafc')
    
    # Valeur courante
    ax4.text(p_grr, 0.6, f'{p_grr:.1f}%', 
             ha='center', va='center', fontsize=24, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                      edgecolor='#1a1a2e', alpha=0.9))
    
    ax4.set_title('📊 Jauge de Performance', fontsize=14, fontweight='bold', pad=20)
    
    # Sous-graphique 3: Courbes par pièce
    ax5 = fig2.add_subplot(gs[1, :])
    
    for i, (op_cols, color, label) in enumerate(zip([op1_cols, op2_cols, op3_cols], 
                                                   colors[:3], ['Opérateur 1', 'Opérateur 2', 'Opérateur 3'])):
        means_by_piece = df[op_cols].mean(axis=1)
        ax5.plot(range(1, len(means_by_piece) + 1), means_by_piece, 
                marker='o', linewidth=2.5, markersize=6, label=label, color=color,
                markerfacecolor='white', markeredgewidth=2)
    
    ax5.set_xlabel('Numéro de Pièce', fontweight='bold')
    ax5.set_ylabel('Valeur Mesurée', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='best')
    ax5.set_title('📈 Mesures par Pièce et Opérateur', fontsize=14, fontweight='bold', pad=20)
    ax5.set_facecolor('#f8fafc')
    
    plt.tight_layout()
    st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close()
    
    # Graphique 3: Matrice de Corrélation (Heatmap)
    if show_advanced:
        st.markdown('<div class="plot-container-pro">', unsafe_allow_html=True)
        fig3, ax6 = plt.subplots(figsize=(12, 8))
        
        # Préparation des données
        all_columns = op1_cols + op2_cols + op3_cols
        corr_matrix = df[all_columns].corr()
        
        # Masque pour le triangle supérieur
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Heatmap avec Seaborn
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, 
                   fmt='.2f', center=0, square=True, linewidths=1,
                   cbar_kws={"shrink": .8}, ax=ax6)
        
        # Amélioration du design
        ax6.set_title('🔥 Matrice de Corrélation entre les Mesures', 
                     fontsize=16, fontweight='bold', pad=20)
        ax6.set_facecolor('#f8fafc')
        
        plt.tight_layout()
        st.pyplot(fig3)
        st.markdown('</div>', unsafe_allow_html=True)
        plt.close()
    
    # ---------------- ANALYSE DÉTAILLÉE ----------------
    st.markdown('<div class="section-header-pro">🔍 Analyse Statistique</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📋 Résultats Complets", "📊 Statistiques", "🎯 Recommandations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Résultats Principaux**")
            results_data = {
                'Paramètre': ['EV (Répétabilité)', 'AV (Reproductibilité)', 'GRR (Système)', 
                             'VP (Pièces)', 'VT (Totale)', '%GRR', 'Ndc', '%Tolérance'],
                'Valeur': [f'{ev:.6f}', f'{av:.6f}', f'{grr:.6f}', 
                          f'{vp:.6f}', f'{vt:.6f}', f'{p_grr:.2f}%', 
                          f'{ndc:.2f}', f'{p_tv:.2f}%'],
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
        
        with col2:
            st.markdown("**📈 Performance des Opérateurs**")
            op_stats_data = {
                'Opérateur': ['👤 Opérateur 1', '👤 Opérateur 2', '👤 Opérateur 3'],
                'Moyenne': [f'{x_bar_op1:.4f}', f'{x_bar_op2:.4f}', f'{x_bar_op3:.4f}'],
                'Étendue Moy.': [f'{r_bar_op1:.4f}', f'{r_bar_op2:.4f}', f'{r_bar_op3:.4f}'],
                'σ': [f'{df[op1_cols].values.flatten().std():.4f}', 
                      f'{df[op2_cols].values.flatten().std():.4f}', 
                      f'{df[op3_cols].values.flatten().std():.4f}']
            }
            
            op_stats_df = pd.DataFrame(op_stats_data)
            st.dataframe(op_stats_df, use_container_width=True)
            
            # Test ANOVA
            st.markdown("**🧪 Test ANOVA**")
            st.write(f"**F-statistique:** {f_stat:.4f}")
            st.write(f"**P-value:** {p_value:.6f}")
            st.write(f"**Différence significative:** {'✅ Oui' if p_value < 0.05 else '❌ Non'}")
    
    with tab2:
        # Statistiques descriptives détaillées
        st.markdown("**📊 Statistiques Descriptives par Opérateur**")
        
        all_stats = []
        for i, (op_cols, op_name) in enumerate(zip([op1_cols, op2_cols, op3_cols], 
                                                  ['Opérateur 1', 'Opérateur 2', 'Opérateur 3']), 1):
            data = df[op_cols].values.flatten()
            stats_dict = {
                'Opérateur': op_name,
                'Moyenne': np.mean(data),
                'Médiane': np.median(data),
                'Écart-type': np.std(data),
                'CV%': (np.std(data) / np.mean(data) * 100) if np.mean(data) != 0 else 0,
                'Min': np.min(data),
                'Max': np.max(data),
                'Étendue': np.ptp(data)
            }
            all_stats.append(stats_dict)
        
        stats_df = pd.DataFrame(all_stats)
        st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True)
    
    with tab3:
        # Recommandations basées sur les résultats
        if p_grr < 10:
            st.success("""
            ## 🎉 **SYSTÈME EXCELLENT**
            
            ### ✅ **Actions recommandées :**
            1. **Maintenance**
               - Continuer les procédures actuelles
               - Maintenir le calendrier de calibration
               - Documenter les bonnes pratiques
            2. **Surveillance**
               - Suivi périodique (trimestriel)
               - Enregistrement des dérives
            3. **Optimisation**
               - Capitaliser sur l'excellence
               - Étendre les bonnes pratiques
            
            ### 📊 **Points forts :**
            - Système très fiable
            - Bonne discrimination des pièces
            - Variation système minimale
            """)
        elif p_grr <= 30:
            st.warning("""
            ## ⚠️ **SYSTÈME ACCEPTABLE**
            
            ### 🔧 **Actions recommandées :**
            1. **Formation**
               - Recyclage des opérateurs
               - Standardisation des méthodes
               - Vérification des compétences
            2. **Équipement**
               - Vérification de calibration
               - Maintenance préventive
               - Nettoyage et étalonnage
            3. **Processus**
               - Amélioration des fixations
               - Conditions de mesure stables
               - Documentation précise
            
            ### 📈 **Points à améliorer :**
            - Réduire la variation entre opérateurs
            - Améliorer la répétabilité
            - Standardiser les procédures
            """)
        else:
            st.error("""
            ## ❌ **SYSTÈME INACCEPTABLE**
            
            ### 🚨 **Actions prioritaires :**
            1. **Équipement (Urgent)**
               - Recalibration immédiate
               - Vérification de l'usure
               - Remplacement si nécessaire
            2. **Formation (Urgent)**
               - Formation intensive
               - Certification obligatoire
               - Supervision rapprochée
            3. **Processus (Urgent)**
               - Redéfinition complète
               - Amélioration des fixations
               - Conditions contrôlées
            4. **Investigation**
               - Étude approfondie des causes
               - Plan d'action détaillé
               - Suivi rigoureux
            
            ### ⚠️ **Risques identifiés :**
            - Mesures non fiables
            - Décisions erronées
            - Qualité compromise
            """)
    
    # ---------------- EXPORT PROFESSIONNEL ----------------
    st.markdown('<div class="section-header-pro">💾 Export des Résultats</div>', unsafe_allow_html=True)
    
    # Création du fichier Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Résultats principaux
        export_df = pd.DataFrame({
            "Paramètre": ["EV", "AV", "GRR", "VP", "VT", "%GRR", "Ndc", "%Tolérance"],
            "Valeur": [ev, av, grr, vp, vt, p_grr, ndc, p_tv],
            "Unité": ["unité", "unité", "unité", "unité", "unité", "%", "sans", "%"],
            "Contribution": [f"{ev/vt*100:.1f}%", f"{av/vt*100:.1f}%", f"{grr/vt*100:.1f}%",
                           f"{vp/vt*100:.1f}%", "100%", "-", "-", "-"],
            "Statut": [
                "Excellent" if ev/vt*100 < 10 else "Acceptable" if ev/vt*100 < 30 else "Inacceptable",
                "Excellent" if av/vt*100 < 10 else "Acceptable" if av/vt*100 < 30 else "Inacceptable",
                "Excellent" if p_grr < 10 else "Acceptable" if p_grr <= 30 else "Inacceptable",
                "-", "-",
                "Excellent" if p_grr < 10 else "Acceptable" if p_grr <= 30 else "Inacceptable",
                "Suffisant" if ndc > 5 else "Insuffisant",
                "Acceptable" if p_tv < 30 else "Inacceptable"
            ]
        })
        export_df.to_excel(writer, sheet_name='Résultats', index=False)
        
        # Données brutes
        df.to_excel(writer, sheet_name='Données_Brutes', index=False)
        
        # Statistiques opérateurs
        op_export = pd.DataFrame({
            'Opérateur': ['Opérateur 1', 'Opérateur 2', 'Opérateur 3'],
            'Moyenne': [x_bar_op1, x_bar_op2, x_bar_op3],
            'Étendue Moyenne': [r_bar_op1, r_bar_op2, r_bar_op3],
            'Écart-type': [
                df[op1_cols].values.flatten().std(),
                df[op2_cols].values.flatten().std(),
                df[op3_cols].values.flatten().std()
            ]
        })
        op_export.to_excel(writer, sheet_name='Stats_Opérateurs', index=False)
        
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
                'Gage R&R Pro v1.0',
                'Excellent' if p_grr < 10 else 'Acceptable' if p_grr <= 30 else 'Inacceptable'
            ]
        })
        metadata.to_excel(writer, sheet_name='Métadonnées', index=False)
    
    output.seek(0)
    
    # Bouton de téléchargement
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📥 Télécharger le Rapport Complet",
            data=output,
            file_name=f"gage_rr_rapport_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        st.markdown("""
        <div style="text-align: center; color: var(--gray); font-size: 0.9rem; margin-top: 1rem;">
            <div>📊 Résultats détaillés • 📈 Statistiques • 🎯 Recommandations</div>
            <div style="margin-top: 0.5rem; font-size: 0.8rem;">
                Rapport professionnel prêt pour présentation
            </div>
        </div>
        """, unsafe_allow_html=True)

# Pied de page
st.markdown("""
<div style="margin-top: 4rem; padding: 2rem; background: linear-gradient(135deg, #1a1a2e, #16213e); 
            border-radius: var(--border-radius); color: white; text-align: center;">
    <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 1rem; color: #4cc9f0;">
        📊 Gage R&R Analytics Pro
    </div>
    <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; max-width: 800px; margin: 0 auto;">
        Outil professionnel d'analyse de la capacité des systèmes de mesure • Conforme aux normes industrielles
    </div>
    <div style="margin-top: 1.5rem; color: rgba(255, 255, 255, 0.6); font-size: 0.8rem;">
        © 2024 • Développé avec Streamlit • Pour les professionnels de la qualité
    </div>
</div>
""", unsafe_allow_html=True)
