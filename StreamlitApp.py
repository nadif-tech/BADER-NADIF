import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import time

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
    
    .sidebar-content {
        padding: 1.5rem;
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 0 20px 20px 0;
        height: 100%;
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
    <div class="main-subtitle">Analyse avancée de la capacité du système de mesure</div>
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
    
    st.markdown("---")
    
    st.markdown('<div style="font-size: 1.2rem; font-weight: 600; color: #2c3e50; margin: 1.5rem 0 1rem 0;">📈 Guide de Lecture</div>', unsafe_allow_html=True)
    
    with st.expander("🔍 Comprendre les indicateurs", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class="stat-card">
                <div style="color: #3498db; font-weight: 600;">EV</div>
                <div style="color: #7f8c8d; font-size: 0.85rem;">Répétabilité</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="stat-card">
                <div style="color: #2ecc71; font-weight: 600;">AV</div>
                <div style="color: #7f8c8d; font-size: 0.85rem;">Reproductibilité</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown("""
            <div class="stat-card">
                <div style="color: #9b59b6; font-weight: 600;">GRR</div>
                <div style="color: #7f8c8d; font-size: 0.85rem;">Variation système</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="stat-card">
                <div style="color: #e74c3c; font-weight: 600;">%GRR</div>
                <div style="color: #7f8c8d; font-size: 0.85rem;">Pourcentage total</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div style="font-size: 1.2rem; font-weight: 600; color: #2c3e50; margin: 1.5rem 0 1rem 0;">🎯 Critères</div>', unsafe_allow_html=True)
    
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

# ---------------- ZONE D'UPLOAD STYLÉE ----------------
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
        
        # Style amélioré pour le DataFrame
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

    # ---------------- VISUALISATIONS AVANCÉES ----------------
    st.markdown('<div class="section-header"><span>📈 Visualisations</span></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique 1 : Composantes de variation (3D style)
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        components = ['EV', 'AV', 'GRR', 'VP', 'VT']
        values = [ev, av, grr, vp, vt]
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']
        
        # Barres avec effet 3D
        bars = ax1.bar(components, values, color=colors, edgecolor='white', 
                      linewidth=2, alpha=0.9, zorder=3)
        
        # Ajouter un dégradé aux barres
        for bar, color in zip(bars, colors):
            bar.set_edgecolor('white')
        
        # Style amélioré
        ax1.grid(True, alpha=0.3, zorder=0)
        ax1.set_facecolor('#f8fafc')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Ajouter les valeurs avec animation visuelle
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{value:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10, color='#2c3e50')
        
        ax1.set_title('📊 Composantes de Variation', fontsize=14, fontweight=600, pad=20)
        plt.tight_layout()
        st.pyplot(fig1)
        st.markdown('</div>', unsafe_allow_html=True)
        plt.close()
        
        # Graphique 2 : Radar des opérateurs
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig2 = plt.figure(figsize=(8, 6))
        
        # Données pour le radar
        categories = ['Moyenne', 'Étendue', 'Précision']
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # Valeurs normalisées
        means_norm = [x_bar_op1, r_bar_op1, 1/r_bar_op1]
        means_norm = [v/max(means_norm) for v in means_norm]
        
        ax2 = plt.subplot(111, polar=True)
        ax2.plot(angles, means_norm + means_norm[:1], 'o-', linewidth=2, label='Opérateur 1', color='#3498db')
        ax2.fill(angles, means_norm + means_norm[:1], alpha=0.25, color='#3498db')
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories)
        ax2.set_title('🎯 Performance Opérateurs', fontsize=14, fontweight=600, pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)
        plt.close()
    
    with col2:
        # Graphique 3 : Camembert amélioré
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        
        labels = ['Variation Système\n(GRR)', 'Variation Pièces\n(VP)']
        sizes = [grr**2, vp**2]
        colors = ['#9b59b6', '#e74c3c']
        explode = (0.1, 0)
        
        wedges, texts, autotexts = ax3.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        centre_circle = plt.Circle((0,0), 0.70, fc='white', edgecolor='white', linewidth=2)
        fig3.gca().add_artist(centre_circle)
        
        ax3.axis('equal')
        ax3.set_title('🥧 Répartition des Variations', fontsize=14, fontweight=600, pad=20)
        plt.tight_layout()
        st.pyplot(fig3)
        st.markdown('</div>', unsafe_allow_html=True)
        plt.close()
        
        # Graphique 4 : Jauge de performance
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        
        # Créer une jauge horizontale
        gauge_colors = ['#2ecc71', '#f1c40f', '#e74c3c']
        gauge_ranges = [(0, 10), (10, 30), (30, 100)]
        
        for (start, end), color in zip(gauge_ranges, gauge_colors):
            ax4.barh(0, end-start, left=start, height=0.3, color=color, edgecolor='white', linewidth=2)
        
        # Aiguille de la jauge
        ax4.axvline(x=p_grr, color='#2c3e50', linestyle='-', linewidth=3, alpha=0.8)
        
        # Style
        ax4.set_xlim(0, 100)
        ax4.set_ylim(-0.5, 0.5)
        ax4.set_yticks([])
        ax4.set_xlabel('% Gage R&R', fontsize=12, fontweight=600)
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Texte de valeur
        ax4.text(p_grr, 0.4, f'{p_grr:.1f}%', 
                ha='center', va='center', fontsize=16, fontweight=700,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2c3e50', alpha=0.9))
        
        ax4.set_title('🎯 Jauge de Performance - %GRR', fontsize=14, fontweight=600, pad=20)
        plt.tight_layout()
        st.pyplot(fig4)
        st.markdown('</div>', unsafe_allow_html=True)
        plt.close()

    # ---------------- RÉSULTATS PRINCIPAUX ----------------
    st.markdown('<div class="section-header"><span>📊 Résultats Principaux</span></div>', unsafe_allow_html=True)
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_data = [
        ("EV", ev, "#3498db", "Répétabilité"),
        ("AV", av, "#2ecc71", "Reproductibilité"),
        ("GRR", grr, "#9b59b6", "Variation Système"),
        ("%GRR", p_grr, "#e74c3c", "Pourcentage Total")
    ]
    
    for col, (label, value, color, desc) in zip([col1, col2, col3, col4], metrics_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{desc}</div>
                <div class="metric-value" style="background: linear-gradient(135deg, {color}, #2c3e50); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {value:.3f}{'%' if label == '%GRR' else ''}
                </div>
                <div style="color: #95a5a6; font-size: 0.9rem; margin-top: 0.5rem;">
                    <strong>{label}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Barre de progression avec animation
    progress_html = f"""
    <div style="margin: 2rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <div style="font-weight: 600; color: #2c3e50;">Progression du %GRR</div>
            <div style="font-weight: 600; color: #e74c3c;">{p_grr:.1f}%</div>
        </div>
        <div class="progress-container">
            <div class="progress-bar" style="width: {min(p_grr, 100)}%"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.85rem; color: #7f8c8d;">
            <div>0%</div>
            <div>10%</div>
            <div>30%</div>
            <div>100%</div>
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)
    
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

    # ---------------- STATISTIQUES DÉTAILLÉES ----------------
    st.markdown('<div class="section-header"><span>📋 Statistiques Détaillées</span></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tableau des opérateurs
        st.markdown("**👥 Performance par Opérateur**")
        operators_data = []
        for i, (op_cols, op_name) in enumerate(zip([op1_cols, op2_cols, op3_cols], 
                                                  ['Opérateur 1', 'Opérateur 2', 'Opérateur 3']), 1):
            op_data = df[op_cols].values.flatten()
            operators_data.append({
                'Opérateur': f'👤 {op_name}',
                'Moyenne': f'{np.mean(op_data):.4f}',
                'Étendue': f'{[r_bar_op1, r_bar_op2, r_bar_op3][i-1]:.4f}',
                'σ': f'{np.std(op_data):.4f}'
            })
        
        operators_df = pd.DataFrame(operators_data)
        st.dataframe(
            operators_df.style
            .background_gradient(subset=['Moyenne', 'Étendue', 'σ'], cmap='YlOrRd')
            .set_properties(**{'text-align': 'center'}),
            use_container_width=True
        )
    
    with col2:
        # Indicateurs secondaires
        st.markdown("**📈 Indicateurs Complémentaires**")
        
        secondary_metrics = [
            ("VP (Pièces)", f"{vp:.4f}", "#e74c3c"),
            ("VT (Totale)", f"{vt:.4f}", "#f39c12"),
            ("R̄ (Étendue)", f"{r_double_bar:.4f}", "#3498db"),
            ("Pièces (n)", str(n_pieces), "#95a5a6")
        ]
        
        for label, value, color in secondary_metrics:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; 
                        border-left: 4px solid {color}; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 600; color: #2c3e50;">{label}</div>
                    <div style="font-weight: 700; color: {color}; font-size: 1.1rem;">{value}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---------------- EXPORT STYLÉ ----------------
    st.markdown('<div class="section-header"><span>💾 Export des Résultats</span></div>', unsafe_allow_html=True)
    
    # Création du fichier Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df = pd.DataFrame({
            "Paramètre": ["EV", "AV", "GRR", "VP", "VT", "%GRR"],
            "Valeur": [ev, av, grr, vp, vt, p_grr],
            "Unité": ["unité", "unité", "unité", "unité", "unité", "%"],
            "Statut": [
                "✓ Acceptable" if ev/vt*100 < 30 else "✗ Inacceptable",
                "✓ Acceptable" if av/vt*100 < 30 else "✗ Inacceptable",
                "✓ Excellent" if p_grr < 10 else ("⚠ Conditionnel" if p_grr <= 30 else "✗ Inacceptable"),
                "-", "-",
                f"{p_grr:.1f}%"
            ]
        })
        export_df.to_excel(writer, sheet_name='Résultats', index=False)
        
        # Ajouter d'autres feuilles
        df.to_excel(writer, sheet_name='Données Brutes', index=False)
        
        summary_df = pd.DataFrame({
            'Info': ['Date', 'Pièces', 'Opérateurs', 'Essais', 'Facteur k'],
            'Valeur': [pd.Timestamp.now().strftime('%Y-%m-%d'), n_pieces, n_operateurs, n_essais, confidence_factor]
        })
        summary_df.to_excel(writer, sheet_name='Résumé', index=False)
    
    output.seek(0)
    
    # Bouton de téléchargement stylisé
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <a href='data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{output.getvalue().hex()}' 
           download='resultats_gage_rr_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
           class='download-btn'>
           📥 Télécharger le Rapport Complet
        </a>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem; margin-top: 1rem;">
            Inclut : Résultats détaillés • Données brutes • Résumé de l'étude
        </div>
        """, unsafe_allow_html=True)

# Pied de page élégant
st.markdown("""
<div style="margin-top: 4rem; padding: 2rem; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); 
            border-radius: 20px; text-align: center; border-top: 1px solid #e0e6ed;">
    <div style="font-size: 0.9rem; color: #7f8c8d;">
        <div style="display: flex; justify-content: center; align-items: center; gap: 10px; margin-bottom: 0.5rem;">
            <div>📊</div>
            <div><strong>Gage R&R - Méthode des Étendues</strong></div>
            <div>⚡</div>
        </div>
        <div>Analyse avancée de la capacité du système de mesure • Version Premium</div>
        <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.7;">
            Développé avec Streamlit • Optimisé pour la qualité industrielle
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
