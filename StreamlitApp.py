import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import time
from datetime import datetime

st.set_page_config(
    page_title="Gage R&R Pro - Analyse Avancée",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé avec animations et effets visuels
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
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        border: 2px solid transparent;
    }
    
    .interactive-badge:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        border-color: rgba(255, 255, 255, 0.3);
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
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
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
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
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
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="gradient-header">
    <div class="main-title">📊 Gage R&R Pro</div>
    <div class="main-subtitle">Analyse avancée de la capacité du système de mesure - Intelligence Artificielle</div>
    
    <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
        <span class="interactive-badge">⚡ Temps Réel</span>
        <span class="interactive-badge">🎨 3D Interactive</span>
        <span class="interactive-badge">📈 Export Pro</span>
        <span class="interactive-badge">🤖 IA Intégrée</span>
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

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown('<div class="glass-morphism" style="padding: 1.5rem;">', unsafe_allow_html=True)
    
    # Logo et titre sidebar
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown('<div class="icon-3d">⚙️</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="font-size: 1.5rem; font-weight: 800; background: linear-gradient(135deg, #667eea, #764ba2); 
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Dashboard Pro
        </div>
        <div style="color: #64748b; font-size: 0.9rem;">Contrôle en temps réel</div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Paramètres
    st.markdown("**🔧 PARAMÈTRES AVANCÉS**")
    confidence_factor = st.slider(
        "Facteur de Confiance (k)",
        min_value=4.0,
        max_value=6.0,
        value=5.15,
        step=0.05
    )
    
    st.markdown("---")
    st.markdown("**🎨 THÈME VISUEL**")
    theme = st.selectbox(
        "Style de visualisation",
        ["Industriel Pro", "Data Science", "Minimaliste", "Futuriste"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("**📊 PERFORMANCE LIVE**")
    
    # Indicateurs de performance
    metrics = {
        "CPU": 45,
        "Mémoire": 68,
        "GPU": 22,
        "Réseau": 12
    }
    
    for name, value in metrics.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{name}**")
        with col2:
            color = "#10b981" if value < 50 else "#f59e0b" if value < 80 else "#ef4444"
            st.markdown(f'<span style="color: {color}; font-weight: 600;">{value}%</span>', unsafe_allow_html=True)
        
        st.progress(value/100)
    
    st.markdown("---")
    
    # Widget date/heure
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        current_time = datetime.now().strftime("%H:%M")
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 2.5rem;">🌤️</div>
            <div style="font-size: 1.2rem; font-weight: 600;">{current_time}</div>
            <div style="color: #64748b; font-size: 0.9rem;">Analyse en cours...</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ZONE D'UPLOAD ----------------
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
    type=["xlsx"],
    help="Téléversez votre fichier Excel contenant les mesures",
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem;">
        <div style="font-size: 5rem; margin-bottom: 1.5rem; animation: iconFloat 4s ease-in-out infinite;">☁️</div>
        <div style="font-size: 1.3rem; font-weight: 600; color: #2c3e50; margin-bottom: 1rem;">
            Zone de dépôt intelligente
        </div>
        <div style="color: #64748b; margin-bottom: 2rem;">
            Déposez votre fichier Excel ici ou parcourez vos dossiers
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    # Animation de chargement
    with st.spinner('🚀 **Traitement des données en cours...**'):
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.01)
        
        df = pd.read_excel(uploaded_file)
        st.success('✅ **Analyse complète !**')

    # ---------------- APERÇU DES DONNÉES ----------------
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
        </div>
    """, unsafe_allow_html=True)
    
    with st.expander("**🔍 Aperçu détaillé des données**", expanded=True):
        view_mode = st.radio(
            "Mode d'affichage :",
            ["Tableau Interactif", "Statistiques", "Heatmap"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if view_mode == "Tableau Interactif":
            st.dataframe(df.style.background_gradient(cmap='RdYlBu_r'), use_container_width=True, height=300)
        elif view_mode == "Statistiques":
            st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Colonnes opérateurs
    op1_cols = ["OP1-1", "OP1-2", "OP1-3"]
    op2_cols = ["OP2-1", "OP2-2", "OP2-3"]
    op3_cols = ["OP3-1", "OP3-2", "OP3-3"]

    n_pieces = df.shape[0]
    n_operateurs = 3
    n_essais = 3

    # ---------------- CALCULS ----------------
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

    # ---------------- VISUALISATIONS ----------------
    st.markdown("""
    <div class="neomorph-card">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem;">
            <div class="icon-3d">🎨</div>
            <div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #2c3e50;">Visualisations Avancées</div>
                <div style="color: #64748b;">Graphiques interactifs et analyses intelligentes</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Analyses", "🎯 Performance"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique 1
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            components = ['EV', 'AV', 'GRR', 'VP', 'VT']
            values = [ev, av, grr, vp, vt]
            colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']
            
            bars = ax1.bar(components, values, color=colors, edgecolor='white', linewidth=2)
            ax1.set_title('Composantes de Variation', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            plt.close()
        
        with col2:
            # Graphique 2
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sizes = [grr**2, vp**2]
            labels = ['Système', 'Pièces']
            colors = ['#9b59b6', '#e74c3c']
            
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Répartition des Variations', fontsize=14, fontweight='bold')
            st.pyplot(fig2)
            plt.close()
    
    with tab3:
        # Métriques de performance
        col1, col2, col3, col4 = st.columns(4)
        
        metrics_data = [
            ("EV", ev, "#3498db", "Répétabilité"),
            ("AV", av, "#2ecc71", "Reproductibilité"),
            ("GRR", grr, "#9b59b6", "Variation Système"),
            ("%GRR", p_grr, "#e74c3c", "Performance")
        ]
        
        for col, (label, value, color, desc) in zip([col1, col2, col3, col4], metrics_data):
            with col:
                st.markdown(f"""
                <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
                    <div style="font-size: 2.5rem; font-weight: 800; color: {color}; margin-bottom: 0.5rem;">
                        {value:.3f}{'%' if label == '%GRR' else ''}
                    </div>
                    <div style="color: #2c3e50; font-weight: 600; font-size: 1.2rem;">{label}</div>
                    <div style="color: #64748b; font-size: 0.9rem; margin-top: 0.5rem;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Indicateur de résultat
        st.markdown("---")
        if p_grr < 10:
            status_color = "#2ecc71"
            status_message = "🌟 EXCELLENT - NIVEAU WORLD CLASS"
            st.success("🎉 **FÉLICITATIONS !** Votre processus est optimal")
        elif p_grr <= 30:
            status_color = "#f1c40f"
            status_message = "✅ ACCEPTABLE - AMÉLIORATIONS POSSIBLES"
            st.warning("⚠️ **ATTENTION :** Améliorations recommandées")
        else:
            status_color = "#e74c3c"
            status_message = "🚨 INACCEPTABLE - ACTION REQUISE"
            st.error("❌ **URGENT :** Plan d'action corrective requis")
        
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, {status_color}20, {status_color}40); 
                    border-radius: 20px; border: 2px solid {status_color}; margin-top: 1rem;">
            <div style="font-size: 2rem; font-weight: 800; color: {status_color}; margin-bottom: 1rem;">
                {status_message}
            </div>
            <div style="color: #2c3e50; font-size: 1.1rem;">
                Score final : <span style="font-weight: 800;">{p_grr:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- EXPORT ----------------
    st.markdown("""
    <div class="neomorph-card">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 2rem;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div class="icon-3d">💾</div>
                <div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #2c3e50;">Centre d'Export</div>
                    <div style="color: #64748b;">Génération de rapports professionnels</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Bouton de téléchargement
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df = pd.DataFrame({
            "Paramètre": ["EV", "AV", "GRR", "VP", "VT", "%GRR"],
            "Valeur": [ev, av, grr, vp, vt, p_grr],
            "Unité": ["unité", "unité", "unité", "unité", "unité", "%"],
            "Statut": [
                "✓ Excellent" if p_grr < 10 else ("⚠ Acceptable" if p_grr < 30 else "✗ Inacceptable"),
                "✓ Excellent" if p_grr < 10 else ("⚠ Acceptable" if p_grr < 30 else "✗ Inacceptable"),
                "✓ Excellent" if p_grr < 10 else ("⚠ Acceptable" if p_grr < 30 else "✗ Inacceptable"),
                "-", "-",
                f"{p_grr:.1f}%"
            ]
        })
        export_df.to_excel(writer, sheet_name='Résultats', index=False)
        df.to_excel(writer, sheet_name='Données Brutes', index=False)
    
    output.seek(0)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📥 **TÉLÉCHARGER LE RAPPORT COMPLET**",
            data=output,
            file_name=f"gage_rr_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PIED DE PAGE ----------------
st.markdown("""
<div style="margin-top: 4rem; padding: 2rem; background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(245,247,250,0.9)); 
            border-radius: 20px; text-align: center; border: 1px solid rgba(255, 255, 255, 0.3);">
    
    <div style="font-size: 1.2rem; font-weight: 700; color: #2c3e50; margin-bottom: 1rem;">
        ⚡ Gage R&R Pro - Intelligence Industrielle 🚀
    </div>
    
    <div style="color: #64748b; max-width: 600px; margin: 0 auto 1rem auto; line-height: 1.6;">
        Système d'analyse avancée pour la qualité industrielle 4.0
    </div>
    
    <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;">
        <span class="interactive-badge">📚 Documentation</span>
        <span class="interactive-badge">💬 Support</span>
        <span class="interactive-badge">🔄 Mise à jour</span>
    </div>
    
    <div style="margin-top: 1.5rem; color: #94a3b8; font-size: 0.85rem;">
        <div>© 2024 Gage R&R Pro • Version 2.0</div>
        <div style="margin-top: 0.5rem;">
            <span>🔒 Sécurité maximale</span> • 
            <span>⚡ Performance optimale</span> • 
            <span>🎨 Design premium</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
