import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gage R&R - Étendues", layout="wide")

# CSS personnalisé pour améliorer l'esthétique
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2c3e50;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 30px;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #3498db;
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .result-indicator {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.1em;
    }
    
    .good {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    
    .warning {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffeaa7;
    }
    
    .bad {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    
    .stDataFrame {
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    
    .plot-container {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Titre avec style
st.markdown('<div class="main-title">📊 Étude Gage R&R – Méthode des Étendues</div>', unsafe_allow_html=True)

# ---------------- d2 FUNCTION ----------------
def get_d2(z, w):
    if z > 15 and w == 3:
        return 1.693
    if z == 1 and w == 3:
        return 1.91
    if z == 1 and w == 10:
        return 3.18
    return 1.0

# ---------------- PARAMÈTRES ----------------
with st.sidebar:
    st.header("⚙️ Paramètres")
    st.markdown("---")
    confidence_factor = st.slider(
        "Facteur de confiance (k)",
        min_value=4.0,
        max_value=6.0,
        value=5.15,
        step=0.05,
        help="Valeur recommandée: 5.15 pour 99% de couverture"
    )
    
    st.markdown("---")
    st.header("📊 Légende")
    st.markdown("""
    **EV**: Répétabilité (Équipement)  
    **AV**: Reproductibilité (Opérateurs)  
    **GRR**: Variation du système  
    **VP**: Variation des pièces  
    **VT**: Variation totale
    """)
    
    st.markdown("---")
    st.header("✅ Critères")
    st.markdown("""
    - **< 10%** : Excellent
    - **10-30%** : Acceptable
    - **> 30%** : Inacceptable
    """)

# ---------------- IMPORT EXCEL ----------------
uploaded_file = st.file_uploader("📥 Importer le fichier Excel Gage R&R", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    with st.expander("📄 Aperçu des données", expanded=True):
        # Style pour le dataframe
        styled_df = df.style.background_gradient(subset=pd.IndexSlice[:, df.columns.str.contains('OP')], cmap='Blues')
        st.dataframe(styled_df, use_container_width=True)

    # Colonnes opérateurs
    op1_cols = ["OP1-1", "OP1-2", "OP1-3"]
    op2_cols = ["OP2-1", "OP2-2", "OP2-3"]
    op3_cols = ["OP3-1", "OP3-2", "OP3-3"]

    n_pieces = df.shape[0]
    n_operateurs = 3
    n_essais = 3

    # ---------------- MOYENNES & ÉTENDUES ----------------
    df["R_OP1"] = df[op1_cols].max(axis=1) - df[op1_cols].min(axis=1)
    df["R_OP2"] = df[op2_cols].max(axis=1) - df[op2_cols].min(axis=1)
    df["R_OP3"] = df[op3_cols].max(axis=1) - df[op3_cols].min(axis=1)

    r_bar_op1 = df["R_OP1"].mean()
    r_bar_op2 = df["R_OP2"].mean()
    r_bar_op3 = df["R_OP3"].mean()

    x_bar_op1 = df[op1_cols].values.mean()
    x_bar_op2 = df[op2_cols].values.mean()
    x_bar_op3 = df[op3_cols].values.mean()

    # ---------------- CALCULS GRR ----------------
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

    # ---------------- VARIABILITÉ PIÈCES ----------------
    df["Moy_Piece"] = df[op1_cols + op2_cols + op3_cols].mean(axis=1)
    rp = df["Moy_Piece"].max() - df["Moy_Piece"].min()

    d2_vp = get_d2(1, n_pieces)
    vp = (confidence_factor * rp) / d2_vp

    vt = np.sqrt(grr ** 2 + vp ** 2)
    p_grr = (grr / vt) * 100

    # ---------------- GRAPHIQUES ----------------
    st.divider()
    st.subheader("📈 Visualisations")
    
    # Création des graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique 1: Composantes de variation
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        st.markdown("**📊 Composantes de Variation**")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        components = ['EV', 'AV', 'GRR', 'VP', 'VT']
        values = [ev, av, grr, vp, vt]
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']
        
        bars = ax1.bar(components, values, color=colors, edgecolor='black')
        ax1.set_ylabel('Valeur')
        ax1.set_title('Composantes de Variation')
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig1)
        st.markdown("</div>", unsafe_allow_html=True)
        plt.close()
        
        # Graphique 2: Répartition en pourcentage
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        st.markdown("**🥧 Répartition des Variations**")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        labels = ['GRR', 'VP']
        sizes = [grr**2, vp**2]
        colors = ['#9b59b6', '#e74c3c']
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors,
                                           autopct='%1.1f%%', startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax2.axis('equal')
        plt.tight_layout()
        st.pyplot(fig2)
        st.markdown("</div>", unsafe_allow_html=True)
        plt.close()
    
    with col2:
        # Graphique 3: Performance des opérateurs
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        st.markdown("**👥 Performance par Opérateur**")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        operators = ['Opérateur 1', 'Opérateur 2', 'Opérateur 3']
        r_values = [r_bar_op1, r_bar_op2, r_bar_op3]
        
        bars = ax3.bar(operators, r_values, color=['#3498db', '#2ecc71', '#9b59b6'], edgecolor='black')
        ax3.set_ylabel('Étendue Moyenne')
        ax3.set_title('Étendues par Opérateur')
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, r_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(r_values)*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig3)
        st.markdown("</div>", unsafe_allow_html=True)
        plt.close()
        
        # Graphique 4: % Gage R&R avec barre de progression
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        st.markdown("**🎯 % Gage R&R**")
        
        # Créer une barre de progression horizontale
        fig4, ax4 = plt.subplots(figsize=(8, 1))
        
        # Zones de couleur
        ax4.barh(0, 10, height=0.6, color='#2ecc71', edgecolor='black')
        ax4.barh(0, 20, left=10, height=0.6, color='#f1c40f', edgecolor='black')
        ax4.barh(0, 70, left=30, height=0.6, color='#e74c3c', edgecolor='black')
        
        # Marqueur pour la valeur actuelle
        ax4.axvline(x=p_grr, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax4.text(p_grr, 0, f'  {p_grr:.1f}%', 
                verticalalignment='center', fontweight='bold', fontsize=10)
        
        ax4.set_xlim(0, 100)
        ax4.set_ylim(-0.5, 0.5)
        ax4.set_xlabel('% Gage R&R')
        ax4.set_yticks([])
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        st.pyplot(fig4)
        st.markdown("</div>", unsafe_allow_html=True)
        plt.close()

    # ---------------- AFFICHAGE DES RÉSULTATS ----------------
    st.divider()
    st.subheader("📊 Résultats Gage R&R")

    # Affichage des métriques dans des cartes stylisées
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("**EV – Répétabilité**", f"{ev:.4f}", 
                 help="Variation due à l'équipement de mesure")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("**AV – Reproductibilité**", f"{av:.4f}",
                 help="Variation entre les opérateurs")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("**Gage R&R**", f"{grr:.4f}",
                 help="Variation totale du système de mesure")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("**Variabilité Totale**", f"{vt:.4f}",
                 help="Variation totale du processus")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("**% Gage R&R**", f"{p_grr:.2f}%",
                 help="Pourcentage de variation du système par rapport à la variation totale")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Affichage du statut avec style
        if p_grr < 10:
            status_class = "good"
            status_icon = "✅"
            status_text = "Système de mesure EXCELLENT"
            st.balloons()
        elif p_grr <= 30:
            status_class = "warning"
            status_icon = "⚠️"
            status_text = "Système ACCEPTABLE avec amélioration"
        else:
            status_class = "bad"
            status_icon = "❌"
            status_text = "Système NON ACCEPTABLE"
        
        st.markdown(f'<div class="result-indicator {status_class}">{status_icon} {status_text}</div>', 
                   unsafe_allow_html=True)
        
        # Tableau des statistiques par opérateur
        st.markdown("**📋 Statistiques par Opérateur**")
        stats_data = {
            'Opérateur': ['Opérateur 1', 'Opérateur 2', 'Opérateur 3'],
            'Moyenne': [x_bar_op1, x_bar_op2, x_bar_op3],
            'Étendue Moyenne': [r_bar_op1, r_bar_op2, r_bar_op3],
            'Écart-Type': [
                df[op1_cols].values.std(),
                df[op2_cols].values.std(),
                df[op3_cols].values.std()
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(
            stats_df.style
            .background_gradient(subset=['Moyenne', 'Étendue Moyenne', 'Écart-Type'], cmap='YlOrRd')
            .format(precision=4),
            use_container_width=True
        )

    # ---------------- EXPORT EXCEL ----------------
    st.divider()
    st.subheader("💾 Export des Résultats")
    
    # Créer un fichier Excel avec plusieurs feuilles
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Feuille des résultats principaux
        export_df = pd.DataFrame({
            "Paramètre": ["EV", "AV", "GRR", "VP", "VT", "%GRR"],
            "Valeur": [ev, av, grr, vp, vt, p_grr],
            "Description": [
                "Répétabilité (Équipement)",
                "Reproductibilité (Opérateurs)",
                "Variation système de mesure",
                "Variation pièces",
                "Variation totale",
                "Pourcentage Gage R&R"
            ],
            "Statut": [
                "✓" if ev/vt*100 < 30 else "✗",
                "✓" if av/vt*100 < 30 else "✗",
                "✓" if p_grr < 30 else "✗",
                "-",
                "-",
                "✓ Excellent" if p_grr < 10 else ("⚠ Acceptable" if p_grr <= 30 else "✗ Inacceptable")
            ]
        })
        export_df.to_excel(writer, sheet_name='Résultats', index=False)
        
        # Feuille des données calculées
        df.to_excel(writer, sheet_name='Données Calculées', index=False)
        
        # Feuille des statistiques par opérateur
        stats_df.to_excel(writer, sheet_name='Statistiques Opérateurs', index=False)
    
    output.seek(0)
    
    # Bouton de téléchargement stylisé
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📥 Télécharger le rapport complet Excel",
            data=output,
            file_name="resultats_gage_rr_complet.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Contient tous les résultats et statistiques détaillées",
            use_container_width=True
        )

else:
    # Affichage quand aucun fichier n'est uploadé
    st.info("📋 **Instructions:**")
    st.markdown("""
    1. **Préparez votre fichier Excel** avec les colonnes suivantes :
       - `OP1-1`, `OP1-2`, `OP1-3` (Opérateur 1)
       - `OP2-1`, `OP2-2`, `OP2-3` (Opérateur 2)
       - `OP3-1`, `OP3-2`, `OP3-3` (Opérateur 3)
    
    2. **Structure des données :**
       - Chaque ligne représente une pièce différente
       - Chaque colonne contient une mesure
       - 3 opérateurs × 3 essais = 9 colonnes
    
    3. **Exemple de format :**
    """)
    
    # Exemple de données
    example_data = {
        'OP1-1': [10.1, 10.2, 10.3, 10.4, 10.5],
        'OP1-2': [10.2, 10.3, 10.4, 10.5, 10.6],
        'OP1-3': [10.0, 10.1, 10.2, 10.3, 10.4],
        'OP2-1': [10.3, 10.4, 10.5, 10.6, 10.7],
        'OP2-2': [10.2, 10.3, 10.4, 10.5, 10.6],
        'OP2-3': [10.1, 10.2, 10.3, 10.4, 10.5],
        'OP3-1': [10.0, 10.1, 10.2, 10.3, 10.4],
        'OP3-2': [10.1, 10.2, 10.3, 10.4, 10.5],
        'OP3-3': [10.2, 10.3, 10.4, 10.5, 10.6]
    }
    
    example_df = pd.DataFrame(example_data)
    st.dataframe(example_df, use_container_width=True)

# Pied de page
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d; font-size: 0.9em;'>"
    "📊 Gage R&R - Méthode des Étendues | Développé avec Streamlit"
    "</div>",
    unsafe_allow_html=True
)
