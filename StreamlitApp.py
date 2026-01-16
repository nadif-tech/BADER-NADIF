import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(
    page_title="Gage R&R - Méthode des Étendues",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
    }
    
    .highlight {
        font-size: 1.8rem;
        font-weight: bold;
        color: #FF6B6B;
    }
    
    .good {
        color: #10B981;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .warning {
        color: #F59E0B;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .bad {
        color: #EF4444;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .progress-container {
        background: #F3F4F6;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    
    .custom-progress {
        height: 20px;
        border-radius: 10px;
        background: linear-gradient(90deg, #10B981 0%, #F59E0B 50%, #EF4444 100%);
        transition: width 0.5s ease;
    }
    
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .download-button {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .download-button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Titre avec style
st.markdown('<h1 class="main-header">📊 Étude Gage R&R – Méthode des Étendues</h1>', unsafe_allow_html=True)

# ---------------- d2 FUNCTION ----------------
def get_d2(z, w):
    d2_values = {
        (15, 3): 1.693,
        (1, 3): 1.91,
        (1, 10): 3.18
    }
    return d2_values.get((z, w), 1.0)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ⚙️ Paramètres de l'étude")
    st.markdown("---")
    
    confidence_factor = st.slider(
        "Facteur de confiance (k)",
        min_value=4.0,
        max_value=6.0,
        value=5.15,
        step=0.05,
        help="Valeur recommandée: 5.15 (99% de couverture)"
    )
    
    st.markdown("---")
    st.markdown("### 📈 Légende")
    st.markdown("""
    **EV**: Répétabilité (Equipement Variation)
    
    **AV**: Reproductibilité (Appraiser Variation)
    
    **GRR**: Variation Totale du Système
    
    **VP**: Variation des Pièces
    
    **VT**: Variation Totale
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Critères d'acceptation")
    col_ok, col_warn, col_bad = st.columns(3)
    with col_ok:
        st.markdown("""
        <div style='text-align: center;'>
            <div style='background-color: #10B981; color: white; padding: 5px; border-radius: 5px;'>
                <strong>✓ < 10%</strong><br>Excellent
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_warn:
        st.markdown("""
        <div style='text-align: center;'>
            <div style='background-color: #F59E0B; color: white; padding: 5px; border-radius: 5px;'>
                <strong>⚠ 10-30%</strong><br>Conditionnel
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_bad:
        st.markdown("""
        <div style='text-align: center;'>
            <div style='background-color: #EF4444; color: white; padding: 5px; border-radius: 5px;'>
                <strong>✗ > 30%</strong><br>Inacceptable
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- IMPORT EXCEL ----------------
st.markdown("## 📥 Importation des Données")

uploaded_file = st.file_uploader(
    "Téléversez votre fichier Excel",
    type=["xlsx"],
    help="Le fichier doit contenir les colonnes OP1-1, OP1-2, OP1-3, OP2-1, OP2-2, OP2-3, OP3-1, OP3-2, OP3-3"
)

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        
        # Aperçu des données avec style
        st.markdown("## 📄 Aperçu des Données")
        with st.expander("Voir les données", expanded=True):
            # Appliquer un style aux données
            styled_df = df.style.background_gradient(
                subset=df.columns[df.columns.str.contains('OP')], 
                cmap='Blues'
            ).format(precision=3)
            
            st.dataframe(styled_df, use_container_width=True)
        
        # Vérification des colonnes
        op1_cols = ["OP1-1", "OP1-2", "OP1-3"]
        op2_cols = ["OP2-1", "OP2-2", "OP2-3"]
        op3_cols = ["OP3-1", "OP3-2", "OP3-3"]
        
        # Validation des colonnes
        missing_cols = []
        for col_set in [op1_cols, op2_cols, op3_cols]:
            for col in col_set:
                if col not in df.columns:
                    missing_cols.append(col)
        
        if missing_cols:
            st.error(f"❌ Colonnes manquantes: {', '.join(missing_cols)}")
        else:
            n_pieces = df.shape[0]
            n_operateurs = 3
            n_essais = 3
            
            # ---------------- CALCULS ----------------
            # Moyennes et étendues par opérateur
            df["R_OP1"] = df[op1_cols].max(axis=1) - df[op1_cols].min(axis=1)
            df["R_OP2"] = df[op2_cols].max(axis=1) - df[op2_cols].min(axis=1)
            df["R_OP3"] = df[op3_cols].max(axis=1) - df[op3_cols].min(axis=1)
            
            r_bar_op1 = df["R_OP1"].mean()
            r_bar_op2 = df["R_OP2"].mean()
            r_bar_op3 = df["R_OP3"].mean()
            
            x_bar_op1 = df[op1_cols].values.mean()
            x_bar_op2 = df[op2_cols].values.mean()
            x_bar_op3 = df[op3_cols].values.mean()
            
            # GRR Calculations
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
            
            # ---------------- VISUALISATIONS AVEC MATPLOTLIB ----------------
            st.markdown("---")
            st.markdown("## 📈 Visualisations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Diagramme en barres des variations
                st.markdown("#### 📊 Composantes de Variation")
                fig1, ax1 = plt.subplots(figsize=(8, 4))
                components = ['EV', 'AV', 'GRR', 'VP', 'VT']
                values = [ev, av, grr, vp, vt]
                colors = ['#3B82F6', '#10B981', '#8B5CF6', '#F59E0B', '#EF4444']
                
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
                plt.close()
                
                # Graphique des étendues par opérateur
                st.markdown("#### 🔍 Étendues par Opérateur")
                fig3, ax3 = plt.subplots(figsize=(8, 4))
                operators = ['Opérateur 1', 'Opérateur 2', 'Opérateur 3']
                r_values = [r_bar_op1, r_bar_op2, r_bar_op3]
                ax3.plot(operators, r_values, marker='o', linewidth=2, markersize=8, color='#3B82F6')
                ax3.fill_between(operators, r_values, alpha=0.2, color='#3B82F6')
                ax3.set_ylabel('Étendue Moyenne')
                ax3.set_title('Performance des Opérateurs')
                ax3.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close()
            
            with col2:
                # Camembert pour la répartition des variations
                st.markdown("#### 🥧 Répartition des Variations")
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                labels = ['GRR', 'VP']
                sizes = [grr**2, vp**2]
                colors = ['#8B5CF6', '#F59E0B']
                explode = (0.1, 0)
                
                wedges, texts, autotexts = ax2.pie(
                    sizes, explode=explode, labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=90
                )
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                ax2.axis('equal')
                ax2.set_title('Répartition des Variations')
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
                
                # Graphique à barres horizontales pour le %GRR
                st.markdown("#### 🎯 % Gage R&R")
                fig4, ax4 = plt.subplots(figsize=(8, 2))
                
                # Créer une barre horizontale colorée
                x_pos = 0
                bar_height = 0.6
                
                # Zones de couleur
                ax4.barh(x_pos, 10, height=bar_height, color='#10B981', edgecolor='black')
                ax4.barh(x_pos, 20, left=10, height=bar_height, color='#F59E0B', edgecolor='black')
                ax4.barh(x_pos, 70, left=30, height=bar_height, color='#EF4444', edgecolor='black')
                
                # Marqueur pour la valeur actuelle
                ax4.axvline(x=p_grr, color='black', linestyle='--', linewidth=2, alpha=0.7)
                ax4.text(p_grr, x_pos, f'  {p_grr:.1f}%', 
                        verticalalignment='center', fontweight='bold', fontsize=10)
                
                ax4.set_xlim(0, 100)
                ax4.set_ylim(-0.5, 0.5)
                ax4.set_xlabel('% Gage R&R')
                ax4.set_yticks([])
                ax4.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                st.pyplot(fig4)
                plt.close()
            
            # ---------------- RÉSULTATS DÉTAILLÉS ----------------
            st.markdown("---")
            st.markdown("## 📊 Résultats Détailés")
            
            # Métriques dans des colonnes avec style
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("📏 EV – Répétabilité", f"{ev:.4f}", 
                         help="Variation due à l'équipement de mesure")
                st.metric("👥 AV – Reproductibilité", f"{av:.4f}",
                         help="Variation entre les opérateurs")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("⚙️ Gage R&R", f"{grr:.4f}",
                         help="Variation totale du système de mesure")
                st.metric("🔧 VP – Variation Pièces", f"{vp:.4f}",
                         help="Variation entre les pièces")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🌐 VT – Variation Totale", f"{vt:.4f}",
                         help="Variation totale du processus")
                
                # Barre de progression personnalisée
                st.markdown(f"**🎯 % Gage R&R: {p_grr:.2f}%**")
                
                # Barre de progression colorée
                progress_width = min(p_grr / 100, 1.0)
                st.markdown(f"""
                <div class="progress-container">
                    <div class="custom-progress" style="width: {progress_width*100}%"></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Message de statut
                if p_grr < 10:
                    st.markdown('<p class="good">✅ Excellent - Système accepté</p>', unsafe_allow_html=True)
                    st.balloons()
                elif p_grr <= 30:
                    st.markdown('<p class="warning">⚠️ Conditionnel - Amélioration recommandée</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="bad">❌ Inacceptable - Action corrective requise</p>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # ---------------- TABLEAU DES MOYENNES ----------------
            st.markdown("### 📋 Statistiques par Opérateur")
            
            # Calcul des statistiques
            stats_data = []
            for i, (op_cols, op_name) in enumerate(zip(
                [op1_cols, op2_cols, op3_cols],
                ['Opérateur 1', 'Opérateur 2', 'Opérateur 3']
            ), 1):
                op_data = df[op_cols].values.flatten()
                stats_data.append({
                    'Opérateur': op_name,
                    'Moyenne': df[op_cols].values.mean(),
                    'Médiane': np.median(op_data),
                    'Étendue Moyenne': [r_bar_op1, r_bar_op2, r_bar_op3][i-1],
                    'Écart-Type': np.std(op_data),
                    'Min': np.min(op_data),
                    'Max': np.max(op_data)
                })
            
            stats_df = pd.DataFrame(stats_data)
            
            # Afficher le tableau avec style
            st.dataframe(
                stats_df.style
                .background_gradient(subset=['Moyenne', 'Étendue Moyenne', 'Écart-Type'], cmap='YlOrRd')
                .format(precision=4),
                use_container_width=True
            )
            
            # ---------------- EXPORT EXCEL ----------------
            st.markdown("---")
            st.markdown("## 💾 Export des Résultats")
            
            # Créer un fichier Excel en mémoire
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Résultats principaux
                resultats_df = pd.DataFrame({
                    'Paramètre': ['EV', 'AV', 'GRR', 'VP', 'VT', '%GRR'],
                    'Valeur': [ev, av, grr, vp, vt, p_grr],
                    'Unité': ['unité', 'unité', 'unité', 'unité', 'unité', '%'],
                    'Statut': [
                        '✓ Acceptable' if ev/vt*100 < 30 else '✗ Inacceptable',
                        '✓ Acceptable' if av/vt*100 < 30 else '✗ Inacceptable',
                        '✓ Acceptable' if p_grr < 30 else '✗ Inacceptable',
                        'N/A', 'N/A',
                        '✓ Excellent' if p_grr < 10 else ('⚠️ Conditionnel' if p_grr <= 30 else '✗ Inacceptable')
                    ]
                })
                resultats_df.to_excel(writer, sheet_name='Résultats', index=False)
                
                # Statistiques par opérateur
                stats_df.to_excel(writer, sheet_name='Statistiques_Opérateurs', index=False)
                
                # Données brutes avec calculs
                df.to_excel(writer, sheet_name='Données_Brutes', index=False)
                
                # Détails des calculs
                calculs_df = pd.DataFrame({
                    'Calcul': [
                        'R_bar (étendue moyenne)',
                        'EV (Répétabilité)',
                        'X_range (étendue des moyennes)',
                        'AV (Reproductibilité)',
                        'GRR',
                        'RP (étendue des pièces)',
                        'VP (Variation Pièces)',
                        'VT (Variation Totale)',
                        '%GRR'
                    ],
                    'Formule': [
                        f'({r_bar_op1:.3f} + {r_bar_op2:.3f} + {r_bar_op3:.3f}) / 3',
                        f'{confidence_factor} × {r_double_bar:.3f} / {d2_ev}',
                        f'{max(means_ops):.3f} - {min(means_ops):.3f}',
                        f'√(({confidence_factor}×{x_range:.3f}/{d2_av})² - ({ev:.3f}²/({n_pieces}×{n_essais})))',
                        f'√({ev:.3f}² + {av:.3f}²)',
                        f'{df["Moy_Piece"].max():.3f} - {df["Moy_Piece"].min():.3f}',
                        f'{confidence_factor} × {rp:.3f} / {d2_vp}',
                        f'√({grr:.3f}² + {vp:.3f}²)',
                        f'({grr:.3f} / {vt:.3f}) × 100'
                    ],
                    'Valeur': [
                        f'{r_double_bar:.4f}',
                        f'{ev:.4f}',
                        f'{x_range:.4f}',
                        f'{av:.4f}',
                        f'{grr:.4f}',
                        f'{rp:.4f}',
                        f'{vp:.4f}',
                        f'{vt:.4f}',
                        f'{p_grr:.2f}%'
                    ]
                })
                calculs_df.to_excel(writer, sheet_name='Détails_Calculs', index=False)
            
            output.seek(0)
            
            # Bouton de téléchargement
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📥 Télécharger le Rapport Complet Excel",
                    data=output,
                    file_name="resultats_gage_rr_complet.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Contient tous les résultats, statistiques et données brutes"
                )
    
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement du fichier: {str(e)}")
        st.info("ℹ️ Veuillez vérifier que le fichier Excel est correctement formaté.")

else:
    # Instructions quand aucun fichier n'est uploadé
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div style='text-align: center;'>
            <div style='font-size: 4rem;'>📋</div>
            <h3>Instructions</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### Format requis pour le fichier Excel:
        
        1. **Colonnes obligatoires:**
           - `OP1-1`, `OP1-2`, `OP1-3` (Opérateur 1)
           - `OP2-1`, `OP2-2`, `OP2-3` (Opérateur 2)
           - `OP3-1`, `OP3-2`, `OP3-3` (Opérateur 3)
        
        2. **Structure des données:**
           - Chaque ligne = une pièce différente
           - Chaque colonne = une mesure par opérateur
           - 3 opérateurs × 3 essais = 9 colonnes
        
        3. **Exemple de données:**
        """)
        
        # Afficher un exemple de tableau
        example_data = {
            'Pièce': [1, 2, 3, 4, 5],
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
        
        # Créer et offrir un template
        template_df = example_df.drop('Pièce', axis=1)
        template_buffer = BytesIO()
        template_df.to_excel(template_buffer, index=False)
        template_buffer.seek(0)
        
        st.download_button(
            "📝 Télécharger le Template Excel",
            template_buffer,
            "template_gage_rr.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Utilisez ce template pour format correctement vos données"
        )

# Pied de page
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "📊 Gage R&R - Méthode des Étendues | Développé avec Streamlit, Pandas & NumPy"
    "</div>",
    unsafe_allow_html=True
)
