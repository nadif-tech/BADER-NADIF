import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

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
    }
    
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .highlight {
        font-size: 1.8rem;
        font-weight: bold;
        color: #FF6B6B;
    }
    
    .good {
        color: #10B981;
        font-weight: bold;
    }
    
    .warning {
        color: #F59E0B;
        font-weight: bold;
    }
    
    .bad {
        color: #EF4444;
        font-weight: bold;
    }
    
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
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
    - **EV**: Répétabilité (Equipement Variation)
    - **AV**: Reproductibilité (Appraiser Variation)
    - **GRR**: Variation Totale du Système
    - **VP**: Variation des Pièces
    - **VT**: Variation Totale
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Critères d'acceptation")
    st.markdown("""
    - ✅ **< 10%**: Excellent
    - ⚠️ **10-30%**: Conditionnel
    - ❌ **> 30%**: Inacceptable
    """)

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
            st.dataframe(
                df.style
                .background_gradient(subset=pd.IndexSlice[:, df.columns.str.contains('OP')], cmap='Blues')
                .format(precision=3),
                use_container_width=True
            )
        
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
            st.error(f"Colonnes manquantes: {', '.join(missing_cols)}")
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
            
            # ---------------- GRAPHIQUES ----------------
            st.markdown("---")
            st.markdown("## 📈 Visualisations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Graphique en barres des variations
                fig1 = go.Figure(data=[
                    go.Bar(name='Variations', 
                          x=['EV', 'AV', 'GRR', 'VP', 'VT'], 
                          y=[ev, av, grr, vp, vt],
                          marker_color=['#3B82F6', '#10B981', '#8B5CF6', '#F59E0B', '#EF4444'])
                ])
                fig1.update_layout(
                    title='📊 Composantes de Variation',
                    xaxis_title='Composante',
                    yaxis_title='Valeur',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Diagramme en radar pour les performances des opérateurs
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=[r_bar_op1, r_bar_op2, r_bar_op3],
                    theta=['Opérateur 1', 'Opérateur 2', 'Opérateur 3'],
                    fill='toself',
                    name='Étendues Moyennes',
                    line_color='#3B82F6'
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, max([r_bar_op1, r_bar_op2, r_bar_op3])*1.2])
                    ),
                    title='📡 Performance par Opérateur',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with col2:
                # Camembert pour la répartition des variations
                labels = ['Variation Mesure (GRR)', 'Variation Pièces (VP)']
                values = [grr**2, vp**2]  # Utilisation des variances
                colors = ['#8B5CF6', '#F59E0B']
                
                fig2 = go.Figure(data=[go.Pie(
                    labels=labels, 
                    values=values,
                    hole=.3,
                    marker_colors=colors,
                    textinfo='percent+label',
                    textposition='inside'
                )])
                fig2.update_layout(
                    title='🥧 Répartition des Variations',
                    template='plotly_white',
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Jauge pour le %GRR
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=p_grr,
                    title={'text': "🎯 % Gage R&R"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#3B82F6"},
                        'steps': [
                            {'range': [0, 10], 'color': "#10B981"},
                            {'range': [10, 30], 'color': "#F59E0B"},
                            {'range': [30, 100], 'color': "#EF4444"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 30
                        }
                    }
                ))
                fig_gauge.update_layout(height=400)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            # ---------------- RÉSULTATS DÉTAILLÉS ----------------
            st.markdown("---")
            st.markdown("## 📊 Résultats Détailés")
            
            # Métriques dans des colonnes avec style
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("EV – Répétabilité", f"{ev:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("AV – Reproductibilité", f"{av:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Gage R&R", f"{grr:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Variabilité Pièces", f"{vp:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Variabilité Totale", f"{vt:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("% Gage R&R", f"{p_grr:.2f}%")
                
                # Indicateur visuel
                progress_value = min(p_grr / 100, 1.0)
                st.progress(progress_value)
                
                # Message de statut
                if p_grr < 10:
                    st.markdown('<p class="good">✅ Excellent - Système accepté</p>', unsafe_allow_html=True)
                elif p_grr <= 30:
                    st.markdown('<p class="warning">⚠️ Conditionnel - Amélioration recommandée</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="bad">❌ Inacceptable - Action corrective requise</p>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # ---------------- TABLEAU DES MOYENNES ----------------
            st.markdown("### 📋 Statistiques par Opérateur")
            stats_df = pd.DataFrame({
                'Opérateur': ['Opérateur 1', 'Opérateur 2', 'Opérateur 3'],
                'Moyenne': [x_bar_op1, x_bar_op2, x_bar_op3],
                'Étendue Moyenne': [r_bar_op1, r_bar_op2, r_bar_op3],
                'Écart-Type': [
                    df[op1_cols].values.std(),
                    df[op2_cols].values.std(),
                    df[op3_cols].values.std()
                ]
            })
            
            st.dataframe(
                stats_df.style
                .format(precision=4)
                .background_gradient(subset=['Moyenne', 'Étendue Moyenne', 'Écart-Type'], cmap='YlOrRd'),
                use_container_width=True
            )
            
            # ---------------- EXPORT EXCEL ----------------
            st.markdown("---")
            st.markdown("## 💾 Export des Résultats")
            
            # Création du fichier Excel avec onglets multiples
            with pd.ExcelWriter('resultats_gage_rr_complet.xlsx', engine='openpyxl') as writer:
                # Résultats principaux
                resultats_df = pd.DataFrame({
                    'Paramètre': ['EV', 'AV', 'GRR', 'VP', 'VT', '%GRR'],
                    'Valeur': [ev, av, grr, vp, vt, p_grr],
                    'Statut': [
                        'Acceptable' if ev/vt*100 < 30 else 'Inacceptable',
                        'Acceptable' if av/vt*100 < 30 else 'Inacceptable',
                        'Acceptable' if p_grr < 30 else 'Inacceptable',
                        'N/A', 'N/A',
                        'Excellent' if p_grr < 10 else ('Conditionnel' if p_grr <= 30 else 'Inacceptable')
                    ]
                })
                resultats_df.to_excel(writer, sheet_name='Résultats', index=False)
                
                # Statistiques par opérateur
                stats_df.to_excel(writer, sheet_name='Statistiques_Opérateurs', index=False)
                
                # Données brutes avec calculs
                df.to_excel(writer, sheet_name='Données_Brutes', index=False)
            
            with open('resultats_gage_rr_complet.xlsx', 'rb') as f:
                excel_data = f.read()
            
            st.download_button(
                label="📥 Télécharger le Rapport Complet Excel",
                data=excel_data,
                file_name="resultats_gage_rr_complet.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Contient tous les résultats, statistiques et données brutes"
            )
    
    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier: {str(e)}")
        st.info("Veuillez vérifier que le fichier Excel est correctement formaté.")

else:
    # Instructions quand aucun fichier n'est uploadé
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 3rem;'>
            <h3>📋 Instructions</h3>
            <p>1. Préparez votre fichier Excel avec le format suivant :</p>
            <p>2. Les colonnes doivent être nommées :</p>
            <p><strong>OP1-1, OP1-2, OP1-3</strong> (Opérateur 1)</p>
            <p><strong>OP2-1, OP2-2, OP2-3</strong> (Opérateur 2)</p>
            <p><strong>OP3-1, OP3-2, OP3-3</strong> (Opérateur 3)</p>
            <p>3. Chaque ligne représente une pièce différente</p>
            <p>4. Téléversez le fichier pour démarrer l'analyse</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Template de fichier Excel
        template_data = {
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
        template_df = pd.DataFrame(template_data)
        
        with st.expander("Voir un exemple de format"):
            st.dataframe(template_df, use_container_width=True)
            
            # Télécharger le template
            buffer = BytesIO()
            template_df.to_excel(buffer, index=False)
            buffer.seek(0)
            
            st.download_button(
                "📝 Télécharger le Template Excel",
                buffer,
                "template_gage_rr.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Pied de page
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "📊 Gage R&R - Méthode des Étendues | Développé avec Streamlit"
    "</div>",
    unsafe_allow_html=True
)
