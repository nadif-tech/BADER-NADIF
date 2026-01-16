import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- Configuration page ---
st.set_page_config(
    page_title="Gage R&R – Dashboard Avancé", 
    layout="wide",
    page_icon="📊"
)

# --- CSS personnalisé pour améliorer l'esthétique ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    .good-metric {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    }
    
    .warning-metric {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
    }
    
    .bad-metric {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
    }
    
    .section-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-top: 2rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3B82F6;
    }
    
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton button {
        background: linear-gradient(90deg, #3B82F6, #1D4ED8);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- En-tête avec style amélioré ---
st.markdown('<h1 class="main-header">📊 Dashboard Gage R&R - Analyse de Système de Mesure</h1>', unsafe_allow_html=True)
st.markdown("---")

# --- Fonction d2 ---
def get_d2(z, w):
    d2_table = {
        (15, 3): 1.693,
        (1, 3): 1.91,
        (1, 10): 3.18
    }
    return d2_table.get((z, w), 1.0)

# --- Sidebar paramètres avec style amélioré ---
with st.sidebar:
    st.markdown("### ⚙️ **Paramètres d'analyse**")
    
    with st.expander("Paramètres statistiques", expanded=True):
        confidence_factor = st.number_input(
            "Facteur de confiance (k)",
            value=5.15, 
            step=0.01,
            help="Facteur multiplicateur pour calculer les intervalles de confiance (généralement 5.15 pour 99% de couverture)"
        )
        
        bins_hist = st.slider(
            "Nombre de classes histogramme", 
            5, 30, 10,
            help="Nombre de barres pour les histogrammes"
        )
    
    st.markdown("---")
    
    with st.expander("Personnalisation graphique", expanded=True):
        theme_color = st.selectbox(
            "Couleur du thème",
            ["Blues", "Viridis", "Plasma", "Inferno", "Magma"],
            index=0
        )
        
        chart_style = st.selectbox(
            "Style des graphiques",
            ["Plotly", "Matplotlib", "Seaborn"],
            index=0
        )
    
    st.markdown("---")
    st.markdown("### 📖 **Guide d'interprétation**")
    with st.expander("Critères d'acceptation"):
        st.markdown("""
        - **%GRR < 10%** : ✅ Acceptable
        - **10% ≤ %GRR ≤ 30%** : ⚠️ À considérer
        - **%GRR > 30%** : ❌ Inacceptable
        - **NdC ≥ 5** : ✅ Acceptable
        """)

# --- Import Excel ---
uploaded_file = st.file_uploader(
    "📥 **Importer le fichier Excel Gage R&R**", 
    type=["xlsx", "xls"],
    help="Format attendu : Colonnes OP1-1, OP1-2, OP1-3, OP2-1, etc."
)

if uploaded_file:
    # Chargement avec barre de progression
    with st.spinner('Chargement des données...'):
        df = pd.read_excel(uploaded_file)
        st.success('✅ Données chargées avec succès!')
    
    # --- Affichage des données avec onglets ---
    st.markdown('<h2 class="section-header">📋 Données de Mesure</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Aperçu des données", "Statistiques descriptives", "Vue détaillée"])
    
    with tab1:
        st.dataframe(
            df.style
            .background_gradient(subset=pd.IndexSlice[:, df.columns.str.contains('OP')], cmap='Blues')
            .format(precision=3),
            use_container_width=True,
            height=300
        )
    
    with tab2:
        if 'OP' in ''.join(df.columns):
            op_cols = [col for col in df.columns if 'OP' in col]
            stats_df = df[op_cols].agg(['mean', 'std', 'min', 'max', 'count']).round(3)
            st.dataframe(stats_df, use_container_width=True)
    
    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre de pièces", df.shape[0])
        with col2:
            st.metric("Nombre d'opérateurs", 3)
        with col3:
            st.metric("Nombre de répétitions", 3)
    
    # --- Configuration des colonnes ---
    op1_cols = ["OP1-1", "OP1-2", "OP1-3"]
    op2_cols = ["OP2-1", "OP2-2", "OP2-3"]
    op3_cols = ["OP3-1", "OP3-2", "OP3-3"]
    n_pieces = df.shape[0]
    n_operateurs = 3
    n_essais = 3
    
    # --- Calculs ---
    with st.spinner('Calcul des indicateurs Gage R&R...'):
        # Calcul des plages par opérateur
        df["R_OP1"] = df[op1_cols].max(axis=1) - df[op1_cols].min(axis=1)
        df["R_OP2"] = df[op2_cols].max(axis=1) - df[op2_cols].min(axis=1)
        df["R_OP3"] = df[op3_cols].max(axis=1) - df[op3_cols].min(axis=1)
        
        # Moyennes des plages
        r_bar_op1 = df["R_OP1"].mean()
        r_bar_op2 = df["R_OP2"].mean()
        r_bar_op3 = df["R_OP3"].mean()
        
        # Moyennes des mesures
        x_bar_op1 = df[op1_cols].values.mean()
        x_bar_op2 = df[op2_cols].values.mean()
        x_bar_op3 = df[op3_cols].values.mean()
        
        # Calcul R double barre
        r_double_bar = (r_bar_op1 + r_bar_op2 + r_bar_op3) / n_operateurs
        
        # Calcul EV (Variabilité des équipements)
        d2_ev = get_d2(n_pieces * n_operateurs, n_essais)
        ev = (confidence_factor * r_double_bar) / d2_ev
        
        # Calcul AV (Variabilité des opérateurs)
        x_range = max([x_bar_op1, x_bar_op2, x_bar_op3]) - min([x_bar_op1, x_bar_op2, x_bar_op3])
        d2_av = get_d2(1, n_operateurs)
        av_term = (confidence_factor * x_range / d2_av) ** 2
        ev_corr = (ev ** 2) / (n_pieces * n_essais)
        av = np.sqrt(max(0, av_term - ev_corr))
        
        # Calcul GRR
        grr = np.sqrt(ev ** 2 + av ** 2)
        
        # Calcul VP (Variabilité des pièces)
        df["Moy_Piece"] = df[op1_cols + op2_cols + op3_cols].mean(axis=1)
        rp = df["Moy_Piece"].max() - df["Moy_Piece"].min()
        vp = (confidence_factor * rp) / get_d2(1, n_pieces)
        
        # Calcul VT (Variabilité totale)
        vt = np.sqrt(grr ** 2 + vp ** 2)
        
        # Pourcentages
        p_ev = (ev / vt) * 100 if vt != 0 else 0
        p_av = (av / vt) * 100 if vt != 0 else 0
        p_vp = (vp / vt) * 100 if vt != 0 else 0
        p_grr = (grr / vt) * 100 if vt != 0 else 0
        
        # Calcul NdC (Nombre de catégories distinctes)
        ndc = 1.41 * (vp / grr) if grr != 0 else 0
    
    # --- Cartes métriques avec couleurs conditionnelles ---
    st.markdown('<h2 class="section-header">📊 Résultats Principaux</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        grr_class = "good-metric" if p_grr < 10 else "warning-metric" if p_grr <= 30 else "bad-metric"
        st.markdown(f"""
        <div class="metric-card {grr_class}">
            <div class="metric-label">%GRR</div>
            <div class="metric-value">{p_grr:.1f}%</div>
            <div class="metric-label">{"✅ Acceptable" if p_grr < 10 else "⚠️ À considérer" if p_grr <= 30 else "❌ Inacceptable"}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        ndc_class = "good-metric" if ndc >= 5 else "bad-metric"
        st.markdown(f"""
        <div class="metric-card {ndc_class}">
            <div class="metric-label">NdC</div>
            <div class="metric-value">{ndc:.1f}</div>
            <div class="metric-label">{"✅ ≥ 5" if ndc >= 5 else "❌ < 5"}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">%EV</div>
            <div class="metric-value">{p_ev:.1f}%</div>
            <div class="metric-label">Équipement</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">%AV</div>
            <div class="metric-value">{p_av:.1f}%</div>
            <div class="metric-label">Opérateurs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">%VP</div>
            <div class="metric-value">{p_vp:.1f}%</div>
            <div class="metric-label">Pièces</div>
        </div>
        """, unsafe_allow_html=True)
    
    # --- Graphiques et visualisations ---
    st.markdown('<h2 class="section-header">📈 Visualisations Analytiques</h2>', unsafe_allow_html=True)
    
    tab_graph1, tab_graph2, tab_graph3 = st.tabs(["Analyse de variance", "Contrôle par opérateur", "Distribution"])
    
    with tab_graph1:
        # Diagramme en barres des contributions
        fig1 = go.Figure(data=[
            go.Bar(
                name='Contribution',
                x=['Équipement (EV)', 'Opérateurs (AV)', 'Pièces (VP)', 'GRR'],
                y=[p_ev, p_av, p_vp, p_grr],
                text=[f'{p_ev:.1f}%', f'{p_av:.1f}%', f'{p_vp:.1f}%', f'{p_grr:.1f}%'],
                textposition='auto',
                marker_color=['#3B82F6', '#10B981', '#8B5CF6', '#EF4444']
            )
        ])
        
        fig1.update_layout(
            title='Analyse de Variance (% Contribution)',
            yaxis_title='Pourcentage (%)',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab_graph2:
        # Graphique par opérateur
        fig2 = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Opérateur 1', 'Opérateur 2', 'Opérateur 3'),
            shared_yaxes=True
        )
        
        for i, (op_cols, op_name) in enumerate(zip([op1_cols, op2_cols, op3_cols], ['OP1', 'OP2', 'OP3']), 1):
            for j, col in enumerate(op_cols, 1):
                fig2.add_trace(
                    go.Scatter(
                        x=df.index + 1,
                        y=df[col],
                        mode='markers+lines',
                        name=f'{op_name}-{j}',
                        showlegend=(i==1)
                    ),
                    row=1, col=i
                )
        
        fig2.update_layout(
            title='Mesures par Opérateur',
            height=400,
            template='plotly_white'
        )
        fig2.update_xaxes(title_text='Numéro de pièce')
        fig2.update_yaxes(title_text='Valeur mesurée', row=1, col=1)
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab_graph3:
        col_hist1, col_hist2 = st.columns(2)
        
        with col_hist1:
            # Histogramme des moyennes par pièce
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(
                x=df["Moy_Piece"],
                nbinsx=bins_hist,
                name='Distribution',
                marker_color='#3B82F6',
                opacity=0.7
            ))
            
            fig3.update_layout(
                title='Distribution des moyennes par pièce',
                xaxis_title='Valeur moyenne',
                yaxis_title='Fréquence',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col_hist2:
            # Box plot par opérateur
            fig4 = go.Figure()
            
            all_data = []
            labels = []
            for op_cols, op_name in zip([op1_cols, op2_cols, op3_cols], ['OP1', 'OP2', 'OP3']):
                for col in op_cols:
                    all_data.append(df[col])
                    labels.append(op_name)
            
            fig4.add_trace(go.Box(
                y=pd.concat(all_data),
                x=labels,
                name='Mesures',
                boxpoints='outliers',
                marker_color='#10B981'
            ))
            
            fig4.update_layout(
                title='Box Plot par Opérateur',
                xaxis_title='Opérateur',
                yaxis_title='Valeur mesurée',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    # --- Tableaux détaillés ---
    st.markdown('<h2 class="section-header">📋 Résultats Détailés</h2>', unsafe_allow_html=True)
    
    col_table1, col_table2 = st.columns(2)
    
    with col_table1:
        st.markdown("**📌 Valeurs absolues**")
        main_results = pd.DataFrame({
            "Indicateur": ["EV", "AV", "VP", "GRR", "VT"],
            "Valeur": [round(ev, 4), round(av, 4), round(vp, 4), round(grr, 4), round(vt, 4)],
            "Description": [
                "Variabilité équipement",
                "Variabilité opérateurs", 
                "Variabilité pièces",
                "Variabilité totale mesure",
                "Variabilité totale"
            ]
        })
        st.dataframe(main_results, use_container_width=True, hide_index=True)
    
    with col_table2:
        st.markdown("**📌 Pourcentages et NdC**")
        percentage_results = pd.DataFrame({
            "Indicateur": ["%EV", "%AV", "%VP", "%GRR", "NdC"],
            "Valeur": [f"{p_ev:.1f}%", f"{p_av:.1f}%", f"{p_vp:.1f}%", f"{p_grr:.1f}%", f"{ndc:.1f}"],
            "Statut": [
                "✅" if p_ev < 10 else "⚠️" if p_ev <= 30 else "❌",
                "✅" if p_av < 10 else "⚠️" if p_av <= 30 else "❌",
                "✅" if p_vp > 60 else "⚠️" if p_vp >= 40 else "❌",
                "✅" if p_grr < 10 else "⚠️" if p_grr <= 30 else "❌",
                "✅" if ndc >= 5 else "❌"
            ]
        })
        st.dataframe(percentage_results, use_container_width=True, hide_index=True)
    
    # --- Export avec options ---
    st.markdown('<h2 class="section-header">💾 Export des Résultats</h2>', unsafe_allow_html=True)
    
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        # Export Excel
        export_df = pd.DataFrame({
            "Paramètre": ["EV", "AV", "VP", "GRR", "VT", "%EV", "%AV", "%VP", "%GRR", "NdC"],
            "Valeur": [ev, av, vp, grr, vt, p_ev, p_av, p_vp, p_grr, ndc],
            "Unité": ["abs", "abs", "abs", "abs", "abs", "%", "%", "%", "%", ""]
        })
        
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            export_df.to_excel(writer, sheet_name='Résultats', index=False)
            df.to_excel(writer, sheet_name='Données brutes', index=False)
            
            # Ajouter un graphique dans Excel
            workbook = writer.book
            worksheet = writer.sheets['Résultats']
            
            # Créer un graphique pour Excel
            chart = workbook.add_chart({'type': 'column'})
            chart.add_series({
                'values': 'Résultats!$B$2:$B$5',
                'categories': 'Résultats!$A$2:$A$5',
                'name': 'Valeurs absolues'
            })
            worksheet.insert_chart('F2', chart)
        
        buffer.seek(0)
        
        st.download_button(
            label="📥 Télécharger rapport complet Excel",
            data=buffer,
            file_name="rapport_gage_rr_complet.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col_export2:
        # Export PDF simulé (rapport HTML)
        html_report = f"""
        <html>
        <head><title>Rapport Gage R&R</title></head>
        <body>
            <h1>Rapport d'Analyse Gage R&R</h1>
            <h2>Résumé des résultats</h2>
            <p>%GRR: {p_grr:.1f}% - {'Acceptable' if p_grr < 10 else 'À considérer' if p_grr <= 30 else 'Inacceptable'}</p>
            <p>NdC: {ndc:.1f} - {'Acceptable' if ndc >= 5 else 'Inacceptable'}</p>
            <h3>Détails:</h3>
            <p>%EV: {p_ev:.1f}%</p>
            <p>%AV: {p_av:.1f}%</p>
            <p>%VP: {p_vp:.1f}%</p>
        </body>
        </html>
        """
        
        st.download_button(
            label="📄 Générer rapport PDF",
            data=html_report,
            file_name="rapport_gage_rr.html",
            mime="text/html"
        )
    
    with col_export3:
        # Copier les résultats
        if st.button("📋 Copier les résultats"):
            results_text = f"""
            RÉSULTATS GAGE R&R
            ------------------
            %GRR: {p_grr:.1f}%
            NdC: {ndc:.1f}
            %EV: {p_ev:.1f}%
            %AV: {p_av:.1f}%
            %VP: {p_vp:.1f}%
            
            Conclusion: {'✅ ACCEPTABLE' if p_grr < 10 and ndc >= 5 else '⚠️ À CONSIDÉRER' if p_grr <= 30 else '❌ INACCEPTABLE'}
            """
            st.code(results_text)
            st.success("Résultats copiables dans le code ci-dessus")
    
    # --- Conclusion avec recommandations ---
    st.markdown("---")
    st.markdown('<h2 class="section-header">🎯 Conclusion et Recommandations</h2>', unsafe_allow_html=True)
    
    if p_grr < 10:
        st.success("""
        ### ✅ SYSTÈME DE MESURE ACCEPTABLE
        **Recommandations:**
        - Le système de mesure est adéquat pour son utilisation prévue
        - Maintenir les procédures de mesure actuelles
        - Continuer le suivi périodique selon le plan de surveillance
        """)
    elif p_grr <= 30:
        st.warning("""
        ### ⚠️ SYSTÈME DE MESURE À CONSIDÉRER
        **Recommandations:**
        - Analyser la cause principale de la variabilité
        - Considérer l'impact sur la décision produit
        - Mettre en place des actions correctives si nécessaire
        - Revalider après modifications
        """)
    else:
        st.error("""
        ### ❌ SYSTÈME DE MESURE INACCEPTABLE
        **Actions requises:**
        - Arrêter l'utilisation du système pour des décisions produit
        - Identifier et corriger les causes de variabilité
        - Ré-étalonner ou remplacer l'équipement si nécessaire
        - Re-former les opérateurs
        - Refaire l'étude après corrections
        """)
    
else:
    # Écran d'accueil sans fichier
    st.info("👆 Veuillez importer un fichier Excel pour commencer l'analyse Gage R&R")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("### 📋 Format de fichier attendu")
        st.markdown("""
        Votre fichier Excel doit contenir les colonnes suivantes:
        
        | Pièce | OP1-1 | OP1-2 | OP1-3 | OP2-1 | OP2-2 | OP2-3 | OP3-1 | OP3-2 | OP3-3 |
        |-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
        | 1     | 10.1  | 10.2  | 10.0  | 10.3  | 10.1  | 10.2  | 10.0  | 10.1  | 10.2  |
        | 2     | 20.2  | 20.3  | 20.1  | 20.4  | 20.2  | 20.3  | 20.1  | 20.2  | 20.3  |
        | ...   | ...   | ...   | ...   | ...   | ...   | ...   | ...   | ...   | ...   |
        """)
    
    with col_info2:
        st.markdown("### 🎯 Objectifs de l'analyse")
        st.markdown("""
        1. **Évaluer la capabilité** du système de mesure
        2. **Décomposer la variabilité** en composantes
        3. **Déterminer l'acceptabilité** selon les normes
        4. **Identifier les améliorations** possibles
        
        **Critères d'acceptation:**
        - %GRR < 10% : Acceptable
        - 10-30% : À considérer
        - >30% : Inacceptable
        - NdC ≥ 5 : Acceptable
        """)

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280; padding: 1rem;'>
        <p>📊 Dashboard Gage R&R - Analyse de Système de Mesure | Développé avec Streamlit</p>
        <p style='font-size: 0.9rem;'>Pour toute question ou support technique, contactez l'équipe qualité.</p>
    </div>
    """,
    unsafe_allow_html=True
)
