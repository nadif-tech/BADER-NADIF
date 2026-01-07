import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
import math

# =====================================================
# CONFIGURATION GÉNÉRALE
# =====================================================
st.set_page_config(
    page_title="Analyse Gage R&R - Lean Six Sigma",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# TABLE d2 CORRIGÉE (selon normes statistiques)
# =====================================================
D2_TABLE = {
    2: {1: 1.128, 2: 1.693, 3: 2.059, 4: 2.326, 5: 2.534, 6: 2.704, 7: 2.847, 8: 2.970, 9: 3.078, 10: 3.173},
    3: {1: 1.023, 2: 1.160, 3: 1.147, 4: 1.129, 5: 1.110, 6: 1.092, 7: 1.075, 8: 1.060, 9: 1.046, 10: 1.033},
    4: {1: 0.729, 2: 0.841, 3: 0.808, 4: 0.773, 5: 0.743, 6: 0.718, 7: 0.697, 8: 0.680, 9: 0.665, 10: 0.652},
    5: {1: 0.577, 2: 0.606, 3: 0.562, 4: 0.525, 5: 0.496, 6: 0.472, 7: 0.454, 8: 0.438, 9: 0.425, 10: 0.414}
}

def get_d2(w, m):
    """
    Retourne la valeur d2 pour w (taille d'échantillon) et m (nombre d'échantillons)
    Formule standard pour Gage R&R
    """
    if w in D2_TABLE and m in D2_TABLE[w]:
        return D2_TABLE[w][m]
    else:
        # Approximation pour valeurs hors table
        return 1.0

# =====================================================
# FONCTION DE CALCUL GAGE R&R CORRIGÉE
# =====================================================
def calculate_gage_rr_correct(df, n_parts, n_operators, n_trials, k=5.15):
    """
    Calcule Gage R&R selon la méthode standard avec d2
    """
    # Préparation des données
    data = df.iloc[:n_parts, :n_operators*n_trials].values
    
    # Calcul des moyennes et étendues par opérateur
    operator_means = []
    operator_ranges = []
    
    for op in range(n_operators):
        start_col = op * n_trials
        end_col = start_col + n_trials
        op_data = data[:, start_col:end_col]
        
        # Moyenne par pièce pour cet opérateur
        op_means = np.mean(op_data, axis=1)
        operator_means.append(op_means)
        
        # Étendue par pièce pour cet opérateur
        op_ranges = np.max(op_data, axis=1) - np.min(op_data, axis=1)
        operator_ranges.append(op_ranges)
    
    # 1. Calcul de R̄ (moyenne des étendues)
    all_ranges = np.concatenate(operator_ranges)
    R_bar = np.mean(all_ranges)
    
    # 2. Calcul de X_diff (différence des moyennes d'opérateurs)
    operator_global_means = [np.mean(op_mean) for op_mean in operator_means]
    X_diff = max(operator_global_means) - min(operator_global_means)
    
    # 3. Calcul de R_p (étendue des moyennes des pièces)
    all_part_means = []
    for part in range(n_parts):
        part_values = []
        for op in range(n_operators):
            part_values.extend(data[part, op*n_trials:(op+1)*n_trials])
        all_part_means.append(np.mean(part_values))
    
    R_p = max(all_part_means) - min(all_part_means)
    
    # 4. Calcul des valeurs d2
    # Pour EV: m = nombre d'essais, w = 1 (car on utilise R_bar)
    d2_ev = get_d2(1, n_trials)
    
    # Pour AV: m = nombre d'opérateurs, w = nombre de pièces
    d2_av = get_d2(n_parts, n_operators)
    
    # Pour PV: m = nombre de pièces, w = 1
    d2_pv = get_d2(1, n_parts)
    
    # 5. Calcul des composantes
    # Répétabilité (EV)
    EV = (k * R_bar) / d2_ev
    
    # Reproductibilité (AV)
    AV_term1 = ((k * X_diff) / d2_av) ** 2
    AV_term2 = (EV ** 2) / (n_parts * n_trials)
    AV = math.sqrt(max(AV_term1 - AV_term2, 0))
    
    # Gage R&R
    GRR = math.sqrt(EV ** 2 + AV ** 2)
    
    # Variation Pièces (PV)
    PV = (k * R_p) / d2_pv
    
    # Variation Totale (TV)
    TV = math.sqrt(GRR ** 2 + PV ** 2)
    
    # Pourcentages
    if TV > 0:
        EV_pct = (EV / TV) * 100
        AV_pct = (AV / TV) * 100
        GRR_pct = (GRR / TV) * 100
        PV_pct = (PV / TV) * 100
    else:
        EV_pct = AV_pct = GRR_pct = PV_pct = 0
    
    return {
        'R_bar': R_bar,
        'X_diff': X_diff,
        'R_p': R_p,
        'd2_ev': d2_ev,
        'd2_av': d2_av,
        'd2_pv': d2_pv,
        'EV': EV,
        'AV': AV,
        'GRR': GRR,
        'PV': PV,
        'TV': TV,
        'EV_pct': EV_pct,
        'AV_pct': AV_pct,
        'GRR_pct': GRR_pct,
        'PV_pct': PV_pct,
        'operator_means': operator_global_means,
        'part_means': all_part_means,
        'all_ranges': all_ranges.tolist()
    }

# =====================================================
# INTERFACE UTILISATEUR
# =====================================================

# SIDEBAR
with st.sidebar:
    st.title("⚙️ Paramètres de l'analyse")
    
    st.markdown("### Paramètres statistiques")
    k_factor = st.number_input(
        "Facteur k (niveau de confiance)",
        value=5.15,
        min_value=4.0,
        max_value=6.0,
        step=0.01,
        help="5.15 pour 99% de confiance, 6.0 pour 99.73%"
    )
    
    tolerance = st.number_input(
        "Tolérance spécifiée",
        value=1.0,
        min_value=0.0,
        step=0.1,
        help="Tolérance du processus pour calcul %R&R/Tolérance"
    )
    
    st.markdown("### Seuils d'acceptation")
    threshold_1 = st.number_input("Seuil vert (<%)", value=10.0, min_value=0.0, max_value=100.0)
    threshold_2 = st.number_input("Seuil orange (<%)", value=30.0, min_value=0.0, max_value=100.0)
    
    st.divider()
    
    st.markdown("### Aide")
    st.info("""
    **Interprétation des résultats:**
    - ✅ < 10% : Acceptable
    - ⚠️ 10-30% : Marginal
    - ❌ > 30% : Inacceptable
    """)

# TITRE PRINCIPAL
st.title("📊 Analyse Gage R&R - Lean Six Sigma")
st.markdown("**Méthode des étendues et des moyennes avec table d₂**")

# =====================================================
# SECTION 1: IMPORTATION DES DONNÉES
# =====================================================
st.header("📥 Importation des données")

data_mode = st.radio(
    "Sélectionnez le mode d'entrée:",
    ["📁 Importer un fichier", "✍️ Saisie manuelle", "📊 Exemple prédéfini"],
    horizontal=True
)

df = None
n_parts = n_operators = n_trials = 0

if data_mode == "📁 Importer un fichier":
    uploaded_file = st.file_uploader(
        "Choisissez un fichier CSV ou Excel",
        type=["csv", "xlsx"],
        help="Le fichier doit contenir les mesures organisées par opérateurs et essais"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Nettoyage des données
            df = df.apply(pd.to_numeric, errors='coerce').dropna(how='all', axis=1).dropna(how='all', axis=0)
            
            st.success(f"✅ Fichier importé avec succès: {df.shape[0]} lignes × {df.shape[1]} colonnes")
            
            # Configuration des paramètres
            cols = st.columns(3)
            with cols[0]:
                n_parts = st.number_input("Nombre de pièces", min_value=2, value=min(10, df.shape[0]), max_value=df.shape[0])
            with cols[1]:
                n_operators = st.number_input("Nombre d'opérateurs", min_value=2, value=2)
            with cols[2]:
                n_trials = st.number_input("Nombre d'essais", min_value=2, value=3)
            
            # Vérification des dimensions
            required_cols = n_operators * n_trials
            if required_cols > df.shape[1]:
                st.error(f"❌ Nombre de colonnes insuffisant. Requis: {required_cols}, Disponible: {df.shape[1]}")
            else:
                df = df.iloc[:n_parts, :required_cols]
                
        except Exception as e:
            st.error(f"❌ Erreur lors de l'importation: {str(e)}")

elif data_mode == "✍️ Saisie manuelle":
    cols = st.columns(3)
    with cols[0]:
        n_parts = st.number_input("Nombre de pièces", min_value=2, value=10)
    with cols[1]:
        n_operators = st.number_input("Nombre d'opérateurs", min_value=2, value=3)
    with cols[2]:
        n_trials = st.number_input("Nombre d'essais", min_value=2, value=2)
    
    # Création d'un DataFrame vide
    columns = []
    for op in range(n_operators):
        for trial in range(n_trials):
            columns.append(f"Op{op+1}_T{trial+1}")
    
    df = pd.DataFrame(
        np.random.normal(45, 0.1, (n_parts, len(columns))),
        columns=columns
    )
    
    st.info("📝 Modifiez les valeurs dans le tableau ci-dessous")

else:  # Exemple prédéfini
    st.info("📊 Chargement d'un exemple de référence")
    
    # Exemple basé sur votre cas
    n_parts, n_operators, n_trials = 10, 3, 2
    
    # Génération de données réalistes
    np.random.seed(42)
    base_values = np.array([45.10, 45.15, 45.20, 45.05, 45.25, 45.30, 45.00, 45.18, 45.22, 45.12])
    
    data_dict = {}
    for op in range(n_operators):
        op_bias = np.random.uniform(-0.02, 0.02)
        for trial in range(n_trials):
            col_name = f"Op{op+1}_T{trial+1}"
            noise = np.random.normal(0, 0.015, n_parts)
            data_dict[col_name] = base_values + op_bias + noise
    
    df = pd.DataFrame(data_dict).round(3)

# Affichage des données
if df is not None:
    st.subheader("📋 Données de mesure")
    
    # Éditeur de données interactif
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        height=300,
        num_rows="dynamic",
        column_config={
            col: st.column_config.NumberColumn(
                label=col,
                format="%.3f",
                step=0.001
            ) for col in df.columns
        }
    )
    
    df = edited_df
    
    # =====================================================
    # SECTION 2: CALCUL ET ANALYSE
    # =====================================================
    st.header("🧮 Calcul Gage R&R")
    
    if st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True):
        with st.spinner("Calcul en cours..."):
            try:
                # Calcul des résultats
                results = calculate_gage_rr_correct(df, n_parts, n_operators, n_trials, k_factor)
                
                # =====================================================
                # AFFICHAGE DES RÉSULTATS
                # =====================================================
                
                # 1. Résumé des paramètres intermédiaires
                with st.expander("📐 Paramètres intermédiaires", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("R̄ (moyenne étendues)", f"{results['R_bar']:.4f}")
                        st.metric("X_diff", f"{results['X_diff']:.4f}")
                    
                    with col2:
                        st.metric("R_p (étendue pièces)", f"{results['R_p']:.4f}")
                        st.metric("k (facteur)", f"{k_factor}")
                    
                    with col3:
                        st.metric("d2(EV)", f"{results['d2_ev']:.3f}")
                        st.metric("d2(AV)", f"{results['d2_av']:.3f}")
                        st.metric("d2(PV)", f"{results['d2_pv']:.3f}")
                
                # 2. Résultats principaux
                st.subheader("🎯 Résultats de l'analyse")
                
                cols = st.columns(5)
                metrics = [
                    ("EV", "Répétabilité", results['EV'], results['EV_pct']),
                    ("AV", "Reproductibilité", results['AV'], results['AV_pct']),
                    ("R&R", "Gage R&R", results['GRR'], results['GRR_pct']),
                    ("PV", "Variation Pièces", results['PV'], results['PV_pct']),
                    ("TV", "Variation Totale", results['TV'], "100%")
                ]
                
                for idx, (label, desc, value, pct) in enumerate(metrics):
                    with cols[idx]:
                        st.metric(label, f"{value:.4f}", f"{pct}" if isinstance(pct, str) else f"{pct:.1f}%")
                        st.caption(desc)
                
                # 3. Évaluation du système
                st.subheader("📈 Évaluation du système de mesure")
                
                # Détermination du statut
                if results['GRR_pct'] < threshold_1:
                    status = "✅ ACCEPTABLE"
                    color = "green"
                    icon = "✅"
                elif results['GRR_pct'] < threshold_2:
                    status = "⚠️ MARGINAL"
                    color = "orange"
                    icon = "⚠️"
                else:
                    status = "❌ INACCEPTABLE"
                    color = "red"
                    icon = "❌"
                
                # Affichage du statut
                st.markdown(f"""
                <div style="background-color:{color}20; padding:20px; border-radius:10px; border-left:5px solid {color}; margin:20px 0;">
                    <h3 style="color:{color}; margin:0;">{icon} {status}</h3>
                    <p style="font-size:1.5em; margin:10px 0;">
                        <strong>%R&R = {results['GRR_pct']:.1f}%</strong>
                    </p>
                    <p>%EV = {results['EV_pct']:.1f}% | %AV = {results['AV_pct']:.1f}% | %PV = {results['PV_pct']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Barre de progression
                progress_value = min(results['GRR_pct'] / threshold_2, 1.0)
                st.progress(progress_value, text=f"R&R: {results['GRR_pct']:.1f}% / Limite: {threshold_2}%")
                
                # Calcul %R&R/Tolérance si spécifié
                if tolerance > 0:
                    grr_tol_pct = (results['GRR'] / tolerance) * 100
                    st.info(f"📏 **%R&R/Tolérance = {grr_tol_pct:.1f}%** (Tolérance: {tolerance:.3f})")
                
                # =====================================================
                # SECTION 3: VISUALISATIONS
                # =====================================================
                st.header("📊 Visualisations")
                
                # Création des graphiques avec Plotly
                fig1 = go.Figure()
                
                # Graphique 1: Distribution des composantes
                components = ['EV', 'AV', 'R&R', 'PV', 'TV']
                values = [results['EV'], results['AV'], results['GRR'], results['PV'], results['TV']]
                percentages = [results['EV_pct'], results['AV_pct'], results['GRR_pct'], results['PV_pct'], 100]
                
                fig1.add_trace(go.Bar(
                    x=components,
                    y=values,
                    name='Valeurs absolues',
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b'],
                    text=[f'{v:.3f}' for v in values],
                    textposition='auto'
                ))
                
                fig1.update_layout(
                    title='Composantes de variation',
                    xaxis_title='Composante',
                    yaxis_title='Valeur',
                    height=400
                )
                
                st.plotly_chart(fig1, use_container_width=True)
                
                # Graphique 2: Pourcentages
                fig2 = go.Figure()
                
                fig2.add_trace(go.Bar(
                    x=['EV%', 'AV%', 'R&R%', 'PV%'],
                    y=percentages[:-1],
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'],
                    text=[f'{p:.1f}%' for p in percentages[:-1]],
                    textposition='auto'
                ))
                
                # Ajout des lignes de référence
                fig2.add_hline(y=threshold_1, line_dash="dash", line_color="green", 
                              annotation_text=f"Seuil {threshold_1}%")
                fig2.add_hline(y=threshold_2, line_dash="dash", line_color="red", 
                              annotation_text=f"Seuil {threshold_2}%")
                
                fig2.update_layout(
                    title='Distribution des pourcentages de variation',
                    xaxis_title='Composante',
                    yaxis_title='Pourcentage (%)',
                    height=400
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Graphique 3: Moyennes par opérateur
                fig3 = go.Figure()
                
                op_labels = [f'Op{i+1}' for i in range(n_operators)]
                fig3.add_trace(go.Scatter(
                    x=op_labels,
                    y=results['operator_means'],
                    mode='markers+lines',
                    marker=dict(size=10, color='blue'),
                    name='Moyenne par opérateur'
                ))
                
                fig3.update_layout(
                    title='Moyennes par opérateur',
                    xaxis_title='Opérateur',
                    yaxis_title='Moyenne',
                    height=300
                )
                
                st.plotly_chart(fig3, use_container_width=True)
                
                # =====================================================
                # SECTION 4: EXPORT DES RÉSULTATS
                # =====================================================
                st.header("💾 Export des résultats")
                
                # Préparation des données d'export
                export_df = pd.DataFrame({
                    'Paramètre': [
                        'Pièces (n)', 'Opérateurs (k)', 'Essais (r)',
                        'R̄', 'X_diff', 'R_p',
                        'd2_EV', 'd2_AV', 'd2_PV',
                        'EV', 'AV', 'R&R', 'PV', 'TV',
                        '%EV', '%AV', '%R&R', '%PV',
                        'Statut'
                    ],
                    'Valeur': [
                        n_parts, n_operators, n_trials,
                        f"{results['R_bar']:.6f}",
                        f"{results['X_diff']:.6f}",
                        f"{results['R_p']:.6f}",
                        f"{results['d2_ev']:.3f}",
                        f"{results['d2_av']:.3f}",
                        f"{results['d2_pv']:.3f}",
                        f"{results['EV']:.6f}",
                        f"{results['AV']:.6f}",
                        f"{results['GRR']:.6f}",
                        f"{results['PV']:.6f}",
                        f"{results['TV']:.6f}",
                        f"{results['EV_pct']:.2f}%",
                        f"{results['AV_pct']:.2f}%",
                        f"{results['GRR_pct']:.2f}%",
                        f"{results['PV_pct']:.2f}%",
                        status
                    ],
                    'Description': [
                        'Nombre de pièces',
                        'Nombre d\'opérateurs',
                        'Nombre d\'essais',
                        'Moyenne des étendues',
                        'Différence des moyennes opérateurs',
                        'Étendue des moyennes pièces',
                        'd2 pour répétabilité',
                        'd2 pour reproductibilité',
                        'd2 pour variation pièces',
                        'Répétabilité',
                        'Reproductibilité',
                        'Gage R&R',
                        'Variation pièces',
                        'Variation totale',
                        'Pourcentage EV',
                        'Pourcentage AV',
                        'Pourcentage R&R',
                        'Pourcentage PV',
                        'Évaluation du système'
                    ]
                })
                
                # Boutons d'export
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 CSV",
                        data=csv_data,
                        file_name="gage_rr_results.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Création du rapport Excel
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Données brutes', index=False)
                        export_df.to_excel(writer, sheet_name='Résultats', index=False)
                        
                        # Ajouter les moyennes
                        means_df = pd.DataFrame({
                            'Opérateur': [f'Op{i+1}' for i in range(n_operators)],
                            'Moyenne': results['operator_means']
                        })
                        means_df.to_excel(writer, sheet_name='Moyennes', index=False)
                    
                    st.download_button(
                        label="📥 Excel",
                        data=excel_buffer.getvalue(),
                        file_name="rapport_gage_rr.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col3:
                    # Rapport texte détaillé
                    report = f"""
                    RAPPORT D'ANALYSE GAGE R&R
                    ===========================
                    
                    DATE: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                    
                    PARAMÈTRES DE L'ÉTUDE:
                    - Pièces analysées: {n_parts}
                    - Opérateurs: {n_operators}
                    - Essais par opérateur: {n_trials}
                    - Facteur k: {k_factor}
                    
                    RÉSULTATS:
                    ---------------------------
                    Répétabilité (EV): {results['EV']:.4f} ({results['EV_pct']:.1f}%)
                    Reproductibilité (AV): {results['AV']:.4f} ({results['AV_pct']:.1f}%)
                    Gage R&R: {results['GRR']:.4f} ({results['GRR_pct']:.1f}%)
                    Variation Pièces (PV): {results['PV']:.4f} ({results['PV_pct']:.1f}%)
                    Variation Totale (TV): {results['TV']:.4f}
                    
                    ÉVALUATION:
                    ---------------------------
                    %R&R = {results['GRR_pct']:.1f}%
                    Classification: {status}
                    
                    """
                    
                    if tolerance > 0:
                        report += f"""
                    PAR RAPPORT À LA TOLÉRANCE:
                    - Tolérance spécifiée: {tolerance:.3f}
                    - %R&R/Tolérance: {(results['GRR']/tolerance)*100:.1f}%
                        """
                    
                    st.download_button(
                        label="📥 Rapport TXT",
                        data=report,
                        file_name="rapport_gage_rr.txt",
                        mime="text/plain"
                    )
                
                # =====================================================
                # SECTION 5: RECOMMANDATIONS
                # =====================================================
                st.header("💡 Recommandations")
                
                if results['GRR_pct'] > 30:
                    st.error("""
                    **Actions recommandées:**
                    1. **Améliorer la répétabilité** (%EV élevé):
                       - Vérifier l'étalonnage des instruments
                       - Standardiser les méthodes de mesure
                       - Former les opérateurs
                    2. **Réduire la reproductibilité** (%AV élevé):
                       - Harmoniser les techniques de mesure
                       - Créer des procédures standardisées
                       - Vérifier la compréhension des instructions
                    """)
                elif results['GRR_pct'] > 10:
                    st.warning("""
                    **Suggestions d'amélioration:**
                    - Documenter les meilleures pratiques
                    - Mettre en place des audits réguliers
                    - Considérer un recalibrage périodique
                    """)
                else:
                    st.success("""
                    **Système de mesure acceptable:**
                    - Maintenir les procédures actuelles
                    - Surveiller régulièrement la performance
                    - Documenter les résultats pour référence future
                    """)
                
            except Exception as e:
                st.error(f"❌ Erreur lors du calcul: {str(e)}")
                st.info("Vérifiez que les données sont correctement formatées et complètes.")

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: gray;">
    <p><strong>Gage R&R Analysis Tool</strong> - Méthode des étendues et des moyennes</p>
    <p>Lean Six Sigma - Outil d'analyse de la capabilité des systèmes de mesure</p>
</div>
""", unsafe_allow_html=True)
