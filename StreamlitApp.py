import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# TABLE d2 CORRIGÉE (basée sur votre image)
# =====================================================
# Table d2 complète basée sur l'image fournie
D2_TABLE_COMPLETE = {
    1: {1: 1.41, 2: 1.91, 3: 2.24, 4: 2.48, 5: 2.67, 6: 2.83, 7: 2.96, 8: 3.08, 9: 3.18, 10: 3.27, 11: 3.35, 12: 3.42, 13: 3.49, 14: 3.55, 15: 3.61},
    2: {1: 1.28, 2: 1.81, 3: 2.15, 4: 2.40, 5: 2.60, 6: 2.77, 7: 2.91, 8: 3.02, 9: 3.13, 10: 3.22, 11: 3.30, 12: 3.38, 13: 3.45, 14: 3.51, 15: 3.57},
    3: {1: 1.23, 2: 1.77, 3: 2.12, 4: 2.38, 5: 2.58, 6: 2.75, 7: 2.89, 8: 3.01, 9: 3.11, 10: 3.21, 11: 3.29, 12: 3.37, 13: 3.43, 14: 3.50, 15: 3.56},
    4: {1: 1.21, 2: 1.75, 3: 2.11, 4: 2.37, 5: 2.57, 6: 2.74, 7: 2.88, 8: 3.00, 9: 3.10, 10: 3.20, 11: 3.28, 12: 3.36, 13: 3.43, 14: 3.49, 15: 3.55},
    5: {1: 1.19, 2: 1.74, 3: 2.10, 4: 2.36, 5: 2.56, 6: 2.73, 7: 2.87, 8: 2.99, 9: 3.10, 10: 3.19, 11: 3.28, 12: 3.36, 13: 3.42, 14: 3.49, 15: 3.55},
    6: {1: 1.18, 2: 1.73, 3: 2.09, 4: 2.35, 5: 2.56, 6: 2.73, 7: 2.87, 8: 2.99, 9: 3.10, 10: 3.19, 11: 3.27, 12: 3.35, 13: 3.42, 14: 3.49, 15: 3.55},
    7: {1: 1.17, 2: 1.73, 3: 2.09, 4: 2.35, 5: 2.55, 6: 2.72, 7: 2.87, 8: 2.99, 9: 3.10, 10: 3.19, 11: 3.27, 12: 3.35, 13: 3.42, 14: 3.48, 15: 3.54},
    8: {1: 1.17, 2: 1.72, 3: 2.08, 4: 2.35, 5: 2.55, 6: 2.72, 7: 2.87, 8: 2.98, 9: 3.09, 10: 3.19, 11: 3.27, 12: 3.35, 13: 3.42, 14: 3.48, 15: 3.54},
    9: {1: 1.16, 2: 1.72, 3: 2.08, 4: 2.34, 5: 2.55, 6: 2.72, 7: 2.86, 8: 2.98, 9: 3.09, 10: 3.19, 11: 3.27, 12: 3.35, 13: 3.42, 14: 3.48, 15: 3.54},
    10: {1: 1.16, 2: 1.72, 3: 2.08, 4: 2.34, 5: 2.55, 6: 2.72, 7: 2.86, 8: 2.98, 9: 3.09, 10: 3.18, 11: 3.27, 12: 3.34, 13: 3.42, 14: 3.48, 15: 3.54},
    11: {1: 1.15, 2: 1.71, 3: 2.08, 4: 2.34, 5: 2.55, 6: 2.72, 7: 2.86, 8: 2.98, 9: 3.09, 10: 3.18, 11: 3.27, 12: 3.34, 13: 3.41, 14: 3.48, 15: 3.54},
    12: {1: 1.15, 2: 1.71, 3: 2.07, 4: 2.34, 5: 2.55, 6: 2.72, 7: 2.85, 8: 2.98, 9: 3.09, 10: 3.18, 11: 3.27, 12: 3.34, 13: 3.41, 14: 3.48, 15: 3.54},
    13: {1: 1.15, 2: 1.71, 3: 2.07, 4: 2.34, 5: 2.55, 6: 2.71, 7: 2.85, 8: 2.98, 9: 3.09, 10: 3.18, 11: 3.27, 12: 3.34, 13: 3.41, 14: 3.48, 15: 3.53},
    14: {1: 1.15, 2: 1.71, 3: 2.07, 4: 2.34, 5: 2.54, 6: 2.71, 7: 2.85, 8: 2.98, 9: 3.09, 10: 3.18, 11: 3.27, 12: 3.34, 13: 3.41, 14: 3.48, 15: 3.53},
    15: {1: 1.15, 2: 1.71, 3: 2.07, 4: 2.34, 5: 2.54, 6: 2.71, 7: 2.85, 8: 2.98, 9: 3.08, 10: 3.18, 11: 3.26, 12: 3.34, 13: 3.41, 14: 3.48, 15: 3.53}
}

# Valeurs pour W > 15 (dernière ligne du tableau)
D2_TABLE_LARGE = {
    1: 1.128, 2: 1.693, 3: 2.059, 4: 2.326, 5: 2.534, 6: 2.704, 7: 2.847, 8: 2.970,
    9: 3.078, 10: 3.173, 11: 3.258, 12: 3.336, 13: 3.407, 14: 3.472
}

def get_d2(z, w):
    """
    Retourne la valeur d2 pour:
    - z = nombre d'échantillons (première colonne du tableau)
    - w = taille de l'échantillon (en-tête du tableau)
    
    Selon votre tableau, pour Gage R&R:
    - Pour EV (répétabilité): z = 1, w = nombre d'essais
    - Pour AV (reproductibilité): z = nombre de pièces, w = nombre d'opérateurs
    - Pour PV (variation pièces): z = 1, w = nombre de pièces
    """
    # Pour w > 15, utiliser la dernière ligne du tableau
    if w > 15:
        # Pour z > 15 aussi, utiliser les valeurs de la dernière ligne
        if z > 15:
            z = 15
        # Pour w > 15, retourner la valeur de D2_TABLE_LARGE
        # On prend la valeur pour le w donné, ou la plus proche si > 14
        if w in D2_TABLE_LARGE:
            return D2_TABLE_LARGE[w]
        else:
            # Pour w > 14, utiliser la dernière valeur disponible
            return D2_TABLE_LARGE[14]
    
    # Pour z > 15, utiliser z = 15
    if z > 15:
        z = 15
    
    # Chercher la valeur dans la table complète
    if z in D2_TABLE_COMPLETE and w in D2_TABLE_COMPLETE[z]:
        return D2_TABLE_COMPLETE[z][w]
    elif z in D2_TABLE_COMPLETE and w <= 15:
        # Si w existe dans le tableau pour ce z
        # Trouver la valeur la plus proche
        available_w = [k for k in D2_TABLE_COMPLETE[z].keys() if k <= w]
        if available_w:
            closest_w = max(available_w)
            return D2_TABLE_COMPLETE[z][closest_w]
    
    # Fallback: utiliser la valeur pour z=1
    if w in D2_TABLE_COMPLETE.get(1, {}):
        return D2_TABLE_COMPLETE[1][w]
    elif w <= 15:
        # Approximation linéaire
        return 1.0 + (w - 1) * 0.15
    else:
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
    
    # 4. Calcul des valeurs d2 AVEC LA NOUVELLE MÉTHODE
    # Pour EV: z = 1 (car on utilise R_bar), w = nombre d'essais
    d2_ev = get_d2(1, n_trials)
    
    # Pour AV: z = nombre de pièces, w = nombre d'opérateurs
    d2_av = get_d2(n_parts, n_operators)
    
    # Pour PV: z = 1 (car on utilise R_p), w = nombre de pièces
    d2_pv = get_d2(1, n_parts)
    
    # 5. Calcul des composantes
    # Répétabilité (EV)
    if d2_ev > 0:
        EV = (k * R_bar) / d2_ev
    else:
        EV = 0
    
    # Reproductibilité (AV)
    if d2_av > 0:
        AV_term1 = ((k * X_diff) / d2_av) ** 2
        AV_term2 = (EV ** 2) / (n_parts * n_trials)
        AV = math.sqrt(max(AV_term1 - AV_term2, 0))
    else:
        AV = 0
    
    # Gage R&R
    GRR = math.sqrt(EV ** 2 + AV ** 2)
    
    # Variation Pièces (PV)
    if d2_pv > 0:
        PV = (k * R_p) / d2_pv
    else:
        PV = 0
    
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
    
    # Afficher un aperçu de la table d2
    with st.expander("📊 Aperçu de la table d2"):
        st.caption("Valeurs d2 pour Z=1 (utilisées pour EV et PV):")
        df_d2_preview = pd.DataFrame({
            'W': list(range(1, 16)),
            'd2': [D2_TABLE_COMPLETE[1].get(i, 0) for i in range(1, 16)]
        })
        st.dataframe(df_d2_preview, hide_index=True, use_container_width=True)
        st.caption(f"Pour W>15: {D2_TABLE_LARGE[14]:.3f} (valeur maximale)")
    
    st.divider()
    
    st.markdown("### Aide")
    st.info("""
    **Interprétation des résultats:**
    - ✅ < 10% : Acceptable
    - ⚠️ 10-30% : Marginal
    - ❌ > 30% : Inacceptable
    
    **Valeurs d2:**
    - EV: d2(1, nombre d'essais)
    - AV: d2(nombre de pièces, nombre d'opérateurs)
    - PV: d2(1, nombre de pièces)
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
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("R̄ (moyenne étendues)", f"{results['R_bar']:.6f}")
                        st.metric("X_diff", f"{results['X_diff']:.6f}")
                    
                    with col2:
                        st.metric("R_p (étendue pièces)", f"{results['R_p']:.6f}")
                        st.metric("k (facteur)", f"{k_factor}")
                    
                    with col3:
                        st.metric("d2(EV)", f"{results['d2_ev']:.6f}")
                        st.metric("d2(AV)", f"{results['d2_av']:.6f}")
                    
                    with col4:
                        st.metric("d2(PV)", f"{results['d2_pv']:.6f}")
                        st.metric("Pièces/Op/Essais", f"{n_parts}/{n_operators}/{n_trials}")
                
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
                        st.metric(label, f"{value:.6f}", f"{pct}" if isinstance(pct, str) else f"{pct:.2f}%")
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
                        <strong>%R&R = {results['GRR_pct']:.2f}%</strong>
                    </p>
                    <p>%EV = {results['EV_pct']:.2f}% | %AV = {results['AV_pct']:.2f}% | %PV = {results['PV_pct']:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Barre de progression
                progress_value = min(results['GRR_pct'] / threshold_2, 1.0)
                st.progress(progress_value, text=f"R&R: {results['GRR_pct']:.2f}% / Limite: {threshold_2}%")
                
                # Calcul %R&R/Tolérance si spécifié
                if tolerance > 0:
                    grr_tol_pct = (results['GRR'] / tolerance) * 100
                    st.info(f"📏 **%R&R/Tolérance = {grr_tol_pct:.2f}%** (Tolérance: {tolerance:.6f})")
                
                # =====================================================
                # SECTION 3: VISUALISATIONS (avec Matplotlib)
                # =====================================================
                st.header("📊 Visualisations")
                
                # Création des graphiques avec Matplotlib
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle('Analyse Gage R&R - Résultats', fontsize=16)
                
                # Graphique 1: Composantes de variation
                ax1 = axes[0, 0]
                components = ['EV', 'AV', 'R&R', 'PV', 'TV']
                values = [results['EV'], results['AV'], results['GRR'], results['PV'], results['TV']]
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
                
                bars1 = ax1.bar(components, values, color=colors, alpha=0.8)
                ax1.set_ylabel('Valeur')
                ax1.set_title('Composantes de variation (absolues)')
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Ajout des valeurs sur les barres
                for bar, val in zip(bars1, values):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2, height, f'{val:.4f}',
                            ha='center', va='bottom', fontsize=9)
                
                # Graphique 2: Pourcentages
                ax2 = axes[0, 1]
                comps_pct = ['EV%', 'AV%', 'R&R%', 'PV%']
                vals_pct = [results['EV_pct'], results['AV_pct'], results['GRR_pct'], results['PV_pct']]
                colors_pct = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
                
                bars2 = ax2.bar(comps_pct, vals_pct, color=colors_pct, alpha=0.8)
                ax2.set_ylabel('Pourcentage (%)')
                ax2.set_title('Distribution des variations (%)')
                ax2.axhline(y=threshold_1, color='green', linestyle='--', alpha=0.7, label=f'Seuil {threshold_1}%')
                ax2.axhline(y=threshold_2, color='red', linestyle='--', alpha=0.7, label=f'Seuil {threshold_2}%')
                ax2.grid(True, alpha=0.3, axis='y')
                ax2.legend()
                
                for bar, val in zip(bars2, vals_pct):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2, height, f'{val:.1f}%',
                            ha='center', va='bottom', fontsize=9)
                
                # Graphique 3: Moyennes par opérateur
                ax3 = axes[1, 0]
                op_indices = np.arange(n_operators)
                ax3.bar(op_indices, results['operator_means'], color='skyblue', alpha=0.7)
                ax3.set_xlabel('Opérateur')
                ax3.set_ylabel('Moyenne')
                ax3.set_title('Moyennes globales par opérateur')
                ax3.set_xticks(op_indices)
                ax3.set_xticklabels([f'Op{i+1}' for i in op_indices])
                ax3.grid(True, alpha=0.3, axis='y')
                
                for i, mean in enumerate(results['operator_means']):
                    ax3.text(i, mean, f'{mean:.4f}', ha='center', va='bottom')
                
                # Graphique 4: Moyennes par pièce
                ax4 = axes[1, 1]
                part_indices = np.arange(n_parts)
                ax4.plot(part_indices, results['part_means'], 'o-', color='green', linewidth=2)
                ax4.set_xlabel('Pièce')
                ax4.set_ylabel('Moyenne')
                ax4.set_title('Moyennes par pièce (tous opérateurs)')
                ax4.set_xticks(part_indices)
                ax4.set_xticklabels([f'P{i+1}' for i in part_indices], rotation=45)
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # =====================================================
                # SECTION 4: EXPORT DES RÉSULTATS
                # =====================================================
                st.header("💾 Export des résultats")
                
                # Préparation des données d'export
                export_df = pd.DataFrame({
                    'Paramètre': [
                        'Pièces (n)', 'Opérateurs (k)', 'Essais (r)',
                        'R̄ (moyenne étendues)', 'X_diff (diff moyennes)', 'R_p (étendue pièces)',
                        'd2_EV', 'd2_AV', 'd2_PV',
                        'EV (Répétabilité)', 'AV (Reproductibilité)', 'R&R (Gage R&R)', 
                        'PV (Variation Pièces)', 'TV (Variation Totale)',
                        '%EV', '%AV', '%R&R', '%PV',
                        'Statut', 'k facteur'
                    ],
                    'Valeur': [
                        n_parts, n_operators, n_trials,
                        f"{results['R_bar']:.6f}",
                        f"{results['X_diff']:.6f}",
                        f"{results['R_p']:.6f}",
                        f"{results['d2_ev']:.6f}",
                        f"{results['d2_av']:.6f}",
                        f"{results['d2_pv']:.6f}",
                        f"{results['EV']:.6f}",
                        f"{results['AV']:.6f}",
                        f"{results['GRR']:.6f}",
                        f"{results['PV']:.6f}",
                        f"{results['TV']:.6f}",
                        f"{results['EV_pct']:.2f}%",
                        f"{results['AV_pct']:.2f}%",
                        f"{results['GRR_pct']:.2f}%",
                        f"{results['PV_pct']:.2f}%",
                        status,
                        f"{k_factor}"
                    ],
                    'Description': [
                        'Nombre de pièces',
                        'Nombre d\'opérateurs',
                        'Nombre d\'essais',
                        'Moyenne des étendues par opérateur et pièce',
                        'Différence entre les moyennes maximales et minimales des opérateurs',
                        'Étendue des moyennes de toutes les pièces',
                        'Facteur d2 pour la répétabilité (z=1, w=essais)',
                        'Facteur d2 pour la reproductibilité (z=pièces, w=opérateurs)',
                        'Facteur d2 pour la variation pièces (z=1, w=pièces)',
                        'Équipment Variation (Répétabilité)',
                        'Appraiser Variation (Reproductibilité)',
                        'Gage Repeatability & Reproducibility',
                        'Part Variation (Variation entre pièces)',
                        'Total Variation',
                        'Pourcentage de répétabilité',
                        'Pourcentage de reproductibilité',
                        'Pourcentage de Gage R&R',
                        'Pourcentage de variation pièces',
                        'Évaluation du système de mesure',
                        'Facteur k de niveau de confiance'
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
                        
                        # Ajouter la table d2 utilisée
                        d2_used_df = pd.DataFrame({
                            'Calcul': ['EV', 'AV', 'PV'],
                            'z (échantillons)': [1, n_parts, 1],
                            'w (taille)': [n_trials, n_operators, n_parts],
                            'd2 valeur': [results['d2_ev'], results['d2_av'], results['d2_pv']]
                        })
                        d2_used_df.to_excel(writer, sheet_name='Valeurs d2 utilisées', index=False)
                    
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
                    
                    VALEURS d2 UTILISÉES:
                    - EV (Répétabilité): d2(z=1, w={n_trials}) = {results['d2_ev']:.4f}
                    - AV (Reproductibilité): d2(z={n_parts}, w={n_operators}) = {results['d2_av']:.4f}
                    - PV (Variation Pièces): d2(z=1, w={n_parts}) = {results['d2_pv']:.4f}
                    
                    RÉSULTATS INTERMÉDIAIRES:
                    ---------------------------
                    R̄ (moyenne étendues): {results['R_bar']:.6f}
                    X_diff (différence moyennes): {results['X_diff']:.6f}
                    R_p (étendue pièces): {results['R_p']:.6f}
                    
                    RÉSULTATS FINAUX:
                    ---------------------------
                    Répétabilité (EV): {results['EV']:.6f} ({results['EV_pct']:.2f}%)
                    Reproductibilité (AV): {results['AV']:.6f} ({results['AV_pct']:.2f}%)
                    Gage R&R: {results['GRR']:.6f} ({results['GRR_pct']:.2f}%)
                    Variation Pièces (PV): {results['PV']:.6f} ({results['PV_pct']:.2f}%)
                    Variation Totale (TV): {results['TV']:.6f}
                    
                    ÉVALUATION:
                    ---------------------------
                    %R&R = {results['GRR_pct']:.2f}%
                    Classification: {status}
                    
                    MOYENNES PAR OPÉRATEUR:
                    """
                    
                    for i, mean in enumerate(results['operator_means']):
                        report += f"\n  - Opérateur {i+1}: {mean:.4f}"
                    
                    if tolerance > 0:
                        report += f"""
                    
                    PAR RAPPORT À LA TOLÉRANCE:
                    - Tolérance spécifiée: {tolerance:.6f}
                    - %R&R/Tolérance: {(results['GRR']/tolerance)*100:.2f}%
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
                    **Actions recommandées (Système INACCEPTABLE):**
                    
                    1. **Si %EV est élevé (>20%):**
                       - Vérifier l'étalonnage des instruments
                       - Standardiser les méthodes de mesure
                       - Former les opérateurs sur l'utilisation correcte
                       - Vérifier la stabilité de l'équipement
                    
                    2. **Si %AV est élevé (>20%):**
                       - Harmoniser les techniques de mesure entre opérateurs
                       - Créer des procédures standardisées détaillées
                       - Vérifier la compréhension des instructions
                       - Mettre en place des formations communes
                    
                    3. **Actions générales:**
                       - Revoir le système de mesure complet
                       - Considérer un équipement plus précis
                       - Augmenter le nombre d'essais ou d'opérateurs
                       - Améliorer la formation des opérateurs
                    """)
                elif results['GRR_pct'] > 10:
                    st.warning("""
                    **Suggestions d'amélioration (Système MARGINAL):**
                    
                    1. **Actions correctives:**
                       - Documenter les meilleures pratiques
                       - Mettre en place des audits réguliers du processus de mesure
                       - Considérer un recalibrage périodique plus fréquent
                       - Standardiser les conditions de mesure (température, humidité, etc.)
                    
                    2. **Surveillance:**
                       - Surveiller régulièrement la performance du système
                       - Mettre en place des contrôles statistiques du processus de mesure
                       - Documenter les dérives potentielles
                    
                    3. **Amélioration continue:**
                       - Recueillir les retours des opérateurs
                       - Identifier les sources de variation résiduelles
                       - Planifier des améliorations incrémentales
                    """)
                else:
                    st.success("""
                    **Système de mesure ACCEPTABLE:**
                    
                    1. **Maintenance:**
                       - Maintenir les procédures actuelles
                       - Continuer le programme d'étalonnage régulier
                       - Documenter les résultats pour référence future
                    
                    2. **Surveillance:**
                       - Surveiller régulièrement la performance
                       - Mettre en place des indicateurs de performance clés
                       - Réviser périodiquement les procédures
                    
                    3. **Amélioration continue:**
                       - Identifier les opportunités d'amélioration mineures
                       - Maintenir la formation des opérateurs
                       - Documenter les meilleures pratiques
                    """)
                
                # Information supplémentaire sur les valeurs d2
                with st.expander("📊 Informations sur les valeurs d2 utilisées"):
                    st.markdown("""
                    **Signification des paramètres d2:**
                    - **z**: Nombre d'échantillons (première colonne du tableau)
                    - **w**: Taille de l'échantillon (en-tête du tableau)
                    
                    **Pour cette analyse:**
                    - **EV**: Répétabilité → d2(z=1, w=nombre d'essais) = **{:.4f}**
                    - **AV**: Reproductibilité → d2(z=nombre de pièces, w=nombre d'opérateurs) = **{:.4f}**
                    - **PV**: Variation Pièces → d2(z=1, w=nombre de pièces) = **{:.4f}**
                    
                    **Source:** Table d2 standard pour les méthodes de contrôle statistique
                    """.format(results['d2_ev'], results['d2_av'], results['d2_pv']))
                
            except Exception as e:
                st.error(f"❌ Erreur lors du calcul: {str(e)}")
                st.info("Vérifiez que les données sont correctement formatées et complètes.")

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: gray;">
    <p><strong>Gage R&R Analysis Tool</strong> - Méthode des étendues et des moyennes avec table d₂</p>
    <p>Lean Six Sigma - Outil d'analyse de la capabilité des systèmes de mesure</p>
    <p>Version 2.0 - Table d₂ corrigée selon normes statistiques</p>
</div>
""", unsafe_allow_html=True)
