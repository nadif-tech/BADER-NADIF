import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import math

# =====================================================
# CONFIGURATION PAGE
# =====================================================
st.set_page_config(
    page_title="Gage R&R – Analyse du Système de Mesure",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# SIDEBAR - INFORMATIONS ET PARAMÈTRES
# =====================================================
with st.sidebar:
    st.title("ℹ️ Guide Gage R&R")
    
    st.markdown("### Formules utilisées")
    
    with st.expander("📐 Formules de calcul", expanded=False):
        st.markdown("""
        **Répétabilité (EV - Equipment Variation):**
        ```
        R̄ = (ΣR_i)/n_parts
        EV = R̄ × K1
        ```
        
        **Reproductibilité (AV - Appraiser Variation):**
        ```
        X_diff = X̄_max - X̄_min
        AV = √[(X_diff × K2)² - (EV²/(n_parts × n_trials))]
        ```
        
        **Variation pièces (PV - Part Variation):**
        ```
        R_p = X̄_part_max - X̄_part_min
        PV = R_p × K3
        ```
        
        **Variation totale (TV - Total Variation):**
        ```
        GRR = √(EV² + AV²)
        TV = √(GRR² + PV²)
        %GRR = (GRR/TV) × 100%
        ```
        """)
    
    with st.expander("🎯 Constantes AIAG", expanded=False):
        st.markdown("""
        **K1 (pour EV):**
        - 2 essais: 0.886
        - 3 essais: 0.590
        - 4 essais: 0.485
        
        **K2 (pour AV):**
        - 2 opérateurs: 0.707
        - 3 opérateurs: 0.523
        - 4 opérateurs: 0.446
        
        **K3 (pour PV):** 0.590
        """)
    
    with st.expander("📈 Critères d'acceptation", expanded=False):
        st.markdown("""
        **Selon l'AIAG:**
        - ✅ **< 10%** : Système acceptable
        - ⚠️ **10% - 30%** : Acceptable sous conditions
        - ❌ **> 30%** : Système inacceptable
        
        **Autres normes:**
        - VDA 5: < 20%
        - ISO/TS 16949: < 30%
        """)
    
    st.divider()
    
    st.markdown("### ⚙️ Paramètres avancés")
    
    # Choix des constantes
    use_aiag_constants = st.checkbox("Utiliser les constantes AIAG", value=True)
    
    if not use_aiag_constants:
        k1_custom = st.number_input("K1 personnalisé", value=0.886, format="%.3f")
        k2_custom = st.number_input("K2 personnalisé", value=0.523, format="%.3f")
        k3_custom = st.number_input("K3 personnalisé", value=0.590, format="%.3f")
    else:
        k1_custom = k2_custom = k3_custom = None
    
    # Tolérance optionnelle
    tol_spec = st.number_input("Tolérance spécifiée (optionnel)", 
                              value=0.0, 
                              help="Pour calculer %GRR/Tolérance")

# =====================================================
# HEADER PRINCIPAL
# =====================================================
st.title("📊 Gage R&R - Analyse du Système de Mesure")
st.markdown("**Méthode des étendues et des moyennes (selon AIAG)**")

# =====================================================
# IMPORT DES DONNÉES
# =====================================================
st.subheader("📤 Importation des données")

upload_option = st.radio(
    "Choix du mode d'entrée",
    ["📁 Importer un fichier", "✍️ Saisie manuelle", "📊 Générer des données test"],
    horizontal=True
)

df = None
n_parts = 0
n_operators = 0
n_trials = 0

if upload_option == "📁 Importer un fichier":
    uploaded_file = st.file_uploader(
        "Importer un fichier CSV ou Excel",
        type=["csv", "xlsx", "xls"],
        help="Format attendu: colonnes = opérateurs × essais, lignes = pièces"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine="openpyxl")
            
            # Nettoyage des données
            df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
            df = df.apply(pd.to_numeric, errors='coerce')
            
            # Afficher les colonnes détectées
            st.success(f"✅ Fichier importé : {df.shape[0]} pièces, {df.shape[1]} colonnes")
            st.info(f"📋 Colonnes détectées : {list(df.columns)}")
            
            # Demander à l'utilisateur de spécifier la structure
            st.subheader("🔧 Configuration de la structure des données")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                n_parts_input = st.number_input("Nombre de pièces", 
                                               min_value=2, 
                                               max_value=df.shape[0], 
                                               value=df.shape[0])
            with col2:
                n_operators_input = st.number_input("Nombre d'opérateurs", 
                                                   min_value=1, 
                                                   max_value=df.shape[1], 
                                                   value=min(3, df.shape[1]))
            with col3:
                n_trials_input = st.number_input("Nombre de répétitions", 
                                                min_value=1, 
                                                max_value=df.shape[1], 
                                                value=min(2, df.shape[1]))
            
            if n_operators_input * n_trials_input != df.shape[1]:
                st.warning(f"⚠️ Attention : {n_operators_input} opérateurs × {n_trials_input} essais = {n_operators_input * n_trials_input} colonnes, mais {df.shape[1]} colonnes détectées")
            
            n_parts = n_parts_input
            n_operators = n_operators_input
            n_trials = n_trials_input
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'import: {str(e)}")

elif upload_option == "✍️ Saisie manuelle":
    cols = st.columns(3)
    with cols[0]:
        n_parts = st.number_input("Nombre de pièces", min_value=2, max_value=50, value=5)
    with cols[1]:
        n_operators = st.number_input("Nombre d'opérateurs", min_value=2, max_value=10, value=3)
    with cols[2]:
        n_trials = st.number_input("Nombre de répétitions", min_value=2, max_value=10, value=2)
    
    total_cols = n_operators * n_trials
    
    # Création d'un DataFrame vide avec noms de colonnes
    col_names = [f"Op{o+1}_T{t+1}" for o in range(n_operators) for t in range(n_trials)]
    df = pd.DataFrame(np.zeros((n_parts, total_cols)), columns=col_names)
    
    st.info("⚠️ Modifiez les valeurs dans le tableau ci-dessous")

else:  # Générer des données test
    cols = st.columns(3)
    with cols[0]:
        n_parts = st.number_input("Pièces (test)", min_value=5, max_value=20, value=10)
    with cols[1]:
        n_operators = st.number_input("Opérateurs (test)", min_value=2, max_value=5, value=3)
    with cols[2]:
        n_trials = st.number_input("Essais (test)", min_value=2, max_value=5, value=3)
    
    # Génération de données réalistes avec variation
    np.random.seed(42)
    base_values = np.random.normal(100, 5, n_parts)  # Valeurs réelles des pièces
    
    data = {}
    for op in range(n_operators):
        op_bias = np.random.normal(0, 0.5)  # Biais par opérateur
        for t in range(n_trials):
            noise = np.random.normal(0, 0.2, n_parts)  # Bruit de mesure
            col_name = f"Op{op+1}_T{t+1}"
            data[col_name] = base_values + op_bias + noise
    
    df = pd.DataFrame(data)
    st.success(f"✅ Données test générées : {n_parts} pièces × {n_operators} opérateurs × {n_trials} essais")

# =====================================================
# AFFICHAGE ET ÉDITION DES DONNÉES
# =====================================================
if df is not None:
    st.subheader("📥 Données de mesure")
    
    # Si on a importé un fichier, permettre de renommer les colonnes
    if upload_option == "📁 Importer un fichier" and df is not None:
        st.info("🔄 Vous pouvez renommer les colonnes pour qu'elles suivent le format 'OpX_TY'")
        
        # Afficher le mapping actuel des colonnes
        st.write("**Mapping actuel des colonnes :**")
        current_cols = list(df.columns)
        
        # Créer un formulaire pour renommer les colonnes
        new_col_names = []
        col_mapping = {}
        
        for i, col in enumerate(current_cols):
            # Suggérer un nom basé sur la position
            op_num = (i // n_trials) + 1
            trial_num = (i % n_trials) + 1
            suggested_name = f"Op{op_num}_T{trial_num}"
            
            new_name = st.text_input(f"Colonne {i+1} ('{col}') →", 
                                    value=suggested_name,
                                    key=f"rename_{i}")
            new_col_names.append(new_name)
            col_mapping[col] = new_name
        
        if st.button("🔄 Appliquer les nouveaux noms de colonnes"):
            df = df.rename(columns=col_mapping)
            st.success("✅ Noms de colonnes mis à jour")
            st.rerun()
    
    # Édition des données
    st.write("**Tableau des mesures :**")
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        height=400,
        num_rows="dynamic",
        column_config={
            col: st.column_config.NumberColumn(
                col,
                help=f"Mesure {col}",
                format="%.4f",
                min_value=-10000.0,
                max_value=10000.0
            ) for col in df.columns
        }
    )
    
    df = edited_df  # Mettre à jour avec les données éditées
    
    # S'assurer que toutes les colonnes sont numériques
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remplacer les NaN par 0 pour éviter les erreurs
    df = df.fillna(0)
    
    # Vérifier que nous avons les paramètres nécessaires
    if n_operators == 0 or n_trials == 0:
        st.error("❌ Veuillez spécifier le nombre d'opérateurs et d'essais")
        st.stop()
    
    # =====================================================
    # CONSTANTES AIAG
    # =====================================================
    if use_aiag_constants:
        K1 = {2: 0.886, 3: 0.590, 4: 0.485}
        K2 = {2: 0.707, 3: 0.523, 4: 0.446}
        K3 = 0.590
        
        k1_val = K1.get(n_trials, 0.886)
        k2_val = K2.get(n_operators, 0.707)
        k3_val = K3
    else:
        k1_val = k1_custom
        k2_val = k2_custom
        k3_val = k3_custom
    
    # =====================================================
    # CALCUL GAGE R&R
    # =====================================================
    if st.button("🚀 Calculer l'analyse Gage R&R", type="primary", use_container_width=True):
        
        try:
            # Initialisation des tableaux
            ranges_per_part = []
            operator_means = []
            all_measurements = []
            
            # Vérifier que nous avons assez de colonnes
            total_cols_needed = n_operators * n_trials
            if len(df.columns) < total_cols_needed:
                st.error(f"❌ Pas assez de colonnes. Besoin de {total_cols_needed} colonnes, mais seulement {len(df.columns)} disponibles.")
                st.stop()
            
            # Afficher la structure détectée
            st.info(f"🔍 Structure détectée : {n_operators} opérateurs × {n_trials} essais = {total_cols_needed} colonnes")
            
            # Organiser les colonnes par opérateur
            # D'abord, vérifier quelles colonnes nous avons
            available_cols = list(df.columns)
            st.write(f"Colonnes disponibles : {available_cols}")
            
            # Essayer différentes stratégies pour organiser les colonnes
            op_cols_list = []
            
            # Stratégie 1 : Si les colonnes sont nommées OpX_TY
            op_pattern_cols = {}
            for col in available_cols:
                if 'op' in col.lower() and ('t' in col.lower() or 'essai' in col.lower() or 'trial' in col.lower()):
                    # Essayer d'extraire le numéro d'opérateur
                    import re
                    op_match = re.search(r'op\s*(\d+)', col.lower())
                    trial_match = re.search(r't\s*(\d+)|essai\s*(\d+)|trial\s*(\d+)', col.lower())
                    
                    if op_match and trial_match:
                        op_num = int(op_match.group(1))
                        trial_num = int(trial_match.group(1) or trial_match.group(2) or trial_match.group(3))
                        
                        if op_num not in op_pattern_cols:
                            op_pattern_cols[op_num] = {}
                        op_pattern_cols[op_num][trial_num] = col
            
            if op_pattern_cols:
                # Nous avons des colonnes avec le bon format
                for op_num in sorted(op_pattern_cols.keys()):
                    if op_num <= n_operators:
                        op_trials = op_pattern_cols[op_num]
                        # Prendre les n_trials premiers essais
                        trial_cols = [op_trials.get(t, None) for t in range(1, n_trials + 1)]
                        trial_cols = [c for c in trial_cols if c is not None]
                        if trial_cols:
                            op_cols_list.append(trial_cols)
            
            # Stratégie 2 : Si nous n'avons pas trouvé de pattern, diviser équitablement
            if not op_cols_list and len(available_cols) >= total_cols_needed:
                st.info("⚙️ Organisation automatique des colonnes par opérateur")
                # Diviser les colonnes en n_operators groupes de n_trials
                for op_idx in range(n_operators):
                    start_idx = op_idx * n_trials
                    end_idx = start_idx + n_trials
                    op_cols = available_cols[start_idx:end_idx]
                    op_cols_list.append(op_cols)
            
            # Si nous n'avons toujours pas d'organisation, demander à l'utilisateur
            if not op_cols_list:
                st.error("❌ Impossible d'organiser automatiquement les colonnes.")
                st.info("Veuillez renommer vos colonnes avec le format 'OpX_TY' ou spécifier manuellement.")
                
                # Créer une interface pour spécifier manuellement
                st.subheader("🔧 Spécification manuelle des colonnes par opérateur")
                
                op_cols_list = []
                for op_idx in range(n_operators):
                    with st.expander(f"Opérateur {op_idx + 1}", expanded=op_idx == 0):
                        selected_cols = st.multiselect(
                            f"Sélectionnez les {n_trials} colonnes pour l'opérateur {op_idx + 1}",
                            options=available_cols,
                            default=available_cols[op_idx*n_trials:min((op_idx+1)*n_trials, len(available_cols))],
                            key=f"op_{op_idx}_cols"
                        )
                        if len(selected_cols) != n_trials:
                            st.warning(f"⚠️ Veuillez sélectionner exactement {n_trials} colonnes")
                        op_cols_list.append(selected_cols)
                
                if st.button("✅ Confirmer l'organisation"):
                    st.rerun()
                else:
                    st.stop()
            
            # Afficher l'organisation choisie
            st.write("**Organisation des colonnes :**")
            for op_idx, op_cols in enumerate(op_cols_list):
                st.write(f"Opérateur {op_idx + 1} : {op_cols}")
            
            # Maintenant, effectuer les calculs avec l'organisation choisie
            for op_idx, op_cols in enumerate(op_cols_list):
                if len(op_cols) != n_trials:
                    st.error(f"❌ L'opérateur {op_idx + 1} n'a pas {n_trials} colonnes")
                    st.stop()
                
                df_op = df[op_cols]
                
                # Étendues par pièce pour cet opérateur
                op_ranges = df_op.max(axis=1) - df_op.min(axis=1)
                ranges_per_part.append(op_ranges)
                
                # Moyennes par pièce pour cet opérateur
                op_means = df_op.mean(axis=1)
                operator_means.append(op_means)
                
                # Toutes les mesures pour statistiques
                all_measurements.extend(df_op.values.flatten())
            
            # Vérifier que nous avons des données
            if not all_measurements:
                st.error("❌ Aucune donnée valide trouvée pour le calcul")
                st.stop()
            
            # -------------------------------------------------
            # 1. RÉPÉTABILITÉ (EV)
            # -------------------------------------------------
            # Moyenne des étendues par pièce (moyenne des opérateurs)
            if ranges_per_part:
                R_bar_matrix = pd.concat(ranges_per_part, axis=1)
                R_bar_per_part = R_bar_matrix.mean(axis=1)
                R_bar_global = R_bar_per_part.mean()
                
                EV = R_bar_global * k1_val
            else:
                EV = 0
                R_bar_global = 0
            
            # -------------------------------------------------
            # 2. REPRODUCTIBILITÉ (AV)
            # -------------------------------------------------
            # Moyennes globales par opérateur
            if operator_means:
                op_global_means = [m.mean() for m in operator_means]
                X_diff = max(op_global_means) - min(op_global_means)
                
                # Calcul AV avec vérification de la racine carrée
                av_term = (X_diff * k2_val) ** 2 - (EV ** 2 / (n_parts * n_trials))
                AV = math.sqrt(max(av_term, 0)) if av_term > 0 else 0
            else:
                AV = 0
                X_diff = 0
            
            # -------------------------------------------------
            # 3. VARIATION PIÈCES (PV)
            # -------------------------------------------------
            # Moyenne de toutes les mesures par pièce
            part_data = []
            for part_idx in range(min(n_parts, len(df))):
                part_vals = []
                for op_cols in op_cols_list:
                    part_vals.extend(df.loc[part_idx, op_cols].values)
                if part_vals:
                    part_data.append(np.mean(part_vals))
            
            if part_data:
                R_p = max(part_data) - min(part_data)
                PV = R_p * k3_val
            else:
                R_p = 0
                PV = 0
            
            # -------------------------------------------------
            # 4. VARIATION TOTALE (TV) ET %GRR
            # -------------------------------------------------
            GRR = math.sqrt(EV**2 + AV**2)
            TV = math.sqrt(GRR**2 + PV**2)
            
            if TV > 0:
                GRR_percent = (GRR / TV) * 100
                EV_percent = (EV / TV) * 100
                AV_percent = (AV / TV) * 100
                PV_percent = (PV / TV) * 100
            else:
                GRR_percent = EV_percent = AV_percent = PV_percent = 0
            
            # Calcul supplémentaire %GRR/Tolérance si spécifiée
            if tol_spec > 0:
                GRR_tol_percent = (GRR / tol_spec) * 100
            else:
                GRR_tol_percent = None
            
            # -------------------------------------------------
            # 5. STATISTIQUES DES DONNÉES
            # -------------------------------------------------
            if all_measurements:
                all_measurements_array = np.array(all_measurements)
                data_stats = {
                    "Moyenne": np.mean(all_measurements_array),
                    "Écart-type": np.std(all_measurements_array, ddof=1),
                    "Min": np.min(all_measurements_array),
                    "Max": np.max(all_measurements_array),
                    "Étendue": np.ptp(all_measurements_array)
                }
            else:
                data_stats = {
                    "Moyenne": 0,
                    "Écart-type": 0,
                    "Min": 0,
                    "Max": 0,
                    "Étendue": 0
                }
            
            # =====================================================
            # AFFICHAGE DES RÉSULTATS
            # =====================================================
            st.subheader("📊 Résultats de l'analyse")
            
            # Métriques principales
            cols = st.columns(5)
            with cols[0]:
                st.metric("Répétabilité (EV)", f"{EV:.4f}", f"{EV_percent:.1f}%")
            with cols[1]:
                st.metric("Reproductibilité (AV)", f"{AV:.4f}", f"{AV_percent:.1f}%")
            with cols[2]:
                st.metric("Gage R&R (GRR)", f"{GRR:.4f}", f"{GRR_percent:.1f}%")
            with cols[3]:
                st.metric("Variation Pièces (PV)", f"{PV:.4f}", f"{PV_percent:.1f}%")
            with cols[4]:
                st.metric("Variation Totale (TV)", f"{TV:.4f}", "100%")
            
            # Indicateur de qualité
            st.subheader("📈 Évaluation du système de mesure")
            
            if GRR_percent < 10:
                st.success(f"✅ **SYSTÈME ACCEPTABLE** - %GRR = {GRR_percent:.1f}% (< 10%)")
                st.progress(GRR_percent / 30)
            elif GRR_percent < 30:
                st.warning(f"⚠️ **ACCEPTABLE SOUS CONDITIONS** - %GRR = {GRR_percent:.1f}% (entre 10% et 30%)")
                st.progress(GRR_percent / 30)
            else:
                st.error(f"❌ **SYSTÈME INACCEPTABLE** - %GRR = {GRR_percent:.1f}% (> 30%)")
                st.progress(1.0)
            
            if GRR_tol_percent is not None:
                st.info(f"📏 %GRR/Tolérance = {GRR_tol_percent:.1f}% (tolérance spécifiée: {tol_spec})")
            
            # =====================================================
            # TABLEAUX DÉTAILLÉS
            # =====================================================
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Tableau détaillé", "📐 Calculs intermédiaires", "📈 Statistiques", "🔧 Paramètres"])
            
            with tab1:
                # S'assurer que nous avons le bon nombre de pièces
                actual_n_parts = min(n_parts, len(part_data))
                detailed_df = pd.DataFrame({
                    "Pièce": range(1, actual_n_parts + 1),
                    "Moyenne Pièce": part_data[:actual_n_parts],
                    "Étendue Moyenne (R̄)": R_bar_per_part.values[:actual_n_parts] if len(R_bar_per_part) > 0 else [0]*actual_n_parts,
                    "EV (par pièce)": [EV] * actual_n_parts,
                    "AV (par pièce)": [AV] * actual_n_parts,
                    "GRR (par pièce)": [GRR] * actual_n_parts,
                    "PV (par pièce)": [PV] * actual_n_parts,
                    "TV (par pièce)": [TV] * actual_n_parts
                })
                st.dataframe(detailed_df, use_container_width=True)
            
            with tab2:
                calc_df = pd.DataFrame({
                    "Paramètre": ["R̄ (moyenne des étendues)", "X_diff (différence des moyennes op.)", 
                                 "K1 utilisé", "K2 utilisé", "K3 utilisé", "R_p (étendue pièces)"],
                    "Valeur": [f"{R_bar_global:.4f}", f"{X_diff:.4f}", 
                              f"{k1_val}", f"{k2_val}", f"{k3_val}", f"{R_p:.4f}"],
                    "Formule": ["ΣR_i / n_parts", "max(X̄_op) - min(X̄_op)", 
                               f"K1({n_trials})", f"K2({n_operators})", "0.590", "max(X̄_part) - min(X̄_part)"]
                })
                st.dataframe(calc_df, use_container_width=True)
            
            with tab3:
                stats_df = pd.DataFrame(list(data_stats.items()), 
                                       columns=["Statistique", "Valeur"])
                st.dataframe(stats_df, use_container_width=True)
            
            with tab4:
                param_df = pd.DataFrame({
                    "Paramètre": ["Pièces", "Opérateurs", "Essais", "Total mesures"],
                    "Valeur": [n_parts, n_operators, n_trials, n_parts * n_operators * n_trials]
                })
                st.dataframe(param_df, use_container_width=True)
            
            # =====================================================
            # VISUALISATIONS
            # =====================================================
            st.subheader("📊 Visualisations")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Diagramme à barres des composantes
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                components = ['EV', 'AV', 'PV', 'GRR', 'TV']
                values = [EV, AV, PV, GRR, TV]
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                
                bars = ax1.bar(components, values, color=colors)
                ax1.set_ylabel('Variation')
                ax1.set_title('Composantes de variation (absolues)')
                
                # Ajout des valeurs sur les barres
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.3f}', ha='center', va='bottom')
                
                st.pyplot(fig1)
            
            with viz_col2:
                # Diagramme à barres des pourcentages
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                components_pct = ['EV%', 'AV%', 'PV%', 'GRR%']
                values_pct = [EV_percent, AV_percent, PV_percent, GRR_percent]
                colors_pct = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                
                bars_pct = ax2.bar(components_pct, values_pct, color=colors_pct)
                ax2.set_ylabel('Pourcentage (%)')
                ax2.set_title('Distribution des variations (%)')
                ax2.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Limite 10%')
                ax2.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Limite 30%')
                
                # Ajout des valeurs sur les barres
                for bar, val in zip(bars_pct, values_pct):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.1f}%', ha='center', va='bottom')
                
                ax2.legend()
                st.pyplot(fig2)
            
            # Graphique des moyennes par opérateur
            if operator_means:
                st.subheader("📈 Moyennes par opérateur")
                
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                actual_n_parts = min(n_parts, len(operator_means[0]))
                x_positions = np.arange(actual_n_parts)
                width = 0.8 / min(n_operators, len(operator_means))
                
                for op_idx in range(min(n_operators, len(operator_means))):
                    op_means = operator_means[op_idx][:actual_n_parts]
                    ax3.bar(x_positions + op_idx * width, op_means, 
                           width=width, label=f'Op {op_idx+1}', 
                           alpha=0.7)
                
                ax3.set_xlabel('Pièces')
                ax3.set_ylabel('Moyenne des mesures')
                ax3.set_title('Moyennes par opérateur pour chaque pièce')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                st.pyplot(fig3)
            
            # Graphique des étendues par pièce
            if ranges_per_part:
                st.subheader("📉 Étendues par pièce")
                
                fig4, ax4 = plt.subplots(figsize=(10, 4))
                actual_n_parts = min(n_parts, len(ranges_per_part[0]))
                parts = range(1, actual_n_parts + 1)
                
                for op_idx in range(min(n_operators, len(ranges_per_part))):
                    ax4.plot(parts, ranges_per_part[op_idx][:actual_n_parts], 
                            marker='o', label=f'Op {op_idx+1}', 
                            alpha=0.7, linewidth=2)
                
                ax4.set_xlabel('Pièces')
                ax4.set_ylabel('Étendue')
                ax4.set_title('Étendues par pièce et par opérateur')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                st.pyplot(fig4)
            
            # =====================================================
            # RAPPORT DÉTAILLÉ
            # =====================================================
            with st.expander("📄 Rapport détaillé de l'analyse", expanded=False):
                st.markdown(f"""
                ## Rapport d'analyse Gage R&R
                
                **Date de l'analyse:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                **Paramètres de l'étude:**
                - Nombre de pièces: {n_parts}
                - Nombre d'opérateurs: {n_operators}
                - Nombre de répétitions: {n_trials}
                - Constantes utilisées: K1={k1_val}, K2={k2_val}, K3={k3_val}
                
                **Résultats:**
                - Répétabilité (EV): {EV:.4f} ({EV_percent:.1f}%)
                - Reproductibilité (AV): {AV:.4f} ({AV_percent:.1f}%)
                - Variation Gage R&R (GRR): {GRR:.4f} ({GRR_percent:.1f}%)
                - Variation Pièces (PV): {PV:.4f} ({PV_percent:.1f}%)
                - Variation Totale (TV): {TV:.4f}
                
                **Conclusion:**
                Le système de mesure est **{'acceptable' if GRR_percent < 10 else 'acceptable sous conditions' if GRR_percent < 30 else 'inacceptable'}** 
                avec un %GRR de {GRR_percent:.1f}%.
                """)
            
            # =====================================================
            # EXPORT DES RÉSULTATS
            # =====================================================
            st.subheader("💾 Export des résultats")
            
            # Préparation des données pour export
            results_dict = {
                "Paramètre": ["EV", "AV", "GRR", "PV", "TV", "%GRR", "%EV", "%AV", "%PV"],
                "Valeur": [EV, AV, GRR, PV, TV, GRR_percent, EV_percent, AV_percent, PV_percent],
                "Unité": ["absolu", "absolu", "absolu", "absolu", "absolu", "%", "%", "%", "%"]
            }
            
            summary_df = pd.DataFrame(results_dict)
            
            # Boutons d'export
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            
            with col_exp1:
                st.download_button(
                    label="📥 Exporter résultats (CSV)",
                    data=summary_df.to_csv(index=False),
                    file_name="gage_rr_results.csv",
                    mime="text/csv"
                )
            
            with col_exp2:
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Données brutes', index=False)
                    detailed_df.to_excel(writer, sheet_name='Analyse détaillée', index=False)
                    summary_df.to_excel(writer, sheet_name='Résumé', index=False)
                
                st.download_button(
                    label="📥 Exporter rapport complet (Excel)",
                    data=excel_buffer.getvalue(),
                    file_name="rapport_gage_rr_complet.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col_exp3:
                # Génération d'un rapport texte
                report_text = f"""
                RAPPORT GAGE R&R
                =================
                Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                PARAMÈTRES DE L'ÉTUDE:
                - Pièces: {n_parts}
                - Opérateurs: {n_operators}
                - Essais: {n_trials}
                
                RÉSULTATS:
                - Répétabilité (EV): {EV:.4f} ({EV_percent:.1f}%)
                - Reproductibilité (AV): {AV:.4f} ({AV_percent:.1f}%)
                - Gage R&R (GRR): {GRR:.4f} ({GRR_percent:.1f}%)
                - Variation pièces (PV): {PV:.4f} ({PV_percent:.1f}%)
                - Variation totale (TV): {TV:.4f}
                
                CONCLUSION:
                %GRR = {GRR_percent:.1f}% -> Système {'acceptable' if GRR_percent < 10 else 'acceptable sous conditions' if GRR_percent < 30 else 'inacceptable'}
                """
                
                st.download_button(
                    label="📥 Exporter rapport (TXT)",
                    data=report_text,
                    file_name="rapport_gage_rr.txt",
                    mime="text/plain"
                )
        
        except Exception as e:
            st.error(f"❌ Erreur lors du calcul : {str(e)}")
            st.info("Veuillez vérifier que vos données sont correctement formatées et que tous les paramètres sont définis.")

else:
    st.info("👈 Veuillez importer ou saisir des données pour commencer l'analyse")

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
    <p>Gage R&R - Méthode des étendues et des moyennes | Basé sur les recommandations AIAG</p>
    <p>Outils pour l'amélioration de la qualité et la maîtrise statistique des processus</p>
</div>
""", unsafe_allow_html=True)
