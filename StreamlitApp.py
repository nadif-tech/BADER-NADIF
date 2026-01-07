import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import math

# =====================================================
# TABLE DES VALEURS d2 (AIAG)
# =====================================================
# Table d2 pour le calcul de la variation des pièces
# g = nombre de sous-groupes (pièces), m = taille du sous-groupe (mesures par pièce)
D2_TABLE = {
    2: 1.41, 3: 1.91, 4: 2.24, 5: 2.48, 6: 2.67,
    7: 2.83, 8: 2.96, 9: 3.08, 10: 3.18, 11: 3.27,
    12: 3.35, 13: 3.42, 14: 3.49, 15: 3.55
}

# =====================================================
# CONSTANTES K (AIAG)
# =====================================================
K1 = {2: 0.8862, 3: 0.5908, 4: 0.4857}
K2 = {2: 0.7071, 3: 0.5231, 4: 0.4467}
K3 = 0.5231  # Pour 2 opérateurs

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
    
    st.markdown("### 📐 Formules de calcul")
    
    with st.expander("Formules principales", expanded=True):
        st.markdown("""
        **1. Répétabilité (EV - Equipment Variation):**
        ```
        EV = R̄ × K₁
        R̄ = (ΣR_i)/n_parts
        ```
        
        **2. Reproductibilité (AV - Appraiser Variation):**
        ```
        AV = √[(X̄_diff × K₂)² - (EV²/(n_parts × n_trials))]
        X̄_diff = max(X̄_op) - min(X̄_op)
        ```
        
        **3. Variation pièces (PV - Part Variation):**
        ```
        PV = R_p × K₃
        R_p = max(X̄_part) - min(X̄_part)
        ```
        
        **4. Gage R&R et Variation totale:**
        ```
        R&R = √(EV² + AV²)
        V_T = √(R&R² + PV²)
        %R&R = (R&R / V_T) × 100%
        ```
        """)
    
    with st.expander("🎯 Constantes AIAG", expanded=False):
        st.markdown("""
        **K₁ (pour EV - Répétabilité):**
        ```
        n_trials = 2 → K₁ = 0.8862
        n_trials = 3 → K₁ = 0.5908
        n_trials = 4 → K₁ = 0.4857
        ```
        
        **K₂ (pour AV - Reproductibilité):**
        ```
        n_operators = 2 → K₂ = 0.7071
        n_operators = 3 → K₂ = 0.5231
        n_operators = 4 → K₂ = 0.4467
        ```
        
        **K₃ (pour PV - Variation pièces):**
        ```
        K₃ = 0.5231 (pour 2 opérateurs)
        ```
        """)
    
    with st.expander("📊 Table des valeurs d₂", expanded=False):
        st.markdown("""
        | m  | d₂    |
        |----|-------|
        | 2  | 1.41  |
        | 3  | 1.91  |
        | 4  | 2.24  |
        | 5  | 2.48  |
        | 6  | 2.67  |
        | 7  | 2.83  |
        | 8  | 2.96  |
        | 9  | 3.08  |
        | 10 | 3.18  |
        | 11 | 3.27  |
        | 12 | 3.35  |
        | 13 | 3.42  |
        | 14 | 3.49  |
        | 15 | 3.55  |
        
        *m = nombre de mesures par pièce*
        """)
    
    with st.expander("📈 Critères d'acceptation", expanded=False):
        st.markdown("""
        **Selon l'AIAG:**
        - ✅ **< 10%** : Système acceptable
        - ⚠️ **10% - 30%** : Acceptable sous conditions
        - ❌ **> 30%** : Système inacceptable
        
        **Selon VDA 5:**
        - ✅ **< 20%** : Système acceptable
        
        **Selon ISO/TS 16949:**
        - ✅ **< 30%** : Système acceptable
        """)
    
    st.divider()
    
    st.markdown("### ⚙️ Paramètres avancés")
    
    # Choix de la méthode de calcul
    method = st.radio(
        "Méthode de calcul",
        ["AIAG standard", "Avec d₂"],
        help="AIAG: utilise K1,K2,K3 constants. Avec d₂: utilise la table d2 pour PV"
    )
    
    # Tolérance optionnelle
    tol_spec = st.number_input(
        "Tolérance spécifiée (optionnel)", 
        value=0.0,
        help="Pour calculer %R&R/Tolérance"
    )
    
    # Niveau de confiance
    confidence = st.slider(
        "Niveau de confiance (%)",
        min_value=90,
        max_value=99,
        value=95,
        help="Pour les calculs statistiques"
    )

# =====================================================
# HEADER PRINCIPAL
# =====================================================
st.title("📊 Gage R&R - Analyse du Système de Mesure")
st.markdown("**Méthode des étendues et des moyennes selon AIAG**")

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
            
            # Configuration de la structure
            st.subheader("🔧 Configuration de la structure")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                n_parts = st.number_input(
                    "Nombre de pièces", 
                    min_value=2, 
                    max_value=df.shape[0], 
                    value=df.shape[0]
                )
            with col2:
                n_operators = st.number_input(
                    "Nombre d'opérateurs", 
                    min_value=1, 
                    max_value=df.shape[1], 
                    value=min(3, df.shape[1])
                )
            with col3:
                n_trials = st.number_input(
                    "Nombre de répétitions", 
                    min_value=1, 
                    max_value=df.shape[1], 
                    value=min(2, df.shape[1])
                )
            
            # Ajuster le dataframe si nécessaire
            if n_parts < df.shape[0]:
                df = df.iloc[:n_parts, :]
            if n_operators * n_trials < df.shape[1]:
                df = df.iloc[:, :n_operators * n_trials]
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'import: {str(e)}")

elif upload_option == "✍️ Saisie manuelle":
    cols = st.columns(3)
    with cols[0]:
        n_parts = st.number_input("Nombre de pièces", min_value=2, max_value=50, value=10)
    with cols[1]:
        n_operators = st.number_input("Nombre d'opérateurs", min_value=2, max_value=10, value=3)
    with cols[2]:
        n_trials = st.number_input("Nombre de répétitions", min_value=2, max_value=10, value=3)
    
    total_cols = n_operators * n_trials
    
    # Création d'un DataFrame vide
    col_names = [f"Op{o+1}_T{t+1}" for o in range(n_operators) for t in range(n_trials)]
    df = pd.DataFrame(np.random.randn(n_parts, total_cols) * 0.5 + 100, columns=col_names)
    
    st.info("⚠️ Modifiez les valeurs dans le tableau ci-dessous")

else:  # Générer des données test
    cols = st.columns(3)
    with cols[0]:
        n_parts = st.number_input("Pièces (test)", min_value=5, max_value=30, value=10)
    with cols[1]:
        n_operators = st.number_input("Opérateurs (test)", min_value=2, max_value=5, value=3)
    with cols[2]:
        n_trials = st.number_input("Essais (test)", min_value=2, max_value=5, value=3)
    
    # Génération de données réalistes
    np.random.seed(42)
    base_values = np.random.normal(100, 10, n_parts)  # Valeurs réelles des pièces
    
    data = {}
    for op in range(n_operators):
        op_bias = np.random.normal(0, 1)  # Biais par opérateur
        for t in range(n_trials):
            noise = np.random.normal(0, 0.5, n_parts)  # Bruit de mesure
            col_name = f"Op{op+1}_T{t+1}"
            data[col_name] = base_values + op_bias + noise
    
    df = pd.DataFrame(data)
    st.success(f"✅ Données test générées : {n_parts} pièces × {n_operators} opérateurs × {n_trials} essais")

# =====================================================
# AFFICHAGE ET ÉDITION DES DONNÉES
# =====================================================
if df is not None:
    st.subheader("📥 Données de mesure")
    
    # Redimensionner le dataframe si nécessaire
    if 'n_parts' in locals() and 'n_operators' in locals() and 'n_trials' in locals():
        total_cols_needed = n_operators * n_trials
        
        # S'assurer que nous avons le bon nombre de colonnes
        if len(df.columns) > total_cols_needed:
            df = df.iloc[:, :total_cols_needed]
        elif len(df.columns) < total_cols_needed:
            # Ajouter des colonnes manquantes
            missing_cols = total_cols_needed - len(df.columns)
            for i in range(missing_cols):
                df[f'Col_{len(df.columns)+1}'] = 0.0
        
        # Redimensionner les lignes
        if len(df) > n_parts:
            df = df.iloc[:n_parts, :]
        elif len(df) < n_parts:
            # Ajouter des lignes manquantes
            missing_rows = n_parts - len(df)
            new_rows = pd.DataFrame(np.zeros((missing_rows, len(df.columns))), columns=df.columns)
            df = pd.concat([df, new_rows], ignore_index=True)
    
    # Renommer les colonnes pour un format standard
    if not df.columns[0].startswith('Op'):
        col_names = [f"Op{(i//n_trials)+1}_T{(i%n_trials)+1}" for i in range(len(df.columns))]
        df.columns = col_names
    
    # Édition des données
    st.write("**Tableau des mesures :**")
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        height=400,
        num_rows="fixed",
        column_config={
            col: st.column_config.NumberColumn(
                col,
                help=f"Mesure {col}",
                format="%.3f",
                step=0.001
            ) for col in df.columns
        }
    )
    
    df = edited_df
    
    # =====================================================
    # CALCUL GAGE R&R
    # =====================================================
    if st.button("🚀 Calculer l'analyse Gage R&R", type="primary", use_container_width=True):
        
        try:
            # S'assurer que les données sont numériques
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.fillna(0)
            
            # Vérifier les paramètres
            if n_parts < 2 or n_operators < 2 or n_trials < 2:
                st.error("❌ Paramètres insuffisants. Minimum: 2 pièces, 2 opérateurs, 2 essais")
                st.stop()
            
            # =====================================================
            # 1. ORGANISATION DES DONNÉES
            # =====================================================
            st.subheader("🔍 Organisation des données")
            
            # Regrouper les colonnes par opérateur
            op_data = []
            op_ranges = []
            op_means_by_part = []
            all_measurements = []
            
            for op in range(n_operators):
                # Sélectionner les colonnes pour cet opérateur
                op_cols = [f"Op{op+1}_T{t+1}" for t in range(n_trials)]
                
                # Vérifier que les colonnes existent
                missing_cols = [col for col in op_cols if col not in df.columns]
                if missing_cols:
                    st.error(f"❌ Colonnes manquantes pour l'opérateur {op+1}: {missing_cols}")
                    st.stop()
                
                df_op = df[op_cols]
                op_data.append(df_op)
                
                # Calculer les étendues par pièce pour cet opérateur
                ranges_op = df_op.max(axis=1) - df_op.min(axis=1)
                op_ranges.append(ranges_op)
                
                # Calculer les moyennes par pièce pour cet opérateur
                means_op = df_op.mean(axis=1)
                op_means_by_part.append(means_op)
                
                # Collecter toutes les mesures
                all_measurements.extend(df_op.values.flatten())
            
            # =====================================================
            # 2. CALCULS INTERMÉDIAIRES
            # =====================================================
            # Moyenne des étendues par pièce (sur tous les opérateurs)
            R_bar_matrix = pd.concat(op_ranges, axis=1)
            R_bar = R_bar_matrix.mean(axis=1)  # Étendue moyenne par pièce
            R_bar_global = R_bar.mean()        # Étendue moyenne globale
            
            # Moyennes globales par opérateur
            op_global_means = [means.mean() for means in op_means_by_part]
            X_diff = max(op_global_means) - min(op_global_means)
            
            # Moyennes par pièce (tous opérateurs confondus)
            part_means = []
            for part_idx in range(n_parts):
                part_vals = []
                for op in range(n_operators):
                    part_vals.extend(df.iloc[part_idx, op*n_trials:(op+1)*n_trials].values)
                part_means.append(np.mean(part_vals))
            
            R_p = max(part_means) - min(part_means)  # Étendue des moyennes des pièces
            
            # =====================================================
            # 3. CALCUL DES COMPOSANTES DE VARIATION
            # =====================================================
            st.subheader("📐 Calculs détaillés")
            
            # Déterminer les constantes K
            k1 = K1.get(n_trials, 4.56/n_trials)  # Approximation si n_trials > 4
            k2 = K2.get(n_operators, 3.65/n_operators)  # Approximation si n_operators > 4
            
            if method == "Avec d₂":
                # Utiliser d2 pour PV
                if n_parts in D2_TABLE:
                    d2 = D2_TABLE[n_parts]
                else:
                    # Approximation pour n_parts > 15
                    d2 = 3.55 + 0.06 * (n_parts - 15)
                k3 = 5.15 / d2  # Facteur pour 99% de la distribution
            else:
                # Utiliser K3 standard
                k3 = K3
            
            # Afficher les constantes utilisées
            const_df = pd.DataFrame({
                "Constante": ["K₁", "K₂", "K₃", "R̄", "X_diff", "R_p"],
                "Valeur": [f"{k1:.4f}", f"{k2:.4f}", f"{k3:.4f}", 
                          f"{R_bar_global:.4f}", f"{X_diff:.4f}", f"{R_p:.4f}"],
                "Description": [
                    f"Pour {n_trials} essais",
                    f"Pour {n_operators} opérateurs",
                    "Pour variation pièces" if method == "Avec d₂" else "Standard AIAG",
                    "Étendue moyenne",
                    "Différence des moyennes opérateurs",
                    "Étendue des moyennes pièces"
                ]
            })
            st.dataframe(const_df, use_container_width=True)
            
            # 3.1 Répétabilité (EV)
            EV = R_bar_global * k1
            
            # 3.2 Reproductibilité (AV)
            av_term = (X_diff * k2) ** 2 - (EV ** 2 / (n_parts * n_trials))
            AV = math.sqrt(max(av_term, 0))
            
            # 3.3 Gage R&R
            GRR = math.sqrt(EV ** 2 + AV ** 2)
            
            # 3.4 Variation pièces (PV)
            PV = R_p * k3
            
            # 3.5 Variation totale (TV)
            TV = math.sqrt(GRR ** 2 + PV ** 2)
            
            # 3.6 Pourcentages
            if TV > 0:
                EV_pct = (EV / TV) * 100
                AV_pct = (AV / TV) * 100
                GRR_pct = (GRR / TV) * 100
                PV_pct = (PV / TV) * 100
            else:
                EV_pct = AV_pct = GRR_pct = PV_pct = 0
            
            # 3.7 %R&R/Tolérance si spécifiée
            if tol_spec > 0:
                GRR_tol_pct = (GRR / tol_spec) * 100
                EV_tol_pct = (EV / tol_spec) * 100
                AV_tol_pct = (AV / tol_spec) * 100
            else:
                GRR_tol_pct = EV_tol_pct = AV_tol_pct = None
            
            # =====================================================
            # 4. AFFICHAGE DES RÉSULTATS
            # =====================================================
            st.subheader("📊 Résultats de l'analyse")
            
            # Métriques principales
            cols = st.columns(5)
            metrics = [
                ("EV", EV, EV_pct, "#1f77b4"),
                ("AV", AV, AV_pct, "#ff7f0e"),
                ("GRR", GRR, GRR_pct, "#d62728"),
                ("PV", PV, PV_pct, "#2ca02c"),
                ("TV", TV, 100, "#9467bd")
            ]
            
            for i, (name, value, pct, color) in enumerate(metrics):
                with cols[i]:
                    st.metric(
                        label=name,
                        value=f"{value:.4f}",
                        delta=f"{pct:.1f}%" if name != "TV" else None
                    )
            
            # Indicateur de qualité
            st.subheader("📈 Évaluation du système de mesure")
            
            # Barre de progression colorée
            if GRR_pct < 10:
                color = "green"
                status = "✅ **SYSTÈME ACCEPTABLE**"
            elif GRR_pct < 30:
                color = "orange"
                status = "⚠️ **ACCEPTABLE SOUS CONDITIONS**"
            else:
                color = "red"
                status = "❌ **SYSTÈME INACCEPTABLE**"
            
            st.markdown(f"""
            <div style="background-color:{color}20; padding:15px; border-radius:10px; border-left:5px solid {color};">
                <h4 style="margin:0; color:{color}">{status}</h4>
                <p style="margin:5px 0 0 0; font-size:1.2em;">
                    %R&R = <strong>{GRR_pct:.1f}%</strong> | %EV = {EV_pct:.1f}% | %AV = {AV_pct:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Barre de progression
            progress_val = min(GRR_pct / 30, 1.0)
            st.progress(progress_val, text=f"%R&R: {GRR_pct:.1f}%")
            
            if tol_spec > 0:
                st.info(f"📏 **%R&R/Tolérance = {GRR_tol_pct:.1f}%** (tolérance spécifiée: {tol_spec:.3f})")
            
            # =====================================================
            # 5. TABLEAUX DÉTAILLÉS
            # =====================================================
            st.subheader("📋 Tableaux détaillés")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Par pièce", "📈 Par opérateur", "🧮 Calculs", "📊 Statistiques"])
            
            with tab1:
                # Tableau par pièce
                detail_df = pd.DataFrame({
                    "Pièce": range(1, n_parts + 1),
                    "Moyenne": part_means,
                    "Étendue moyenne (R̄)": R_bar.values,
                    "EV contribution": [EV] * n_parts,
                    "AV contribution": [AV] * n_parts,
                    "GRR contribution": [GRR] * n_parts,
                    "PV contribution": [PV] * n_parts
                })
                st.dataframe(detail_df, use_container_width=True)
            
            with tab2:
                # Tableau par opérateur
                op_stats = []
                for op in range(n_operators):
                    op_vals = df.iloc[:, op*n_trials:(op+1)*n_trials].values.flatten()
                    op_stats.append({
                        "Opérateur": f"Op{op+1}",
                        "Moyenne": np.mean(op_vals),
                        "Écart-type": np.std(op_vals, ddof=1),
                        "Min": np.min(op_vals),
                        "Max": np.max(op_vals),
                        "Étendue moyenne": op_ranges[op].mean()
                    })
                op_df = pd.DataFrame(op_stats)
                st.dataframe(op_df, use_container_width=True)
            
            with tab3:
                # Calculs intermédiaires
                calc_df = pd.DataFrame({
                    "Étape": [
                        "1. Étendue moyenne (R̄)",
                        "2. Différence des moyennes (X_diff)",
                        "3. Étendue des moyennes pièces (R_p)",
                        "4. Répétabilité (EV = R̄ × K₁)",
                        "5. Reproductibilité (AV = √[(X_diff × K₂)² - EV²/(n×r)])",
                        "6. Gage R&R (√[EV² + AV²])",
                        "7. Variation pièces (PV = R_p × K₃)",
                        "8. Variation totale (√[GRR² + PV²])"
                    ],
                    "Calcul": [
                        f"{R_bar_global:.4f} = Moyenne des étendues",
                        f"{X_diff:.4f} = {max(op_global_means):.4f} - {min(op_global_means):.4f}",
                        f"{R_p:.4f} = {max(part_means):.4f} - {min(part_means):.4f}",
                        f"{EV:.4f} = {R_bar_global:.4f} × {k1:.4f}",
                        f"{AV:.4f} = √[({X_diff:.4f}×{k2:.4f})² - {EV**2/(n_parts*n_trials):.4f}]",
                        f"{GRR:.4f} = √[{EV:.4f}² + {AV:.4f}²]",
                        f"{PV:.4f} = {R_p:.4f} × {k3:.4f}",
                        f"{TV:.4f} = √[{GRR:.4f}² + {PV:.4f}²]"
                    ],
                    "Résultat": [
                        f"{R_bar_global:.4f}",
                        f"{X_diff:.4f}",
                        f"{R_p:.4f}",
                        f"{EV:.4f}",
                        f"{AV:.4f}",
                        f"{GRR:.4f}",
                        f"{PV:.4f}",
                        f"{TV:.4f}"
                    ]
                })
                st.dataframe(calc_df, use_container_width=True, height=400)
            
            with tab4:
                # Statistiques globales
                all_vals = np.array(all_measurements)
                stats_data = {
                    "Statistique": [
                        "Nombre total de mesures",
                        "Moyenne globale",
                        "Écart-type global",
                        "Coefficient de variation",
                        "Minimum",
                        "Maximum",
                        "Étendue totale",
                        "Capabilité potentielle (Cp) si tolérance"
                    ],
                    "Valeur": [
                        f"{len(all_vals)}",
                        f"{np.mean(all_vals):.4f}",
                        f"{np.std(all_vals, ddof=1):.4f}",
                        f"{(np.std(all_vals, ddof=1)/np.mean(all_vals)*100 if np.mean(all_vals)!=0 else 0):.2f}%",
                        f"{np.min(all_vals):.4f}",
                        f"{np.max(all_vals):.4f}",
                        f"{np.ptp(all_vals):.4f}",
                        f"{tol_spec/(6*np.std(all_vals, ddof=1)):.2f}" if tol_spec>0 else "N/A"
                    ]
                }
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
            
            # =====================================================
            # 6. VISUALISATIONS
            # =====================================================
            st.subheader("📈 Visualisations")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Diagramme à barres des composantes
                fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
                
                # Composantes absolues
                components = ['EV', 'AV', 'GRR', 'PV', 'TV']
                values = [EV, AV, GRR, PV, TV]
                colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd']
                
                ax1.bar(components, values, color=colors)
                ax1.set_ylabel('Valeur absolue')
                ax1.set_title('Composantes de variation (absolues)')
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Ajouter les valeurs sur les barres
                for i, (comp, val) in enumerate(zip(components, values)):
                    ax1.text(i, val, f'{val:.3f}', ha='center', va='bottom')
                
                # Composantes en pourcentage
                components_pct = ['EV%', 'AV%', 'GRR%', 'PV%']
                values_pct = [EV_pct, AV_pct, GRR_pct, PV_pct]
                colors_pct = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
                
                bars = ax2.bar(components_pct, values_pct, color=colors_pct)
                ax2.set_ylabel('Pourcentage (%)')
                ax2.set_title('Distribution des variations (%)')
                ax2.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Limite 10%')
                ax2.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Limite 30%')
                ax2.grid(True, alpha=0.3, axis='y')
                ax2.legend()
                
                # Ajouter les valeurs sur les barres
                for bar, val in zip(bars, values_pct):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.1f}%', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig1)
            
            with viz_col2:
                # Graphique des moyennes par opérateur
                fig2, ax = plt.subplots(figsize=(10, 6))
                
                x = np.arange(n_parts)
                width = 0.8 / n_operators
                
                for op in range(n_operators):
                    offset = (op - (n_operators-1)/2) * width
                    ax.bar(x + offset, op_means_by_part[op], 
                          width=width, label=f'Op {op+1}', alpha=0.7)
                
                ax.set_xlabel('Pièce')
                ax.set_ylabel('Moyenne des mesures')
                ax.set_title('Moyennes par opérateur et par pièce')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_xticks(x)
                ax.set_xticklabels([f'P{i+1}' for i in range(n_parts)])
                
                plt.tight_layout()
                st.pyplot(fig2)
            
            # Graphique des étendues
            fig3, ax = plt.subplots(figsize=(12, 5))
            
            for op in range(n_operators):
                ax.plot(range(1, n_parts + 1), op_ranges[op], 
                       marker='o', label=f'Op {op+1}', linewidth=2, alpha=0.7)
            
            ax.set_xlabel('Pièce')
            ax.set_ylabel('Étendue')
            ax.set_title('Étendues par pièce et par opérateur')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(1, n_parts + 1))
            
            plt.tight_layout()
            st.pyplot(fig3)
            
            # =====================================================
            # 7. EXPORT DES RÉSULTATS
            # =====================================================
            st.subheader("💾 Export des résultats")
            
            # Préparer les données pour export
            results_summary = pd.DataFrame({
                "Composante": ["EV", "AV", "GRR", "PV", "TV", 
                              "%EV", "%AV", "%GRR", "%PV"],
                "Valeur": [EV, AV, GRR, PV, TV,
                          EV_pct, AV_pct, GRR_pct, PV_pct],
                "Unité": ["absolu", "absolu", "absolu", "absolu", "absolu",
                         "%", "%", "%", "%"]
            })
            
            # Boutons d'export
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            
            with col_exp1:
                # Export CSV
                csv_data = results_summary.to_csv(index=False)
                st.download_button(
                    label="📥 Exporter résultats (CSV)",
                    data=csv_data,
                    file_name="gage_rr_results.csv",
                    mime="text/csv"
                )
            
            with col_exp2:
                # Export Excel
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Données brutes', index=False)
                    detail_df.to_excel(writer, sheet_name='Analyse par pièce', index=False)
                    results_summary.to_excel(writer, sheet_name='Résumé', index=False)
                    calc_df.to_excel(writer, sheet_name='Calculs détaillés', index=False)
                
                st.download_button(
                    label="📥 Exporter rapport complet (Excel)",
                    data=excel_buffer.getvalue(),
                    file_name="rapport_gage_rr_complet.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col_exp3:
                # Export rapport texte
                report_text = f"""
                ===================================
                RAPPORT D'ANALYSE GAGE R&R
                ===================================
                Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                Méthode: {method}
                Niveau de confiance: {confidence}%
                
                PARAMÈTRES DE L'ÉTUDE:
                - Nombre de pièces: {n_parts}
                - Nombre d'opérateurs: {n_operators}
                - Nombre de répétitions: {n_trials}
                - Total mesures: {n_parts * n_operators * n_trials}
                
                CONSTANTES UTILISÉES:
                - K₁ (répétabilité): {k1:.4f}
                - K₂ (reproductibilité): {k2:.4f}
                - K₃ (variation pièces): {k3:.4f}
                
                RÉSULTATS:
                - Répétabilité (EV): {EV:.4f} ({EV_pct:.1f}%)
                - Reproductibilité (AV): {AV:.4f} ({AV_pct:.1f}%)
                - Gage R&R: {GRR:.4f} ({GRR_pct:.1f}%)
                - Variation pièces (PV): {PV:.4f} ({PV_pct:.1f}%)
                - Variation totale (TV): {TV:.4f}
                
                """
                
                if tol_spec > 0:
                    report_text += f"""
                PAR RAPPORT À LA TOLÉRANCE ({tol_spec:.3f}):
                - %EV/Tolérance: {EV_tol_pct:.1f}%
                - %AV/Tolérance: {AV_tol_pct:.1f}%
                - %R&R/Tolérance: {GRR_tol_pct:.1f}%
                
                """
                
                report_text += f"""
                CONCLUSION:
                %R&R = {GRR_pct:.1f}% → Système {'ACCEPTABLE' if GRR_pct < 10 else 'ACCEPTABLE SOUS CONDITIONS' if GRR_pct < 30 else 'INACCEPTABLE'}
                
                ===================================
                """
                
                st.download_button(
                    label="📥 Exporter rapport (TXT)",
                    data=report_text,
                    file_name="rapport_gage_rr.txt",
                    mime="text/plain"
                )
            
            # =====================================================
            # 8. RECOMMANDATIONS
            # =====================================================
            with st.expander("💡 Recommandations", expanded=True):
                if GRR_pct > 30:
                    st.error("**Actions prioritaires nécessaires :**")
                    st.markdown("""
                    1. **Investiguez la source de variation :**
                       - Si %EV élevé → Vérifiez l'instrument de mesure
                       - Si %AV élevé → Formez les opérateurs, standardisez les méthodes
                    2. **Améliorez la précision :**
                       - Calibrez l'équipement
                       - Utilisez un instrument plus précis
                    3. **Revoyez la méthode :**
                       - Clarifiez les instructions
                       - Améliorez le support des pièces
                    """)
                elif GRR_pct > 10:
                    st.warning("**Améliorations recommandées :**")
                    st.markdown("""
                    1. **Surveillance continue :**
                       - Mettez en place des contrôles réguliers
                       - Documentez les procédures
                    2. **Formation :**
                       - Rafraîchissez la formation des opérateurs
                       - Vérifiez la compréhension des méthodes
                    3. **Maintenance préventive :**
                       - Calendrier de calibration strict
                       - Entretien régulier de l'équipement
                    """)
                else:
                    st.success("**Maintenance du système :**")
                    st.markdown("""
                    1. **Surveillance :**
                       - Continuez les vérifications régulières
                       - Documentez toute dérive
                    2. **Amélioration continue :**
                       - Recherchez des opportunités d'amélioration
                       - Partagez les bonnes pratiques
                    """)
        
        except Exception as e:
            st.error(f"❌ Erreur lors du calcul : {str(e)}")
            import traceback
            with st.expander("Détails de l'erreur"):
                st.code(traceback.format_exc())

else:
    st.info("👈 Veuillez importer ou saisir des données pour commencer l'analyse")

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
    <p>Gage R&R - Méthode des étendues et des moyennes | Basé sur AIAG et normes qualité</p>
    <p>© 2024 - Outil pour l'amélioration continue et la maîtrise statistique des processus</p>
</div>
""", unsafe_allow_html=True)
