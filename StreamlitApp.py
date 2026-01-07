import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import math

# =====================================================
# TABLE COMPLÈTE DES VALEURS d2
# =====================================================
# Table d2 selon votre image (z = g, w = m)
D2_TABLE = {
    # w (nombre d'essais) → valeurs pour différents z (nombre de sous-groupes)
    # Les clés sont des tuples (z, w)
    1: {1: 1.41, 2: 1.28, 3: 1.23, 4: 1.21, 5: 1.19, 6: 1.18, 7: 1.17, 
         8: 1.16, 9: 1.16, 10: 1.15, 11: 1.15, 12: 1.15, 13: 1.15, 14: 1.15, 15: 1.15},
    2: {1: 1.91, 2: 1.81, 3: 1.77, 4: 1.75, 5: 1.74, 6: 1.73, 7: 1.73,
         8: 1.72, 9: 1.72, 10: 1.72, 11: 1.71, 12: 1.71, 13: 1.71, 14: 1.71, 15: 1.71},
    3: {1: 2.24, 2: 2.15, 3: 2.12, 4: 2.11, 5: 2.10, 6: 2.09, 7: 2.09,
         8: 2.08, 9: 2.08, 10: 2.08, 11: 2.08, 12: 2.07, 13: 2.07, 14: 2.07, 15: 2.07},
    4: {1: 2.48, 2: 2.40, 3: 2.38, 4: 2.37, 5: 2.36, 6: 2.35, 7: 2.35,
         8: 2.34, 9: 2.34, 10: 2.34, 11: 2.34, 12: 2.34, 13: 2.34, 14: 2.34, 15: 2.34},
    5: {1: 2.67, 2: 2.60, 3: 2.58, 4: 2.57, 5: 2.56, 6: 2.56, 7: 2.55,
         8: 2.55, 9: 2.55, 10: 2.55, 11: 2.55, 12: 2.55, 13: 2.55, 14: 2.54, 15: 2.54},
    6: {1: 2.83, 2: 2.77, 3: 2.75, 4: 2.74, 5: 2.78, 6: 2.73, 7: 2.72,
         8: 2.72, 9: 2.72, 10: 2.72, 11: 2.72, 12: 2.72, 13: 2.71, 14: 2.71, 15: 2.71},
    7: {1: 2.96, 2: 2.91, 3: 2.89, 4: 2.88, 5: 2.87, 6: 2.87, 7: 2.87,
         8: 2.86, 9: 2.86, 10: 2.86, 11: 2.86, 12: 2.85, 13: 2.85, 14: 2.85, 15: 2.85},
    8: {1: 3.08, 2: 3.02, 3: 3.01, 4: 3.00, 5: 2.99, 6: 2.99, 7: 2.99,
         8: 2.98, 9: 2.98, 10: 2.98, 11: 2.98, 12: 2.98, 13: 2.98, 14: 2.98, 15: 2.98},
    9: {1: 3.18, 2: 3.13, 3: 3.11, 4: 3.10, 5: 3.10, 6: 3.10, 7: 3.10,
         8: 3.09, 9: 3.09, 10: 3.09, 11: 3.09, 12: 3.09, 13: 3.09, 14: 3.09, 15: 3.09},
    10: {1: 3.27, 2: 3.22, 3: 3.21, 4: 3.20, 5: 3.19, 6: 3.19, 7: 3.19,
          8: 3.18, 9: 3.18, 10: 3.18, 11: 3.18, 12: 3.18, 13: 3.18, 14: 3.18, 15: 3.18},
    11: {1: 3.35, 2: 3.30, 3: 3.29, 4: 3.28, 5: 3.28, 6: 3.27, 7: 3.27,
          8: 3.27, 9: 3.27, 10: 3.27, 11: 3.27, 12: 3.27, 13: 3.27, 14: 3.27, 15: 3.26},
    12: {1: 3.42, 2: 3.38, 3: 3.37, 4: 3.36, 5: 3.36, 6: 3.35, 7: 3.35,
          8: 3.35, 9: 3.34, 10: 3.34, 11: 3.34, 12: 3.34, 13: 3.34, 14: 3.34, 15: 3.34},
    13: {1: 3.49, 2: 3.45, 3: 3.43, 4: 3.43, 5: 3.42, 6: 3.42, 7: 3.42,
          8: 3.42, 9: 3.41, 10: 3.41, 11: 3.41, 12: 3.41, 13: 3.41, 14: 3.41, 15: 3.41},
    14: {1: 3.55, 2: 3.51, 3: 3.50, 4: 3.49, 5: 3.49, 6: 3.49, 7: 3.48,
          8: 3.48, 9: 3.48, 10: 3.48, 11: 3.48, 12: 3.48, 13: 3.48, 14: 3.48, 15: 3.48}
}

# Valeurs pour g > 15 (approximation)
D2_LARGE_G = {
    1: 1.128, 2: 1.693, 3: 2.059, 4: 2.326, 5: 2.534,
    6: 2.704, 7: 2.847, 8: 2.970, 9: 3.078, 10: 3.173,
    11: 3.258, 12: 3.336, 13: 3.407, 14: 3.472, 15: 3.535
}

# =====================================================
# FONCTION POUR OBTENIR d2
# =====================================================
def get_d2(z, w):
    """
    Retourne la valeur d2 selon la table.
    z = nombre de sous-groupes (g)
    w = taille du sous-groupe (m)
    """
    # Limiter w à 15 (max dans la table)
    w = min(w, 15)
    
    if z <= 15:
        # Utiliser la table complète
        return D2_TABLE.get(w, {}).get(z, D2_LARGE_G.get(w, 1.0))
    else:
        # Utiliser les valeurs pour g > 15
        return D2_LARGE_G.get(w, 1.0)

# =====================================================
# CONFIGURATION PAGE
# =====================================================
st.set_page_config(
    page_title="Gage R&R - Méthode d2 (Étendues et Moyennes)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# SIDEBAR - GUIDE ET FORMULES
# =====================================================
with st.sidebar:
    st.title("ℹ️ Guide Gage R&R - Méthode d2")
    
    with st.expander("📐 Formules exactes (selon votre méthode)", expanded=True):
        st.markdown("""
        **1. Répétabilité (EV):**
        ```
        Répétabilité = (5.15 × R̄) / d2₁
        ```
        - **R̄** = moyenne des moyennes des étendues par tous les opérateurs
        - **d2₁** = avec Z = n × k, w = r
        - n = pièces, k = opérateurs, r = essais
        
        **2. Reproductibilité (AV):**
        ```
        Reproductibilité = √[((5.15 × X_étendue)/d2₂)² - (Répétabilité²/(n×r))]
        ```
        - **X_étendue** = max(moyenne_op) - min(moyenne_op)
        - **d2₂** = avec Z = 1, w = k (nombre d'opérateurs)
        
        **3. Variation Pièces (PV):**
        ```
        V_p = (5.15 × R_p) / d2₃
        ```
        - **R_p** = max(moyenne_pièce) - min(moyenne_pièce)
        - **d2₃** = avec Z = 1, w = n (nombre de pièces)
        
        **4. Gage R&R:**
        ```
        R&R = √(Répétabilité² + Reproductibilité²)
        ```
        
        **5. Variation Totale (TV):**
        ```
        V_T = √(R&R² + V_p²)
        ```
        """)
    
    with st.expander("📊 Table des valeurs d2", expanded=False):
        st.markdown("""
        | m \\ g | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
        |--------|---|---|---|---|---|---|---|---|---|----|----|----|----|----|----|
        | 2 | 1.41 | 1.28 | 1.23 | 1.21 | 1.19 | 1.18 | 1.17 | 1.16 | 1.16 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 |
        | 3 | 1.91 | 1.81 | 1.77 | 1.75 | 1.74 | 1.73 | 1.73 | 1.72 | 1.72 | 1.72 | 1.71 | 1.71 | 1.71 | 1.71 | 1.71 |
        | 4 | 2.24 | 2.15 | 2.12 | 2.11 | 2.10 | 2.09 | 2.09 | 2.08 | 2.08 | 2.08 | 2.08 | 2.07 | 2.07 | 2.07 | 2.07 |
        | 5 | 2.48 | 2.40 | 2.38 | 2.37 | 2.36 | 2.35 | 2.35 | 2.34 | 2.34 | 2.34 | 2.34 | 2.34 | 2.34 | 2.34 | 2.34 |
        | 6 | 2.67 | 2.60 | 2.58 | 2.57 | 2.56 | 2.56 | 2.55 | 2.55 | 2.55 | 2.55 | 2.55 | 2.55 | 2.55 | 2.54 | 2.54 |
        | 7 | 2.83 | 2.77 | 2.75 | 2.74 | 2.78 | 2.73 | 2.72 | 2.72 | 2.72 | 2.72 | 2.72 | 2.72 | 2.71 | 2.71 | 2.71 |
        | 8 | 2.96 | 2.91 | 2.89 | 2.88 | 2.87 | 2.87 | 2.87 | 2.86 | 2.86 | 2.86 | 2.86 | 2.85 | 2.85 | 2.85 | 2.85 |
        | 9 | 3.08 | 3.02 | 3.01 | 3.00 | 2.99 | 2.99 | 2.99 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 |
        | 10| 3.18 | 3.13 | 3.11 | 3.10 | 3.10 | 3.10 | 3.10 | 3.09 | 3.09 | 3.09 | 3.09 | 3.09 | 3.09 | 3.09 | 3.09 |
        | 11| 3.27 | 3.22 | 3.21 | 3.20 | 3.19 | 3.19 | 3.19 | 3.18 | 3.18 | 3.18 | 3.18 | 3.18 | 3.18 | 3.18 | 3.18 |
        | 12| 3.35 | 3.30 | 3.29 | 3.28 | 3.28 | 3.27 | 3.27 | 3.27 | 3.27 | 3.27 | 3.27 | 3.27 | 3.27 | 3.27 | 3.27 |
        | 13| 3.42 | 3.38 | 3.37 | 3.36 | 3.36 | 3.35 | 3.35 | 3.35 | 3.34 | 3.34 | 3.34 | 3.34 | 3.34 | 3.34 | 3.34 |
        | 14| 3.49 | 3.45 | 3.43 | 3.43 | 3.42 | 3.42 | 3.42 | 3.42 | 3.41 | 3.41 | 3.41 | 3.41 | 3.41 | 3.41 | 3.41 |
        | 15| 3.55 | 3.51 | 3.50 | 3.49 | 3.49 | 3.49 | 3.48 | 3.48 | 3.48 | 3.48 | 3.48 | 3.48 | 3.48 | 3.48 | 3.48 |
        
        **g = nombre de sous-groupes (Z)**  
        **m = taille du sous-groupe (W)**
        """)
    
    with st.expander("🎯 Critères d'acceptation", expanded=False):
        st.markdown("""
        **Selon l'AIAG:**
        - ✅ **< 10%** : Système acceptable
        - ⚠️ **10% - 30%** : Acceptable sous conditions
        - ❌ **> 30%** : Système inacceptable
        """)
    
    st.divider()
    
    st.markdown("### ⚙️ Paramètres")
    
    # Facteur de confiance (5.15 pour 99%)
    confidence_factor = st.number_input(
        "Facteur de confiance (k)",
        value=5.15,
        min_value=4.0,
        max_value=6.0,
        step=0.01,
        help="Valeur 5.15 pour 99% de confiance, 4.0 pour 95%"
    )
    
    tol_spec = st.number_input(
        "Tolérance spécifiée (optionnel)", 
        value=0.0,
        help="Pour calculer %R&R/Tolérance"
    )

# =====================================================
# HEADER PRINCIPAL
# =====================================================
st.title("📊 Étude de la précision - Gage R&R")
st.markdown("**Méthode des étendues et des moyennes avec table d₂**")

# =====================================================
# IMPORT DES DONNÉES
# =====================================================
st.subheader("📤 Importation des données")

upload_option = st.radio(
    "Choix du mode d'entrée",
    ["📁 Importer un fichier", "✍️ Saisie manuelle"],
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
        help="Format: colonnes = Op1_Essai1, Op1_Essai2, Op2_Essai1, ..."
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine="openpyxl")
            
            # Nettoyage
            df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
            df = df.apply(pd.to_numeric, errors='coerce')
            
            st.success(f"✅ Fichier importé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
            
            # Détection automatique
            st.subheader("🔧 Configuration")
            
            # Compter les opérateurs uniques
            op_patterns = {}
            for col in df.columns:
                col_str = str(col).lower()
                if 'op' in col_str or 'operateur' in col_str or 'operator' in col_str:
                    # Essayer d'extraire le numéro d'opérateur
                    import re
                    op_match = re.search(r'op[^\d]*(\d+)', col_str)
                    if op_match:
                        op_num = int(op_match.group(1))
                        if op_num not in op_patterns:
                            op_patterns[op_num] = 0
                        op_patterns[op_num] += 1
            
            if op_patterns:
                n_operators = len(op_patterns)
                n_trials = list(op_patterns.values())[0]  # Prendre le premier comme référence
                n_parts = df.shape[0]
                
                st.info(f"""
                **Détection automatique:**
                - Pièces: {n_parts}
                - Opérateurs: {n_operators}
                - Essais par opérateur: {n_trials}
                """)
            else:
                # Configuration manuelle
                cols = st.columns(3)
                with cols[0]:
                    n_parts = st.number_input("Nombre de pièces", min_value=2, value=df.shape[0])
                with cols[1]:
                    n_operators = st.number_input("Nombre d'opérateurs", min_value=2, value=3)
                with cols[2]:
                    n_trials = st.number_input("Nombre d'essais", min_value=2, value=2)
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'import: {str(e)}")

else:  # Saisie manuelle
    cols = st.columns(3)
    with cols[0]:
        n_parts = st.number_input("Nombre de pièces", min_value=2, max_value=50, value=10)
    with cols[1]:
        n_operators = st.number_input("Nombre d'opérateurs", min_value=2, max_value=10, value=3)
    with cols[2]:
        n_trials = st.number_input("Nombre d'essais", min_value=2, max_value=10, value=3)
    
    total_cols = n_operators * n_trials
    
    # Création d'un DataFrame vide
    col_names = []
    for op in range(n_operators):
        for t in range(n_trials):
            col_names.append(f"Op{op+1}_Essai{t+1}")
    
    # Valeurs par défaut réalistes
    np.random.seed(42)
    base_values = np.random.normal(100, 5, n_parts)  # Valeurs réelles des pièces
    
    data = {}
    for i, col in enumerate(col_names):
        op_num = (i // n_trials) + 1
        op_bias = np.random.normal(0, 0.3)  # Petit biais par opérateur
        noise = np.random.normal(0, 0.2, n_parts)  # Bruit de mesure
        data[col] = base_values + op_bias + noise
    
    df = pd.DataFrame(data)
    st.info("⚠️ Modifiez les valeurs dans le tableau ci-dessous")

# =====================================================
# AFFICHAGE ET ÉDITION DES DONNÉES
# =====================================================
if df is not None:
    st.subheader("📥 Données de mesure")
    
    # Afficher les premières lignes
    st.dataframe(df, use_container_width=True, height=300)
    
    # =====================================================
    # CALCUL GAGE R&R (MÉTHODE EXACTE)
    # =====================================================
    if st.button("🚀 Calculer Gage R&R avec méthode d2", type="primary", use_container_width=True):
        
        try:
            # =====================================================
            # 1. PRÉPARATION DES DONNÉES
            # =====================================================
            st.subheader("🔍 Préparation des données")
            
            # S'assurer que les données sont numériques
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
            df_numeric = df_numeric.fillna(0)
            
            # Organiser les données par opérateur
            op_ranges = []  # Étendues par opérateur et par pièce
            op_means = []   # Moyennes par opérateur et par pièce
            op_global_means = []  # Moyenne globale par opérateur
            
            for op in range(n_operators):
                # Sélectionner les colonnes de cet opérateur
                op_cols = [f"Op{op+1}_Essai{t+1}" for t in range(n_trials)]
                
                # Si les colonnes n'existent pas, essayer d'autres formats
                if not all(col in df_numeric.columns for col in op_cols):
                    # Essayer de trouver les colonnes par position
                    start_idx = op * n_trials
                    end_idx = start_idx + n_trials
                    if end_idx <= len(df_numeric.columns):
                        op_cols = df_numeric.columns[start_idx:end_idx].tolist()
                    else:
                        st.error(f"❌ Impossible de trouver les colonnes pour l'opérateur {op+1}")
                        st.stop()
                
                df_op = df_numeric[op_cols]
                
                # Calculer les étendues par pièce
                ranges = df_op.max(axis=1) - df_op.min(axis=1)
                op_ranges.append(ranges)
                
                # Calculer les moyennes par pièce
                means = df_op.mean(axis=1)
                op_means.append(means)
                
                # Moyenne globale de l'opérateur
                op_global_means.append(means.mean())
            
            # =====================================================
            # 2. CALCUL DES PARAMÈTRES INTERMÉDIAIRES
            # =====================================================
            # a) R̄ : moyenne des moyennes des étendues
            # D'abord, moyenne des étendues par opérateur
            op_range_means = [r.mean() for r in op_ranges]
            # Puis moyenne de ces moyennes
            R_bar = np.mean(op_range_means)
            
            # b) X_étendue : différence entre les moyennes max et min des opérateurs
            X_range = max(op_global_means) - min(op_global_means)
            
            # c) R_p : différence entre les moyennes max et min des pièces
            # Calculer les moyennes par pièce (tous opérateurs confondus)
            part_means = []
            for part_idx in range(n_parts):
                part_values = []
                for op in range(n_operators):
                    part_values.extend(df_numeric.iloc[part_idx, op*n_trials:(op+1)*n_trials])
                part_means.append(np.mean(part_values))
            
            R_p = max(part_means) - min(part_means)
            
            # Afficher les paramètres intermédiaires
            st.info(f"""
            **Paramètres intermédiaires:**
            - R̄ (moyenne des moyennes des étendues) = {R_bar:.4f}
            - X_étendue (différence moyennes opérateurs) = {X_range:.4f}
            - R_p (étendue des moyennes pièces) = {R_p:.4f}
            """)
            
            # =====================================================
            # 3. CALCUL DES VALEURS d2
            # =====================================================
            st.subheader("📊 Valeurs d2 utilisées")
            
            # d2₁ pour la répétabilité : Z = n × k, w = r
            z1 = n_parts * n_operators
            w1 = n_trials
            d2_1 = get_d2(z1, w1)
            
            # d2₂ pour la reproductibilité : Z = 1, w = k
            z2 = 1
            w2 = n_operators
            d2_2 = get_d2(z2, w2)
            
            # d2₃ pour la variation pièces : Z = 1, w = n
            z3 = 1
            w3 = n_parts
            d2_3 = get_d2(z3, w3)
            
            # Afficher les valeurs d2
            d2_df = pd.DataFrame({
                "Composante": ["Répétabilité (EV)", "Reproductibilité (AV)", "Variation Pièces (PV)"],
                "Z (g)": [z1, z2, z3],
                "W (m)": [w1, w2, w3],
                "d2": [d2_1, d2_2, d2_3],
                "Formule": [
                    f"Z = n×k = {n_parts}×{n_operators}, W = r = {n_trials}",
                    f"Z = 1, W = k = {n_operators}",
                    f"Z = 1, W = n = {n_parts}"
                ]
            })
            st.dataframe(d2_df, use_container_width=True)
            
            # =====================================================
            # 4. CALCUL DES COMPOSANTES
            # =====================================================
            st.subheader("🧮 Calcul des composantes")
            
            # 4.1 Répétabilité (EV)
            EV = (confidence_factor * R_bar) / d2_1
            
            # 4.2 Reproductibilité (AV)
            av_numerateur = (confidence_factor * X_range) / d2_2
            av_soustraction = (EV ** 2) / (n_parts * n_trials)
            av_term = av_numerateur ** 2 - av_soustraction
            
            if av_term >= 0:
                AV = math.sqrt(av_term)
            else:
                AV = 0
                st.warning("⚠️ Le terme sous la racine pour AV est négatif. AV est fixé à 0.")
            
            # 4.3 Gage R&R
            GRR = math.sqrt(EV ** 2 + AV ** 2)
            
            # 4.4 Variation Pièces (PV)
            PV = (confidence_factor * R_p) / d2_3
            
            # 4.5 Variation Totale (TV)
            TV = math.sqrt(GRR ** 2 + PV ** 2)
            
            # 4.6 Pourcentages
            if TV > 0:
                EV_pct = (EV / TV) * 100
                AV_pct = (AV / TV) * 100
                GRR_pct = (GRR / TV) * 100
                PV_pct = (PV / TV) * 100
            else:
                EV_pct = AV_pct = GRR_pct = PV_pct = 0
            
            # 4.7 %R&R/Tolérance si spécifiée
            if tol_spec > 0:
                GRR_tol_pct = (GRR / tol_spec) * 100
                ndc = (1.41 * PV) / GRR if GRR > 0 else 0  # Nombre de catégories distinctes
            else:
                GRR_tol_pct = None
                ndc = None
            
            # =====================================================
            # 5. AFFICHAGE DES RÉSULTATS DÉTAILLÉS
            # =====================================================
            st.subheader("📊 Résultats détaillés")
            
            # Tableau des calculs étape par étape
            calc_steps = pd.DataFrame({
                "Étape": [
                    "1. Répétabilité (EV)",
                    "2. Reproductibilité (AV)",
                    "3. Gage R&R",
                    "4. Variation Pièces (PV)",
                    "5. Variation Totale (TV)"
                ],
                "Formule": [
                    f"({confidence_factor} × {R_bar:.4f}) / {d2_1:.4f}",
                    f"√[(({confidence_factor} × {X_range:.4f})/{d2_2:.4f})² - ({EV:.4f}²/({n_parts}×{n_trials}))]",
                    f"√({EV:.4f}² + {AV:.4f}²)",
                    f"({confidence_factor} × {R_p:.4f}) / {d2_3:.4f}",
                    f"√({GRR:.4f}² + {PV:.4f}²)"
                ],
                "Calcul": [
                    f"{confidence_factor * R_bar:.4f} / {d2_1:.4f}",
                    f"√[({confidence_factor * X_range / d2_2:.4f})² - {av_soustraction:.4f}]",
                    f"√({EV**2:.4f} + {AV**2:.4f})",
                    f"{confidence_factor * R_p:.4f} / {d2_3:.4f}",
                    f"√({GRR**2:.4f} + {PV**2:.4f})"
                ],
                "Résultat": [
                    f"{EV:.4f}",
                    f"{AV:.4f}",
                    f"{GRR:.4f}",
                    f"{PV:.4f}",
                    f"{TV:.4f}"
                ]
            })
            
            st.dataframe(calc_steps, use_container_width=True)
            
            # =====================================================
            # 6. RÉSULTATS FINAUX
            # =====================================================
            st.subheader("🎯 Résultats finaux")
            
            # Métriques principales
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Répétabilité (EV)", f"{EV:.4f}", f"{EV_pct:.1f}%")
                st.caption(f"d2 = {d2_1:.4f}")
            
            with col2:
                st.metric("Reproductibilité (AV)", f"{AV:.4f}", f"{AV_pct:.1f}%")
                st.caption(f"d2 = {d2_2:.4f}")
            
            with col3:
                st.metric("Gage R&R", f"{GRR:.4f}", f"{GRR_pct:.1f}%")
                st.caption(f"√(EV² + AV²)")
            
            with col4:
                st.metric("Variation Pièces (PV)", f"{PV:.4f}", f"{PV_pct:.1f}%")
                st.caption(f"d2 = {d2_3:.4f}")
            
            with col5:
                st.metric("Variation Totale (TV)", f"{TV:.4f}", "100%")
                st.caption(f"√(R&R² + PV²)")
            
            # Évaluation
            st.subheader("📈 Évaluation du système")
            
            if GRR_pct < 10:
                status = "✅ ACCEPTABLE"
                color = "green"
                emoji = "✅"
            elif GRR_pct < 30:
                status = "⚠️ ACCEPTABLE SOUS CONDITIONS"
                color = "orange"
                emoji = "⚠️"
            else:
                status = "❌ INACCEPTABLE"
                color = "red"
                emoji = "❌"
            
            st.markdown(f"""
            <div style="background-color:{color}20; padding:20px; border-radius:10px; border-left:5px solid {color}; margin:20px 0;">
                <h3 style="color:{color}; margin:0;">{emoji} {status}</h3>
                <p style="font-size:1.2em; margin:10px 0;">
                    <strong>%R&R = {GRR_pct:.1f}%</strong><br>
                    <small>%EV = {EV_pct:.1f}% | %AV = {AV_pct:.1f}% | %PV = {PV_pct:.1f}%</small>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Barre de progression
            progress_value = min(GRR_pct / 30, 1.0)
            st.progress(progress_value, text=f"%R&R: {GRR_pct:.1f}% / Limite: 30%")
            
            # Informations supplémentaires
            if tol_spec > 0:
                st.info(f"📏 **%R&R/Tolérance = {GRR_tol_pct:.1f}%** (tolérance = {tol_spec:.3f})")
            
            if ndc:
                st.info(f"🔢 **Nombre de catégories distinctes (ndc) = {ndc:.1f}**")
            
            # =====================================================
            # 7. VISUALISATIONS
            # =====================================================
            st.subheader("📈 Visualisations")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # 7.1 Composantes de variation (absolues)
            components = ['EV', 'AV', 'GRR', 'PV', 'TV']
            values = [EV, AV, GRR, PV, TV]
            colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd']
            
            bars1 = ax1.bar(components, values, color=colors)
            ax1.set_ylabel('Valeur absolue')
            ax1.set_title('Composantes de variation (absolues)')
            ax1.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars1, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom')
            
            # 7.2 Composantes de variation (%)
            components_pct = ['EV%', 'AV%', 'GRR%', 'PV%']
            values_pct = [EV_pct, AV_pct, GRR_pct, PV_pct]
            colors_pct = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
            
            bars2 = ax2.bar(components_pct, values_pct, color=colors_pct)
            ax2.set_ylabel('Pourcentage (%)')
            ax2.set_title('Distribution des variations (%)')
            ax2.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='10%')
            ax2.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='30%')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.legend()
            
            for bar, val in zip(bars2, values_pct):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}%', ha='center', va='bottom')
            
            # 7.3 Moyennes par opérateur
            op_indices = np.arange(n_operators)
            ax3.bar(op_indices, op_global_means, color='skyblue', alpha=0.7)
            ax3.set_xlabel('Opérateur')
            ax3.set_ylabel('Moyenne')
            ax3.set_title('Moyennes globales par opérateur')
            ax3.set_xticks(op_indices)
            ax3.set_xticklabels([f'Op{i+1}' for i in op_indices])
            ax3.grid(True, alpha=0.3, axis='y')
            
            for i, mean in enumerate(op_global_means):
                ax3.text(i, mean, f'{mean:.2f}', ha='center', va='bottom')
            
            # 7.4 Moyennes par pièce
            part_indices = np.arange(n_parts)
            ax4.plot(part_indices, part_means, 'o-', color='green', linewidth=2, markersize=6)
            ax4.set_xlabel('Pièce')
            ax4.set_ylabel('Moyenne')
            ax4.set_title('Moyennes par pièce (tous opérateurs)')
            ax4.set_xticks(part_indices)
            ax4.set_xticklabels([f'P{i+1}' for i in part_indices])
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # =====================================================
            # 8. EXPORT DES RÉSULTATS
            # =====================================================
            st.subheader("💾 Export des résultats")
            
            # Préparer les données pour export
            results_df = pd.DataFrame({
                "Paramètre": ["EV", "AV", "GRR", "PV", "TV", 
                             "%EV", "%AV", "%GRR", "%PV", "R̄", "X_range", "R_p"],
                "Valeur": [EV, AV, GRR, PV, TV,
                          EV_pct, AV_pct, GRR_pct, PV_pct, R_bar, X_range, R_p],
                "Description": [
                    "Répétabilité", "Reproductibilité", "Gage R&R", "Variation Pièces", "Variation Totale",
                    "Pourcentage EV", "Pourcentage AV", "Pourcentage GRR", "Pourcentage PV",
                    "Moyenne des moyennes des étendues", "Différence des moyennes opérateurs", "Étendue des moyennes pièces"
                ]
            })
            
            d2_results_df = pd.DataFrame({
                "Composante": ["EV", "AV", "PV"],
                "d2": [d2_1, d2_2, d2_3],
                "Z (g)": [z1, z2, z3],
                "W (m)": [w1, w2, w3]
            })
            
            # Boutons d'export
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export CSV des résultats
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Résultats (CSV)",
                    data=csv_data,
                    file_name="gage_rr_resultats.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export Excel complet
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    df_numeric.to_excel(writer, sheet_name='Données brutes', index=False)
                    results_df.to_excel(writer, sheet_name='Résultats', index=False)
                    d2_results_df.to_excel(writer, sheet_name='Valeurs d2', index=False)
                    calc_steps.to_excel(writer, sheet_name='Calculs détaillés', index=False)
                
                st.download_button(
                    label="📥 Rapport complet (Excel)",
                    data=excel_buffer.getvalue(),
                    file_name="rapport_gage_rr_complet.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col3:
                # Export rapport texte
                report_text = f"""
                ============================================
                RAPPORT D'ANALYSE GAGE R&R - MÉTHODE d2
                ============================================
                Date: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')}
                Facteur de confiance: {confidence_factor} (99%)
                
                PARAMÈTRES DE L'ÉTUDE:
                - Nombre de pièces (n): {n_parts}
                - Nombre d'opérateurs (k): {n_operators}
                - Nombre d'essais (r): {n_trials}
                - Total mesures: {n_parts * n_operators * n_trials}
                
                VALEURS d2 UTILISÉES:
                - Répétabilité: d2 = {d2_1:.4f} (Z={z1}, W={w1})
                - Reproductibilité: d2 = {d2_2:.4f} (Z={z2}, W={w2})
                - Variation Pièces: d2 = {d2_3:.4f} (Z={z3}, W={w3})
                
                PARAMÈTRES INTERMÉDIAIRES:
                - R̄ = {R_bar:.4f}
                - X_étendue = {X_range:.4f}
                - R_p = {R_p:.4f}
                
                RÉSULTATS:
                - Répétabilité (EV) = {EV:.4f} ({EV_pct:.1f}%)
                - Reproductibilité (AV) = {AV:.4f} ({AV_pct:.1f}%)
                - Gage R&R = {GRR:.4f} ({GRR_pct:.1f}%)
                - Variation Pièces (PV) = {PV:.4f} ({PV_pct:.1f}%)
                - Variation Totale (TV) = {TV:.4f}
                
                ÉVALUATION:
                %R&R = {GRR_pct:.1f}%
                Classification: {status}
                """
                
                if tol_spec > 0:
                    report_text += f"""
                PAR RAPPORT À LA TOLÉRANCE:
                - Tolérance spécifiée: {tol_spec:.3f}
                - %R&R/Tolérance: {GRR_tol_pct:.1f}%
                    """
                
                report_text += f"""
                
                ============================================
                """
                
                st.download_button(
                    label="📥 Rapport (TXT)",
                    data=report_text,
                    file_name="rapport_gage_rr.txt",
                    mime="text/plain"
                )
            
            # =====================================================
            # 9. INTERPRÉTATION ET RECOMMANDATIONS
            # =====================================================
            with st.expander("💡 Interprétation et recommandations", expanded=True):
                st.markdown("### 🔍 Analyse des résultats")
                
                if GRR_pct < 10:
                    st.success("""
                    **Système de mesure EXCELLENT**
                    
                    **Actions recommandées:**
                    - Continuer la surveillance régulière
                    - Maintenir les procédures actuelles
                    - Documenter les bonnes pratiques
                    """)
                elif GRR_pct < 30:
                    st.warning("""
                    **Système de mesure ACCEPTABLE SOUS CONDITIONS**
                    
                    **Analyse des composantes:**
                    """)
                    
                    if EV_pct > AV_pct:
                        st.info("""
                        **Problème principal: RÉPÉTABILITÉ (EV)**
                        - L'instrument de mesure peut être instable
                        - Le processus de mesure manque de précision
                        
                        **Actions recommandées:**
                        1. Vérifier la calibration de l'équipement
                        2. Inspecter l'usure des outils
                        3. Standardiser la méthode de mesure
                        """)
                    else:
                        st.info("""
                        **Problème principal: REPRODUCTIBILITÉ (AV)**
                        - Les opérateurs ont des méthodes différentes
                        - Manque de formation ou de standardisation
                        
                        **Actions recommandées:**
                        1. Former les opérateurs de manière uniforme
                        2. Créer des instructions de travail claires
                        3. Mettre en place des audits croisés
                        """)
                else:
                    st.error("""
                    **Système de mesure INACCEPTABLE**
                    
                    **Actions URGENTES nécessaires:**
                    1. **Arrêter** l'utilisation du système actuel
                    2. **Investigation complète** des causes
                    3. **Remplacement** ou **réparation** de l'équipement si nécessaire
                    4. **Reformation** complète des opérateurs
                    
                    **Points à vérifier:**
                    - État de l'équipement de mesure
                    - Compétence des opérateurs
                    - Procédures de mesure
                    - Conditions environnementales
                    """)
                
                st.markdown("---")
                st.markdown("""
                **📊 Guide d'interprétation:**
                - **%EV élevé**: Problème avec l'équipement de mesure
                - **%AV élevé**: Problème avec les opérateurs ou la méthode
                - **%PV faible**: Les pièces ne sont pas assez différentes pour évaluer correctement le système
                """)
        
        except Exception as e:
            st.error(f"❌ Erreur lors du calcul : {str(e)}")
            import traceback
            with st.expander("Détails techniques de l'erreur"):
                st.code(traceback.format_exc())

else:
    st.info("👈 Veuillez importer ou saisir des données pour commencer l'analyse")

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
    <p><strong>Étude de la précision - Gage R&R | Méthode des étendues et des moyennes avec table d₂</strong></p>
    <p>Méthode conforme aux spécifications fournies | Facteur de confiance: 5.15 (99%)</p>
</div>
""", unsafe_allow_html=True)
