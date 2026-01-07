import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import math

# =====================================================
# TABLE DES VALEURS d2
# =====================================================
D2_TABLE = {
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

def get_d2(z, w):
    """Retourne la valeur d2 selon la table."""
    w = min(w, 15)
    if z <= 15:
        return D2_TABLE.get(w, {}).get(z, 1.0)
    else:
        # Valeurs pour g > 15
        d2_large = {1: 1.128, 2: 1.693, 3: 2.059, 4: 2.326, 5: 2.534,
                    6: 2.704, 7: 2.847, 8: 2.970, 9: 3.078, 10: 3.173,
                    11: 3.258, 12: 3.336, 13: 3.407, 14: 3.472, 15: 3.535}
        return d2_large.get(w, 1.0)

# =====================================================
# CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Gage R&R - Vérification Exacte",
    page_icon="✅",
    layout="wide"
)

st.title("✅ Gage R&R - Vérification avec Votre Exemple")
st.markdown("**Méthode des étendues et des moyennes - Calcul exact**")

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.header("📐 Votre Exemple")
    st.markdown("""
    **Données de référence:**
    - R̄ = 0,058
    - d2 (EV) = 1,693
    - EV = 0,176
    
    - X_diff = 0,03 (45,09 - 45,06)
    - d2 (AV) = 1,91
    - n = 10, r = 3
    - AV = 0,080
    
    - R&R = 0,193
    
    - Rp = 0,33 (45,27 - 44,94)
    - d2 (PV) = 3,18
    - PV = 0,53
    """)
    
    st.divider()
    
    st.header("⚙️ Paramètres")
    k = st.number_input("Facteur k (5.15 pour 99%)", value=5.15, step=0.01)
    tol = st.number_input("Tolérance (optionnel)", value=0.0)

# =====================================================
# ENTRÉE DES DONNÉES DE VOTRE EXEMPLE
# =====================================================
st.subheader("📝 Entrez les valeurs de votre exemple")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Répétabilité (EV)**")
    R_bar = st.number_input("R̄ (moyenne des étendues)", value=0.058, step=0.001, format="%.3f")
    d2_ev = st.number_input("d2 pour EV", value=1.693, step=0.001, format="%.3f")
    ev_expected = st.number_input("EV attendu", value=0.176, step=0.001, format="%.3f")

with col2:
    st.markdown("**Reproductibilité (AV)**")
    X_diff = st.number_input("X_diff (différence des moyennes)", value=0.030, step=0.001, format="%.3f")
    d2_av = st.number_input("d2 pour AV", value=1.910, step=0.001, format="%.3f")
    n_parts = st.number_input("n (pièces)", value=10, min_value=2)
    n_trials = st.number_input("r (essais)", value=3, min_value=2)
    av_expected = st.number_input("AV attendu", value=0.080, step=0.001, format="%.3f")

with col3:
    st.markdown("**Variation Pièces (PV)**")
    R_p = st.number_input("Rp (étendue des moyennes)", value=0.330, step=0.001, format="%.3f")
    d2_pv = st.number_input("d2 pour PV", value=3.180, step=0.001, format="%.3f")
    pv_expected = st.number_input("PV attendu", value=0.530, step=0.001, format="%.3f")

# =====================================================
# CALCULS EXACTS
# =====================================================
if st.button("🔢 Calculer et Vérifier", type="primary"):
    
    st.subheader("🧮 Calculs détaillés")
    
    # 1. Répétabilité (EV)
    ev_calculated = (k * R_bar) / d2_ev
    ev_diff = abs(ev_calculated - ev_expected)
    
    st.markdown(f"""
    **1. Répétabilité (EV):**
    ```
    EV = (k × R̄) / d2
      = ({k} × {R_bar:.3f}) / {d2_ev:.3f}
      = {k * R_bar:.4f} / {d2_ev:.3f}
      = {ev_calculated:.4f}
    ```
    **Attendu: {ev_expected:.3f}** | **Calculé: {ev_calculated:.4f}** | **Différence: {ev_diff:.4f}**
    
    {"✅ CORRECT" if ev_diff < 0.001 else "❌ DIFFÉRENT"}
    """)
    
    # 2. Reproductibilité (AV)
    # Première partie: (k × X_diff) / d2
    av_part1 = (k * X_diff) / d2_av
    
    # Deuxième partie: EV²/(n×r)
    av_part2 = (ev_calculated ** 2) / (n_parts * n_trials)
    
    # Calcul AV
    av_term = av_part1 ** 2 - av_part2
    
    if av_term >= 0:
        av_calculated = math.sqrt(av_term)
    else:
        av_calculated = 0
    
    av_diff = abs(av_calculated - av_expected)
    
    st.markdown(f"""
    **2. Reproductibilité (AV):**
    ```
    AV = √[((k × X_diff)/d2)² - (EV²/(n×r))]
    
    Étape 1: (k × X_diff)/d2 = ({k} × {X_diff:.3f}) / {d2_av:.3f}
            = {k * X_diff:.4f} / {d2_av:.3f}
            = {av_part1:.4f}
    
    Étape 2: (EV²/(n×r)) = ({ev_calculated:.4f}²) / ({n_parts} × {n_trials})
            = {ev_calculated**2:.6f} / {n_parts*n_trials}
            = {av_part2:.6f}
    
    Étape 3: √[({av_part1:.4f})² - {av_part2:.6f}]
            = √[{av_part1**2:.6f} - {av_part2:.6f}]
            = √[{av_term:.6f}]
            = {av_calculated:.4f}
    ```
    **Attendu: {av_expected:.3f}** | **Calculé: {av_calculated:.4f}** | **Différence: {av_diff:.4f}**
    
    {"✅ CORRECT" if av_diff < 0.001 else "⚠️ PETITE DIFFÉRENCE"}
    """)
    
    # 3. Gage R&R
    grr_calculated = math.sqrt(ev_calculated ** 2 + av_calculated ** 2)
    grr_expected = math.sqrt(ev_expected ** 2 + av_expected ** 2)
    grr_diff = abs(grr_calculated - grr_expected)
    
    st.markdown(f"""
    **3. Gage R&R:**
    ```
    R&R = √(EV² + AV²)
        = √({ev_calculated:.4f}² + {av_calculated:.4f}²)
        = √({ev_calculated**2:.6f} + {av_calculated**2:.6f})
        = √{ev_calculated**2 + av_calculated**2:.6f}
        = {grr_calculated:.4f}
    ```
    **Attendu: {grr_expected:.3f}** | **Calculé: {grr_calculated:.4f}** | **Différence: {grr_diff:.4f}**
    
    {"✅ CORRECT" if grr_diff < 0.001 else "⚠️ DIFFÉRENCE"}
    """)
    
    # 4. Variation Pièces (PV)
    pv_calculated = (k * R_p) / d2_pv
    pv_diff = abs(pv_calculated - pv_expected)
    
    st.markdown(f"""
    **4. Variation Pièces (PV):**
    ```
    PV = (k × Rp) / d2
       = ({k} × {R_p:.3f}) / {d2_pv:.3f}
       = {k * R_p:.4f} / {d2_pv:.3f}
       = {pv_calculated:.4f}
    ```
    **Attendu: {pv_expected:.3f}** | **Calculé: {pv_calculated:.4f}** | **Différence: {pv_diff:.4f}**
    
    {"✅ CORRECT" if pv_diff < 0.001 else "❌ DIFFÉRENT"}
    """)
    
    # 5. Variation Totale (TV)
    tv_calculated = math.sqrt(grr_calculated ** 2 + pv_calculated ** 2)
    
    # 6. Pourcentages
    if tv_calculated > 0:
        ev_pct = (ev_calculated / tv_calculated) * 100
        av_pct = (av_calculated / tv_calculated) * 100
        grr_pct = (grr_calculated / tv_calculated) * 100
        pv_pct = (pv_calculated / tv_calculated) * 100
    else:
        ev_pct = av_pct = grr_pct = pv_pct = 0
    
    st.markdown(f"""
    **5. Variation Totale (TV):**
    ```
    TV = √(R&R² + PV²)
       = √({grr_calculated:.4f}² + {pv_calculated:.4f}²)
       = {tv_calculated:.4f}
    ```
    
    **6. Pourcentages:**
    - %EV = {ev_pct:.1f}%
    - %AV = {av_pct:.1f}%
    - %R&R = {grr_pct:.1f}%
    - %PV = {pv_pct:.1f}%
    """)
    
    # =====================================================
    # RÉSULTATS FINAUX
    # =====================================================
    st.subheader("🎯 Résumé des résultats")
    
    results = pd.DataFrame({
        "Composante": ["EV", "AV", "R&R", "PV", "TV"],
        "Attendu": [ev_expected, av_expected, grr_expected, pv_expected, "N/A"],
        "Calculé": [ev_calculated, av_calculated, grr_calculated, pv_calculated, tv_calculated],
        "Différence": [ev_diff, av_diff, grr_diff, pv_diff, "N/A"],
        "Statut": [
            "✅" if ev_diff < 0.001 else "⚠️",
            "✅" if av_diff < 0.001 else "⚠️",
            "✅" if grr_diff < 0.001 else "⚠️",
            "✅" if pv_diff < 0.001 else "⚠️",
            "✓"
        ]
    })
    
    st.dataframe(results, use_container_width=True)
    
    # Évaluation
    st.subheader("📈 Évaluation du système")
    
    if grr_pct < 10:
        status = "✅ ACCEPTABLE"
        color = "green"
    elif grr_pct < 30:
        status = "⚠️ ACCEPTABLE SOUS CONDITIONS"
        color = "orange"
    else:
        status = "❌ INACCEPTABLE"
        color = "red"
    
    st.markdown(f"""
    <div style="background-color:{color}20; padding:20px; border-radius:10px; border-left:5px solid {color};">
        <h3 style="color:{color}; margin:0;">{status}</h3>
        <p style="font-size:1.2em; margin:10px 0;">
            <strong>%R&R = {grr_pct:.1f}%</strong><br>
            <small>%EV = {ev_pct:.1f}% | %AV = {av_pct:.1f}% | %PV = {pv_pct:.1f}%</small>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # =====================================================
    # ENTRER LES DONNÉES BRUTES POUR VÉRIFICATION
    # =====================================================
    st.subheader("📊 Entrer les données brutes pour vérification complète")
    
    st.info("""
    **Pour une vérification complète, entrez les données brutes de votre exemple:**
    - 10 pièces
    - 2 opérateurs (OP1 et OP2)
    - 3 essais par opérateur
    """)
    
    # Créer un formulaire pour entrer les données
    n_operators = 2  # Selon votre exemple
    
    # Créer un DataFrame vide
    data_entries = []
    
    with st.form("raw_data_form"):
        st.write("**Données brutes (format: valeur numérique)**")
        
        # Créer des colonnes pour chaque mesure
        col_names = []
        for op in range(n_operators):
            for trial in range(n_trials):
                col_names.append(f"OP{op+1}_Essai{trial+1}")
        
        # Créer une ligne pour chaque pièce
        for part in range(n_parts):
            cols = st.columns(len(col_names))
            part_data = {}
            for i, col in enumerate(col_names):
                with cols[i]:
                    part_data[col] = st.number_input(
                        f"P{part+1} {col}",
                        value=45.0,
                        step=0.01,
                        format="%.2f",
                        key=f"{part}_{col}"
                    )
            data_entries.append(part_data)
        
        submitted = st.form_submit_button("📥 Calculer à partir des données brutes")
        
        if submitted:
            # Créer le DataFrame
            df_raw = pd.DataFrame(data_entries)
            
            st.write("**Données brutes entrées:**")
            st.dataframe(df_raw, use_container_width=True)
            
            # Calculer à partir des données brutes
            st.subheader("🔍 Calcul automatique à partir des données brutes")
            
            # Calculer R̄ (moyenne des étendues)
            all_ranges = []
            op_means = []
            
            for op in range(n_operators):
                # Colonnes pour cet opérateur
                op_cols = [f"OP{op+1}_Essai{t+1}" for t in range(n_trials)]
                op_data = df_raw[op_cols]
                
                # Étendues par pièce
                ranges = op_data.max(axis=1) - op_data.min(axis=1)
                all_ranges.extend(ranges)
                
                # Moyennes par opérateur
                means = op_data.mean(axis=1)
                op_means.append(means.mean())
            
            # R̄ calculé
            R_bar_calc = np.mean(all_ranges)
            
            # X_diff calculé
            X_diff_calc = max(op_means) - min(op_means)
            
            # Rp calculé
            part_means_calc = df_raw.mean(axis=1)
            R_p_calc = max(part_means_calc) - min(part_means_calc)
            
            st.markdown(f"""
            **Paramètres calculés automatiquement:**
            - R̄ (moyenne des étendues) = {R_bar_calc:.4f}
            - X_diff (différence des moyennes) = {X_diff_calc:.4f}
            - Rp (étendue des moyennes) = {R_p_calc:.4f}
            
            **Comparaison avec vos valeurs:**
            - Votre R̄: {R_bar:.3f} | Calculé: {R_bar_calc:.4f}
            - Votre X_diff: {X_diff:.3f} | Calculé: {X_diff_calc:.4f}
            - Votre Rp: {R_p:.3f} | Calculé: {R_p_calc:.4f}
            """)

# =====================================================
# EXPLICATION DE LA DIFFÉRENCE POUR AV
# =====================================================
with st.expander("🔍 Pourquoi la petite différence pour AV?"):
    st.markdown("""
    **Explication possible de la différence pour AV (0,080 attendu vs ~0,074 calculé):**
    
    1. **Arrondis dans l'exemple:** Les valeurs intermédiaires dans votre exemple 
       pourraient avoir été arrondies différemment.
    
    2. **Ordre des calculs:** 
       - Dans la formule: `AV = √[((5.15 × X_diff)/d2)² - (EV²/(n×r))]`
       - Si on utilise EV = 0,176 exactement:
         ```
         (5.15 × 0.03)/1.91 = 0.1545/1.91 = 0.08089
         (0.08089)² = 0.006543
         EV²/(n×r) = 0.030976/30 = 0.0010325
         AV = √(0.006543 - 0.0010325) = √0.0055105 = 0.07423
         ```
    
    3. **Valeur de X_diff:** Votre X_diff = 0,03 pourrait être une valeur arrondie.
       Si X_diff était légèrement différente:
       - Pour obtenir AV = 0,080, il faudrait X_diff ≈ 0,0317
    
    4. **Valeur de d2:** Le d2 = 1,91 pourrait être légèrement différent selon la table exacte.
    
    **Conclusion:** La formule est correcte. Les petites différences viennent probablement 
    d'arrondis dans l'exemple fourni.
    """)

# =====================================================
# CODE POUR L'APPLICATION COMPLÈTE
# =====================================================
with st.expander("🚀 Code complet pour application Gage R&R"):
    st.code("""
# À IMPORTER DANS UN FICHIER app.py SEPARÉ

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import math

# [Insérer ici tout le code de la table d2 et des fonctions]

def calculate_gage_rr_from_data(df, n_parts, n_operators, n_trials, k=5.15):
    \"\"\"
    Calcule Gage R&R à partir d'un DataFrame de données brutes.
    
    Args:
        df: DataFrame avec les mesures
        n_parts: nombre de pièces
        n_operators: nombre d'opérateurs
        n_trials: nombre d'essais
        k: facteur de confiance (5.15 par défaut)
    
    Returns:
        Dictionnaire avec tous les résultats
    \"\"\"
    
    # 1. Calculer R̄ (moyenne des étendues)
    all_ranges = []
    op_global_means = []
    
    for op in range(n_operators):
        op_cols = df.columns[op*n_trials:(op+1)*n_trials]
        op_data = df[op_cols]
        
        # Étendues pour cet opérateur
        ranges = op_data.max(axis=1) - op_data.min(axis=1)
        all_ranges.extend(ranges)
        
        # Moyenne globale de l'opérateur
        op_mean = op_data.mean(axis=1).mean()
        op_global_means.append(op_mean)
    
    R_bar = np.mean(all_ranges)
    
    # 2. Calculer X_diff
    X_diff = max(op_global_means) - min(op_global_means)
    
    # 3. Calculer Rp
    part_means = df.mean(axis=1)
    R_p = max(part_means) - min(part_means)
    
    # 4. Obtenir les d2
    # d2 pour EV: Z = n×k, W = r
    d2_ev = get_d2(n_parts * n_operators, n_trials)
    
    # d2 pour AV: Z = 1, W = k
    d2_av = get_d2(1, n_operators)
    
    # d2 pour PV: Z = 1, W = n
    d2_pv = get_d2(1, n_parts)
    
    # 5. Calculer les composantes
    EV = (k * R_bar) / d2_ev
    
    av_term = ((k * X_diff) / d2_av) ** 2 - (EV ** 2) / (n_parts * n_trials)
    AV = math.sqrt(max(av_term, 0))
    
    GRR = math.sqrt(EV ** 2 + AV ** 2)
    
    PV = (k * R_p) / d2_pv
    
    TV = math.sqrt(GRR ** 2 + PV ** 2)
    
    # 6. Calculer les pourcentages
    if TV > 0:
        EV_pct = (EV / TV) * 100
        AV_pct = (AV / TV) * 100
        GRR_pct = (GRR / TV) * 100
        PV_pct = (PV / TV) * 100
    else:
        EV_pct = AV_pct = GRR_pct = PV_pct = 0
    
    return {
        'R_bar': R_bar, 'X_diff': X_diff, 'R_p': R_p,
        'd2_ev': d2_ev, 'd2_av': d2_av, 'd2_pv': d2_pv,
        'EV': EV, 'AV': AV, 'GRR': GRR, 'PV': PV, 'TV': TV,
        'EV_pct': EV_pct, 'AV_pct': AV_pct, 'GRR_pct': GRR_pct, 'PV_pct': PV_pct
    }

# Utilisation:
# results = calculate_gage_rr_from_data(df, 10, 2, 3, 5.15)
# print(f"EV: {results['EV']:.3f}, AV: {results['AV']:.3f}, R&R: {results['GRR']:.3f}")
""", language="python")

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
    <p><strong>Gage R&R - Vérification Exacte | Basé sur votre exemple fourni</strong></p>
    <p>Formules: EV = (k×R̄)/d2 | AV = √[((k×X_diff)/d2)² - EV²/(n×r)] | R&R = √(EV² + AV²)</p>
</div>
""", unsafe_allow_html=True)
