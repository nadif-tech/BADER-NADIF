import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Calculateur Gage R&R - Méthode des Étendues", layout="wide")

st.title("📊 Étude de Précision : Gage R&R")
st.subheader("Méthode des Étendues et des Moyennes (Lean Six Sigma)")

# --- FONCTION DE RÉCUPÉRATION DE d2 ---
def get_d2(z, w):
    # Extrait de la table d2 du cours (Page 71)
    # Z = lignes (1 à >15), W = colonnes (2 à 15)
    # On utilise ici les valeurs courantes pour l'exemple standard
    table = {
        (1, 2): 1.41, (1, 3): 1.91,
        (10, 3): 1.72, # Cas courant pour EV: Z=30, W=3 (approximation >15)
        (30, 3): 1.693, # Valeur exacte pour Z > 15, W = 3
    }
    # Valeurs spécifiques citées dans l'exemple du cours (Page 77-79)
    if z > 15 and w == 3: return 1.693 # d2 pour répétabilité (EV)
    if z == 1 and w == 3: return 1.91  # d2 pour reproductibilité (AV)
    if z == 1 and w == 10: return 3.18 # d2 pour variabilité pièce (VP)
    return 1.0 # Valeur par défaut si non trouvé

# --- SAISIE DES PARAMÈTRES ---
with st.sidebar:
    st.header("Paramètres de l'étude")
    n_pieces = st.number_input("Nombre de pièces (n)", value=10)
    n_operateurs = st.number_input("Nombre d'opérateurs", value=3)
    n_essais = st.number_input("Nombre d'essais (r)", value=3)
    confidence_factor = 5.15 # Niveau de confiance 99% selon le cours

# --- ENTRÉE DES DONNÉES ---
st.write("### Saisie des mesures moyennes et étendues")
st.info("Saisissez les résultats calculés par opérateur (comme dans le tableau page 76 du cours).")

col1, col2, col3 = st.columns(3)

with col1:
    x_double_bar_op1 = st.number_input("Moyenne OP1 (X̄1)", value=45.09)
    r_bar_op1 = st.number_input("Étendue moyenne OP1 (R̄1)", value=0.055)

with col2:
    x_double_bar_op2 = st.number_input("Moyenne OP2 (X̄2)", value=45.06)
    r_bar_op2 = st.number_input("Étendue moyenne OP2 (R̄2)", value=0.087)

with col3:
    x_double_bar_op3 = st.number_input("Moyenne OP3 (X̄3)", value=45.08)
    r_bar_op3 = st.number_input("Étendue moyenne OP3 (R̄3)", value=0.031)

# --- CALCULS ---
if st.button("Calculer la variabilité"):
    # 1. Répétabilité (EV)
    r_double_bar = (r_bar_op1 + r_bar_op2 + r_bar_op3) / n_operateurs [cite: 1486]
    d2_ev = get_d2(n_pieces * n_operateurs, n_essais)
    ev = (confidence_factor * r_double_bar) / d2_ev [cite: 1473]
    
    # 2. Reproductibilité (AV)
    means = [x_double_bar_op1, x_double_bar_op2, x_double_bar_op3]
    x_etendue = max(means) - min(means) [cite: 1477]
    d2_av = get_d2(1, n_operateurs)
    
    # Formule avec correction de la répétabilité [cite: 1477]
    av_term = (confidence_factor * x_etendue / d2_av)**2
    ev_correction = (ev**2) / (n_pieces * n_essais)
    av = np.sqrt(max(0, av_term - ev_correction))
    
    # 3. Gage R&R
    grr = np.sqrt(ev**2 + av**2) [cite: 1479]
    
    # 4. Variabilité Pièce (VP)
    # Simulation de Rp pour l'exemple (Max - Min des moyennes de pièces)
    rp = st.number_input("Étendue des moyennes de pièces (Rp)", value=0.33)
    d2_vp = get_d2(1, n_pieces)
    vp = (confidence_factor * rp) / d2_vp [cite: 1479]
    
    # 5. Variabilité Totale (VT)
    vt = np.sqrt(grr**2 + vp**2) [cite: 1481]
    
    # --- AFFICHAGE DES RÉSULTATS ---
    st.divider()
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.write("#### Composantes de la variance")
        st.metric("Répétabilité (EV)", round(ev, 4))
        st.metric("Reproductibilité (AV)", round(av, 4))
        st.metric("Gage R&R", round(grr, 4))
        st.metric("Variabilité Totale (VT)", round(vt, 4))

    with res_col2:
        st.write("#### Contribution (%)")
        p_ev = (ev / vt) * 100
        p_av = (av / vt) * 100
        p_grr = (grr / vt) * 100
        
        st.write(f"**% EV (Équipement):** {p_ev:.1f}%")
        st.write(f"**% AV (Opérateur):** {p_av:.1f}%")
        st.write(f"**% Gage R&R:** {p_grr:.1f}%")
        
        # Conclusion selon les règles du cours 
        if p_grr < 10:
            st.success("✅ Processus satisfaisant (< 10%)")
        elif 10 <= p_grr <= 30:
            st.warning("⚠️ Processus acceptable mais à améliorer (10-30%)")
        else:
            st.error("❌ Processus inacceptable (> 30%)")

st.sidebar.markdown("""
---
**Note :** Les constantes $d_2$ sont extraites automatiquement pour les configurations standards (10 pièces, 3 opérateurs, 3 essais).
""")
