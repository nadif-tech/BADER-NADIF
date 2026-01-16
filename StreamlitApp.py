import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Configuration initiale
st.set_page_config(page_title="Gage R&R Lean Six Sigma", layout="wide")

st.title("📊 Calculateur Gage R&R (Import/Export Excel)")
st.write("Calcul de la fiabilité du système de mesure selon la méthode des étendues[cite: 699, 715].")

# --- FONCTION d2 ---
def get_d2(z, w):
    # Valeurs d2 selon la table du cours (Page 71)
    if z > 15 and w == 3: return 1.693  # Répétabilité [cite: 712]
    if z == 1 and w == 3: return 1.91   # Reproductibilité [cite: 712, 720]
    if z == 1 and w == 10: return 3.18  # Variabilité pièce [cite: 712, 728]
    return 1.128

# --- 1. IMPORTATION DES DONNÉES ---
st.header("📂 1. Importation des mesures")
uploaded_file = st.file_uploader("Importer un fichier Excel de mesures", type="xlsx")

if uploaded_file:
    df_mesures = pd.read_excel(uploaded_file)
    st.write("Aperçu de vos données :")
    st.dataframe(df_mesures.head())

# --- 2. PARAMÈTRES ET SAISIE ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    # Paramètres recommandés : 10 pièces, 3 opérateurs, 3 séries (Page 69)
    n = st.number_input("Nombre de pièces (n)", value=10) [cite: 695]
    op_count = st.number_input("Nombre d'opérateurs", value=3) [cite: 695]
    r_count = st.number_input("Nombre d'essais (r)", value=3) [cite: 695]
    k = 5.15 # Niveau de confiance 99% [cite: 706, 739]

col1, col2, col3 = st.columns(3)
with col1:
    x1 = st.number_input("Moyenne OP1 (X̄1)", value=45.09) [cite: 751]
    r1 = st.number_input("R̄ OP1", value=0.055) [cite: 751]
with col2:
    x2 = st.number_input("Moyenne OP2 (X̄2)", value=45.06) [cite: 751]
    r2 = st.number_input("R̄ OP2", value=0.087) [cite: 751]
with col3:
    x3 = st.number_input("Moyenne OP3 (X̄3)", value=45.08) [cite: 751]
    r3 = st.number_input("R̄ OP3", value=0.031) [cite: 751]

# --- 3. CALCULS ---
if st.button("Lancer les calculs"):
    # EV - Répétabilité (Page 70)
    r_double_bar = (r1 + r2 + r3) / op_count [cite: 751]
    d2_ev = get_d2(n * op_count, r_count) [cite: 705]
    ev = (k * r_double_bar) / d2_ev [cite: 702]
    
    # AV - Reproductibilité (Page 72)
    x_range = max([x1, x2, x3]) - min([x1, x2, x3]) [cite: 718, 763]
    d2_av = get_d2(1, op_count) [cite: 720]
    av_raw = (k * x_range / d2_av)**2 - (ev**2 / (n * r_count)) [cite: 717]
    av = np.sqrt(max(0, av_raw)) [cite: 717]
    
    # R&R et Variabilité Totale (Page 73-74)
    rr = np.sqrt(ev**2 + av**2) [cite: 725]
    rp_val = 0.33 # Exemple Rp page 79 [cite: 772]
    d2_vp = get_d2(1, n) [cite: 728]
    vp = (k * rp_val) / d2_vp [cite: 726]
    vt = np.sqrt(rr**2 + vp**2) [cite: 733]
    
    # Préparation des résultats pour export
    res_data = {
        "Composante": ["EV (Équipement)", "AV (Opérateur)", "Gage R&R", "Variabilité Pièce (VP)", "Variabilité Totale (VT)"],
        "Valeur": [ev, av, rr, vp, vt],
        "% Contribution": [(ev/vt)*100, (av/vt)*100, (rr/vt)*100, (vp/vt)*100, 100] [cite: 782, 783, 784]
    }
    df_res = pd.DataFrame(res_data)
    
    st.divider()
    st.subheader("📋 Résultats de l'analyse")
    st.table(df_res)

    # Interprétation du %R&R (Page 81)
    p_rr = (rr / vt) * 100
    if p_rr < 10:
        st.success(f"Résultat : {p_rr:.2f}% - Processus SATISFAISANT [cite: 791]")
    elif 10 <= p_rr <= 30:
        st.warning(f"Résultat : {p_rr:.2f}% - Processus ACCEPTABLE (à améliorer) [cite: 792]")
    else:
        st.error(f"Résultat : {p_rr:.2f}% - Processus INACCEPTABLE [cite: 793]")

    # --- 4. EXPORTATION ---
    st.header("💾 2. Exportation des résultats")
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_res.to_excel(writer, index=False, sheet_name='Gage_RR_Results')
    
    st.download_button(
        label="📥 Télécharger les résultats (Excel)",
        data=output.getvalue(),
        file_name="resultats_gage_rr.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
