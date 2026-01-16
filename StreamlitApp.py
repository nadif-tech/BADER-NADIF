import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Gage R&R Pro - Import/Export", layout="wide")

st.title("📊 Étude Gage R&R avec Import/Export Excel")
st.write("Méthode des étendues et des moyennes[cite: 699].")

# --- FONCTIONS DE CALCUL ---
def get_d2(z, w):
    # Valeurs d2 selon la table du cours (Page 71) [cite: 712]
    if z > 15 and w == 3: return 1.693
    if z == 1 and w == 3: return 1.91
    if z == 1 and w == 10: return 3.18
    return 1.128

# --- SECTION 1 : IMPORTATION ---
st.header("1. Importation des données")
uploaded_file = st.file_uploader("Choisissez un fichier Excel (.xlsx)", type="xlsx")

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    st.write("Aperçu des données importées :")
    st.dataframe(df_input.head())
    
    # Hypothèse : Le fichier contient les colonnes 'Moyenne' et 'Etendue' par opérateur
    # Vous pouvez adapter selon la structure de votre fichier Excel
else:
    st.info("Utilisez les valeurs par défaut ou importez un fichier pour commencer.")

# --- SECTION 2 : PARAMÈTRES ET SAISIE ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    n = st.number_input("Nombre de pièces (n)", value=10) [cite: 695]
    op_count = st.number_input("Nombre d'opérateurs", value=3) [cite: 695]
    r_count = st.number_input("Nombre d'essais (r)", value=3) [cite: 695]
    k = 5.15 # Facteur de confiance 99% [cite: 706]

col1, col2, col3 = st.columns(3)
with col1:
    x1 = st.number_input("Moyenne OP1", value=45.09)
    r1 = st.number_input("R̄ OP1", value=0.055)
with col2:
    x2 = st.number_input("Moyenne OP2", value=45.06)
    r2 = st.number_input("R̄ OP2", value=0.087)
with col3:
    x3 = st.number_input("Moyenne OP3", value=45.08)
    r3 = st.number_input("R̄ OP3", value=0.031)

# --- CALCULS ---
if st.button("Calculer et Préparer l'Export"):
    # EV - Répétabilité [cite: 700, 702]
    r_double_bar = (r1 + r2 + r3) / op_count
    d2_ev = get_d2(n * op_count, r_count)
    ev = (k * r_double_bar) / d2_ev
    
    # AV - Reproductibilité [cite: 716, 717]
    x_range = max([x1, x2, x3]) - min([x1, x2, x3])
    d2_av = get_d2(1, op_count)
    av_val = np.sqrt(max(0, (k * x_range / d2_av)**2 - (ev**2 / (n * r_count))))
    
    # R&R et VT [cite: 725, 733]
    rr = np.sqrt(ev**2 + av_val**2)
    # Exemple Rp du cours [cite: 772]
    rp_val = 0.33 
    d2_vp = get_d2(1, n)
    vp = (k * rp_val) / d2_vp
    vt = np.sqrt(rr**2 + vp**2)
    
    # Résultats pour export
    results = {
        "Indicateur": ["EV (Répétabilité)", "AV (Reproductibilité)", "Gage R&R", "Variabilité Pièce", "VT (Totale)"],
        "Valeur": [ev, av_val, rr, vp, vt],
        "% Contribution": [(ev/vt)*100, (av_val/vt)*100, (rr/vt)*100, (vp/vt)*100, 100]
    }
    df_results = pd.DataFrame(results)
    
    st.table(df_results)

    # --- SECTION 3 : EXPORTATION ---
    st.header("2. Exportation des résultats")
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_results.to_excel(writer, index=False, sheet_name='Resultats_Gage_RR')
    
    st.download_button(
        label="📥 Télécharger les résultats en Excel",
        data=output.getvalue(),
        file_name="resultats_gage_rr.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- RAPPEL DES RÈGLES DE DÉCISION ---
st.divider()
st.write("### Rappel des critères d'acceptation[cite: 790]:")
st.info("- **< 10%** : Processus satisfaisant [cite: 791]\n- **10% - 30%** : Acceptable mais améliorable [cite: 792]\n- **> 30%** : Inacceptable [cite: 793]")
