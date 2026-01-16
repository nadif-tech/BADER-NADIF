import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Configuration de la page
st.set_page_config(page_title="Gage R&R - Analyse de Fiabilité", layout="wide")

st.title("📊 Système de Mesure : Gage R&R")
st.write("Analyse de la capabilité du processus de mesure (Méthode des étendues).")

# --- FONCTION d2 (Table Page 71) ---
def get_d2(z, w):
    if z > 15 and w == 3: return 1.693 # Répétabilité (Page 70)
    if z == 1 and w == 3: return 1.91  # Reproductibilité (Page 72)
    if z == 1 and w == 10: return 3.18 # Pièce (Page 73)
    return 1.128

# --- 1. IMPORTATION DU FICHIER TEST ---
st.header("📂 Importation des données de test")
uploaded_file = st.file_uploader("Chargez votre fichier TESET.xlsx", type=["xlsx", "csv"])

if uploaded_file:
    # Lecture du fichier (CSV ou Excel)
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("Données chargées :")
    st.dataframe(df)

# --- 2. PARAMÈTRES (Sidebar) ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    # Valeurs recommandées par le cours (Page 69)
    n = st.number_input("Nombre de pièces (n)", value=10)
    op_count = st.number_input("Nombre d'opérateurs", value=3)
    r_count = st.number_input("Nombre d'essais (r)", value=3)
    k = 5.15 # Niveau de confiance 99% (Page 70)

# --- 3. SAISIE DES MOYENNES (Exemple Page 76) ---
st.subheader("📝 Saisie des calculs intermédiaires")
c1, c2, c3 = st.columns(3)
with c1:
    x1 = st.number_input("X̄̄ Opérateur 1", value=45.09)
    r1 = st.number_input("R̄ Opérateur 1", value=0.055)
with c2:
    x2 = st.number_input("X̄̄ Opérateur 2", value=45.06)
    r2 = st.number_input("R̄ Opérateur 2", value=0.087)
with c3:
    x3 = st.number_input("X̄̄ Opérateur 3", value=45.08)
    r3 = st.number_input("R̄ Opérateur 3", value=0.031)

# --- 4. CALCULS ET EXPORT ---
if st.button("Calculer la Fiabilité (R&R)"):
    # EV - Répétabilité (Page 70)
    r_double_bar = (r1 + r2 + r3) / op_count
    d2_ev = get_d2(n * op_count, r_count)
    ev = (k * r_double_bar) / d2_ev
    
    # AV - Reproductibilité (Page 72)
    x_range = max([x1, x2, x3]) - min([x1, x2, x3])
    d2_av = get_d2(1, op_count)
    av_raw = (k * x_range / d2_av)**2 - (ev**2 / (n * r_count))
    av = np.sqrt(max(0, av_raw))
    
    # Gage R&R et VT (Page 73-74)
    rr = np.sqrt(ev**2 + av**2)
    vt = np.sqrt(rr**2 + 0.53**2) # 0.53 est un exemple de Vp (Page 79)
    
    p_rr = (rr / vt) * 100

    # Affichage des résultats
    st.divider()
    st.metric("Gage R&R (%)", f"{p_rr:.2f}%")
    
    if p_rr < 10:
        st.success("✅ Processus satisfaisant [cite: 791]")
    elif 10 <= p_rr <= 30:
        st.warning("⚠️ Processus acceptable mais à améliorer [cite: 792]")
    else:
        st.error("❌ Processus inacceptable [cite: 793]")

    # EXPORT EXCEL
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_res = pd.DataFrame({"Composante": ["EV", "AV", "R&R"], "Valeur": [ev, av, rr]})
        df_res.to_excel(writer, index=False, sheet_name='Resultats')
    
    st.download_button("📥 Télécharger les résultats", output.getvalue(), "resultats.xlsx")
