import streamlit as st
import pandas as pd
import numpy as np

# ================= CONFIG =================
st.set_page_config(
    page_title="Gage R&R – Méthode des Étendues",
    layout="wide"
)

st.title("📊 Étude de Précision – Gage R&R")
st.subheader("Méthode des Étendues et des Moyennes (Lean Six Sigma)")

# ================= d2 FUNCTION =================
def get_d2(z, w):
    if z > 15 and w == 3:
        return 1.693
    if z == 1 and w == 3:
        return 1.91
    if z == 1 and w == 10:
        return 3.18
    return 1.0

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Paramètres")
    n_pieces = st.number_input("Nombre de pièces (n)", 1, 50, 10)
    n_operateurs = st.number_input("Nombre d'opérateurs", 1, 10, 3)
    n_essais = st.number_input("Nombre d'essais (r)", 1, 10, 3)
    confidence_factor = 5.15

    st.divider()
    st.subheader("📥 Import Excel")
    uploaded_file = st.file_uploader(
        "Importer fiche Gage R&R",
        type=["xlsx"]
    )

# ================= DEFAULT VALUES =================
x_double_bar_op1 = 45.09
r_bar_op1 = 0.055
x_double_bar_op2 = 45.06
r_bar_op2 = 0.087
x_double_bar_op3 = 45.08
r_bar_op3 = 0.031
rp = 0.33

# ================= IMPORT DATA =================
if uploaded_file:
    df_import = pd.read_excel(uploaded_file)

    x_double_bar_op1 = df_import.loc[0, "Xbar_OP1"]
    r_bar_op1 = df_import.loc[0, "Rbar_OP1"]
    x_double_bar_op2 = df_import.loc[0, "Xbar_OP2"]
    r_bar_op2 = df_import.loc[0, "Rbar_OP2"]
    x_double_bar_op3 = df_import.loc[0, "Xbar_OP3"]
    r_bar_op3 = df_import.loc[0, "Rbar_OP3"]
    rp = df_import.loc[0, "Rp"]

    st.success("✅ Données Excel importées avec succès")

# ================= INPUT UI =================
st.write("### ✍️ Données de mesure")

col1, col2, col3 = st.columns(3)

with col1:
    x_double_bar_op1 = st.number_input("Moyenne OP1 (X̄1)", value=float(x_double_bar_op1))
    r_bar_op1 = st.number_input("Étendue OP1 (R̄1)", value=float(r_bar_op1))

with col2:
    x_double_bar_op2 = st.number_input("Moyenne OP2 (X̄2)", value=float(x_double_bar_op2))
    r_bar_op2 = st.number_input("Étendue OP2 (R̄2)", value=float(r_bar_op2))

with col3:
    x_double_bar_op3 = st.number_input("Moyenne OP3 (X̄3)", value=float(x_double_bar_op3))
    r_bar_op3 = st.number_input("Étendue OP3 (R̄3)", value=float(r_bar_op3))

rp = st.number_input("Étendue des moyennes de pièces (Rp)", value=float(rp))

# ================= CALCUL =================
if st.button("📊 Calculer Gage R&R"):
    r_double_bar = (r_bar_op1 + r_bar_op2 + r_bar_op3) / n_operateurs
    d2_ev = get_d2(n_pieces * n_operateurs, n_essais)
    ev = (confidence_factor * r_double_bar) / d2_ev

    means = [x_double_bar_op1, x_double_bar_op2, x_double_bar_op3]
    x_etendue = max(means) - min(means)
    d2_av = get_d2(1, n_operateurs)

    av_term = (confidence_factor * x_etendue / d2_av) ** 2
    ev_corr = (ev ** 2) / (n_pieces * n_essais)
    av = np.sqrt(max(0, av_term - ev_corr))

    grr = np.sqrt(ev ** 2 + av ** 2)

    d2_vp = get_d2(1, n_pieces)
    vp = (confidence_factor * rp) / d2_vp

    vt = np.sqrt(grr ** 2 + vp ** 2)

    # ================= RESULTS =================
    st.divider()
    colA, colB = st.columns(2)

    with colA:
        st.subheader("📌 Résultats")
        st.metric("EV – Répétabilité", round(ev, 4))
        st.metric("AV – Reproductibilité", round(av, 4))
        st.metric("Gage R&R", round(grr, 4))
        st.metric("Variabilité Totale", round(vt, 4))

    with colB:
        st.subheader("📈 Contribution (%)")
        p_grr = (grr / vt) * 100

        st.write(f"% EV : {(ev/vt)*100:.1f}%")
        st.write(f"% AV : {(av/vt)*100:.1f}%")
        st.write(f"% GRR : {p_grr:.1f}%")

        if p_grr < 10:
            st.success("✅ Processus capable")
        elif p_grr <= 30:
            st.warning("⚠️ Processus acceptable")
        else:
            st.error("❌ Processus non capable")

    # ================= EXPORT EXCEL =================
    df_export = pd.DataFrame({
        "EV": [ev],
        "AV": [av],
        "GRR": [grr],
        "VP": [vp],
        "VT": [vt],
        "%GRR": [p_grr]
    })

    st.download_button(
        "📤 Exporter résultats Excel",
        data=df_export.to_excel(index=False, engine="xlsxwriter"),
        file_name="resultats_gage_rr.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
