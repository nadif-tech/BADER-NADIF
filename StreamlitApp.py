import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Gage R&R - Étendues", layout="wide")
st.title("📊 Étude Gage R&R – Méthode des Étendues")

# ---------------- d2 FUNCTION ----------------
def get_d2(z, w):
    if z > 15 and w == 3:
        return 1.693
    if z == 1 and w == 3:
        return 1.91
    if z == 1 and w == 10:
        return 3.18
    return 1.0

# ---------------- PARAMÈTRES ----------------
with st.sidebar:
    st.header("⚙️ Paramètres")
    confidence_factor = 5.15

# ---------------- IMPORT EXCEL ----------------
uploaded_file = st.file_uploader("📥 Importer le fichier Excel Gage R&R", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Aperçu des données")
    st.dataframe(df)

    # Colonnes opérateurs
    op1_cols = ["OP1-1", "OP1-2", "OP1-3"]
    op2_cols = ["OP2-1", "OP2-2", "OP2-3"]
    op3_cols = ["OP3-1", "OP3-2", "OP3-3"]

    n_pieces = df.shape[0]
    n_operateurs = 3
    n_essais = 3

    # ---------------- MOYENNES & ÉTENDUES ----------------
    df["R_OP1"] = df[op1_cols].max(axis=1) - df[op1_cols].min(axis=1)
    df["R_OP2"] = df[op2_cols].max(axis=1) - df[op2_cols].min(axis=1)
    df["R_OP3"] = df[op3_cols].max(axis=1) - df[op3_cols].min(axis=1)

    r_bar_op1 = df["R_OP1"].mean()
    r_bar_op2 = df["R_OP2"].mean()
    r_bar_op3 = df["R_OP3"].mean()

    x_bar_op1 = df[op1_cols].values.mean()
    x_bar_op2 = df[op2_cols].values.mean()
    x_bar_op3 = df[op3_cols].values.mean()

    # ---------------- CALCULS GRR ----------------
    r_double_bar = (r_bar_op1 + r_bar_op2 + r_bar_op3) / n_operateurs
    d2_ev = get_d2(n_pieces * n_operateurs, n_essais)
    ev = (confidence_factor * r_double_bar) / d2_ev

    means_ops = [x_bar_op1, x_bar_op2, x_bar_op3]
    x_range = max(means_ops) - min(means_ops)
    d2_av = get_d2(1, n_operateurs)

    av_term = (confidence_factor * x_range / d2_av) ** 2
    ev_corr = (ev ** 2) / (n_pieces * n_essais)
    av = np.sqrt(max(0, av_term - ev_corr))

    grr = np.sqrt(ev ** 2 + av ** 2)

    # ---------------- VARIABILITÉ PIÈCES ----------------
    df["Moy_Piece"] = df[op1_cols + op2_cols + op3_cols].mean(axis=1)
    rp = df["Moy_Piece"].max() - df["Moy_Piece"].min()

    d2_vp = get_d2(1, n_pieces)
    vp = (confidence_factor * rp) / d2_vp

    vt = np.sqrt(grr ** 2 + vp ** 2)
    p_grr = (grr / vt) * 100

    # ---------------- AFFICHAGE ----------------
    st.divider()
    st.subheader("📊 Résultats Gage R&R")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("EV – Répétabilité", round(ev, 4))
        st.metric("AV – Reproductibilité", round(av, 4))
        st.metric("Gage R&R", round(grr, 4))
        st.metric("Variabilité Totale", round(vt, 4))

    with col2:
        st.metric("% Gage R&R", f"{p_grr:.2f}%")
        if p_grr < 10:
            st.success("✅ Système de mesure acceptable")
        elif p_grr <= 30:
            st.warning("⚠️ Acceptable avec amélioration")
        else:
            st.error("❌ Système non acceptable")

    # ---------------- EXPORT EXCEL ----------------
    export_df = pd.DataFrame({
        "EV": [ev],
        "AV": [av],
        "GRR": [grr],
        "VP": [vp],
        "VT": [vt],
        "%GRR": [p_grr]
    })

    buffer = BytesIO()
    export_df.to_excel(buffer, index=False)
    buffer.seek(0)

    st.download_button(
        "📤 Télécharger résultats Excel",
        buffer,
        "resultats_gage_rr.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
