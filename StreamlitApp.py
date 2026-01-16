import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Gage R&R – Étendues", layout="wide")
st.title("📊 Étude Gage R&R – Méthode des Étendues (AIAG)")

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
uploaded_file = st.file_uploader(
    "📥 Importer le fichier Excel Gage R&R",
    type=["xlsx"]
)

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Aperçu des données")
    st.dataframe(df, use_container_width=True)

    # Colonnes opérateurs (FORMAT EXACT DU FICHIER)
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

    # ---------------- STATISTIQUES ----------------
    p_ev = (ev / vt) * 100
    p_av = (av / vt) * 100
    p_vp = (vp / vt) * 100
    p_grr = (grr / vt) * 100
    ndc = 1.41 * (vp / grr) if grr != 0 else 0

    # ---------------- AFFICHAGE ----------------
    st.divider()
    st.subheader("📊 Résultats Gage R&R")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EV", f"{ev:.4f}")
    c2.metric("AV", f"{av:.4f}")
    c3.metric("GRR", f"{grr:.4f}")
    c4.metric("VT", f"{vt:.4f}")

    st.subheader("📈 Indicateurs (%)")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("% EV", f"{p_ev:.1f}%")
    c6.metric("% AV", f"{p_av:.1f}%")
    c7.metric("% VP", f"{p_vp:.1f}%")
    c8.metric("NdC", f"{ndc:.1f}")

    if p_grr < 10:
        st.success("✅ Système de mesure acceptable")
    elif p_grr <= 30:
        st.warning("⚠️ Acceptable avec amélioration")
    else:
        st.error("❌ Système de mesure non acceptable")

    # ---------------- GRAPHE CONTRIBUTION ----------------
    st.subheader("📊 Contribution des sources de variation (%)")
    contrib_df = pd.DataFrame({
        "Source": ["EV", "AV", "VP"],
        "Pourcentage": [p_ev, p_av, p_vp]
    })
    st.bar_chart(contrib_df.set_index("Source"))

    # ---------------- BOXPLOT ----------------
    st.subheader("📦 Distribution des mesures par opérateur")
    box_df = pd.DataFrame({
        "OP1": df[op1_cols].values.flatten(),
        "OP2": df[op2_cols].values.flatten(),
        "OP3": df[op3_cols].values.flatten()
    })
    st.box_chart(box_df)

    # ---------------- INTERACTION ----------------
    st.subheader("🔁 Interaction Pièce × Opérateur")
    interaction_df = pd.DataFrame({
        "Pièce": df["N° Pièce"],
        "OP1": df[op1_cols].mean(axis=1),
        "OP2": df[op2_cols].mean(axis=1),
        "OP3": df[op3_cols].mean(axis=1)
    })
    st.line_chart(interaction_df.set_index("Pièce"))

    # ---------------- EXPORT EXCEL ----------------
    export_df = pd.DataFrame({
        "EV": [ev],
        "AV": [av],
        "VP": [vp],
        "GRR": [grr],
        "VT": [vt],
        "%EV": [p_ev],
        "%AV": [p_av],
        "%VP": [p_vp],
        "%GRR": [p_grr],
        "NdC": [ndc]
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
