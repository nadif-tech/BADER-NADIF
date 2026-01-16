import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# --- Configuration page ---
st.set_page_config(page_title="Gage R&R – Étendues", layout="wide")
st.title("📊 Étude Gage R&R – Méthode des Étendues (AIAG)")

# --- Fonction d2 ---
def get_d2(z, w):
    if z > 15 and w == 3:
        return 1.693
    if z == 1 and w == 3:
        return 1.91
    if z == 1 and w == 10:
        return 3.18
    return 1.0

# --- Paramètres ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    confidence_factor = 5.15

# --- Import Excel ---
uploaded_file = st.file_uploader(
    "📥 Importer le fichier Excel Gage R&R",
    type=["xlsx"]
)

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("📄 Aperçu des données")
    st.dataframe(df, use_container_width=True)

    # Colonnes exactes
    op1_cols = ["OP1-1", "OP1-2", "OP1-3"]
    op2_cols = ["OP2-1", "OP2-2", "OP2-3"]
    op3_cols = ["OP3-1", "OP3-2", "OP3-3"]

    n_pieces = df.shape[0]
    n_operateurs = 3
    n_essais = 3

    # --- Moyennes et Étendues ---
    df["R_OP1"] = df[op1_cols].max(axis=1) - df[op1_cols].min(axis=1)
    df["R_OP2"] = df[op2_cols].max(axis=1) - df[op2_cols].min(axis=1)
    df["R_OP3"] = df[op3_cols].max(axis=1) - df[op3_cols].min(axis=1)

    r_bar_op1 = df["R_OP1"].mean()
    r_bar_op2 = df["R_OP2"].mean()
    r_bar_op3 = df["R_OP3"].mean()

    x_bar_op1 = df[op1_cols].values.mean()
    x_bar_op2 = df[op2_cols].values.mean()
    x_bar_op3 = df[op3_cols].values.mean()

    # --- Calculs Gage R&R ---
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

    # --- Variabilité pièces ---
    df["Moy_Piece"] = df[op1_cols + op2_cols + op3_cols].mean(axis=1)
    rp = df["Moy_Piece"].max() - df["Moy_Piece"].min()
    d2_vp = get_d2(1, n_pieces)
    vp = (confidence_factor * rp) / d2_vp

    vt = np.sqrt(grr ** 2 + vp ** 2)

    # --- Statistiques ---
    p_ev = (ev / vt) * 100
    p_av = (av / vt) * 100
    p_vp = (vp / vt) * 100
    p_grr = (grr / vt) * 100
    ndc = 1.41 * (vp / grr) if grr != 0 else 0

    # --- Affichage résultats ---
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

    # --- Graphique contribution ---
    st.subheader("📊 Contribution des sources de variation (%)")
    contrib_df = pd.DataFrame({
        "Source": ["EV", "AV", "VP"],
        "Pourcentage": [p_ev, p_av, p_vp]
    })
    fig_contrib, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(contrib_df["Source"], contrib_df["Pourcentage"], color=['skyblue','salmon','lightgreen'])
    ax.set_ylim(0,100)
    ax.set_ylabel("Pourcentage (%)")
    ax.set_title("Contribution des sources de variation (%)")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', fontweight='bold')
    st.pyplot(fig_contrib)

    # --- Boxplot ---
    st.subheader("📦 Distribution des mesures par opérateur")
    fig_box, ax = plt.subplots(figsize=(8,5))
    ax.boxplot(
        [df[op1_cols].values.flatten(),
         df[op2_cols].values.flatten(),
         df[op3_cols].values.flatten()],
        labels=["OP1", "OP2", "OP3"],
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', color='blue'),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(color='blue'),
        capprops=dict(color='blue')
    )
    ax.set_title("Boxplot des mesures par opérateur")
    ax.set_ylabel("Valeur mesurée")
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_box)

    # --- Histogramme ---
    st.subheader("📊 Histogramme des mesures par opérateur")
    fig_hist, ax = plt.subplots(figsize=(8,5))
    ax.hist(df[op1_cols].values.flatten(), bins=10, alpha=0.5, color='skyblue', label='OP1')
    ax.hist(df[op2_cols].values.flatten(), bins=10, alpha=0.5, color='salmon', label='OP2')
    ax.hist(df[op3_cols].values.flatten(), bins=10, alpha=0.5, color='lightgreen', label='OP3')
    ax.set_title("Histogramme des mesures par opérateur")
    ax.set_xlabel("Valeur mesurée")
    ax.set_ylabel("Fréquence")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_hist)

    # --- Interaction Pièce × Opérateur ---
    st.subheader("🔁 Interaction Pièce × Opérateur")
    interaction_df = pd.DataFrame({
        "Pièce": df["N° Pièce"],
        "OP1": df[op1_cols].mean(axis=1),
        "OP2": df[op2_cols].mean(axis=1),
        "OP3": df[op3_cols].mean(axis=1)
    })
    fig_line, ax = plt.subplots(figsize=(8,5))
    ax.plot(interaction_df["Pièce"], interaction_df["OP1"], marker='o', label='OP1', color='skyblue')
    ax.plot(interaction_df["Pièce"], interaction_df["OP2"], marker='s', label='OP2', color='salmon')
    ax.plot(interaction_df["Pièce"], interaction_df["OP3"], marker='^', label='OP3', color='lightgreen')
    ax.set_title("Interaction Pièce × Opérateur")
    ax.set_xlabel("Pièce")
    ax.set_ylabel("Valeur mesurée")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    st.pyplot(fig_line)

    # --- Export Excel ---
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
        "📤 Télécharger les résultats Excel",
        buffer,
        "resultats_gage_rr.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
