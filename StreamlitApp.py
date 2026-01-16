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
    confidence_factor = st.number_input("Facteur de confiance", value=5.15, step=0.01)

# --- Import Excel ---
uploaded_file = st.file_uploader(
    "📥 Importer le fichier Excel Gage R&R",
    type=["xlsx"]
)

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("📄 Aperçu des données")
    st.dataframe(df, use_container_width=True)

    # Colonnes
    op1_cols = ["OP1-1", "OP1-2", "OP1-3"]
    op2_cols = ["OP2-1", "OP2-2", "OP2-3"]
    op3_cols = ["OP3-1", "OP3-2", "OP3-3"]

    n_pieces = df.shape[0]
    n_operateurs = 3
    n_essais = 3

    # --- Calculs de base ---
    df["R_OP1"] = df[op1_cols].max(axis=1) - df[op1_cols].min(axis=1)
    df["R_OP2"] = df[op2_cols].max(axis=1) - df[op2_cols].min(axis=1)
    df["R_OP3"] = df[op3_cols].max(axis=1) - df[op3_cols].min(axis=1)

    r_bar_op1 = df["R_OP1"].mean()
    r_bar_op2 = df["R_OP2"].mean()
    r_bar_op3 = df["R_OP3"].mean()

    x_bar_op1 = df[op1_cols].values.mean()
    x_bar_op2 = df[op2_cols].values.mean()
    x_bar_op3 = df[op3_cols].values.mean()

    r_double_bar = (r_bar_op1 + r_bar_op2 + r_bar_op3) / n_operateurs
    d2_ev = get_d2(n_pieces * n_operateurs, n_essais)
    ev = (confidence_factor * r_double_bar) / d2_ev

    x_range = max([x_bar_op1, x_bar_op2, x_bar_op3]) - min([x_bar_op1, x_bar_op2, x_bar_op3])
    d2_av = get_d2(1, n_operateurs)
    av_term = (confidence_factor * x_range / d2_av) ** 2
    ev_corr = (ev ** 2) / (n_pieces * n_essais)
    av = np.sqrt(max(0, av_term - ev_corr))

    grr = np.sqrt(ev ** 2 + av ** 2)

    # --- Variabilité pièces ---
    df["Moy_Piece"] = df[op1_cols + op2_cols + op3_cols].mean(axis=1)
    rp = df["Moy_Piece"].max() - df["Moy_Piece"].min()
    vp = (confidence_factor * rp) / get_d2(1, n_pieces)
    vt = np.sqrt(grr ** 2 + vp ** 2)

    # --- Pourcentages ---
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

    # --- Couleur et style pour les graphiques ---
    colors = ['#1f77b4','#ff7f0e','#2ca02c']

    # --- Contribution barres ---
    st.subheader("📊 Contribution des sources (%)")
    fig1, ax = plt.subplots(figsize=(5,3))
    bars = ax.bar(['EV','AV','VP'], [p_ev,p_av,p_vp], color=colors, edgecolor='black')
    ax.set_ylim(0,100)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', fontweight='bold')
    ax.set_ylabel("Pourcentage")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig1)

    # --- Boxplot compact ---
    st.subheader("📦 Distribution mesures")
    fig2, ax = plt.subplots(figsize=(5,3))
    bplot = ax.boxplot(
        [df[op1_cols].values.flatten(),
         df[op2_cols].values.flatten(),
         df[op3_cols].values.flatten()],
        labels=["OP1","OP2","OP3"],
        patch_artist=True,
        medianprops=dict(color='red', linewidth=2)
    )
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig2)

    # --- Histogramme compact ---
    st.subheader("📊 Histogramme mesures")
    fig3, ax = plt.subplots(figsize=(5,3))
    ax.hist(df[op1_cols].values.flatten(), bins=10, alpha=0.5, color=colors[0], label='OP1', edgecolor='black')
    ax.hist(df[op2_cols].values.flatten(), bins=10, alpha=0.5, color=colors[1], label='OP2', edgecolor='black')
    ax.hist(df[op3_cols].values.flatten(), bins=10, alpha=0.5, color=colors[2], label='OP3', edgecolor='black')
    ax.set_xlabel("Valeur")
    ax.set_ylabel("Fréquence")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig3)

    # --- Interaction pièce × opérateur ---
    st.subheader("🔁 Interaction Pièce × Opérateur")
    interaction_df = pd.DataFrame({
        "Pièce": df["N° Pièce"],
        "OP1": df[op1_cols].mean(axis=1),
        "OP2": df[op2_cols].mean(axis=1),
        "OP3": df[op3_cols].mean(axis=1)
    })
    fig4, ax = plt.subplots(figsize=(5,3))
    ax.plot(interaction_df["Pièce"], interaction_df["OP1"], marker='o', color=colors[0], label='OP1')
    ax.plot(interaction_df["Pièce"], interaction_df["OP2"], marker='s', color=colors[1], label='OP2')
    ax.plot(interaction_df["Pièce"], interaction_df["OP3"], marker='^', color=colors[2], label='OP3')
    ax.set_xlabel("Pièce")
    ax.set_ylabel("Mesure")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)
    st.pyplot(fig4)

    # --- Carte de contrôle (EV, AV, GRR) ---
    st.subheader("📈 Carte de contrôle")
    fig5, ax = plt.subplots(figsize=(5,3))
    measures = [ev, av, grr]
    mean_val = np.mean(measures)
    ucl = mean_val + 3*np.std(measures)
    lcl = mean_val - 3*np.std(measures)
    ax.plot(['EV','AV','GRR'], measures, marker='o', color='purple', label='Mesures')
    ax.axhline(mean_val, color='green', linestyle='--', label='Moyenne')
    ax.axhline(ucl, color='red', linestyle='--', label='UCL')
    ax.axhline(lcl, color='red', linestyle='--', label='LCL')
    ax.set_ylim(0, max(measures)*1.3)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylabel("Valeur")
    ax.legend(fontsize=8)
    st.pyplot(fig5)

    # --- Export Excel ---
    export_df = pd.DataFrame({
        "EV":[ev],
        "AV":[av],
        "VP":[vp],
        "GRR":[grr],
        "VT":[vt],
        "%EV":[p_ev],
        "%AV":[p_av],
        "%VP":[p_vp],
        "%GRR":[p_grr],
        "NdC":[ndc]
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
