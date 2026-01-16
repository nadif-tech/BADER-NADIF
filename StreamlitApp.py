import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# --- CONFIGURATION PAGE ---
st.set_page_config(
    page_title="Gage R&R – Tableau de Bord",
    layout="wide"
)

st.markdown("<h1 style='text-align: center; color: navy;'>📊 Gage R&R – Méthode des Étendues (AIAG)</h1>", unsafe_allow_html=True)
st.write("---")

# --- FONCTION D2 ---
def get_d2(z, w):
    if z > 15 and w == 3:
        return 1.693
    if z == 1 and w == 3:
        return 1.91
    if z == 1 and w == 10:
        return 3.18
    return 1.0

# --- PARAMÈTRES SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    confidence_factor = st.number_input("Facteur de confiance", value=5.15, step=0.01)

# --- IMPORT EXCEL ---
uploaded_file = st.file_uploader("📥 Importer le fichier Excel Gage R&R", type=["xlsx"])

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

    # --- CALCULS ---
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

    # Variabilité pièces
    df["Moy_Piece"] = df[op1_cols + op2_cols + op3_cols].mean(axis=1)
    rp = df["Moy_Piece"].max() - df["Moy_Piece"].min()
    vp = (confidence_factor * rp) / get_d2(1, n_pieces)
    vt = np.sqrt(grr ** 2 + vp ** 2)

    # Pourcentages
    p_ev = (ev / vt) * 100
    p_av = (av / vt) * 100
    p_vp = (vp / vt) * 100
    p_grr = (grr / vt) * 100
    ndc = 1.41 * (vp / grr) if grr != 0 else 0

    # --- AFFICHAGE DES MÉTRIQUES ---
    st.subheader("📊 Indicateurs clés")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("EV", f"{ev:.4f}", delta=f"{p_ev:.1f}%")
    col2.metric("AV", f"{av:.4f}", delta=f"{p_av:.1f}%")
    col3.metric("GRR", f"{grr:.4f}", delta=f"{p_grr:.1f}%")
    col4.metric("VT", f"{vt:.4f}")

    # --- GRAPHIQUES STYLE DASHBOARD ---
    colors = ['#1f77b4','#ff7f0e','#2ca02c']

    # Contribution des sources
    st.subheader("📊 Contribution des sources (%)")
    fig, ax = plt.subplots(figsize=(4,2))
    bars = ax.bar(['EV','AV','VP'], [p_ev,p_av,p_vp], color=colors, edgecolor='black')
    ax.set_ylim(0,100)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval+1, f"{yval:.1f}%", ha='center', fontsize=8)
    ax.set_ylabel("Pourcentage", fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    # Boxplot compact
    st.subheader("📦 Distribution des mesures")
    fig2, ax = plt.subplots(figsize=(4,2))
    bplot = ax.boxplot([df[op1_cols].values.flatten(),
                        df[op2_cols].values.flatten(),
                        df[op3_cols].values.flatten()],
                        labels=["OP1","OP2","OP3"],
                        patch_artist=True,
                        medianprops=dict(color='red', linewidth=2))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig2, use_container_width=True)

    # Histogramme compact
    st.subheader("📊 Histogramme mesures")
    fig3, ax = plt.subplots(figsize=(4,2))
    ax.hist(df[op1_cols].values.flatten(), bins=8, alpha=0.5, color=colors[0], label='OP1', edgecolor='black')
    ax.hist(df[op2_cols].values.flatten(), bins=8, alpha=0.5, color=colors[1], label='OP2', edgecolor='black')
    ax.hist(df[op3_cols].values.flatten(), bins=8, alpha=0.5, color=colors[2], label='OP3', edgecolor='black')
    ax.set_xlabel("Valeur", fontsize=8)
    ax.set_ylabel("Fréquence", fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig3, use_container_width=True)

    # Interaction Pièce × Opérateur
    st.subheader("🔁 Interaction Pièce × Opérateur")
    interaction_df = pd.DataFrame({
        "Pièce": df["N° Pièce"],
        "OP1": df[op1_cols].mean(axis=1),
        "OP2": df[op2_cols].mean(axis=1),
        "OP3": df[op3_cols].mean(axis=1)
    })
    fig4, ax = plt.subplots(figsize=(4,2))
    ax.plot(interaction_df["Pièce"], interaction_df["OP1"], marker='o', color=colors[0], label='OP1')
    ax.plot(interaction_df["Pièce"], interaction_df["OP2"], marker='s', color=colors[1], label='OP2')
    ax.plot(interaction_df["Pièce"], interaction_df["OP3"], marker='^', color=colors[2], label='OP3')
    ax.set_xlabel("Pièce", fontsize=8)
    ax.set_ylabel("Mesure", fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=7)
    st.pyplot(fig4, use_container_width=True)

    # Carte de contrôle EV/AV/GRR compacte
    st.subheader("📈 Carte de contrôle EV/AV/GRR")
    fig5, ax = plt.subplots(figsize=(4,2))
    measures = [ev, av, grr]
    mean_val = np.mean(measures)
    std_val = np.std(measures)
    ucl = mean_val + 3*std_val
    lcl = max(mean_val - 3*std_val, 0)
    ax.plot(['EV','AV','GRR'], measures, marker='o', color='purple', label='Mesures')
    ax.axhline(mean_val, color='green', linestyle='--', label='Moyenne')
    ax.axhline(ucl, color='red', linestyle='--', label='UCL')
    ax.axhline(lcl, color='red', linestyle='--', label='LCL')
    ax.set_ylim(0, max(ucl*1.1, max(measures)*1.2))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=7)
    st.pyplot(fig5, use_container_width=True)

    # --- EXPORT EXCEL ---
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
