import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# --- Configuration page ---
st.set_page_config(page_title="Gage R&R – Dashboard", layout="wide")
st.title("📊 Gage R&R – Dashboard Compact et Structuré")
st.write("---")

# --- Fonction d2 ---
def get_d2(z, w):
    if z > 15 and w == 3:
        return 1.693
    if z == 1 and w == 3:
        return 1.91
    if z == 1 and w == 10:
        return 3.18
    return 1.0

# --- Sidebar paramètres ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    confidence_factor = st.number_input("Facteur de confiance", value=5.15, step=0.01)
    bins_hist = st.slider("Nombre de classes (bins) histogramme", 5, 20, 8)

# --- Import Excel ---
uploaded_file = st.file_uploader("📥 Importer le fichier Excel Gage R&R", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("📄 Aperçu des données")
    st.dataframe(df, use_container_width=True)

    # Colonnes opérateurs
    op1_cols = ["OP1-1", "OP1-2", "OP1-3"]
    op2_cols = ["OP2-1", "OP2-2", "OP2-3"]
    op3_cols = ["OP3-1", "OP3-2", "OP3-3"]
    n_pieces = df.shape[0]
    n_operateurs = 3
    n_essais = 3

    # --- Calculs ---
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

    df["Moy_Piece"] = df[op1_cols + op2_cols + op3_cols].mean(axis=1)
    rp = df["Moy_Piece"].max() - df["Moy_Piece"].min()
    vp = (confidence_factor * rp) / get_d2(1, n_pieces)
    vt = np.sqrt(grr ** 2 + vp ** 2)
    p_ev = (ev / vt) * 100
    p_av = (av / vt) * 100
    p_vp = (vp / vt) * 100
    p_grr = (grr / vt) * 100
    ndc = 1.41 * (vp / grr) if grr != 0 else 0

    # --- Tableaux bien structurés ---
    st.subheader("📋 Tableaux Résultats Gage R&R")

    # Résultats principaux
    main_results = pd.DataFrame({
        "Indicateur": ["EV", "AV", "GRR", "VT"],
        "Valeur": [round(ev,4), round(av,4), round(grr,4), round(vt,4)]
    })

    # Pourcentages
    percentage_results = pd.DataFrame({
        "Indicateur": ["%EV", "%AV", "%VP", "%GRR", "NdC"],
        "Valeur": [round(p_ev,1), round(p_av,1), round(p_vp,1), round(p_grr,1), round(ndc,1)]
    })

    # Affichage côte à côte
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📌 Résultats principaux**")
        st.table(main_results.style.set_properties(**{'text-align':'center'}).set_table_styles(
            [{'selector':'th','props':[('text-align','center')]}]
        ))

    with col2:
        st.markdown("**📌 Pourcentages (%) et NdC**")
        st.table(percentage_results.style.set_properties(**{'text-align':'center'}).set_table_styles(
            [{'selector':'th','props':[('text-align','center')]}]
        ))

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
