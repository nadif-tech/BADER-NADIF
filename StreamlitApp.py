import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import math

# =====================================================
# CONFIGURATION
# =====================================================
st.set_page_config(page_title="Gage R&R – MSP", layout="wide")

st.title("📊 Analyse du Système de Mesure – Gage R&R")
st.markdown("**Méthode des étendues et des moyennes (AIAG)**")

# =====================================================
# IMPORT DES DONNÉES
# =====================================================
st.subheader("📤 Importer les données")

uploaded_file = st.file_uploader(
    "Importer un fichier CSV ou Excel (format Gage R&R)",
    type=["csv", "xlsx"]
)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl")

    df = df.dropna(axis=1, how="all")  # supprimer colonnes vides

    st.success(f"✅ Fichier importé ({df.shape[1]} colonnes détectées)")
else:
    df = None

# =====================================================
# PARAMÈTRES (AUTO OU MANUEL)
# =====================================================
st.subheader("⚙️ Paramètres")

if df is not None:
    n_parts = df.shape[0]

    total_cols = df.shape[1]
    n_trials = st.number_input("Nombre de répétitions", 2, 10, 2)
    n_operators = total_cols // n_trials

    st.info(
        f"📌 Détection automatique : "
        f"{n_operators} opérateurs × {n_trials} essais"
    )
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        n_parts = st.number_input("Nombre de pièces", 2, 50, 5)
    with c2:
        n_operators = st.number_input("Nombre d'opérateurs", 2, 10, 3)
    with c3:
        n_trials = st.number_input("Nombre de répétitions", 2, 10, 2)

    total_cols = n_operators * n_trials
    df = pd.DataFrame(np.zeros((n_parts, total_cols)))

# =====================================================
# NORMALISATION DES COLONNES
# =====================================================
expected_columns = [
    f"Op{op+1}_Essai{t+1}"
    for op in range(n_operators)
    for t in range(n_trials)
]

df = df.iloc[:, :len(expected_columns)]
df.columns = expected_columns

# =====================================================
# TABLEAU DE SAISIE
# =====================================================
st.subheader("📥 Tableau de saisie des mesures")
df = st.data_editor(df, use_container_width=True)

# =====================================================
# EXPORT DES MESURES
# =====================================================
def export_excel(data):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        data.to_excel(writer, index=False)
    return output.getvalue()

st.download_button(
    "⬇️ Export Excel",
    export_excel(df),
    "mesures_gage_rr.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# =====================================================
# CONSTANTES AIAG
# =====================================================
K1 = {2: 0.886, 3: 0.590, 4: 0.485}
K2 = {2: 0.707, 3: 0.523, 4: 0.446}
K3 = 0.590

# =====================================================
# CALCUL GAGE R&R
# =====================================================
if st.button("🔢 Calculer Gage R&R"):
    ranges = []
    operator_means = []

    for op in range(n_operators):
        cols = [f"Op{op+1}_Essai{t+1}" for t in range(n_trials)]
        df_op = df[cols]

        ranges.append(df_op.max(axis=1) - df_op.min(axis=1))
        operator_means.append(df_op.mean().mean())

    R_bar = pd.concat(ranges, axis=1).mean().mean()
    EV = R_bar * K1.get(n_trials, 0.886)

    X_diff = max(operator_means) - min(operator_means)
    AV = np.sqrt(
        max(
            (X_diff * K2.get(n_operators, 0.707))**2
            - (EV**2 / (n_parts * n_trials)),
            0
        )
    )

    GRR = np.sqrt(EV**2 + AV**2)

    part_means = df.mean(axis=1)
    PV = (part_means.max() - part_means.min()) * K3

    TV = np.sqrt(GRR**2 + PV**2)
    GRR_percent = (GRR / TV) * 100

    st.subheader("📈 Résultats")

    st.metric("Gage R&R (%)", f"{GRR_percent:.2f}")

    if GRR_percent < 10:
        st.success("✅ Système de mesure acceptable")
    elif GRR_percent < 30:
        st.warning("⚠️ Acceptable sous conditions")
    else:
        st.error("❌ Système de mesure non acceptable")

    fig, ax = plt.subplots()
    ax.bar(["EV", "AV", "PV"], [EV, AV, PV])
    st.pyplot(fig)
