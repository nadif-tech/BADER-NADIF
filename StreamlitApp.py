import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import math

# =====================================================
# CONFIGURATION PAGE
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
    st.info(f"📌 Détection automatique : {n_operators} opérateurs × {n_trials} essais")
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
    f"Op{op+1}_Essai{t+1}" for op in range(n_operators) for t in range(n_trials)
]
df = df.iloc[:, :len(expected_columns)]
df.columns = expected_columns

# =====================================================
# TABLEAU DE SAISIE
# =====================================================
st.subheader("📥 Tableau de saisie des mesures")
st.dataframe(df, use_container_width=True, height=400)

# =====================================================
# EXPORT DES MESURES
# =====================================================
def export_csv(data):
    return data.to_csv(index=False).encode("utf-8")

def export_excel(data):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        data.to_excel(writer, index=False)
    return output.getvalue()

e1, e2 = st.columns(2)
with e1:
    st.download_button(
        "⬇️ Export mesures CSV",
        export_csv(df),
        "mesures_gage_rr.csv",
        "text/csv"
    )
with e2:
    st.download_button(
        "⬇️ Export mesures Excel",
        export_excel(df),
        "mesures_gage_rr.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =====================================================
# CONSTANTES AIAG
# =====================================================
K3 = 0.590  # PV coefficient

# Table des d2 pour n répétitions (AIAG standard)
d2_table = {2:1.128, 3:1.693, 4:2.059, 5:2.326, 6:2.534,
            7:2.704, 8:2.847, 9:2.970, 10:3.078}

# =====================================================
# CALCUL GAGE R&R
# =====================================================
if st.button("🔢 Calculer Gage R&R"):
    ranges = []
    operator_means = []

    # Calcul des étendues et moyennes par opérateur
    for op in range(n_operators):
        cols = [f"Op{op+1}_Essai{t+1}" for t in range(n_trials)]
        df_op = df[cols]
        ranges.append(df_op.max(axis=1) - df_op.min(axis=1))   # étendue pièce par opérateur
        operator_means.append(df_op.mean(axis=1))              # moyenne par pièce

    # Répétabilité (EV)
    R_bar_total = np.mean([r.mean() for r in ranges])  # moyenne des étendues global
    EV = R_bar_total  # EV global (ou appliquer K1 si nécessaire)

    # Reproductibilité (AV) selon formule exacte
    d2 = d2_table.get(n_trials, 1.128)
    AV = np.sqrt(
        (5.15 * R_bar_total / d2)**2 - (EV**2 / (n_parts * n_trials))
    )

    # Gage R&R
    GRR = np.sqrt(EV**2 + AV**2)

    # Variation pièces et totale
    part_means = df.mean(axis=1)
    PV = (part_means.max() - part_means.min()) * K3
    TV = np.sqrt(GRR**2 + PV**2)
    GRR_percent = (GRR / TV) * 100

    # =====================================================
    # TABLEAU DÉTAILLÉ PAR PIÈCE
    # =====================================================
    detailed_df = pd.DataFrame({
        "Pièce": range(1, n_parts + 1),
        "EV (répétabilité)": [EV]*n_parts,
        "AV (reproductibilité)": [AV]*n_parts,
        "GRR": [GRR]*n_parts,
        "PV (variation pièces)": [PV]*n_parts,
        "TV (variation totale)": [TV]*n_parts
    })

    st.subheader("📊 Détail des composantes par pièce")
    st.dataframe(detailed_df, use_container_width=True, height=400)

    # =====================================================
    # EXPORT DES RÉSULTATS
    # =====================================================
    st.download_button(
        "⬇️ Export résultats Excel",
        export_excel(detailed_df),
        "resultats_gage_rr.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # =====================================================
    # AFFICHAGE GLOBALE
    # =====================================================
    st.subheader("📈 Résultats globaux")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("EV", f"{EV:.4f}")
    col2.metric("AV", f"{AV:.4f}")
    col3.metric("GRR", f"{GRR:.4f}")
    col4.metric("% Gage R&R", f"{GRR_percent:.2f}%")

    if GRR_percent < 10:
        st.success("✅ Système de mesure acceptable")
    elif GRR_percent < 30:
        st.warning("⚠️ Acceptable sous conditions")
    else:
        st.error("❌ Système de mesure non acceptable")

    # =====================================================
    # GRAPHIQUES
    # =====================================================
    st.subheader("📊 Graphiques des composantes de variation")
    fig, ax = plt.subplots()
    ax.bar(["EV", "AV", "PV"], [EV, AV, PV], color=["skyblue","orange","green"])
    ax.set_title("Composantes de variation")
    st.pyplot(fig)
