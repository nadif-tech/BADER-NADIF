import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# =====================================================
# CONFIGURATION
# =====================================================
st.set_page_config(page_title="Gage R&R – MSP", layout="wide")

st.title("📊 Analyse du Système de Mesure – Gage R&R")
st.markdown("**Méthode des étendues et des moyennes (AIAG)**")

# =====================================================
# PARAMÈTRES UTILISATEUR
# =====================================================
c1, c2, c3 = st.columns(3)
with c1:
    n_parts = st.number_input("Nombre de pièces", 2, 50, 5)
with c2:
    n_operators = st.number_input("Nombre d'opérateurs", 2, 10, 3)
with c3:
    n_trials = st.number_input("Nombre de répétitions", 2, 10, 2)

expected_columns = [
    f"Op{op+1}_Essai{t+1}"
    for op in range(n_operators)
    for t in range(n_trials)
]

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

    # Nettoyage
    df = df.dropna(axis=1, how="all")  # supprimer colonnes vides

    if df.shape[1] != len(expected_columns):
        st.error(
            f"❌ Colonnes importées : {df.shape[1]} | "
            f"Colonnes attendues : {len(expected_columns)}"
        )
        st.stop()

    df.columns = expected_columns
    st.success("✅ Données importées et normalisées")
else:
    df = pd.DataFrame(np.zeros((n_parts, len(expected_columns))), columns=expected_columns)

# =====================================================
# TABLEAU DE SAISIE
# =====================================================
st.subheader("📥 Tableau de saisie des mesures")
df = st.data_editor(df, use_container_width=True)

# =====================================================
# EXPORT DES MESURES
# =====================================================
st.subheader("📁 Exporter les données")

def export_csv(data):
    return data.to_csv(index=False).encode("utf-8")

def export_excel(data):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        data.to_excel(writer, index=False, sheet_name="Mesures")
    return output.getvalue()

e1, e2 = st.columns(2)
with e1:
    st.download_button(
        "⬇️ Export CSV",
        export_csv(df),
        "mesures_gage_rr.csv",
        "text/csv"
    )

with e2:
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

    # =================================================
    # AFFICHAGE DES RÉSULTATS
    # =================================================
    st.subheader("📈 Résultats")

    r1, r2, r3 = st.columns(3)
    r1.metric("Répétabilité (EV)", f"{EV:.4f}")
    r2.metric("Reproductibilité (AV)", f"{AV:.4f}")
    r3.metric("Gage R&R", f"{GRR:.4f}")

    r4, r5, r6 = st.columns(3)
    r4.metric("Variation Pièces (PV)", f"{PV:.4f}")
    r5.metric("Variation Totale (TV)", f"{TV:.4f}")
    r6.metric("% Gage R&R", f"{GRR_percent:.2f} %")

    if GRR_percent < 10:
        st.success("✅ Système de mesure acceptable")
    elif GRR_percent < 30:
        st.warning("⚠️ Acceptable sous conditions")
    else:
        st.error("❌ Système de mesure non acceptable")

    # =================================================
    # EXPORT DES RÉSULTATS
    # =================================================
    results_df = pd.DataFrame({
        "Indicateur": ["EV", "AV", "GRR", "PV", "TV", "% GRR"],
        "Valeur": [EV, AV, GRR, PV, TV, GRR_percent]
    })

    st.download_button(
        "⬇️ Export Résultats Excel",
        export_excel(results_df),
        "resultats_gage_rr.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # =================================================
    # GRAPHIQUES
    # =================================================
    st.subheader("📊 Graphiques MSP")

    fig1, ax1 = plt.subplots()
    ax1.boxplot(df.mean(axis=1))
    ax1.set_title("Variation pièce à pièce")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.bar(["EV", "AV", "PV"], [EV, AV, PV])
    ax2.set_title("Composantes de variation")
    st.pyplot(fig2)
