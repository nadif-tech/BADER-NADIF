import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gage R&R – MSP", layout="wide")

st.title("📊 Analyse du Système de Mesure – Gage R&R")
st.markdown("**Méthode des étendues et des moyennes (AIAG)**")

# Paramètres
c1, c2, c3 = st.columns(3)
with c1:
    n_parts = st.number_input("Nombre de pièces", 2, 50, 5)
with c2:
    n_operators = st.number_input("Nombre d'opérateurs", 2, 10, 3)
with c3:
    n_trials = st.number_input("Nombre de répétitions", 2, 10, 2)

# Tableau de mesures
st.subheader("📥 Tableau de saisie des mesures")

columns = [f"Op{op+1}_Essai{t+1}" for op in range(n_operators) for t in range(n_trials)]
df = pd.DataFrame(np.zeros((n_parts, len(columns))), columns=columns)
df = st.data_editor(df, use_container_width=True)

# Constantes AIAG
K1 = {2: 0.886, 3: 0.590, 4: 0.485}
K2 = {2: 0.707, 3: 0.523, 4: 0.446}
K3 = 0.590

# Calcul
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
    AV = np.sqrt(max((X_diff * K2.get(n_operators, 0.707))**2 - (EV**2 / (n_parts * n_trials)), 0))

    GRR = np.sqrt(EV**2 + AV**2)

    part_means = df.mean(axis=1)
    PV = (part_means.max() - part_means.min()) * K3

    TV = np.sqrt(GRR**2 + PV**2)
    GRR_percent = (GRR / TV) * 100

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

    st.subheader("📊 Graphiques MSP")

    fig1, ax1 = plt.subplots()
    ax1.boxplot([df.mean(axis=1)])
    ax1.set_title("Variation pièce à pièce")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.bar(["EV", "AV", "PV"], [EV, AV, PV])
    ax2.set_title("Composantes de variation")
    st.pyplot(fig2)
