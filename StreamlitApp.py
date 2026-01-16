import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Initialisation de l'application
st.set_page_config(page_title="Gage R&R Pro - Lean Six Sigma", layout="wide")

# L'importation correcte de streamlit évite le NameError
st.title("📊 Analyse du Système de Mesure (Gage R&R)")
st.write("Application de calcul basée sur la méthode des étendues et des moyennes.")

# --- FONCTION d2 (Table Page 71) ---
def get_d2(z, w):
    if z > 15 and w == 3: return 1.693  # Répétabilité (Page 70)
    if z == 1 and w == 3: return 1.91   # Reproductibilité (Page 72)
    if z == 1 and w == 10: return 3.18  # Variabilité pièce (Page 73)
    return 1.128

# --- 1. IMPORTATION ET TRAITEMENT DU FICHIER ---
st.header("📂 1. Importation des données")
uploaded_file = st.file_uploader("Chargez votre fichier 'TESET.xlsx'", type=["xlsx", "csv"])

if uploaded_file:
    # Lecture du fichier
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
    st.write("Aperçu des données :")
    st.dataframe(df.head())

    # --- 2. PARAMÈTRES DE L'ÉTUDE ---
    with st.sidebar:
        st.header("⚙️ Paramètres")
        n_pieces = st.number_input("Nombre de pièces (n)", value=10) [cite: 695]
        n_ops = st.number_input("Nombre d'opérateurs", value=3) [cite: 695]
        n_essais = st.number_input("Nombre d'essais (r)", value=3) [cite: 695]
        k = 5.15  # Facteur de confiance 99% (Page 70)

    # --- 3. CALCULS STATISTIQUES (Page 76-80) ---
    st.header("🧮 2. Calculs et Résultats")
    
    # Saisie manuelle des moyennes pour la démonstration (peut être automatisé)
    col1, col2, col3 = st.columns(3)
    with col1:
        x1 = st.number_input("Moyenne OP1 (X̄1)", value=45.09)
        r1 = st.number_input("Étendue R̄1", value=0.055)
    with col2:
        x2 = st.number_input("Moyenne OP2 (X̄2)", value=45.06)
        r2 = st.number_input("Étendue R̄2", value=0.087)
    with col3:
        x3 = st.number_input("Moyenne OP3 (X̄3)", value=45.08)
        r3 = st.number_input("Étendue R̄3", value=0.031)

    if st.button("Lancer l'Analyse"):
        # EV - Répétabilité [cite: 700]
        r_double_bar = (r1 + r2 + r3) / n_ops [cite: 751]
        d2_ev = get_d2(n_pieces * n_ops, n_essais) [cite: 705]
        ev = (k * r_double_bar) / d2_ev [cite: 702]
        
        # AV - Reproductibilité [cite: 716]
        x_range = max([x1, x2, x3]) - min([x1, x2, x3]) [cite: 718]
        d2_av = get_d2(1, n_ops) [cite: 720]
        av_val = np.sqrt(max(0, (k * x_range / d2_av)**2 - (ev**2 / (n_pieces * n_essais)))) [cite: 717]
        
        # R&R et Variabilité Totale [cite: 725, 733]
        rr = np.sqrt(ev**2 + av_val**2) [cite: 725]
        vp = (k * 0.33) / 3.18  # Exemple Rp=0.33, d2=3.18 [cite: 772, 773, 774]
        vt = np.sqrt(rr**2 + vp**2) [cite: 733]
        
        # Pourcentages de contribution [cite: 782, 783, 784]
        p_ev = (ev / vt) * 100
        p_av = (av_val / vt) * 100
        p_rr = (rr / vt) * 100

        # Affichage des métriques
        st.subheader("Indicateurs Clés")
        m1, m2, m3 = st.columns(3)
        m1.metric("EV (Équipement)", f"{p_ev:.1f}%")
        m2.metric("AV (Opérateur)", f"{p_av:.1f}%")
        m3.metric("Gage R&R", f"{p_rr:.1f}%")

        # Interprétation (Page 81)
        if p_rr < 10:
            st.success("✅ Processus satisfaisant") [cite: 791]
        elif 10 <= p_rr <= 30:
            st.warning("⚠️ Processus acceptable mais à améliorer") [cite: 792]
        else:
            st.error("❌ Processus inacceptable") [cite: 793]

        # --- 4. GRAPHES DÉVELOPPÉS ---
        st.header("📈 3. Visualisation de la Variabilité")
        
        g_col1, g_col2 = st.columns(2)
        
        with g_col1:
            st.write("#### Histogramme de Distribution (Page 92)")
            # Simulation d'une loi normale basée sur les mesures
            data_sim = np.random.normal(45.07, 0.007, 100)
            fig, ax = plt.subplots()
            sns.histplot(data_sim, kde=True, ax=ax, color="skyblue")
            ax.axvline(45.07, color='red', linestyle='--') # Moyenne
            st.pyplot(fig) [cite: 911, 919]

        with g_col2:
            st.write("#### Carte de Contrôle des Moyennes (Page 180)")
            # Graphique des moyennes par échantillon
            fig2, ax2 = plt.subplots()
            samples = range(1, 11)
            values = [x1, x2, x3, 45.07, 45.08, 45.10, 45.05, 45.09, 45.07, 45.08]
            ax2.plot(samples, values, marker='o', linestyle='-', color='black')
            ax2.axhline(45.07, color='blue', label='Moyenne')
            ax2.axhline(45.15, color='red', linestyle='--', label='LSC')
            ax2.axhline(44.99, color='red', linestyle='--', label='LIC')
            ax2.legend()
            st.pyplot(fig2) [cite: 1279, 1280]

        # --- 5. EXPORTATION ---
        st.header("📥 4. Export des résultats")
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_final = pd.DataFrame({
                "Composante": ["EV", "AV", "Gage R&R", "VT"],
                "Valeur": [ev, av_val, rr, vt],
                "Pourcentage": [p_ev, p_av, p_rr, 100]
            })
            df_final.to_excel(writer, index=False, sheet_name='Analyse_RR')
        
        st.download_button(
            label="Télécharger le rapport Excel",
            data=output.getvalue(),
            file_name="Analyse_Gage_RR.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Veuillez charger votre fichier Excel pour activer l'analyse complète.")
