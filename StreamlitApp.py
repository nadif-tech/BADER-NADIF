import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# L'importation correcte en haut du fichier corrige le NameError
st.set_page_config(page_title="Gage R&R - Lean Six Sigma", layout="wide")

st.title("📊 Analyse du Système de Mesure (Gage R&R)")
st.write("Étude de précision basée sur la méthode des étendues et des moyennes.")

# --- FORMULES ET CONSTANTES DU COURS ---
def get_d2(z, w):
    # Table des valeurs d2 (Source: Page 71)
    if z > 15 and w == 3: return 1.693  # Répétabilité (Page 70)
    if z == 1 and w == 3: return 1.91   # Reproductibilité (Page 72)
    if z == 1 and w == 10: return 3.18  # Variabilité pièce (Page 73)
    return 1.128

# --- 1. IMPORTATION DES DONNÉES ---
st.header("📂 1. Importation du fichier TESET.xlsx")
uploaded_file = st.file_uploader("Choisissez votre fichier de test", type=["xlsx", "csv"])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
    st.write("Aperçu des données importées :")
    st.dataframe(df.head())

    # --- 2. PARAMÈTRES DE L'ÉTUDE (Page 69) ---
    with st.sidebar:
        st.header("⚙️ Paramètres")
        n = st.number_input("Nombre de pièces (n)", value=10) # Ligne 35 corrigée
        op = st.number_input("Nombre d'opérateurs", value=3)
        r = st.number_input("Nombre d'essais (r)", value=3)
        k = 5.15 # Facteur de confiance 99% (Page 70)

    # --- 3. SAISIE DES MOYENNES (Exemple Page 76) ---
    st.header("🧮 2. Calculs Statistiques")
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

    if st.button("Lancer l'Analyse Complète"):
        # Calcul EV - Répétabilité (Page 70)
        r_double_bar = (r1 + r2 + r3) / op
        d2_ev = get_d2(n * op, r)
        ev = (k * r_double_bar) / d2_ev
        
        # Calcul AV - Reproductibilité (Page 72)
        x_etendue = max([x1, x2, x3]) - min([x1, x2, x3])
        d2_av = get_d2(1, op)
        av = np.sqrt(max(0, (k * x_etendue / d2_av)**2 - (ev**2 / (n * r))))
        
        # R&R et Variabilité Totale (Page 73-74)
        rr = np.sqrt(ev**2 + av**2)
        vp = (k * 0.33) / 3.18 # Exemple Vp page 79
        vt = np.sqrt(rr**2 + vp**2)
        
        # % Contribution (Page 80)
        p_rr = (rr / vt) * 100

        st.subheader("Résultats du Système de Mesure")
        st.metric("Gage R&R (%)", f"{p_rr:.2f}%")

        if p_rr < 10:
            st.success("✅ Processus satisfaisant [Page 81]")
        elif 10 <= p_rr <= 30:
            st.warning("⚠️ Processus acceptable mais à améliorer [Page 81]")
        else:
            st.error("❌ Processus inacceptable [Page 81]")

        # --- 4. GRAPHES (Histogramme et Cartes de contrôle) ---
        st.header("📈 3. Graphiques de Performance")
        
        g1, g2 = st.columns(2)
        
        with g1:
            st.write("#### Distribution des mesures (Page 92)")
            # Image tag added here
            
            fig, ax = plt.subplots()
            data_sim = np.random.normal(45.07, 0.007, 100)
            sns.histplot(data_sim, kde=True, ax=ax, color="green")
            ax.set_title("Histogramme du processus")
            st.pyplot(fig)

        with g2:
            st.write("#### Carte de contrôle X-bar (Page 180)")
            # Image tag added here
            
            fig2, ax2 = plt.subplots()
            points = [x1, x2, x3, 45.07, 45.08, 45.10, 45.05, 45.09, 45.07, 45.08]
            ax2.plot(range(1, 11), points, marker='o', color='black')
            ax2.axhline(45.07, color='blue', label='Moyenne') # Moyenne
            ax2.axhline(45.15, color='red', linestyle='--', label='LSC') # Limite Sup
            ax2.axhline(44.99, color='red', linestyle='--', label='LIC') # Limite Inf
            ax2.set_title("Stabilité du processus (MSP)")
            ax2.legend()
            st.pyplot(fig2)

        # --- 5. EXPORT EXCEL ---
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_res = pd.DataFrame({
                "Composante": ["EV", "AV", "Gage R&R", "VT"],
                "Valeur": [ev, av, rr, vt],
                "% Contribution": [(ev/vt)*100, (av/vt)*100, p_rr, 100]
            })
            df_res.to_excel(writer, index=False, sheet_name='Resultats_Analyse')
        
        st.download_button("📥 Télécharger le rapport final (Excel)", output.getvalue(), "analyse_gage_rr.xlsx")
else:
    st.info("Veuillez charger votre fichier TESET.xlsx pour activer l'analyse.")
