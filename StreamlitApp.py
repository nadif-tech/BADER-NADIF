import streamlit as st
import pandas as pd
import numpy as np
import io

st.title("Gage R&R - Méthode des Moyennes et des Étendues")

st.markdown("""
Application pour calculer la précision d'un système de mesure (Gage R&R)  
basée sur les formules que vous avez fournies.
""")

# Table d2 intégrée (exactement comme dans votre image)
d2_table = {
    1: {2:1.41, 3:1.91, 4:2.24, 5:2.48, 6:2.67, 7:2.83, 8:2.96, 9:3.08, 10:3.18, 11:3.27, 12:3.35, 13:3.42, 14:3.49, 15:3.55},
    2: {2:1.28, 3:1.81, 4:2.15, 5:2.40, 6:2.60, 7:2.77, 8:2.91, 9:3.02, 10:3.13, 11:3.22, 12:3.30, 13:3.38, 14:3.45, 15:3.51},
    3: {2:1.23, 3:1.77, 4:2.12, 5:2.38, 6:2.58, 7:2.75, 8:2.89, 9:3.01, 10:3.11, 11:3.21, 12:3.29, 13:3.37, 14:3.43, 15:3.50},
    4: {2:1.21, 3:1.75, 4:2.11, 5:2.37, 6:2.57, 7:2.74, 8:2.88, 9:3.00, 10:3.10, 11:3.20, 12:3.28, 13:3.36, 14:3.43, 15:3.49},
    5: {2:1.19, 3:1.74, 4:2.10, 5:2.36, 6:2.56, 7:2.78, 8:2.87, 9:2.99, 10:3.10, 11:3.19, 12:3.28, 13:3.36, 14:3.42, 15:3.49},
    6: {2:1.18, 3:1.73, 4:2.09, 5:2.35, 6:2.56, 7:2.73, 8:2.87, 9:2.99, 10:3.10, 11:3.19, 12:3.27, 13:3.35, 14:3.42, 15:3.49},
    7: {2:1.17, 3:1.73, 4:2.09, 5:2.35, 6:2.55, 7:2.72, 8:2.87, 9:2.98, 10:3.10, 11:3.19, 12:3.27, 13:3.35, 14:3.42, 15:3.48},
    8: {2:1.17, 3:1.72, 4:2.08, 5:2.35, 6:2.55, 7:2.72, 8:2.87, 9:2.98, 10:3.09, 11:3.19, 12:3.27, 13:3.35, 14:3.42, 15:3.48},
    9: {2:1.16, 3:1.72, 4:2.08, 5:2.34, 6:2.55, 7:2.72, 8:2.86, 9:2.98, 10:3.09, 11:3.19, 12:3.27, 13:3.35, 14:3.42, 15:3.48},
    10: {2:1.16, 3:1.72, 4:2.08, 5:2.34, 6:2.55, 7:2.72, 8:2.86, 9:2.98, 10:3.09, 11:3.18, 12:3.27, 13:3.34, 14:3.42, 15:3.48},
    # Lignes supplémentaires si besoin (jusqu'à >15)
}

# Fonction pour obtenir d2
def get_d2(z, w):
    if z > 15:
        z = ">15"
    if w > 15:
        w = 15
    # Valeurs approximatives pour >15 (basé sur des tables standards)
    if z == ">15":
        approx = {2:1.128, 3:1.693, 4:2.059, 5:2.326, 6:2.534, 7:2.704, 8:2.847, 9:2.97, 10:3.078, 11:3.173, 12:3.258, 13:3.336, 14:3.407, 15:3.472}
        return approx.get(w, 3.472)
    return d2_table.get(z, d2_table[10]).get(w, d2_table[10][15])

# Upload CSV ou saisie manuelle
upload = st.file_uploader("Uploader un fichier CSV avec les données de mesures", type=["csv"])

if upload is not None:
    df = pd.read_csv(upload)
else:
    st.info("Ou entrez manuellement les données (exemple standard : 10 pièces, 3 opérateurs, 3 essais)")
    # Exemple de données pour test
    example_data = pd.DataFrame({
        'Pièce': [1]*9 + [2]*9 + [3]*9 + [4]*9 + [5]*9 + [6]*9 + [7]*9 + [8]*9 + [9]*9 + [10]*9,
        'Opérateur': [1,1,1,2,2,2,3,3,3]*10,
        'Essai': list(range(1,4))*30,
        'Mesure': np.random.normal(50, 2, 90)  # Données aléatoires pour test
    })
    df = st.data_editor(example_data, num_rows="dynamic")

if not df.empty:
    st.write("Données chargées :", df)

    # Paramètres
    col1, col2, col3 = st.columns(3)
    piece_col = col1.selectbox("Colonne Pièce", df.columns)
    oper_col = col2.selectbox("Colonne Opérateur", df.columns)
    trial_col = col3.selectbox("Colonne Essai (si présent, sinon ignorer)", df.columns + ["Aucun"])
    measure_col = st.selectbox("Colonne Mesure", df.columns)

    if st.button("Calculer Gage R&R"):
        data = df[[piece_col, oper_col, measure_col]]
        if trial_col != "Aucun":
            data = df[[piece_col, oper_col, trial_col, measure_col]]

        # Renommer pour simplicité
        data.columns = ["piece", "oper", "trial", "measure"] if trial_col != "Aucun" else ["piece", "oper", "measure"]

        n_pieces = data['piece'].nunique()
        n_opers = data['oper'].nunique()
        n_trials = data['trial'].nunique() if trial_col != "Aucun" else data.groupby(['piece','oper']).size().max()

        # Calcul des moyennes et étendues
        if trial_col != "Aucun":
            # Avec essais explicites
            ranges_per_oper_piece = data.groupby(['oper', 'piece'])['measure'].apply(lambda x: x.max() - x.min())
            avg_range = ranges_per_oper_piece.mean()  # \bar{R}
            oper_means = data.groupby(['oper', 'piece'])['measure'].mean().groupby('oper').mean()
        else:
            # Données déjà moyennées ou un essai par opérateur/pièce
            ranges_per_oper_piece = pd.Series(0, index=data.index)  # Pas d'étendue intra
            avg_range = 0
            oper_means = data.groupby(['oper', 'piece'])['measure'].mean().groupby('oper').mean()

        # Étendue des moyennes opérateurs
        X_etendue = oper_means.max() - oper_means.min()

        # Moyennes des pièces
        piece_means = data.groupby('piece')['measure'].mean()
        Rp = piece_means.max() - piece_means.min()

        # d2 pour répétabilité (z = n_trials, w = n_opers * n_pieces car g = n_opers * n_pieces ranges)
        # Approximation standard AIAG : pour EV, d2 avec subgroup size = trials, number of subgroups large → d2 standard
        # Ici on utilise z = n_trials, w = n_opers (comme souvent dans tables simplifiées)
        d2_repeat = get_d2(n_trials, n_opers)

        # Pour reproductibilité et pièces, même d2 mais ajusté
        d2_reprod = get_d2(1, n_opers)  # car étendue des moyennes opérateurs, subgroup=1
        d2_part = get_d2(1, n_pieces)   # étendue des moyennes pièces

        # Calculs
        repetabilite = 5.15 * avg_range / d2_repeat

        temp = (5.15 * X_etendue / d2_reprod)**2 - (repetabilite**2) / (n_pieces * n_trials)
        reproductibilite = np.sqrt(max(0, temp)) if temp > 0 else 0

        rr = np.sqrt(repetabilite**2 + reproductibilite**2)

        vp = 5.15 * Rp / d2_part

        vt = np.sqrt(rr**2 + vp**2)

        st.markdown("### Résultats")
        st.write(f"Répétabilité : {repetabilite:.4f}")
        st.write(f"Reproductibilité : {reproductibilite:.4f}")
        st.write(f"R&R : {rr:.4f}")
        st.write(f"V_p (Variation Pièces) : {vp:.4f}")
        st.write(f"V_T (Variation Totale) : {vt:.4f}")

        st.write(f"% R&R / VT = {100 * rr / vt:.2f}% (idéalement < 30%)")
