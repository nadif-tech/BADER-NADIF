import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from typing import Tuple, List
import plotly.graph_objects as go
import plotly.express as px

# ===================================================== 
# CONFIGURATION PAGE
# ===================================================== 
st.set_page_config(
    page_title="Gage R&R – MSP", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================== 
# CONSTANTES AIAG
# ===================================================== 
K1 = {2: 0.886, 3: 0.590, 4: 0.485}
K2 = {2: 0.707, 3: 0.523, 4: 0.446}
K3 = 0.590

# ===================================================== 
# FONCTIONS UTILITAIRES
# ===================================================== 
def export_csv(data: pd.DataFrame) -> bytes:
    """Exporte un DataFrame en format CSV"""
    return data.to_csv(index=False).encode("utf-8")

def export_excel(data: pd.DataFrame) -> bytes:
    """Exporte un DataFrame en format Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        data.to_excel(writer, index=False, sheet_name="Données")
    return output.getvalue()

def validate_data(df: pd.DataFrame, n_operators: int, n_trials: int) -> bool:
    """Valide les données importées"""
    if df.empty:
        st.error("❌ Le fichier est vide")
        return False
    
    if df.isnull().values.any():
        st.warning("⚠️ Des valeurs manquantes ont été détectées")
        return False
    
    expected_cols = n_operators * n_trials
    if df.shape[1] != expected_cols:
        st.warning(f"⚠️ Nombre de colonnes attendu : {expected_cols}, reçu : {df.shape[1]}")
    
    return True

def calculate_gage_rr(
    df: pd.DataFrame, 
    n_parts: int, 
    n_operators: int, 
    n_trials: int
) -> Tuple[pd.DataFrame, dict]:
    """
    Calcule les composantes du Gage R&R selon la méthode AIAG
    
    Returns:
        Tuple contenant:
        - DataFrame détaillé par pièce
        - Dictionnaire avec les résultats globaux
    """
    ranges = []
    operator_means = []
    
    # Calcul des étendues et moyennes par opérateur
    for op in range(n_operators):
        cols = [f"Op{op+1}_Essai{t+1}" for t in range(n_trials)]
        df_op = df[cols]
        ranges.append(df_op.max(axis=1) - df_op.min(axis=1))
        operator_means.append(df_op.mean(axis=1))
    
    # Répétabilité (EV - Equipment Variation)
    R_bar = pd.concat(ranges, axis=1).mean(axis=1)
    EV = R_bar.mean() * K1.get(n_trials, 0.886)
    
    # Reproductibilité (AV - Appraiser Variation)
    op_means_global = [m.mean() for m in operator_means]
    X_diff = max(op_means_global) - min(op_means_global)
    
    AV = np.sqrt(max(
        (X_diff * K2.get(n_operators, 0.707))**2 - 
        (EV**2 / (n_parts * n_trials)), 
        0
    ))
    
    # Gage R&R (GRR)
    GRR = np.sqrt(EV**2 + AV**2)
    
    # Variation pièces (PV - Part Variation)
    part_means = df.mean(axis=1)
    PV = (part_means.max() - part_means.min()) * K3
    
    # Variation totale (TV)
    TV = np.sqrt(GRR**2 + PV**2)
    
    # Pourcentages
    EV_percent = (EV / TV) * 100 if TV > 0 else 0
    AV_percent = (AV / TV) * 100 if TV > 0 else 0
    GRR_percent = (GRR / TV) * 100 if TV > 0 else 0
    PV_percent = (PV / TV) * 100 if TV > 0 else 0
    
    # Nombre de catégories distinctes (NDC)
    ndc = int(1.41 * (PV / GRR)) if GRR > 0 else 0
    
    # Création du DataFrame détaillé
    detailed_df = pd.DataFrame({
        "Pièce": range(1, n_parts + 1),
        "Moyenne": part_means.values,
        "Étendue moy.": R_bar.values,
        "EV": [EV] * n_parts,
        "AV": [AV] * n_parts,
        "GRR": [GRR] * n_parts,
        "PV": [PV] * n_parts,
        "TV": [TV] * n_parts
    })
    
    # Résultats globaux
    results = {
        "EV": EV,
        "AV": AV,
        "GRR": GRR,
        "PV": PV,
        "TV": TV,
        "EV_percent": EV_percent,
        "AV_percent": AV_percent,
        "GRR_percent": GRR_percent,
        "PV_percent": PV_percent,
        "NDC": ndc,
        "operator_means": op_means_global,
        "part_means": part_means
    }
    
    return detailed_df, results

def create_variation_chart(results: dict) -> go.Figure:
    """Crée un graphique interactif des composantes de variation"""
    fig = go.Figure(data=[
        go.Bar(
            x=["Répétabilité<br>(EV)", "Reproductibilité<br>(AV)", "Variation Pièces<br>(PV)"],
            y=[results["EV_percent"], results["AV_percent"], results["PV_percent"]],
            marker_color=["#3498db", "#e74c3c", "#2ecc71"],
            text=[f"{results['EV_percent']:.1f}%", 
                  f"{results['AV_percent']:.1f}%", 
                  f"{results['PV_percent']:.1f}%"],
            textposition="outside"
        )
    ])
    
    fig.update_layout(
        title="Composantes de Variation (% de TV)",
        yaxis_title="Pourcentage de la Variation Totale",
        showlegend=False,
        height=400
    )
    
    return fig

def create_operator_comparison_chart(results: dict, n_operators: int) -> go.Figure:
    """Crée un graphique de comparaison des opérateurs"""
    fig = go.Figure(data=[
        go.Bar(
            x=[f"Opérateur {i+1}" for i in range(n_operators)],
            y=results["operator_means"],
            marker_color="#9b59b6",
            text=[f"{val:.3f}" for val in results["operator_means"]],
            textposition="outside"
        )
    ])
    
    fig.update_layout(
        title="Moyennes par Opérateur",
        yaxis_title="Valeur moyenne",
        showlegend=False,
        height=400
    )
    
    return fig

def display_interpretation(grr_percent: float, ndc: int):
    """Affiche l'interprétation des résultats"""
    st.markdown("### 📋 Interprétation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Selon % GRR:**")
        if grr_percent < 10:
            st.success("✅ Système de mesure **ACCEPTABLE**")
            st.info("Le système de mesure est excellent pour cette application.")
        elif grr_percent < 30:
            st.warning("⚠️ Système **ACCEPTABLE SOUS CONDITIONS**")
            st.info("Le système peut être acceptable selon l'application et les coûts d'amélioration.")
        else:
            st.error("❌ Système de mesure **NON ACCEPTABLE**")
            st.info("Des efforts doivent être faits pour améliorer le système de mesure.")
    
    with col2:
        st.markdown("**Selon NDC (Nombre de Catégories Distinctes):**")
        if ndc >= 5:
            st.success(f"✅ NDC = {ndc} - Excellent")
            st.info("Le système peut distinguer adéquatement les pièces.")
        elif ndc >= 4:
            st.warning(f"⚠️ NDC = {ndc} - Acceptable")
            st.info("Le système est marginalement acceptable.")
        else:
            st.error(f"❌ NDC = {ndc} - Insuffisant")
            st.info("Le système ne peut pas distinguer efficacement les pièces.")

# ===================================================== 
# INTERFACE PRINCIPALE
# ===================================================== 
st.title("📊 Analyse du Système de Mesure – Gage R&R")
st.markdown("**Méthode des étendues et des moyennes (AIAG)**")

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Configuration")
    
    mode = st.radio("Mode de saisie", ["Import fichier", "Saisie manuelle"])
    
    if mode == "Saisie manuelle":
        n_parts = st.number_input("Nombre de pièces", 2, 50, 5)
        n_operators = st.number_input("Nombre d'opérateurs", 2, 10, 3)
        n_trials = st.number_input("Nombre de répétitions", 2, 10, 2)
    
    st.markdown("---")
    st.markdown("### 📖 Guide rapide")
    st.markdown("""
    - **EV** : Répétabilité (équipement)
    - **AV** : Reproductibilité (opérateurs)
    - **GRR** : Gage R&R (EV + AV)
    - **PV** : Variation entre pièces
    - **TV** : Variation totale
    """)

# ===================================================== 
# IMPORT OU SAISIE DES DONNÉES
# ===================================================== 
if mode == "Import fichier":
    st.subheader("📤 Importer les données")
    uploaded_file = st.file_uploader(
        "Importer un fichier CSV ou Excel",
        type=["csv", "xlsx"],
        help="Le fichier doit contenir les mesures organisées par opérateur et essai"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine="openpyxl")
            
            df = df.dropna(axis=1, how="all")
            
            n_parts = df.shape[0]
            total_cols = df.shape[1]
            
            col1, col2 = st.columns(2)
            with col1:
                n_trials = st.number_input("Nombre de répétitions", 2, 10, 2, key="trials_import")
            with col2:
                n_operators = total_cols // n_trials
                st.metric("Opérateurs détectés", n_operators)
            
            if validate_data(df, n_operators, n_trials):
                st.success(f"✅ Fichier importé : {n_parts} pièces, {n_operators} opérateurs, {n_trials} essais")
        except Exception as e:
            st.error(f"❌ Erreur lors de l'import : {str(e)}")
            df = None
    else:
        df = None
else:
    total_cols = n_operators * n_trials
    df = pd.DataFrame(np.random.uniform(10, 20, (n_parts, total_cols)))

# ===================================================== 
# NORMALISATION ET AFFICHAGE DES DONNÉES
# ===================================================== 
if df is not None:
    expected_columns = [
        f"Op{op+1}_Essai{t+1}" 
        for op in range(n_operators) 
        for t in range(n_trials)
    ]
    
    df = df.iloc[:, :len(expected_columns)]
    df.columns = expected_columns
    df.index = [f"Pièce {i+1}" for i in range(len(df))]
    
    st.subheader("📥 Données de mesure")
    
    # Utiliser un éditeur de données pour permettre la modification
    edited_df = st.data_editor(
        df, 
        use_container_width=True, 
        height=400,
        num_rows="fixed"
    )
    
    # Export des mesures
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Exporter en CSV",
            export_csv(edited_df),
            "mesures_gage_rr.csv",
            "text/csv",
            use_container_width=True
        )
    with col2:
        st.download_button(
            "⬇️ Exporter en Excel",
            export_excel(edited_df),
            "mesures_gage_rr.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    # ===================================================== 
    # CALCUL ET AFFICHAGE DES RÉSULTATS
    # ===================================================== 
    if st.button("🔢 Calculer Gage R&R", type="primary", use_container_width=True):
        with st.spinner("Calcul en cours..."):
            detailed_df, results = calculate_gage_rr(
                edited_df, 
                n_parts, 
                n_operators, 
                n_trials
            )
        
        st.success("✅ Calculs terminés")
        
        # Résultats globaux
        st.subheader("📈 Résultats globaux")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("EV", f"{results['EV']:.4f}", f"{results['EV_percent']:.1f}%")
        col2.metric("AV", f"{results['AV']:.4f}", f"{results['AV_percent']:.1f}%")
        col3.metric("GRR", f"{results['GRR']:.4f}", f"{results['GRR_percent']:.1f}%")
        col4.metric("PV", f"{results['PV']:.4f}", f"{results['PV_percent']:.1f}%")
        col5.metric("NDC", results['NDC'])
        
        # Interprétation
        display_interpretation(results['GRR_percent'], results['NDC'])
        
        # Graphiques
        st.subheader("📊 Visualisations")
        
        tab1, tab2 = st.tabs(["Composantes de Variation", "Comparaison Opérateurs"])
        
        with tab1:
            st.plotly_chart(create_variation_chart(results), use_container_width=True)
        
        with tab2:
            st.plotly_chart(
                create_operator_comparison_chart(results, n_operators), 
                use_container_width=True
            )
        
        # Tableau détaillé
        st.subheader("📋 Détail des composantes par pièce")
        st.dataframe(
            detailed_df.style.format({
                col: "{:.4f}" for col in detailed_df.columns if col != "Pièce"
            }),
            use_container_width=True,
            height=400
        )
        
        # Export des résultats
        st.download_button(
            "⬇️ Exporter les résultats",
            export_excel(detailed_df),
            "resultats_gage_rr.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
else:
    st.info("👆 Importez un fichier ou configurez la saisie manuelle pour commencer")

# Footer
st.markdown("---")
st.markdown("*Développé selon la méthode AIAG (Automotive Industry Action Group)*")
