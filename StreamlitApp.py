import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import random
import json

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GMAO Pro",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: #0a0e1a;
    color: #e0e6f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1224 0%, #111828 100%);
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 15px;
    color: #7a9cc0;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #111828 0%, #0d1a2e 100%);
    border: 1px solid #1e3a5a;
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00c3ff, #0062ff);
}
.metric-card.warn::before { background: linear-gradient(90deg, #ff9500, #ff5500); }
.metric-card.danger::before { background: linear-gradient(90deg, #ff3030, #c00040); }
.metric-card.success::before { background: linear-gradient(90deg, #00e676, #00b060); }
.metric-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #5a7a9a;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 36px;
    font-weight: 700;
    color: #e8f0ff;
    line-height: 1;
}
.metric-sub {
    font-size: 12px;
    color: #4a6a8a;
    margin-top: 6px;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'Rajdhani', sans-serif;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-critical { background: #3a0010; color: #ff4060; border: 1px solid #ff406040; }
.badge-high { background: #3a1a00; color: #ff8800; border: 1px solid #ff880040; }
.badge-medium { background: #2a2a00; color: #ffcc00; border: 1px solid #ffcc0040; }
.badge-low { background: #003a1a; color: #00cc66; border: 1px solid #00cc6640; }
.badge-open { background: #003060; color: #00aaff; border: 1px solid #00aaff40; }
.badge-progress { background: #2a1a3a; color: #aa66ff; border: 1px solid #aa66ff40; }
.badge-closed { background: #0a1a0a; color: #44aa44; border: 1px solid #44aa4440; }
.badge-planned { background: #001a3a; color: #4488ff; border: 1px solid #4488ff40; }

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    overflow: hidden;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0062ff, #00a8ff);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 8px 20px;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px #0062ff60;
}

/* Form inputs */
.stSelectbox > div > div, .stTextInput > div > div, .stTextArea > div > div {
    background: #111828 !important;
    border: 1px solid #1e3a5a !important;
    border-radius: 8px !important;
    color: #e0e6f0 !important;
}

/* Section titles */
.section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #a0c0e0;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid #1e3a5a;
    padding-bottom: 10px;
    margin-bottom: 20px;
}

/* Alert box */
.alert-box {
    background: #1a0a10;
    border: 1px solid #ff406040;
    border-left: 4px solid #ff4060;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
}

.info-box {
    background: #001828;
    border: 1px solid #00aaff40;
    border-left: 4px solid #00aaff;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
}

.success-box {
    background: #001810;
    border: 1px solid #00cc6640;
    border-left: 4px solid #00cc66;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
}

/* Logo / Brand */
.brand-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    background: linear-gradient(135deg, #00c3ff, #0062ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.brand-sub {
    font-size: 11px;
    letter-spacing: 3px;
    color: #3a5a7a;
    text-transform: uppercase;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #1e2d4a;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE / DONNÉES ────────────────────────────────────────────────────

def init_data():
    """Initialise les données de démonstration."""
    if "equipements" not in st.session_state:
        st.session_state.equipements = pd.DataFrame([
            {"ID": "EQ-001", "Nom": "Compresseur Atlas Copco GA55", "Catégorie": "Pneumatique", "Site": "Usine A", "Statut": "Opérationnel", "Criticité": "Critique", "Installation": "2019-03-15", "Dernière maintenance": "2024-11-01", "Prochaine maintenance": "2025-03-01", "Heures de marche": 12450},
            {"ID": "EQ-002", "Nom": "Convoyeur à bande CB-12", "Catégorie": "Manutention", "Site": "Usine A", "Statut": "En panne", "Criticité": "Haute", "Installation": "2020-07-10", "Dernière maintenance": "2024-09-15", "Prochaine maintenance": "2025-01-15", "Heures de marche": 8320},
            {"ID": "EQ-003", "Nom": "Pompe centrifuge PMP-3", "Catégorie": "Hydraulique", "Site": "Usine B", "Statut": "Opérationnel", "Criticité": "Haute", "Installation": "2018-11-22", "Dernière maintenance": "2024-10-20", "Prochaine maintenance": "2025-04-20", "Heures de marche": 18900},
            {"ID": "EQ-004", "Nom": "Robot soudure RS-7", "Catégorie": "Robotique", "Site": "Usine A", "Statut": "En maintenance", "Criticité": "Critique", "Installation": "2021-02-01", "Dernière maintenance": "2025-01-10", "Prochaine maintenance": "2025-07-10", "Heures de marche": 5600},
            {"ID": "EQ-005", "Nom": "Tour CNC Mazak", "Catégorie": "Usinage", "Site": "Usine C", "Statut": "Opérationnel", "Criticité": "Haute", "Installation": "2017-06-18", "Dernière maintenance": "2024-12-01", "Prochaine maintenance": "2025-06-01", "Heures de marche": 24300},
            {"ID": "EQ-006", "Nom": "Groupe électrogène GE-100", "Catégorie": "Électrique", "Site": "Usine B", "Statut": "Opérationnel", "Criticité": "Critique", "Installation": "2022-01-05", "Dernière maintenance": "2024-08-10", "Prochaine maintenance": "2025-02-10", "Heures de marche": 3200},
            {"ID": "EQ-007", "Nom": "Chaudière vapeur CV-50", "Catégorie": "Thermique", "Site": "Usine C", "Statut": "Opérationnel", "Criticité": "Critique", "Installation": "2016-09-30", "Dernière maintenance": "2024-07-15", "Prochaine maintenance": "2025-01-15", "Heures de marche": 31500},
            {"ID": "EQ-008", "Nom": "Pont roulant PR-10T", "Catégorie": "Levage", "Site": "Usine A", "Statut": "Opérationnel", "Criticité": "Haute", "Installation": "2015-04-12", "Dernière maintenance": "2024-11-30", "Prochaine maintenance": "2025-05-30", "Heures de marche": 41200},
        ])

    if "bons_travaux" not in st.session_state:
        st.session_state.bons_travaux = pd.DataFrame([
            {"BT": "BT-2025-001", "Équipement": "EQ-002", "Titre": "Remplacement courroie convoyeur", "Type": "Correctif", "Priorité": "Haute", "Statut": "En cours", "Demandeur": "M. Dupont", "Technicien": "A. Martin", "Date création": "2025-01-15", "Date prévue": "2025-01-20", "Durée estimée (h)": 4, "Coût estimé (€)": 850},
            {"BT": "BT-2025-002", "Équipement": "EQ-007", "Titre": "Inspection annuelle chaudière", "Type": "Préventif", "Priorité": "Critique", "Statut": "Planifié", "Demandeur": "Système auto", "Technicien": "B. Lefebvre", "Date création": "2025-01-10", "Date prévue": "2025-01-25", "Durée estimée (h)": 8, "Coût estimé (€)": 2400},
            {"BT": "BT-2025-003", "Équipement": "EQ-001", "Titre": "Vidange huile compresseur", "Type": "Préventif", "Priorité": "Moyenne", "Statut": "Terminé", "Demandeur": "Système auto", "Technicien": "C. Bernard", "Date création": "2025-01-05", "Date prévue": "2025-01-08", "Durée estimée (h)": 2, "Coût estimé (€)": 180},
            {"BT": "BT-2025-004", "Équipement": "EQ-004", "Titre": "Calibration robot soudure", "Type": "Correctif", "Priorité": "Critique", "Statut": "En cours", "Demandeur": "Production", "Technicien": "A. Martin", "Date création": "2025-01-12", "Date prévue": "2025-01-18", "Durée estimée (h)": 12, "Coût estimé (€)": 3200},
            {"BT": "BT-2025-005", "Équipement": "EQ-005", "Titre": "Remplacement outil de coupe", "Type": "Préventif", "Priorité": "Basse", "Statut": "Planifié", "Demandeur": "Opérateur", "Technicien": "D. Rousseau", "Date création": "2025-01-14", "Date prévue": "2025-02-01", "Durée estimée (h)": 1, "Coût estimé (€)": 95},
            {"BT": "BT-2025-006", "Équipement": "EQ-003", "Titre": "Vérification étanchéité pompe", "Type": "Préventif", "Priorité": "Haute", "Statut": "Ouvert", "Demandeur": "Contrôle qualité", "Technicien": "Non assigné", "Date création": "2025-01-16", "Date prévue": "2025-01-22", "Durée estimée (h)": 3, "Coût estimé (€)": 420},
            {"BT": "BT-2024-089", "Équipement": "EQ-008", "Titre": "Lubrification pont roulant", "Type": "Préventif", "Priorité": "Moyenne", "Statut": "Terminé", "Demandeur": "Système auto", "Technicien": "B. Lefebvre", "Date création": "2024-12-20", "Date prévue": "2024-12-22", "Durée estimée (h)": 2, "Coût estimé (€)": 240},
        ])

    if "pieces" not in st.session_state:
        st.session_state.pieces = pd.DataFrame([
            {"Référence": "P-001", "Désignation": "Courroie V-Belt A60", "Catégorie": "Transmission", "Stock": 8, "Stock min": 5, "Stock max": 20, "Prix unit. (€)": 42.50, "Fournisseur": "Gates France", "Délai livraison (j)": 3},
            {"Référence": "P-002", "Désignation": "Roulement SKF 6205-2RS", "Catégorie": "Roulements", "Stock": 2, "Stock min": 5, "Stock max": 15, "Prix unit. (€)": 12.80, "Fournisseur": "SKF Maroc", "Délai livraison (j)": 5},
            {"Référence": "P-003", "Désignation": "Joint torique 50x3mm", "Catégorie": "Étanchéité", "Stock": 45, "Stock min": 10, "Stock max": 100, "Prix unit. (€)": 1.20, "Fournisseur": "Trelleborg", "Délai livraison (j)": 7},
            {"Référence": "P-004", "Désignation": "Huile hydraulique HV46", "Catégorie": "Lubrifiants", "Stock": 60, "Stock min": 20, "Stock max": 100, "Prix unit. (€)": 8.50, "Fournisseur": "Total Maroc", "Délai livraison (j)": 2},
            {"Référence": "P-005", "Désignation": "Filtre air compresseur", "Catégorie": "Filtration", "Stock": 3, "Stock min": 4, "Stock max": 12, "Prix unit. (€)": 85.00, "Fournisseur": "Atlas Copco", "Délai livraison (j)": 10},
            {"Référence": "P-006", "Désignation": "Fusible 10A 400V", "Catégorie": "Électrique", "Stock": 30, "Stock min": 20, "Stock max": 60, "Prix unit. (€)": 2.30, "Fournisseur": "Schneider", "Délai livraison (j)": 1},
            {"Référence": "P-007", "Désignation": "Capteur de pression 0-10bar", "Catégorie": "Instrumentation", "Stock": 1, "Stock min": 2, "Stock max": 5, "Prix unit. (€)": 145.00, "Fournisseur": "Endress+Hauser", "Délai livraison (j)": 14},
            {"Référence": "P-008", "Désignation": "Câble électrique 3x2.5mm²", "Catégorie": "Électrique", "Stock": 85, "Stock min": 50, "Stock max": 200, "Prix unit. (€)": 3.80, "Fournisseur": "Nexans", "Délai livraison (j)": 3},
        ])

    if "techniciens" not in st.session_state:
        st.session_state.techniciens = pd.DataFrame([
            {"Nom": "A. Martin", "Spécialité": "Mécanique / Robotique", "Disponible": True, "BT en cours": 2, "Taux efficacité": 92},
            {"Nom": "B. Lefebvre", "Spécialité": "Électrique / Pneumatique", "Disponible": True, "BT en cours": 1, "Taux efficacité": 88},
            {"Nom": "C. Bernard", "Spécialité": "Hydraulique", "Disponible": False, "BT en cours": 0, "Taux efficacité": 95},
            {"Nom": "D. Rousseau", "Spécialité": "Usinage / CNC", "Disponible": True, "BT en cours": 1, "Taux efficacité": 84},
            {"Nom": "E. Petit", "Spécialité": "Chaudronnerie / Thermique", "Disponible": True, "BT en cours": 0, "Taux efficacité": 90},
        ])

init_data()

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand-header">⚙ GMAO PRO</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Gestion de Maintenance</div>', unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1e2d4a; margin:16px 0;'>", unsafe_allow_html=True)

    menu = st.radio(
        "Navigation",
        ["🏠  Tableau de bord", "🔧  Équipements", "📋  Bons de travaux", "📦  Stock & Pièces", "👷  Techniciens", "📅  Planning", "📊  Rapports & KPIs", "⚙️  Paramètres"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#1e2d4a; margin:16px 0;'>", unsafe_allow_html=True)

    # Alertes rapides
    n_pannes = len(st.session_state.equipements[st.session_state.equipements["Statut"] == "En panne"])
    n_stock_bas = len(st.session_state.pieces[st.session_state.pieces["Stock"] < st.session_state.pieces["Stock min"]])
    n_bt_critiques = len(st.session_state.bons_travaux[(st.session_state.bons_travaux["Priorité"] == "Critique") & (st.session_state.bons_travaux["Statut"] != "Terminé")])

    st.markdown(f"""
    <div style='font-family:Rajdhani;font-size:11px;letter-spacing:2px;color:#3a5a7a;text-transform:uppercase;margin-bottom:10px;'>Alertes actives</div>
    <div class='alert-box'>🔴 {n_pannes} équipement(s) en panne</div>
    <div class='alert-box'>🟡 {n_stock_bas} pièce(s) en rupture</div>
    <div class='alert-box'>🟠 {n_bt_critiques} BT critique(s)</div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e2d4a; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:#2a4a6a;'>Dernière sync: {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>", unsafe_allow_html=True)

page = menu.split("  ")[-1].strip()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TABLEAU DE BORD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Tableau de bord":
    st.markdown("<div style='font-family:Rajdhani;font-size:32px;font-weight:700;letter-spacing:3px;color:#a0c8f0;'>TABLEAU DE BORD</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:13px;color:#3a5a7a;margin-bottom:24px;'>{datetime.now().strftime('%A %d %B %Y')}</div>", unsafe_allow_html=True)

    # KPI Row
    eq = st.session_state.equipements
    bt = st.session_state.bons_travaux
    pc = st.session_state.pieces

    taux_dispo = round(len(eq[eq["Statut"] == "Opérationnel"]) / len(eq) * 100, 1)
    bt_ouverts = len(bt[bt["Statut"].isin(["Ouvert", "En cours", "Planifié"])])
    cout_mois = bt[bt["Date création"].str.startswith("2025-01")]["Coût estimé (€)"].sum()
    stocks_critiques = len(pc[pc["Stock"] < pc["Stock min"]])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        color = "success" if taux_dispo >= 80 else "warn" if taux_dispo >= 60 else "danger"
        st.markdown(f"""<div class='metric-card {color}'>
            <div class='metric-label'>Disponibilité équipements</div>
            <div class='metric-value'>{taux_dispo}%</div>
            <div class='metric-sub'>↑ +2.1% vs mois dernier</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card warn'>
            <div class='metric-label'>BT en cours / ouverts</div>
            <div class='metric-value'>{bt_ouverts}</div>
            <div class='metric-sub'>{len(bt[bt["Priorité"]=="Critique"])} critiques</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Coût maintenance (Jan 2025)</div>
            <div class='metric-value'>{cout_mois:,.0f} €</div>
            <div class='metric-sub'>Budget: 15 000 €</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        color2 = "danger" if stocks_critiques > 0 else "success"
        st.markdown(f"""<div class='metric-card {color2}'>
            <div class='metric-label'>Ruptures de stock</div>
            <div class='metric-value'>{stocks_critiques}</div>
            <div class='metric-sub'>Pièces sous seuil min</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("<div class='section-title'>Évolution des interventions (6 mois)</div>", unsafe_allow_html=True)
        mois = ["Août", "Sept", "Oct", "Nov", "Déc", "Jan"]
        correctifs = [8, 12, 7, 5, 9, 6]
        preventifs = [14, 12, 16, 13, 11, 15]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Correctifs", x=mois, y=correctifs, marker_color="#ff5060", opacity=0.85))
        fig.add_trace(go.Bar(name="Préventifs", x=mois, y=preventifs, marker_color="#0088ff", opacity=0.85))
        fig.update_layout(
            barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8ab0d0", family="Rajdhani"), legend=dict(font=dict(color="#8ab0d0")),
            xaxis=dict(gridcolor="#1e2d4a", color="#8ab0d0"),
            yaxis=dict(gridcolor="#1e2d4a", color="#8ab0d0"),
            margin=dict(l=0, r=0, t=10, b=0), height=260
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-title'>Statut des équipements</div>", unsafe_allow_html=True)
        statut_counts = eq["Statut"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=statut_counts.index,
            values=statut_counts.values,
            hole=0.6,
            marker=dict(colors=["#00cc66", "#ff4060", "#aa66ff", "#ff8800"]),
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#8ab0d0", family="Rajdhani"),
            showlegend=True, legend=dict(font=dict(color="#8ab0d0"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=0, b=0), height=260
        )
        st.plotly_chart(fig2, use_container_width=True)

    # BT récents & équipements en panne
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='section-title'>BT prioritaires</div>", unsafe_allow_html=True)
        bt_prioritaires = bt[bt["Statut"] != "Terminé"].sort_values("Priorité", key=lambda x: x.map({"Critique":0,"Haute":1,"Moyenne":2,"Basse":3})).head(4)
        priority_map = {"Critique": "critical", "Haute": "high", "Moyenne": "medium", "Basse": "low"}
        status_map = {"Ouvert": "open", "En cours": "progress", "Terminé": "closed", "Planifié": "planned"}
        for _, row in bt_prioritaires.iterrows():
            p_class = priority_map.get(row["Priorité"], "low")
            s_class = status_map.get(row["Statut"], "open")
            st.markdown(f"""
            <div style='background:#111828;border:1px solid #1e3a5a;border-radius:8px;padding:12px 16px;margin:6px 0;'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div>
                        <span style='font-family:Rajdhani;font-size:13px;color:#4a7aaa;font-weight:600;'>{row['BT']}</span>
                        <span style='font-size:13px;color:#c0d8f0;margin-left:8px;'>{row['Titre'][:35]}...</span>
                    </div>
                    <div>
                        <span class='badge badge-{p_class}'>{row['Priorité']}</span>
                        &nbsp;<span class='badge badge-{s_class}'>{row['Statut']}</span>
                    </div>
                </div>
                <div style='font-size:11px;color:#3a5a7a;margin-top:4px;'>👷 {row['Technicien']} &nbsp;|&nbsp; 📅 {row['Date prévue']} &nbsp;|&nbsp; 💰 {row['Coût estimé (€)']} €</div>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='section-title'>Équipements critiques</div>", unsafe_allow_html=True)
        eq_critiques = eq[eq["Statut"] != "Opérationnel"]
        for _, row in eq_critiques.iterrows():
            color_dot = {"En panne": "#ff4060", "En maintenance": "#aa66ff"}.get(row["Statut"], "#ffaa00")
            st.markdown(f"""
            <div style='background:#111828;border:1px solid #1e3a5a;border-radius:8px;padding:12px 16px;margin:6px 0;'>
                <div style='display:flex;align-items:center;gap:10px;'>
                    <div style='width:10px;height:10px;border-radius:50%;background:{color_dot};box-shadow:0 0 8px {color_dot};flex-shrink:0;'></div>
                    <div>
                        <div style='font-family:Rajdhani;font-size:14px;color:#c0d8f0;font-weight:600;'>{row['Nom']}</div>
                        <div style='font-size:11px;color:#3a5a7a;'>{row['ID']} · {row['Site']} · <span style='color:{color_dot};'>{row['Statut']}</span></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Taux MTBF fictif
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <b style='font-family:Rajdhani;'>MTBF moyen parc:</b> 720 h &nbsp;|&nbsp; <b>MTTR moyen:</b> 3.8 h &nbsp;|&nbsp; <b>Disponibilité:</b> {taux_dispo}%
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ÉQUIPEMENTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Équipements":
    st.markdown("<div style='font-family:Rajdhani;font-size:32px;font-weight:700;letter-spacing:3px;color:#a0c8f0;'>ÉQUIPEMENTS</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📋 Liste des équipements", "➕ Ajouter un équipement"])

    with tab1:
        # Filtres
        col1, col2, col3 = st.columns(3)
        with col1:
            f_site = st.selectbox("Site", ["Tous"] + list(st.session_state.equipements["Site"].unique()))
        with col2:
            f_statut = st.selectbox("Statut", ["Tous"] + list(st.session_state.equipements["Statut"].unique()))
        with col3:
            f_crit = st.selectbox("Criticité", ["Tous"] + list(st.session_state.equipements["Criticité"].unique()))

        df = st.session_state.equipements.copy()
        if f_site != "Tous": df = df[df["Site"] == f_site]
        if f_statut != "Tous": df = df[df["Statut"] == f_statut]
        if f_crit != "Tous": df = df[df["Criticité"] == f_crit]

        # Afficher avec statut coloré
        def style_statut(val):
            colors = {"Opérationnel": "#00cc66", "En panne": "#ff4060", "En maintenance": "#aa66ff"}
            c = colors.get(val, "#aaa")
            return f"color: {c}; font-weight: 600;"

        styled_df = df.style.applymap(style_statut, subset=["Statut"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=350)

        st.markdown(f"<div style='font-size:12px;color:#3a5a7a;'>{len(df)} équipement(s) affiché(s)</div>", unsafe_allow_html=True)

        # Graphe criticité par site
        st.markdown("<br>", unsafe_allow_html=True)
        fig3 = px.bar(
            st.session_state.equipements.groupby(["Site", "Statut"]).size().reset_index(name="count"),
            x="Site", y="count", color="Statut",
            color_discrete_map={"Opérationnel":"#00cc66","En panne":"#ff4060","En maintenance":"#aa66ff"},
            title="Répartition des statuts par site"
        )
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#8ab0d0"), height=300, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.markdown("<div class='section-title'>Nouveau équipement</div>", unsafe_allow_html=True)
        with st.form("form_eq"):
            c1, c2 = st.columns(2)
            with c1:
                nom = st.text_input("Nom de l'équipement *")
                categorie = st.selectbox("Catégorie", ["Pneumatique","Hydraulique","Électrique","Mécanique","Robotique","Usinage","Manutention","Levage","Thermique","Autre"])
                site = st.selectbox("Site", ["Usine A","Usine B","Usine C","Entrepôt"])
                criticite = st.selectbox("Criticité", ["Critique","Haute","Moyenne","Basse"])
            with c2:
                statut = st.selectbox("Statut initial", ["Opérationnel","En maintenance","En panne"])
                date_install = st.date_input("Date d'installation", value=date.today())
                heures = st.number_input("Heures de marche initiales", min_value=0, value=0)
                notes = st.text_area("Notes / Description", height=100)

            submitted = st.form_submit_button("✅ Enregistrer l'équipement")
            if submitted and nom:
                new_id = f"EQ-{len(st.session_state.equipements)+1:03d}"
                new_row = {
                    "ID": new_id, "Nom": nom, "Catégorie": categorie, "Site": site,
                    "Statut": statut, "Criticité": criticite,
                    "Installation": str(date_install),
                    "Dernière maintenance": "N/A",
                    "Prochaine maintenance": "N/A",
                    "Heures de marche": heures
                }
                st.session_state.equipements = pd.concat([st.session_state.equipements, pd.DataFrame([new_row])], ignore_index=True)
                st.markdown(f"<div class='success-box'>✅ Équipement <b>{new_id} - {nom}</b> ajouté avec succès!</div>", unsafe_allow_html=True)
            elif submitted:
                st.markdown("<div class='alert-box'>⚠️ Veuillez saisir au minimum le nom de l'équipement.</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BONS DE TRAVAUX
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Bons de travaux":
    st.markdown("<div style='font-family:Rajdhani;font-size:32px;font-weight:700;letter-spacing:3px;color:#a0c8f0;'>BONS DE TRAVAUX</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📋 Liste des BT", "➕ Créer un BT"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            f_type = st.selectbox("Type", ["Tous","Correctif","Préventif"])
        with c2:
            f_prio = st.selectbox("Priorité", ["Tous","Critique","Haute","Moyenne","Basse"])
        with c3:
            f_stat = st.selectbox("Statut BT", ["Tous","Ouvert","En cours","Planifié","Terminé"])

        df = st.session_state.bons_travaux.copy()
        if f_type != "Tous": df = df[df["Type"] == f_type]
        if f_prio != "Tous": df = df[df["Priorité"] == f_prio]
        if f_stat != "Tous": df = df[df["Statut"] == f_stat]

        # Affichage amélioré
        priority_map = {"Critique": "critical", "Haute": "high", "Moyenne": "medium", "Basse": "low"}
        status_map = {"Ouvert": "open", "En cours": "progress", "Terminé": "closed", "Planifié": "planned"}

        for _, row in df.iterrows():
            p_class = priority_map.get(row["Priorité"], "low")
            s_class = status_map.get(row["Statut"], "open")
            st.markdown(f"""
            <div style='background:#111828;border:1px solid #1e3a5a;border-radius:10px;padding:14px 18px;margin:8px 0;'>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>
                    <div style='display:flex;align-items:center;gap:12px;'>
                        <span style='font-family:Rajdhani;font-size:14px;color:#4a8acd;font-weight:700;'>{row['BT']}</span>
                        <span style='font-family:Rajdhani;font-size:16px;color:#d0e8ff;font-weight:600;'>{row['Titre']}</span>
                        <span class='badge' style='background:#0a1a2a;color:#4488aa;border:1px solid #4488aa30;'>{row['Type']}</span>
                    </div>
                    <div>
                        <span class='badge badge-{p_class}'>{row['Priorité']}</span>
                        &nbsp;<span class='badge badge-{s_class}'>{row['Statut']}</span>
                    </div>
                </div>
                <div style='font-size:12px;color:#3a6a8a;display:flex;gap:20px;'>
                    <span>🔧 {row['Équipement']}</span>
                    <span>👷 {row['Technicien']}</span>
                    <span>📅 Prévu: {row['Date prévue']}</span>
                    <span>⏱ {row['Durée estimée (h)']}h</span>
                    <span>💰 {row['Coût estimé (€)']} €</span>
                    <span>👤 {row['Demandeur']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"<div style='font-size:12px;color:#3a5a7a;margin-top:8px;'>{len(df)} bon(s) de travaux affiché(s)</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='section-title'>Nouveau bon de travaux</div>", unsafe_allow_html=True)
        with st.form("form_bt"):
            c1, c2 = st.columns(2)
            with c1:
                titre = st.text_input("Titre / Description *")
                equipement = st.selectbox("Équipement concerné", st.session_state.equipements["ID"] + " - " + st.session_state.equipements["Nom"])
                type_bt = st.selectbox("Type de maintenance", ["Correctif","Préventif","Prédictif","Amélioratif"])
                priorite = st.selectbox("Priorité", ["Critique","Haute","Moyenne","Basse"])
            with c2:
                technicien = st.selectbox("Technicien assigné", ["Non assigné"] + list(st.session_state.techniciens["Nom"]))
                date_prevue = st.date_input("Date d'intervention prévue", value=date.today() + timedelta(days=3))
                duree = st.number_input("Durée estimée (heures)", min_value=0.5, value=2.0, step=0.5)
                cout = st.number_input("Coût estimé (€)", min_value=0, value=200)
            demandeur = st.text_input("Demandeur", value="Utilisateur")
            description = st.text_area("Description détaillée", height=80)

            submitted = st.form_submit_button("✅ Créer le bon de travaux")
            if submitted and titre:
                new_bt = f"BT-{datetime.now().year}-{len(st.session_state.bons_travaux)+1:03d}"
                eq_id = equipement.split(" - ")[0]
                new_row = {
                    "BT": new_bt, "Équipement": eq_id, "Titre": titre,
                    "Type": type_bt, "Priorité": priorite, "Statut": "Ouvert",
                    "Demandeur": demandeur, "Technicien": technicien,
                    "Date création": str(date.today()), "Date prévue": str(date_prevue),
                    "Durée estimée (h)": duree, "Coût estimé (€)": cout
                }
                st.session_state.bons_travaux = pd.concat([st.session_state.bons_travaux, pd.DataFrame([new_row])], ignore_index=True)
                st.markdown(f"<div class='success-box'>✅ Bon de travaux <b>{new_bt}</b> créé avec succès!</div>", unsafe_allow_html=True)
            elif submitted:
                st.markdown("<div class='alert-box'>⚠️ Le titre est obligatoire.</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: STOCK & PIÈCES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Stock & Pièces":
    st.markdown("<div style='font-family:Rajdhani;font-size:32px;font-weight:700;letter-spacing:3px;color:#a0c8f0;'>STOCK & PIÈCES</div>", unsafe_allow_html=True)

    pc = st.session_state.pieces

    # Alertes de stock
    ruptures = pc[pc["Stock"] < pc["Stock min"]]
    if len(ruptures) > 0:
        for _, row in ruptures.iterrows():
            st.markdown(f"<div class='alert-box'>⚠️ <b>{row['Désignation']}</b> ({row['Référence']}) — Stock: <b>{row['Stock']}</b> unités (min: {row['Stock min']}) · Fournisseur: {row['Fournisseur']} · Délai: {row['Délai livraison (j)']}j</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📦 Inventaire", "➕ Ajouter une pièce"])

    with tab1:
        # Barre de recherche
        search = st.text_input("🔍 Rechercher une pièce", placeholder="Nom, référence, catégorie...")
        df = pc.copy()
        if search:
            df = df[df.apply(lambda r: search.lower() in r.to_string().lower(), axis=1)]

        # Style conditionnel stock
        def style_stock(row):
            styles = [""] * len(row)
            idx = row.index.tolist()
            stock_idx = idx.index("Stock")
            if row["Stock"] < row["Stock min"]:
                styles[stock_idx] = "color: #ff4060; font-weight: 700;"
            elif row["Stock"] < row["Stock min"] * 1.5:
                styles[stock_idx] = "color: #ff8800; font-weight: 600;"
            else:
                styles[stock_idx] = "color: #00cc66;"
            return styles

        st.dataframe(df.style.apply(style_stock, axis=1), use_container_width=True, hide_index=True, height=350)

        # Graphe stock vs seuil
        fig_stock = go.Figure()
        fig_stock.add_trace(go.Bar(name="Stock actuel", x=pc["Désignation"].str[:20], y=pc["Stock"], marker_color="#0088ff"))
        fig_stock.add_trace(go.Scatter(name="Stock minimum", x=pc["Désignation"].str[:20], y=pc["Stock min"], mode="markers+lines", marker=dict(color="#ff4060", size=8), line=dict(color="#ff4060", dash="dash")))
        fig_stock.update_layout(
            title="Niveaux de stock vs seuils minimaux",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8ab0d0"), height=300, margin=dict(l=0,r=0,t=40,b=0),
            xaxis=dict(gridcolor="#1e2d4a"), yaxis=dict(gridcolor="#1e2d4a")
        )
        st.plotly_chart(fig_stock, use_container_width=True)

    with tab2:
        st.markdown("<div class='section-title'>Nouvelle pièce / consommable</div>", unsafe_allow_html=True)
        with st.form("form_piece"):
            c1, c2 = st.columns(2)
            with c1:
                ref = st.text_input("Référence *")
                desig = st.text_input("Désignation *")
                cat = st.selectbox("Catégorie", ["Roulements","Transmission","Étanchéité","Lubrifiants","Filtration","Électrique","Instrumentation","Visserie","Autre"])
                fournisseur = st.text_input("Fournisseur")
            with c2:
                stock_init = st.number_input("Stock initial", min_value=0, value=10)
                stock_min = st.number_input("Stock minimum", min_value=0, value=3)
                stock_max = st.number_input("Stock maximum", min_value=0, value=30)
                prix = st.number_input("Prix unitaire (€)", min_value=0.0, value=10.0, step=0.5)
                delai = st.number_input("Délai livraison (jours)", min_value=1, value=5)

            submitted = st.form_submit_button("✅ Ajouter la pièce")
            if submitted and ref and desig:
                new_row = {
                    "Référence": ref, "Désignation": desig, "Catégorie": cat,
                    "Stock": stock_init, "Stock min": stock_min, "Stock max": stock_max,
                    "Prix unit. (€)": prix, "Fournisseur": fournisseur, "Délai livraison (j)": delai
                }
                st.session_state.pieces = pd.concat([st.session_state.pieces, pd.DataFrame([new_row])], ignore_index=True)
                st.markdown(f"<div class='success-box'>✅ Pièce <b>{ref} - {desig}</b> ajoutée!</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TECHNICIENS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Techniciens":
    st.markdown("<div style='font-family:Rajdhani;font-size:32px;font-weight:700;letter-spacing:3px;color:#a0c8f0;'>TECHNICIENS</div>", unsafe_allow_html=True)

    tech = st.session_state.techniciens

    # Cartes techniciens
    cols = st.columns(len(tech))
    for i, (_, row) in enumerate(tech.iterrows()):
        with cols[i]:
            dispo_color = "#00cc66" if row["Disponible"] else "#ff4060"
            dispo_text = "Disponible" if row["Disponible"] else "Occupé"
            efficacite_color = "#00cc66" if row["Taux efficacité"] >= 90 else "#ff8800" if row["Taux efficacité"] >= 80 else "#ff4060"
            st.markdown(f"""
            <div style='background:#111828;border:1px solid #1e3a5a;border-radius:12px;padding:20px;text-align:center;'>
                <div style='width:60px;height:60px;border-radius:50%;background:linear-gradient(135deg,#0044aa,#0088ff);display:flex;align-items:center;justify-content:center;margin:0 auto 12px;font-size:24px;'>👷</div>
                <div style='font-family:Rajdhani;font-size:16px;font-weight:700;color:#d0e8ff;'>{row['Nom']}</div>
                <div style='font-size:11px;color:#4a7aaa;margin:4px 0;'>{row['Spécialité']}</div>
                <div style='margin:10px 0;'>
                    <span style='display:inline-block;width:8px;height:8px;border-radius:50%;background:{dispo_color};margin-right:5px;'></span>
                    <span style='font-size:12px;color:{dispo_color};'>{dispo_text}</span>
                </div>
                <div style='display:flex;justify-content:space-between;margin-top:12px;padding-top:12px;border-top:1px solid #1e2d4a;'>
                    <div style='text-align:center;'>
                        <div style='font-family:Rajdhani;font-size:20px;font-weight:700;color:#0088ff;'>{row['BT en cours']}</div>
                        <div style='font-size:10px;color:#3a5a7a;'>BT en cours</div>
                    </div>
                    <div style='text-align:center;'>
                        <div style='font-family:Rajdhani;font-size:20px;font-weight:700;color:{efficacite_color};'>{row['Taux efficacité']}%</div>
                        <div style='font-size:10px;color:#3a5a7a;'>Efficacité</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Charge de travail
    st.markdown("<br><div class='section-title'>Charge de travail par technicien</div>", unsafe_allow_html=True)
    tech_bt = st.session_state.bons_travaux[st.session_state.bons_travaux["Statut"].isin(["En cours","Planifié"])].groupby("Technicien").agg({"BT": "count", "Durée estimée (h)": "sum"}).reset_index()
    if len(tech_bt) > 0:
        fig_tech = go.Figure()
        fig_tech.add_trace(go.Bar(name="Nombre de BT", x=tech_bt["Technicien"], y=tech_bt["BT"], marker_color="#0088ff"))
        fig_tech.add_trace(go.Bar(name="Heures estimées", x=tech_bt["Technicien"], y=tech_bt["Durée estimée (h)"], marker_color="#00cc66"))
        fig_tech.update_layout(
            barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8ab0d0"), height=280, margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(gridcolor="#1e2d4a"), yaxis=dict(gridcolor="#1e2d4a"),
            legend=dict(font=dict(color="#8ab0d0"))
        )
        st.plotly_chart(fig_tech, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PLANNING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Planning":
    st.markdown("<div style='font-family:Rajdhani;font-size:32px;font-weight:700;letter-spacing:3px;color:#a0c8f0;'>PLANNING MAINTENANCE</div>", unsafe_allow_html=True)

    bt = st.session_state.bons_travaux.copy()
    bt["Date prévue"] = pd.to_datetime(bt["Date prévue"])
    bt["Date création"] = pd.to_datetime(bt["Date création"])
    bt["Fin prévue"] = bt["Date prévue"] + pd.to_timedelta(bt["Durée estimée (h)"], unit="h")

    bt_planned = bt[bt["Statut"] != "Terminé"].copy()

    if len(bt_planned) > 0:
        color_map = {"Critique": "#ff4060", "Haute": "#ff8800", "Moyenne": "#ffcc00", "Basse": "#00cc66"}

        fig_gantt = go.Figure()
        for i, (_, row) in enumerate(bt_planned.iterrows()):
            color = color_map.get(row["Priorité"], "#4488ff")
            fig_gantt.add_trace(go.Bar(
                name=row["BT"],
                y=[f"{row['BT']} | {row['Titre'][:30]}"],
                x=[(row["Fin prévue"] - row["Date prévue"]).total_seconds() / 86400],
                base=[(row["Date prévue"] - pd.Timestamp("2025-01-01")).total_seconds() / 86400],
                orientation="h",
                marker=dict(color=color, opacity=0.8),
                hovertemplate=f"<b>{row['BT']}</b><br>{row['Titre']}<br>Technicien: {row['Technicien']}<br>Priorité: {row['Priorité']}<extra></extra>",
                showlegend=False
            ))

        start_offset = 0
        end_offset = 45

        fig_gantt.update_layout(
            barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8ab0d0", family="Rajdhani"),
            height=400,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(
                gridcolor="#1e2d4a",
                tickvals=list(range(0, 46, 7)),
                ticktext=[(pd.Timestamp("2025-01-01") + timedelta(days=d)).strftime("%d %b") for d in range(0, 46, 7)],
                color="#8ab0d0"
            ),
            yaxis=dict(color="#8ab0d0", gridcolor="#1e2d4a"),
            title="Diagramme de Gantt - Interventions planifiées"
        )
        st.plotly_chart(fig_gantt, use_container_width=True)
    else:
        st.markdown("<div class='info-box'>Aucun BT planifié en cours.</div>", unsafe_allow_html=True)

    # Calendrier de maintenance préventive
    st.markdown("<br><div class='section-title'>Prochaines maintenances préventives</div>", unsafe_allow_html=True)
    eq = st.session_state.equipements.copy()
    eq_sorted = eq.sort_values("Prochaine maintenance")
    for _, row in eq_sorted.head(6).iterrows():
        try:
            next_date = pd.to_datetime(row["Prochaine maintenance"])
            delta = (next_date - pd.Timestamp.now()).days
            color = "#ff4060" if delta < 7 else "#ff8800" if delta < 30 else "#00cc66"
            urgent = "⚠️" if delta < 7 else "📅"
            st.markdown(f"""
            <div style='background:#111828;border:1px solid #1e3a5a;border-radius:8px;padding:10px 16px;margin:6px 0;display:flex;justify-content:space-between;align-items:center;'>
                <div>
                    <span style='font-family:Rajdhani;font-size:14px;color:#c0d8f0;font-weight:600;'>{urgent} {row['Nom']}</span>
                    <span style='font-size:11px;color:#3a5a7a;margin-left:12px;'>{row['ID']} · {row['Site']}</span>
                </div>
                <div style='font-family:Rajdhani;font-size:14px;color:{color};font-weight:700;'>
                    {row['Prochaine maintenance']} <span style='font-size:12px;font-weight:400;'>({delta}j)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except:
            pass

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RAPPORTS & KPIs
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Rapports & KPIs":
    st.markdown("<div style='font-family:Rajdhani;font-size:32px;font-weight:700;letter-spacing:3px;color:#a0c8f0;'>RAPPORTS & KPIs</div>", unsafe_allow_html=True)

    bt = st.session_state.bons_travaux
    eq = st.session_state.equipements

    # KPIs calculés
    total_bt = len(bt)
    bt_termines = len(bt[bt["Statut"] == "Terminé"])
    taux_completion = round(bt_termines / total_bt * 100, 1) if total_bt > 0 else 0
    cout_total = bt["Coût estimé (€)"].sum()
    duree_totale = bt["Durée estimée (h)"].sum()
    taux_prev = round(len(bt[bt["Type"] == "Préventif"]) / total_bt * 100, 1) if total_bt > 0 else 0
    taux_dispo = round(len(eq[eq["Statut"] == "Opérationnel"]) / len(eq) * 100, 1)

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("Taux de réalisation", f"{taux_completion}%", "BT terminés / total", "success" if taux_completion>=80 else "warn"),
        ("Taux maintenance préventive", f"{taux_prev}%", "Objectif: >70%", "success" if taux_prev>=70 else "warn"),
        ("Coût total estimé", f"{cout_total:,.0f}€", "Toutes interventions", ""),
        ("Disponibilité parc", f"{taux_dispo}%", "Équipements opérationnels", "success" if taux_dispo>=80 else "danger"),
    ]
    for col, (label, val, sub, cls) in zip([c1,c2,c3,c4], kpis):
        with col:
            st.markdown(f"""<div class='metric-card {cls}'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{val}</div>
                <div class='metric-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Répartition par type d'intervention</div>", unsafe_allow_html=True)
        type_counts = bt["Type"].value_counts()
        fig_type = go.Figure(go.Pie(
            labels=type_counts.index, values=type_counts.values, hole=0.5,
            marker=dict(colors=["#0088ff","#00cc66","#ff8800","#aa66ff"])
        ))
        fig_type.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#8ab0d0"), height=280, margin=dict(l=0,r=0,t=10,b=0), legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_type, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Coût par équipement</div>", unsafe_allow_html=True)
        cout_eq = bt.groupby("Équipement")["Coût estimé (€)"].sum().reset_index().sort_values("Coût estimé (€)", ascending=True)
        fig_cout = go.Figure(go.Bar(
            x=cout_eq["Coût estimé (€)"], y=cout_eq["Équipement"],
            orientation="h", marker_color="#0088ff",
            text=cout_eq["Coût estimé (€)"].apply(lambda x: f"{x:,.0f}€"),
            textposition="outside", textfont=dict(color="#8ab0d0")
        ))
        fig_cout.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#8ab0d0"), height=280, margin=dict(l=0,r=30,t=10,b=0), xaxis=dict(gridcolor="#1e2d4a"), yaxis=dict(color="#8ab0d0"))
        st.plotly_chart(fig_cout, use_container_width=True)

    # Tableau de synthèse
    st.markdown("<div class='section-title'>Synthèse mensuelle</div>", unsafe_allow_html=True)
    synth = pd.DataFrame({
        "Mois": ["Août 2024","Sept 2024","Oct 2024","Nov 2024","Déc 2024","Jan 2025"],
        "BT Ouverts": [22, 27, 23, 18, 20, 7],
        "BT Terminés": [20, 24, 22, 17, 19, 3],
        "Coût Total (€)": [8400, 12500, 7800, 6200, 9100, 7145],
        "Taux Réalisation (%)": [91, 89, 96, 94, 95, 43],
        "Disponibilité (%)": [91, 88, 93, 95, 92, 87.5],
    })
    st.dataframe(synth, use_container_width=True, hide_index=True)

    # Export
    st.markdown("<br>", unsafe_allow_html=True)
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        csv_bt = bt.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Exporter BT (CSV)", csv_bt, "bons_travaux.csv", "text/csv")
    with col_e2:
        csv_eq = eq.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Exporter Équipements (CSV)", csv_eq, "equipements.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PARAMÈTRES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Paramètres":
    st.markdown("<div style='font-family:Rajdhani;font-size:32px;font-weight:700;letter-spacing:3px;color:#a0c8f0;'>PARAMÈTRES</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Configuration générale</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Nom de l'entreprise", value="Industrie Maroc SA")
        st.text_input("Email notifications", value="maintenance@industrie.ma")
        st.selectbox("Devise", ["MAD (Dirham)", "EUR (Euro)", "USD (Dollar)"])
    with c2:
        st.selectbox("Langue", ["Français", "Arabe", "Anglais"])
        st.number_input("Seuil alerte stock (%)", value=20, min_value=5, max_value=50)
        st.number_input("Délai rappel maintenance (jours)", value=7, min_value=1, max_value=30)

    st.markdown("<div class='section-title'>Gestion des utilisateurs</div>", unsafe_allow_html=True)
    users = pd.DataFrame([
        {"Utilisateur": "admin@industrie.ma", "Rôle": "Administrateur", "Accès": "Complet"},
        {"Utilisateur": "chef.maintenance@industrie.ma", "Rôle": "Responsable maintenance", "Accès": "BT + Rapports"},
        {"Utilisateur": "technicien@industrie.ma", "Rôle": "Technicien", "Accès": "BT uniquement"},
    ])
    st.dataframe(users, use_container_width=True, hide_index=True)

    if st.button("🔄 Réinitialiser les données de démonstration"):
        for key in ["equipements","bons_travaux","pieces","techniciens"]:
            if key in st.session_state:
                del st.session_state[key]
        init_data()
        st.markdown("<div class='success-box'>✅ Données réinitialisées!</div>", unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div class='info-box'>
        <b style='font-family:Rajdhani;'>GMAO Pro v1.0</b> — Application de Gestion de Maintenance Assistée par Ordinateur<br>
        <span style='font-size:12px;color:#4a7aaa;'>Développé avec Streamlit + Plotly · Compatible Python 3.9+</span>
    </div>
    """, unsafe_allow_html=True)
