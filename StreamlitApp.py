import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import io, base64, json

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, HRFlowable, KeepTogether)
from pypdf import PdfReader, PdfWriter

# ─── CONFIG ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="GMAO Pro+", page_icon="⚙️", layout="wide",
                   initial_sidebar_state="expanded")

# ─── MASTER CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }

/* ── App Background ── */
.stApp {
    background: #060a12;
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(14,52,95,0.15) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(0,168,255,0.05) 0%, transparent 50%),
        linear-gradient(180deg, #060a12 0%, #08101e 100%);
    color: #cdd8e8;
    min-height: 100vh;
}

/* ── Animated grid overlay ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(0,120,200,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,120,200,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #08101e 0%, #060e1a 100%) !important;
    border-right: 1px solid #0d2540;
    box-shadow: 4px 0 24px rgba(0,0,0,0.5);
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0055cc, #00aaff, #0055cc);
    background-size: 200% 100%;
    animation: shimmer 3s infinite linear;
}
@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* ── Main content area ── */
[data-testid="stMainBlockContainer"] { position: relative; z-index: 1; }

/* ── Typography ── */
h1, h2, h3, h4 { font-family: 'Exo 2', sans-serif !important; font-weight: 800 !important; letter-spacing: 1px; }

/* ── Page Title ── */
.page-title {
    font-family: 'Exo 2', sans-serif;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 4px;
    text-transform: uppercase;
    background: linear-gradient(135deg, #60b0ff 0%, #00d4ff 50%, #60b0ff 100%);
    background-size: 200% 100%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientMove 4s ease infinite;
    margin-bottom: 2px;
}
@keyframes gradientMove {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
.page-subtitle {
    font-size: 11px;
    letter-spacing: 3px;
    color: #1a3a5a;
    text-transform: uppercase;
    margin-bottom: 24px;
}

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #0a1525 0%, #08111e 100%);
    border: 1px solid #0d2540;
    border-radius: 16px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,120,255,0.15);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #0055cc, #00aaff);
}
.kpi-card::after {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 100px; height: 100px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,120,255,0.05) 0%, transparent 70%);
}
.kpi-card.warn::before { background: linear-gradient(90deg, #cc5500, #ff8800); }
.kpi-card.danger::before { background: linear-gradient(90deg, #cc0033, #ff2244); }
.kpi-card.success::before { background: linear-gradient(90deg, #006633, #00cc66); }
.kpi-card.purple::before { background: linear-gradient(90deg, #5500aa, #aa44ff); }
.kpi-card.teal::before { background: linear-gradient(90deg, #006677, #00cccc); }

.kpi-icon {
    font-size: 28px;
    margin-bottom: 8px;
    display: block;
    filter: drop-shadow(0 0 8px rgba(0,150,255,0.4));
}
.kpi-label {
    font-family: 'Exo 2', sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #2a4a70;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'Exo 2', sans-serif;
    font-size: 34px;
    font-weight: 800;
    color: #e0eeff;
    line-height: 1;
    letter-spacing: -1px;
}
.kpi-sub {
    font-size: 11px;
    color: #1a3555;
    margin-top: 6px;
    font-weight: 500;
}
.kpi-trend-up { color: #00cc66; }
.kpi-trend-down { color: #ff4455; }

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #0d2540;
}
.section-header-line {
    height: 2px;
    width: 30px;
    background: linear-gradient(90deg, #0055cc, #00aaff);
    border-radius: 2px;
    flex-shrink: 0;
}
.section-title {
    font-family: 'Exo 2', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #5090c0;
}

/* ── Status Badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
    font-family: 'Exo 2', sans-serif;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.badge-critical { background: rgba(200,0,40,0.15); color: #ff3355; border: 1px solid rgba(255,50,80,0.3); }
.badge-high     { background: rgba(200,100,0,0.15); color: #ff8833; border: 1px solid rgba(255,140,40,0.3); }
.badge-medium   { background: rgba(180,150,0,0.15); color: #ffcc00; border: 1px solid rgba(255,200,0,0.3); }
.badge-low      { background: rgba(0,150,80,0.15);  color: #00cc66; border: 1px solid rgba(0,200,100,0.3); }
.badge-open     { background: rgba(0,100,200,0.15); color: #4499ff; border: 1px solid rgba(0,150,255,0.3); }
.badge-progress { background: rgba(100,0,200,0.15); color: #aa66ff; border: 1px solid rgba(150,60,255,0.3); }
.badge-closed   { background: rgba(0,120,60,0.15);  color: #33cc77; border: 1px solid rgba(0,180,80,0.3); }
.badge-planned  { background: rgba(0,80,180,0.15);  color: #3388ff; border: 1px solid rgba(0,120,255,0.3); }
.badge-pending  { background: rgba(160,120,0,0.15); color: #ffaa00; border: 1px solid rgba(220,160,0,0.3); }
.badge-approved { background: rgba(0,140,100,0.15); color: #00ddaa; border: 1px solid rgba(0,200,150,0.3); }
.badge-rejected { background: rgba(180,0,0,0.15);   color: #ff4444; border: 1px solid rgba(220,0,0,0.3); }

/* ── Cards for BT / DI ── */
.item-card {
    background: linear-gradient(135deg, #0a1525 0%, #08111e 100%);
    border: 1px solid #0d2540;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 6px 0;
    transition: all 0.2s;
    cursor: default;
    position: relative;
    overflow: hidden;
}
.item-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, #0055cc, #00aaff);
}
.item-card.critical::before { background: linear-gradient(180deg, #cc0022, #ff3355); }
.item-card.high::before     { background: linear-gradient(180deg, #cc5500, #ff8833); }
.item-card.medium::before   { background: linear-gradient(180deg, #aa8800, #ffcc00); }
.item-card.low::before      { background: linear-gradient(180deg, #006633, #00cc66); }
.item-card:hover {
    border-color: #1a4070;
    box-shadow: 0 4px 20px rgba(0,80,200,0.1);
    transform: translateX(2px);
}

.item-title { font-family:'Exo 2',sans-serif; font-size:14px; font-weight:700; color:#c8e0ff; }
.item-id    { font-family:'JetBrains Mono',monospace; font-size:12px; color:#2a5a8a; }
.item-meta  { font-size:11px; color:#1a3555; margin-top:5px; }
.item-meta span { margin-right:16px; }

/* ── Alert Boxes ── */
.alert-danger  { background:rgba(180,0,30,0.1);  border-left:3px solid #ff2244; border-radius:0 8px 8px 0; padding:10px 14px; margin:6px 0; font-size:12px; color:#ff8899; }
.alert-warning { background:rgba(180,100,0,0.1); border-left:3px solid #ff8800; border-radius:0 8px 8px 0; padding:10px 14px; margin:6px 0; font-size:12px; color:#ffcc77; }
.alert-info    { background:rgba(0,80,180,0.1);  border-left:3px solid #0088ff; border-radius:0 8px 8px 0; padding:10px 14px; margin:6px 0; font-size:12px; color:#77bbff; }
.alert-success { background:rgba(0,120,60,0.1);  border-left:3px solid #00cc66; border-radius:0 8px 8px 0; padding:10px 14px; margin:6px 0; font-size:12px; color:#66ddaa; }

/* ── Brand ── */
.brand-logo {
    font-family: 'Exo 2', sans-serif;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: 4px;
    text-transform: uppercase;
    background: linear-gradient(135deg, #4499ff, #00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.brand-tagline {
    font-size: 9px;
    letter-spacing: 4px;
    color: #0d2540;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── Nav items ── */
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 1px !important;
    color: #3a6090 !important;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #60a0e0 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    font-size: 12px !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #08111e !important;
    border-bottom: 1px solid #0d2540 !important;
    border-radius: 8px 8px 0 0 !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: rgba(0,120,255,0.12) !important;
    border-bottom: 2px solid #0088ff !important;
    color: #60aaff !important;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stTextInput > div > div,
.stNumberInput > div > div,
.stTextArea > div > div {
    background: #0a1525 !important;
    border: 1px solid #0d2540 !important;
    border-radius: 8px !important;
    color: #c0d8f0 !important;
    font-family: 'Exo 2', sans-serif !important;
}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div:focus-within,
.stTextArea > div > div:focus-within {
    border-color: #0055cc !important;
    box-shadow: 0 0 0 2px rgba(0,85,204,0.2) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0044bb, #0077ee) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-size: 12px !important;
    padding: 10px 20px !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0055dd, #0099ff) !important;
    box-shadow: 0 4px 20px rgba(0,100,255,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── Progress bars ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #0044bb, #00aaff) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #0d2540 !important;
    border-radius: 10px !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] th {
    background: #0a1a30 !important;
    color: #4a8abf !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #0a1525;
    border: 1px solid #0d2540;
    border-radius: 10px;
    padding: 12px;
}

/* ── Checkboxes ── */
.stCheckbox label { font-family: 'Exo 2', sans-serif !important; color: #5090c0 !important; font-size: 13px !important; }

/* ── Date input ── */
.stDateInput > div > div { background: #0a1525 !important; border: 1px solid #0d2540 !important; border-radius: 8px !important; }

/* ── Multiselect ── */
.stMultiSelect > div > div { background: #0a1525 !important; border: 1px solid #0d2540 !important; border-radius: 8px !important; }

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #003399, #0055cc) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
}

/* ── PDF button ── */
.pdf-dl-btn {
    display: block;
    background: linear-gradient(135deg, #8b0000, #cc1122);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 11px 20px;
    font-family: 'Exo 2', sans-serif;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-size: 12px;
    text-align: center;
    text-decoration: none;
    transition: all 0.2s;
    cursor: pointer;
    width: 100%;
    margin-top: 8px;
}
.pdf-dl-btn:hover { background: linear-gradient(135deg, #aa0011, #ee1122); box-shadow: 0 4px 20px rgba(200,0,30,0.3); }

/* ── DI specific ── */
.di-card {
    background: linear-gradient(135deg, #0a1525 0%, #08111e 100%);
    border: 1px solid #0d2540;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    position: relative;
    overflow: hidden;
    transition: all 0.2s;
}
.di-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    border-radius: 4px 0 0 4px;
}
.di-card.pending::before  { background: linear-gradient(180deg, #aa8800, #ffcc00); }
.di-card.approved::before { background: linear-gradient(180deg, #006633, #00cc66); }
.di-card.rejected::before { background: linear-gradient(180deg, #880011, #ff2244); }
.di-card.converted::before{ background: linear-gradient(180deg, #002288, #4499ff); }
.di-card:hover { border-color: #1a4070; box-shadow: 0 4px 20px rgba(0,80,200,0.1); transform: translateX(2px); }

/* ── Stat pill ── */
.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,80,180,0.1);
    border: 1px solid rgba(0,120,255,0.15);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    color: #5090c0;
    font-weight: 600;
    font-family: 'Exo 2', sans-serif;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #060a12; }
::-webkit-scrollbar-thumb { background: #0d2540; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #1a4070; }

/* ── Sidebar divider ── */
.sidebar-divider {
    border: none;
    border-top: 1px solid #0d2540;
    margin: 12px 0;
}

/* ── Urgency pulse ── */
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,50,80,0.4); }
    50% { box-shadow: 0 0 0 6px rgba(255,50,80,0); }
}
.pulse { animation: pulse-red 2s infinite; }

/* ── Form section ── */
.form-section {
    background: linear-gradient(135deg, #0a1525 0%, #08111e 100%);
    border: 1px solid #0d2540;
    border-radius: 14px;
    padding: 20px;
    margin: 10px 0;
}

/* ── Timeline ── */
.timeline-item {
    display: flex;
    gap: 12px;
    margin: 8px 0;
    padding: 10px 14px;
    background: #08111e;
    border: 1px solid #0d2540;
    border-radius: 8px;
    font-size: 12px;
}
.timeline-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #0066cc;
    flex-shrink: 0;
    margin-top: 4px;
    box-shadow: 0 0 6px rgba(0,100,200,0.5);
}

/* ── Horizontal rule ── */
hr { border-color: #0d2540 !important; margin: 16px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── PDF COLORS ──────────────────────────────────────────────────────────────
PDF_NAVY   = colors.HexColor("#1B2A4A")
PDF_BLUE   = colors.HexColor("#1565C0")
PDF_LBLUE  = colors.HexColor("#E3F2FD")
PDF_LBLUE2 = colors.HexColor("#F0F8FF")
PDF_ACCENT = colors.HexColor("#C62828")
PDF_GREEN  = colors.HexColor("#2E7D32")
PDF_ORANGE = colors.HexColor("#E65100")
PDF_GRAY   = colors.HexColor("#37474F")
PDF_LGRAY  = colors.HexColor("#F5F7FA")
PDF_WHITE  = colors.white
PDF_BLACK  = colors.HexColor("#1A1A1A")
PDF_MGRAY  = colors.HexColor("#90A4AE")
PDF_TEAL   = colors.HexColor("#00695C")
PDF_PURPLE = colors.HexColor("#4A148C")

# ─── DATA INIT ───────────────────────────────────────────────────────────────
def init_data():
    if "equipements" not in st.session_state:
        st.session_state.equipements = pd.DataFrame([
            {"ID":"EQ-001","Nom":"Compresseur Atlas Copco GA55","Catégorie":"Pneumatique","Site":"Usine A","Statut":"Opérationnel","Criticité":"Critique","Installation":"2019-03-15","Dernière MNT":"2024-11-01","Prochaine MNT":"2025-03-01","Heures":12450,"Valeur (€)":45000,"Resp.":"A. Martin","Localisation":"Hall 1 - Zone B"},
            {"ID":"EQ-002","Nom":"Convoyeur à bande CB-12","Catégorie":"Manutention","Site":"Usine A","Statut":"En panne","Criticité":"Haute","Installation":"2020-07-10","Dernière MNT":"2024-09-15","Prochaine MNT":"2025-01-15","Heures":8320,"Valeur (€)":28000,"Resp.":"B. Lefebvre","Localisation":"Hall 2 - Ligne 3"},
            {"ID":"EQ-003","Nom":"Pompe centrifuge PMP-3","Catégorie":"Hydraulique","Site":"Usine B","Statut":"Opérationnel","Criticité":"Haute","Installation":"2018-11-22","Dernière MNT":"2024-10-20","Prochaine MNT":"2025-04-20","Heures":18900,"Valeur (€)":12000,"Resp.":"C. Bernard","Localisation":"Station pompage"},
            {"ID":"EQ-004","Nom":"Robot soudure RS-7","Catégorie":"Robotique","Site":"Usine A","Statut":"En maintenance","Criticité":"Critique","Installation":"2021-02-01","Dernière MNT":"2025-01-10","Prochaine MNT":"2025-07-10","Heures":5600,"Valeur (€)":95000,"Resp.":"A. Martin","Localisation":"Hall 3 - Cellule R"},
            {"ID":"EQ-005","Nom":"Tour CNC Mazak QT250","Catégorie":"Usinage","Site":"Usine C","Statut":"Opérationnel","Criticité":"Haute","Installation":"2017-06-18","Dernière MNT":"2024-12-01","Prochaine MNT":"2025-06-01","Heures":24300,"Valeur (€)":65000,"Resp.":"D. Rousseau","Localisation":"Atelier usinage"},
            {"ID":"EQ-006","Nom":"Groupe électrogène GE-100","Catégorie":"Électrique","Site":"Usine B","Statut":"Opérationnel","Criticité":"Critique","Installation":"2022-01-05","Dernière MNT":"2024-08-10","Prochaine MNT":"2025-02-10","Heures":3200,"Valeur (€)":38000,"Resp.":"B. Lefebvre","Localisation":"Local GE - Ext."},
            {"ID":"EQ-007","Nom":"Chaudière vapeur CV-50","Catégorie":"Thermique","Site":"Usine C","Statut":"Opérationnel","Criticité":"Critique","Installation":"2016-09-30","Dernière MNT":"2024-07-15","Prochaine MNT":"2025-01-15","Heures":31500,"Valeur (€)":72000,"Resp.":"E. Petit","Localisation":"Chaufferie"},
            {"ID":"EQ-008","Nom":"Pont roulant PR-10T","Catégorie":"Levage","Site":"Usine A","Statut":"Opérationnel","Criticité":"Haute","Installation":"2015-04-12","Dernière MNT":"2024-11-30","Prochaine MNT":"2025-05-30","Heures":41200,"Valeur (€)":55000,"Resp.":"C. Bernard","Localisation":"Hall 1 - Toiture"},
        ])

    if "demandes_intervention" not in st.session_state:
        st.session_state.demandes_intervention = pd.DataFrame([
            {"DI":"DI-2025-001","Titre":"Bruit anormal moteur convoyeur","Équipement":"EQ-002","Site":"Usine A","Urgence":"Haute","Statut":"Approuvée","Demandeur":"M. Hassan","Service":"Production","Date demande":"2025-01-14","Date souhaitée":"2025-01-16","Description":"Bruit métallique inhabituel depuis 2 jours, vibrations anormales détectées","BT généré":"BT-2025-001","Commentaire resp.":"Intervention programmée J+2","Type demandé":"Correctif"},
            {"DI":"DI-2025-002","Titre":"Fuite huile pompe hydraulique","Équipement":"EQ-003","Site":"Usine B","Urgence":"Critique","Statut":"En attente","Demandeur":"M. Alami","Service":"Maintenance","Date demande":"2025-01-16","Date souhaitée":"2025-01-17","Description":"Fuite constatée au niveau du joint principal, risque de contamination sol","BT généré":"","Commentaire resp.":"","Type demandé":"Correctif"},
            {"DI":"DI-2025-003","Titre":"Vibrations excessives ventilateur","Équipement":"EQ-001","Site":"Usine A","Urgence":"Moyenne","Statut":"Approuvée","Demandeur":"Opérateur ligne 2","Service":"Production","Date demande":"2025-01-12","Date souhaitée":"2025-01-20","Description":"Vibrations lors du démarrage, niveau sonore élevé","BT généré":"BT-2025-006","Commentaire resp.":"Prévoir équilibrage","Type demandé":"Correctif"},
            {"DI":"DI-2025-004","Titre":"Maintenance préventive planifiée chaudière","Équipement":"EQ-007","Site":"Usine C","Urgence":"Haute","Statut":"En attente","Demandeur":"Système auto","Service":"Maintenance","Date demande":"2025-01-10","Date souhaitée":"2025-01-25","Description":"Inspection annuelle réglementaire obligatoire avant fin janvier","BT généré":"","Commentaire resp.":"","Type demandé":"Préventif"},
            {"DI":"DI-2024-089","Titre":"Remplacement filtre air compresseur","Équipement":"EQ-001","Site":"Usine A","Urgence":"Basse","Statut":"Clôturée","Demandeur":"Système auto","Service":"Maintenance","Date demande":"2024-12-15","Date souhaitée":"2024-12-20","Description":"Filtre saturé, perte de pression constatée","BT généré":"BT-2025-003","Commentaire resp.":"Terminé le 08/01","Type demandé":"Préventif"},
            {"DI":"DI-2025-005","Titre":"Défaut affichage panneau électrique","Équipement":"EQ-006","Site":"Usine B","Urgence":"Moyenne","Statut":"Rejetée","Demandeur":"Électricien","Service":"Électricité","Date demande":"2025-01-15","Date souhaitée":"2025-01-18","Description":"Écran LCD du tableau ne s'allume plus","BT généré":"","Commentaire resp.":"Hors périmètre maintenance - contacter le SAV fabricant","Type demandé":"Correctif"},
        ])

    if "bons_travaux" not in st.session_state:
        st.session_state.bons_travaux = pd.DataFrame([
            {"BT":"BT-2025-001","Équipement":"EQ-002","DI origine":"DI-2025-001","Titre":"Remplacement courroie convoyeur","Type":"Correctif","Priorité":"Haute","Statut":"En cours","Demandeur":"M. Dupont","Technicien":"A. Martin","Date création":"2025-01-15","Date prévue":"2025-01-20","Durée (h)":4,"Coût estimé (€)":850,"Coût réel (€)":0,"Description":"Courroie principale usée, remplacement urgent"},
            {"BT":"BT-2025-002","Équipement":"EQ-007","DI origine":"","Titre":"Inspection annuelle chaudière","Type":"Préventif","Priorité":"Critique","Statut":"Planifié","Demandeur":"Système auto","Technicien":"B. Lefebvre","Date création":"2025-01-10","Date prévue":"2025-01-25","Durée (h)":8,"Coût estimé (€)":2400,"Coût réel (€)":0,"Description":"Inspection réglementaire annuelle obligatoire"},
            {"BT":"BT-2025-003","Équipement":"EQ-001","DI origine":"DI-2024-089","Titre":"Vidange huile compresseur","Type":"Préventif","Priorité":"Moyenne","Statut":"Terminé","Demandeur":"Système auto","Technicien":"C. Bernard","Date création":"2025-01-05","Date prévue":"2025-01-08","Durée (h)":2,"Coût estimé (€)":180,"Coût réel (€)":165,"Description":"Vidange périodique et remplacement filtre"},
            {"BT":"BT-2025-004","Équipement":"EQ-004","DI origine":"","Titre":"Calibration robot soudure","Type":"Correctif","Priorité":"Critique","Statut":"En cours","Demandeur":"Production","Technicien":"A. Martin","Date création":"2025-01-12","Date prévue":"2025-01-18","Durée (h)":12,"Coût estimé (€)":3200,"Coût réel (€)":0,"Description":"Dérive de précision détectée, recalibrage nécessaire"},
            {"BT":"BT-2025-005","Équipement":"EQ-005","DI origine":"","Titre":"Remplacement outil de coupe","Type":"Préventif","Priorité":"Basse","Statut":"Planifié","Demandeur":"Opérateur","Technicien":"D. Rousseau","Date création":"2025-01-14","Date prévue":"2025-02-01","Durée (h)":1,"Coût estimé (€)":95,"Coût réel (€)":0,"Description":"Usure normale de l'outil selon programme"},
            {"BT":"BT-2025-006","Équipement":"EQ-003","DI origine":"DI-2025-003","Titre":"Vérification étanchéité pompe","Type":"Préventif","Priorité":"Haute","Statut":"Ouvert","Demandeur":"Contrôle qualité","Technicien":"Non assigné","Date création":"2025-01-16","Date prévue":"2025-01-22","Durée (h)":3,"Coût estimé (€)":420,"Coût réel (€)":0,"Description":"Légère fuite constatée, diagnostic nécessaire"},
        ])

    if "pieces" not in st.session_state:
        st.session_state.pieces = pd.DataFrame([
            {"Réf":"P-001","Désignation":"Courroie V-Belt A60","Catégorie":"Transmission","Stock":8,"Min":5,"Max":20,"Prix (€)":42.50,"Fournisseur":"Gates France","Délai (j)":3,"Emplacement":"Étagère A-12"},
            {"Réf":"P-002","Désignation":"Roulement SKF 6205-2RS","Catégorie":"Roulements","Stock":2,"Min":5,"Max":15,"Prix (€)":12.80,"Fournisseur":"SKF Maroc","Délai (j)":5,"Emplacement":"Étagère B-03"},
            {"Réf":"P-003","Désignation":"Joint torique 50x3mm","Catégorie":"Étanchéité","Stock":45,"Min":10,"Max":100,"Prix (€)":1.20,"Fournisseur":"Trelleborg","Délai (j)":7,"Emplacement":"Tiroir C-05"},
            {"Réf":"P-004","Désignation":"Huile hydraulique HV46 (L)","Catégorie":"Lubrifiants","Stock":60,"Min":20,"Max":100,"Prix (€)":8.50,"Fournisseur":"Total Maroc","Délai (j)":2,"Emplacement":"Zone liquides"},
            {"Réf":"P-005","Désignation":"Filtre air compresseur","Catégorie":"Filtration","Stock":3,"Min":4,"Max":12,"Prix (€)":85.00,"Fournisseur":"Atlas Copco","Délai (j)":10,"Emplacement":"Étagère A-08"},
            {"Réf":"P-006","Désignation":"Fusible 10A 400V","Catégorie":"Électrique","Stock":30,"Min":20,"Max":60,"Prix (€)":2.30,"Fournisseur":"Schneider","Délai (j)":1,"Emplacement":"Armoire élec."},
            {"Réf":"P-007","Désignation":"Capteur pression 0-10bar","Catégorie":"Instrumentation","Stock":1,"Min":2,"Max":5,"Prix (€)":145.00,"Fournisseur":"Endress+Hauser","Délai (j)":14,"Emplacement":"Étagère D-01"},
            {"Réf":"P-008","Désignation":"Câble élec. 3x2.5mm² (m)","Catégorie":"Électrique","Stock":85,"Min":50,"Max":200,"Prix (€)":3.80,"Fournisseur":"Nexans","Délai (j)":3,"Emplacement":"Dévidoir E-02"},
        ])

    if "techniciens" not in st.session_state:
        st.session_state.techniciens = pd.DataFrame([
            {"Nom":"A. Martin","Spécialité":"Mécanique / Robotique","Disponible":True,"BT en cours":2,"Efficacité (%)":92,"H/mois":160,"€/h":45,"Certifications":"CACES 3, Habilitation élec."},
            {"Nom":"B. Lefebvre","Spécialité":"Électrique / Pneumatique","Disponible":True,"BT en cours":1,"Efficacité (%)":88,"H/mois":160,"€/h":42,"Certifications":"Habilitation élec. B2V"},
            {"Nom":"C. Bernard","Spécialité":"Hydraulique","Disponible":False,"BT en cours":0,"Efficacité (%)":95,"H/mois":160,"€/h":48,"Certifications":"Hydraulique certifié"},
            {"Nom":"D. Rousseau","Spécialité":"Usinage / CNC","Disponible":True,"BT en cours":1,"Efficacité (%)":84,"H/mois":160,"€/h":40,"Certifications":"Programmeur CNC"},
            {"Nom":"E. Petit","Spécialité":"Chaudronnerie / Thermique","Disponible":True,"BT en cours":0,"Efficacité (%)":90,"H/mois":160,"€/h":44,"Certifications":"OPQIBI, Soudure TIG"},
        ])

    if "historique" not in st.session_state:
        months = ["Juil","Août","Sept","Oct","Nov","Déc","Jan"]
        st.session_state.historique = pd.DataFrame({
            "Mois": months,
            "Correctifs": [8,8,12,7,5,9,6],
            "Préventifs": [12,14,12,16,13,11,15],
            "DI reçues": [10,12,15,9,8,11,8],
            "DI approuvées": [8,10,13,8,7,10,6],
            "Coût (€)": [7200,8400,12500,7800,6200,9100,7145],
            "Dispo (%)": [89,91,88,93,95,92,87.5],
        })

init_data()

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def section_header(icon, title):
    st.markdown(f"""<div class="section-header">
        <div class="section-header-line"></div>
        <span style="font-size:16px;">{icon}</span>
        <span class="section-title">{title}</span>
    </div>""", unsafe_allow_html=True)

def page_title(title, subtitle=""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="page-subtitle">{datetime.now().strftime("%A %d %B %Y — %H:%M")}</div>', unsafe_allow_html=True)

def kpi_card(icon, label, value, sub="", color=""):
    st.markdown(f"""<div class="kpi-card {color}">
        <span class="kpi-icon">{icon}</span>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def item_card(id_text, title, meta_items, badges, prio_class=""):
    badges_html = "".join([f'<span class="badge badge-{b[0]}">{b[1]}</span> ' for b in badges])
    meta_html = "".join([f'<span>{m}</span>' for m in meta_items])
    st.markdown(f"""<div class="item-card {prio_class}">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">
            <div>
                <span class="item-id">{id_text}</span>
                <div class="item-title" style="margin-top:3px;">{title}</div>
            </div>
            <div style="text-align:right;flex-shrink:0;margin-left:12px;">{badges_html}</div>
        </div>
        <div class="item-meta">{meta_html}</div>
    </div>""", unsafe_allow_html=True)

def di_card(di_row):
    status_map = {"En attente":"pending","Approuvée":"approved","Rejetée":"rejected","Clôturée":"closed"}
    urgence_map = {"Critique":"critical","Haute":"high","Moyenne":"medium","Basse":"low"}
    sc = status_map.get(di_row["Statut"], "pending")
    uc = urgence_map.get(di_row["Urgence"], "low")
    bt_info = f"🔗 {di_row['BT généré']}" if di_row.get("BT généré") else "🔗 Pas de BT"
    commentaire = f'<div style="font-size:11px;color:#1a4a6a;margin-top:6px;font-style:italic;">💬 {di_row["Commentaire resp."]}</div>' if di_row.get("Commentaire resp.") else ""
    st.markdown(f"""<div class="di-card {sc}">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
                <span class="item-id">{di_row['DI']} · {di_row['Service']}</span>
                <div class="item-title" style="margin-top:3px;">{di_row['Titre']}</div>
            </div>
            <div style="flex-shrink:0;margin-left:12px;text-align:right;">
                <span class="badge badge-{uc}">{di_row['Urgence']}</span><br>
                <span class="badge badge-{sc}" style="margin-top:4px;display:inline-block;">{di_row['Statut']}</span>
            </div>
        </div>
        <div class="item-meta" style="margin-top:8px;">
            <span>👤 {di_row['Demandeur']}</span>
            <span>🔧 {di_row['Équipement']}</span>
            <span>📅 {di_row['Date demande']}</span>
            <span>⏰ Souhaité: {di_row['Date souhaitée']}</span>
            <span>{bt_info}</span>
        </div>
        <div style="font-size:11px;color:#2a5a7a;margin-top:6px;font-style:italic;">"{di_row['Description'][:80]}..."</div>
        {commentaire}
    </div>""", unsafe_allow_html=True)

def pdf_button(buf, filename, label):
    data = buf.getvalue() if hasattr(buf, 'getvalue') else buf.read()
    b64 = base64.b64encode(data).decode()
    st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="{filename}" class="pdf-dl-btn">📄 {label}</a>',
                unsafe_allow_html=True)

# ─── PLOTLY THEME ─────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#4a7aaa", family="Exo 2"),
    xaxis=dict(gridcolor="#0d2540", color="#2a4a6a", linecolor="#0d2540", zerolinecolor="#0d2540"),
    yaxis=dict(gridcolor="#0d2540", color="#2a4a6a", linecolor="#0d2540", zerolinecolor="#0d2540"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#4a7aaa")),
    margin=dict(l=0, r=0, t=40, b=0),
)

# ─── PDF HELPERS ──────────────────────────────────────────────────────────────
def get_pdf_styles():
    return {
        'h1': ParagraphStyle('h1', fontSize=16, textColor=PDF_WHITE, fontName='Helvetica-Bold',
                             spaceBefore=14, spaceAfter=8, backColor=PDF_NAVY,
                             leftIndent=-10, rightIndent=-10, borderPad=6),
        'h2': ParagraphStyle('h2', fontSize=12, textColor=PDF_NAVY, fontName='Helvetica-Bold',
                             spaceBefore=10, spaceAfter=5),
        'h3': ParagraphStyle('h3', fontSize=10, textColor=PDF_BLUE, fontName='Helvetica-Bold',
                             spaceBefore=6, spaceAfter=3),
        'body': ParagraphStyle('body', fontSize=9.5, textColor=PDF_GRAY, fontName='Helvetica',
                               spaceBefore=2, spaceAfter=2, leading=13),
        'kpi_v': ParagraphStyle('kpi_v', fontSize=20, textColor=PDF_BLUE, fontName='Helvetica-Bold', alignment=TA_CENTER),
        'kpi_l': ParagraphStyle('kpi_l', fontSize=8, textColor=PDF_GRAY, fontName='Helvetica', alignment=TA_CENTER),
        'small': ParagraphStyle('small', fontSize=8, textColor=PDF_MGRAY, fontName='Helvetica', alignment=TA_CENTER),
        'alert': ParagraphStyle('alert', fontSize=10, textColor=PDF_ACCENT, fontName='Helvetica-Bold'),
        'center': ParagraphStyle('center', fontSize=10, alignment=TA_CENTER, fontName='Helvetica'),
    }

def make_hf(canvas_obj, doc, title, subtitle=""):
    canvas_obj.saveState()
    w, h = A4
    canvas_obj.setFillColor(PDF_NAVY)
    canvas_obj.rect(0, h-28*mm, w, 28*mm, fill=1, stroke=0)
    canvas_obj.setFillColor(PDF_WHITE)
    canvas_obj.setFont("Helvetica-Bold", 15)
    canvas_obj.drawString(14*mm, h-15*mm, "⚙  GMAO PRO+")
    canvas_obj.setFont("Helvetica", 10)
    canvas_obj.setFillColor(colors.HexColor("#90CAF9"))
    canvas_obj.drawString(14*mm, h-23*mm, title)
    canvas_obj.setFillColor(PDF_WHITE)
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.drawRightString(w-14*mm, h-15*mm, datetime.now().strftime("%d/%m/%Y %H:%M"))
    if subtitle:
        canvas_obj.setFillColor(colors.HexColor("#90CAF9"))
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.drawRightString(w-14*mm, h-23*mm, subtitle)
    canvas_obj.setStrokeColor(PDF_BLUE)
    canvas_obj.setLineWidth(2)
    canvas_obj.line(0, h-29*mm, w, h-29*mm)
    canvas_obj.setFillColor(PDF_LGRAY)
    canvas_obj.rect(0, 0, w, 11*mm, fill=1, stroke=0)
    canvas_obj.setStrokeColor(PDF_MGRAY)
    canvas_obj.setLineWidth(0.5)
    canvas_obj.line(0, 11*mm, w, 11*mm)
    canvas_obj.setFillColor(PDF_GRAY)
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.drawString(14*mm, 3.5*mm, "Confidentiel — Usage interne")
    canvas_obj.drawCentredString(w/2, 3.5*mm, f"Page {doc.page}")
    canvas_obj.drawRightString(w-14*mm, 3.5*mm, "Industrie Maroc SA")
    canvas_obj.restoreState()

def tbl_style(hc=None, striped=True):
    hc = hc or PDF_NAVY
    ts = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), hc),
        ('TEXTCOLOR', (0,0), (-1,0), PDF_WHITE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
        ('GRID', (0,0), (-1,-1), 0.3, PDF_MGRAY),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ])
    if striped:
        ts.add('ROWBACKGROUNDS', (0,1), (-1,-1), [PDF_WHITE, PDF_LGRAY])
    return ts

def p_color(p): return {"Critique":PDF_ACCENT,"Haute":PDF_ORANGE,"Moyenne":PDF_BLUE,"Basse":PDF_GREEN}.get(p, PDF_GRAY)
def s_color(s): return {"Opérationnel":PDF_GREEN,"En panne":PDF_ACCENT,"En maintenance":colors.HexColor("#6A1B9A"),
                         "Terminé":PDF_GREEN,"En cours":PDF_BLUE,"Planifié":PDF_ORANGE,
                         "Ouvert":PDF_BLUE,"Approuvée":PDF_GREEN,"Rejetée":PDF_ACCENT,
                         "En attente":PDF_ORANGE,"Clôturée":PDF_TEAL}.get(s, PDF_GRAY)

# ═════════════════════════════════════════════════════════════════════════════
# PDF GENERATORS
# ═════════════════════════════════════════════════════════════════════════════

def pdf_rapport_mensuel():
    buf = io.BytesIO()
    S = get_pdf_styles()
    eq = st.session_state.equipements
    bt = st.session_state.bons_travaux
    pc = st.session_state.pieces
    di = st.session_state.demandes_intervention
    hist = st.session_state.historique
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=35*mm, bottomMargin=16*mm, leftMargin=14*mm, rightMargin=14*mm)
    story = []
    w = 167*mm
    def hf(c,d): make_hf(c,d,"RAPPORT MENSUEL DE MAINTENANCE","Janvier 2025")

    # Cover
    cover = Table([["RAPPORT MENSUEL DE MAINTENANCE\nJANVIER 2025"]], colWidths=[w])
    cover.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PDF_NAVY),('TEXTCOLOR',(0,0),(-1,-1),PDF_WHITE),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),18),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),22),('BOTTOMPADDING',(0,0),(-1,-1),22)]))
    story.append(cover); story.append(Spacer(1,5*mm))

    # KPIs
    taux_dispo = round(len(eq[eq["Statut"]=="Opérationnel"])/len(eq)*100,1)
    bt_term = len(bt[bt["Statut"]=="Terminé"])
    cout = bt["Coût estimé (€)"].sum()
    taux_prev = round(len(bt[bt["Type"]=="Préventif"])/len(bt)*100,1)
    n_di = len(di); n_di_app = len(di[di["Statut"]=="Approuvée"])

    kpi_data = [
        [Paragraph(f"{taux_dispo}%",S['kpi_v']), Paragraph(f"{bt_term}/{len(bt)}",S['kpi_v']),
         Paragraph(f"{taux_prev}%",S['kpi_v']), Paragraph(f"{n_di_app}/{n_di}",S['kpi_v']),
         Paragraph(f"{cout:,.0f}€",S['kpi_v'])],
        [Paragraph("Disponibilité",S['kpi_l']), Paragraph("BT réalisés",S['kpi_l']),
         Paragraph("Taux préventif",S['kpi_l']), Paragraph("DI approuvées",S['kpi_l']),
         Paragraph("Coût total",S['kpi_l'])],
    ]
    kt = Table(kpi_data, colWidths=[w/5]*5)
    kt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor("#E3F2FD")),
        ('GRID',(0,0),(-1,-1),0.4,PDF_MGRAY),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.append(kt); story.append(Spacer(1,5*mm))

    story.append(Paragraph("1. SYNTHÈSE ÉQUIPEMENTS", S['h1'])); story.append(Spacer(1,3*mm))
    eq_data = [["ID","Nom","Site","Statut","Criticité","Heures","Proch. MNT"]]
    for _,r in eq.iterrows():
        eq_data.append([r["ID"],r["Nom"][:28],r["Site"],r["Statut"],r["Criticité"],f"{r['Heures']:,}",r["Prochaine MNT"]])
    t = Table(eq_data, colWidths=[15*mm,50*mm,18*mm,22*mm,18*mm,16*mm,28*mm])
    ts = tbl_style()
    for i,r in enumerate(eq_data[1:],1):
        ts.add('TEXTCOLOR',(3,i),(3,i),s_color(r[3])); ts.add('FONTNAME',(3,i),(3,i),'Helvetica-Bold')
        ts.add('TEXTCOLOR',(4,i),(4,i),p_color(r[4]))
    t.setStyle(ts); story.append(t); story.append(Spacer(1,5*mm))

    story.append(Paragraph("2. BONS DE TRAVAUX", S['h1'])); story.append(Spacer(1,3*mm))
    bt_data = [["BT","Titre","Type","Priorité","Statut","Technicien","Coût estimé"]]
    for _,r in bt.iterrows():
        bt_data.append([r["BT"],r["Titre"][:26],r["Type"],r["Priorité"],r["Statut"],r["Technicien"],f"{r['Coût estimé (€)']:,.0f}€"])
    t2 = Table(bt_data, colWidths=[22*mm,43*mm,20*mm,18*mm,18*mm,24*mm,20*mm])
    ts2 = tbl_style()
    for i,r in enumerate(bt_data[1:],1):
        ts2.add('TEXTCOLOR',(3,i),(3,i),p_color(r[3])); ts2.add('TEXTCOLOR',(4,i),(4,i),s_color(r[4]))
        ts2.add('FONTNAME',(3,i),(3,i),'Helvetica-Bold')
    t2.setStyle(ts2); story.append(t2); story.append(Spacer(1,5*mm))

    story.append(Paragraph("3. DEMANDES D'INTERVENTION", S['h1'])); story.append(Spacer(1,3*mm))
    di_data = [["DI","Titre","Équipement","Urgence","Statut","Demandeur","BT généré"]]
    for _,r in di.iterrows():
        di_data.append([r["DI"],r["Titre"][:26],r["Équipement"],r["Urgence"],r["Statut"],r["Demandeur"],r.get("BT généré","—") or "—"])
    t3 = Table(di_data, colWidths=[22*mm,45*mm,18*mm,18*mm,18*mm,24*mm,22*mm])
    ts3 = tbl_style(PDF_TEAL)
    for i,r in enumerate(di_data[1:],1):
        ts3.add('TEXTCOLOR',(3,i),(3,i),p_color(r[3])); ts3.add('TEXTCOLOR',(4,i),(4,i),s_color(r[4]))
        ts3.add('FONTNAME',(3,i),(3,i),'Helvetica-Bold')
    t3.setStyle(ts3); story.append(t3); story.append(Spacer(1,5*mm))

    story.append(Paragraph("4. GESTION DES STOCKS", S['h1'])); story.append(Spacer(1,3*mm))
    pc_data = [["Référence","Désignation","Stock","Min","Statut","Prix","Valeur"]]
    for _,r in pc.iterrows():
        v = r["Stock"]*r["Prix (€)"]; st_txt = "⚠ RUPTURE" if r["Stock"]<r["Min"] else "✓ OK"
        pc_data.append([r["Réf"],r["Désignation"][:26],str(r["Stock"]),str(r["Min"]),st_txt,f"{r['Prix (€)']:.2f}€",f"{v:.0f}€"])
    t4 = Table(pc_data, colWidths=[16*mm,48*mm,14*mm,11*mm,20*mm,18*mm,20*mm])
    ts4 = tbl_style(PDF_PURPLE)
    for i,r in enumerate(pc_data[1:],1):
        if "RUPTURE" in r[4]: ts4.add('TEXTCOLOR',(4,i),(4,i),PDF_ACCENT); ts4.add('FONTNAME',(4,i),(4,i),'Helvetica-Bold')
        else: ts4.add('TEXTCOLOR',(4,i),(4,i),PDF_GREEN)
    t4.setStyle(ts4); story.append(t4)
    story.append(Spacer(1,8*mm))
    story.append(HRFlowable(width=w,color=PDF_BLUE,thickness=1))
    story.append(Spacer(1,3*mm))
    story.append(Paragraph(f"Rapport généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} — GMAO Pro+", S['small']))

    doc.build(story, onFirstPage=hf, onLaterPages=hf)
    buf.seek(0); return buf

def pdf_bon_travaux(bt_id):
    buf = io.BytesIO()
    S = get_pdf_styles()
    bt = st.session_state.bons_travaux
    eq = st.session_state.equipements
    row = bt[bt["BT"]==bt_id].iloc[0]
    def hf(c,d): make_hf(c,d,"BON DE TRAVAUX",bt_id)
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=35*mm, bottomMargin=16*mm, leftMargin=14*mm, rightMargin=14*mm)
    story = []; w = 167*mm
    pc = p_color(row["Priorité"]); sc = s_color(row["Statut"])
    tt = Table([[f"BON DE TRAVAUX — {bt_id}", f"PRIORITÉ: {row['Priorité'].upper()}"]], colWidths=[110*mm,57*mm])
    tt.setStyle(TableStyle([('BACKGROUND',(0,0),(0,0),PDF_NAVY),('BACKGROUND',(1,0),(1,0),pc),
        ('TEXTCOLOR',(0,0),(-1,-1),PDF_WHITE),('FONTNAME',(0,0),(-1,-1),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(0,0),13),('FONTSIZE',(1,0),(1,0),12),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),13),('BOTTOMPADDING',(0,0),(-1,-1),13)]))
    story.append(tt); story.append(Spacer(1,4*mm))
    st_tbl = Table([[f"Statut: {row['Statut'].upper()}  |  Type: {row['Type']}  |  DI Origine: {row.get('DI origine','N/A') or 'Aucune'}"]], colWidths=[w])
    st_tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),sc),('TEXTCOLOR',(0,0),(-1,-1),PDF_WHITE),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),10),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.append(st_tbl); story.append(Spacer(1,5*mm))
    eq_row = eq[eq["ID"]==row["Équipement"]]
    eq_nom = eq_row["Nom"].values[0] if len(eq_row)>0 else "N/A"
    eq_loc = eq_row["Localisation"].values[0] if len(eq_row)>0 else "N/A"
    info = [["Numéro BT:",row["BT"],"Date création:",row["Date création"]],
            ["Titre:",row["Titre"],"Date prévue:",row["Date prévue"]],
            ["Type:",row["Type"],"Durée estimée:",f"{row['Durée (h)']}h"],
            ["Équipement:",f"{row['Équipement']} — {eq_nom}","Localisation:",eq_loc],
            ["Demandeur:",row["Demandeur"],"Coût estimé:",f"{row['Coût estimé (€)']:,.0f}€"],
            ["Technicien:",row["Technicien"],"Coût réel:",f"{row['Coût réel (€)']:,.0f}€" if row["Coût réel (€)"]>0 else "N/A"]]
    it = Table(info, colWidths=[30*mm,52*mm,30*mm,55*mm])
    it.setStyle(TableStyle([('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),('FONTNAME',(2,0),(2,-1),'Helvetica-Bold'),
        ('TEXTCOLOR',(0,0),(0,-1),PDF_NAVY),('TEXTCOLOR',(2,0),(2,-1),PDF_NAVY),
        ('FONTSIZE',(0,0),(-1,-1),9.5),('ROWBACKGROUNDS',(0,0),(-1,-1),[PDF_LGRAY,PDF_WHITE]),
        ('GRID',(0,0),(-1,-1),0.3,PDF_MGRAY),('TOPPADDING',(0,0),(-1,-1),6),
        ('BOTTOMPADDING',(0,0),(-1,-1),6),('LEFTPADDING',(0,0),(-1,-1),8)]))
    story.append(Paragraph("INFORMATIONS GÉNÉRALES", S['h2'])); story.append(it); story.append(Spacer(1,4*mm))
    desc = Table([[row.get("Description","Aucune description.")]], colWidths=[w])
    desc.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor("#E3F2FD")),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),10),
        ('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10),
        ('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),
        ('GRID',(0,0),(-1,-1),0.5,PDF_BLUE)]))
    story.append(Paragraph("DESCRIPTION DES TRAVAUX", S['h2'])); story.append(desc); story.append(Spacer(1,4*mm))
    cl = [["☐","Vérification EPI avant intervention"],["☐","Consignation équipement"],
          ["☐","Diagnostic et identification du problème"],["☐","Préparation outils et pièces"],
          ["☐","Réalisation des travaux"],["☐","Tests de fonctionnement"],
          ["☐","Nettoyage du poste de travail"],["☐","Rapport d'intervention complété"],["☐","Validation responsable"]]
    clt = Table(cl, colWidths=[10*mm,157*mm])
    clt.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),10),
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[PDF_WHITE,PDF_LGRAY]),('GRID',(0,0),(-1,-1),0.3,PDF_MGRAY),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LEFTPADDING',(0,0),(-1,-1),6),
        ('ALIGN',(0,0),(0,-1),'CENTER'),('FONTSIZE',(0,0),(0,-1),14)]))
    story.append(Paragraph("CHECKLIST SÉCURITÉ", S['h2'])); story.append(clt); story.append(Spacer(1,4*mm))
    rp = [["Heure début:","_____________","Heure fin:","_____________"],
          ["Durée réelle:","_____________","Coût réel (€):","_____________"],
          ["Pièces utilisées:","","",""],["","","",""],
          ["Observations:","","",""],["","","",""],["","","",""]]
    rt = Table(rp, colWidths=[32*mm,52*mm,32*mm,51*mm])
    rt.setStyle(TableStyle([('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),('FONTNAME',(2,0),(2,1),'Helvetica-Bold'),
        ('TEXTCOLOR',(0,0),(0,-1),PDF_NAVY),('TEXTCOLOR',(2,0),(2,1),PDF_NAVY),
        ('FONTSIZE',(0,0),(-1,-1),9.5),('SPAN',(1,2),(-1,2)),('SPAN',(0,3),(-1,3)),
        ('SPAN',(1,4),(-1,4)),('SPAN',(0,5),(-1,5)),('SPAN',(0,6),(-1,6)),
        ('GRID',(0,0),(-1,-1),0.3,PDF_MGRAY),('ROWBACKGROUNDS',(0,0),(-1,-1),[PDF_LGRAY,PDF_WHITE]),
        ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('LEFTPADDING',(0,0),(-1,-1),8)]))
    story.append(Paragraph("RAPPORT D'INTERVENTION", S['h2'])); story.append(rt); story.append(Spacer(1,4*mm))
    sg_data = [["Technicien intervenant","Responsable maintenance","Demandeur / Opérateur"],
               ["\n\n\n_____________________","\n\n\n_____________________","\n\n\n_____________________"],
               ["Nom: _______________\nDate: ______________","Nom: _______________\nDate: ______________","Nom: _______________\nDate: ______________"]]
    sg = Table(sg_data, colWidths=[w/3]*3)
    sg.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PDF_NAVY),('TEXTCOLOR',(0,0),(-1,0),PDF_WHITE),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('GRID',(0,0),(-1,-1),0.5,PDF_MGRAY),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.append(Paragraph("SIGNATURES", S['h2'])); story.append(sg)
    doc.build(story, onFirstPage=hf, onLaterPages=hf)
    buf.seek(0); return buf

def pdf_demande_intervention(di_id):
    buf = io.BytesIO()
    S = get_pdf_styles()
    di = st.session_state.demandes_intervention
    eq = st.session_state.equipements
    row = di[di["DI"]==di_id].iloc[0]
    def hf(c,d): make_hf(c,d,"DEMANDE D'INTERVENTION",di_id)
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=35*mm, bottomMargin=16*mm, leftMargin=14*mm, rightMargin=14*mm)
    story = []; w = 167*mm
    uc = p_color(row["Urgence"]); sc = s_color(row["Statut"])

    tt = Table([[f"DEMANDE D'INTERVENTION — {di_id}", f"URGENCE: {row['Urgence'].upper()}"]], colWidths=[110*mm,57*mm])
    tt.setStyle(TableStyle([('BACKGROUND',(0,0),(0,0),PDF_TEAL),('BACKGROUND',(1,0),(1,0),uc),
        ('TEXTCOLOR',(0,0),(-1,-1),PDF_WHITE),('FONTNAME',(0,0),(-1,-1),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(0,0),13),('FONTSIZE',(1,0),(1,0),12),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),13),('BOTTOMPADDING',(0,0),(-1,-1),13)]))
    story.append(tt); story.append(Spacer(1,4*mm))

    st_tbl = Table([[f"Statut: {row['Statut'].upper()}  |  Type: {row['Type demandé']}  |  Service: {row['Service']}"]], colWidths=[w])
    st_tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),sc),('TEXTCOLOR',(0,0),(-1,-1),PDF_WHITE),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),10),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.append(st_tbl); story.append(Spacer(1,5*mm))

    eq_row = eq[eq["ID"]==row["Équipement"]]
    eq_nom = eq_row["Nom"].values[0] if len(eq_row)>0 else "N/A"
    eq_loc = eq_row["Localisation"].values[0] if len(eq_row)>0 else "N/A"
    info = [["Numéro DI:",row["DI"],"Date demande:",row["Date demande"]],
            ["Demandeur:",row["Demandeur"],"Date souhaitée:",row["Date souhaitée"]],
            ["Service:",row["Service"],"Type demandé:",row["Type demandé"]],
            ["Équipement:",f"{row['Équipement']} — {eq_nom}","Localisation:",eq_loc],
            ["Urgence:",row["Urgence"],"BT généré:",row.get("BT généré","Aucun") or "Aucun"],
            ["Site:",row["Site"],"Statut:",row["Statut"]]]
    it = Table(info, colWidths=[30*mm,52*mm,30*mm,55*mm])
    it.setStyle(TableStyle([('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),('FONTNAME',(2,0),(2,-1),'Helvetica-Bold'),
        ('TEXTCOLOR',(0,0),(0,-1),PDF_TEAL),('TEXTCOLOR',(2,0),(2,-1),PDF_TEAL),
        ('FONTSIZE',(0,0),(-1,-1),9.5),('ROWBACKGROUNDS',(0,0),(-1,-1),[PDF_LGRAY,PDF_WHITE]),
        ('GRID',(0,0),(-1,-1),0.3,PDF_MGRAY),('TOPPADDING',(0,0),(-1,-1),6),
        ('BOTTOMPADDING',(0,0),(-1,-1),6),('LEFTPADDING',(0,0),(-1,-1),8)]))
    story.append(Paragraph("INFORMATIONS DE LA DEMANDE", S['h2'])); story.append(it); story.append(Spacer(1,4*mm))

    desc = Table([[row["Description"]]], colWidths=[w])
    desc.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor("#E0F2F1")),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),10),
        ('TOPPADDING',(0,0),(-1,-1),12),('BOTTOMPADDING',(0,0),(-1,-1),12),
        ('LEFTPADDING',(0,0),(-1,-1),14),('RIGHTPADDING',(0,0),(-1,-1),14),
        ('GRID',(0,0),(-1,-1),0.5,PDF_TEAL)]))
    story.append(Paragraph("DESCRIPTION DU PROBLÈME / BESOIN", S['h2'])); story.append(desc); story.append(Spacer(1,4*mm))

    if row.get("Commentaire resp."):
        com = Table([[f"Décision du responsable: {row['Commentaire resp.']}"]], colWidths=[w])
        com.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor("#FFF3E0") if row["Statut"]=="Approuvée" else colors.HexColor("#FFEBEE")),
            ('FONTNAME',(0,0),(-1,-1),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),10),
            ('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10),
            ('LEFTPADDING',(0,0),(-1,-1),12),
            ('GRID',(0,0),(-1,-1),0.5,PDF_ORANGE if row["Statut"]=="Approuvée" else PDF_ACCENT)]))
        story.append(Paragraph("DÉCISION DU RESPONSABLE MAINTENANCE", S['h2'])); story.append(com); story.append(Spacer(1,4*mm))

    # Zone analyse
    ana = [["Analyse préliminaire (à compléter par le responsable maintenance)"],
           ["\n\nCause probable: _______________________________________________\n\nImpact production: ☐ Arrêt total  ☐ Dégradé  ☐ Sans impact\n\nAction requise: ☐ Correctif immédiat  ☐ Planifier  ☐ Préventif  ☐ Refus\n\nPriorité assignée: ☐ Critique  ☐ Haute  ☐ Moyenne  ☐ Basse\n\nEstimation durée intervention: _________h  Estimation coût: _________€\n"]]
    at = Table(ana, colWidths=[w])
    at.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PDF_NAVY),('TEXTCOLOR',(0,0),(-1,0),PDF_WHITE),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,0),10),
        ('ALIGN',(0,0),(-1,0),'CENTER'),('TOPPADDING',(0,0),(-1,0),8),('BOTTOMPADDING',(0,0),(-1,0),8),
        ('FONTNAME',(0,1),(-1,-1),'Helvetica'),('FONTSIZE',(0,1),(-1,-1),9.5),
        ('BACKGROUND',(0,1),(-1,-1),PDF_LGRAY),('TOPPADDING',(0,1),(-1,-1),8),
        ('BOTTOMPADDING',(0,1),(-1,-1),8),('LEFTPADDING',(0,1),(-1,-1),12),
        ('GRID',(0,0),(-1,-1),0.5,PDF_MGRAY)]))
    story.append(at); story.append(Spacer(1,5*mm))

    sg_data = [["Demandeur","Responsable maintenance","Direction"],
               ["\n\n\n_____________________","\n\n\n_____________________","\n\n\n_____________________"],
               ["Nom: _______________\nDate: ______________","Nom: _______________\nDate: ______________","Nom: _______________\nDate: ______________"]]
    sg = Table(sg_data, colWidths=[w/3]*3)
    sg.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PDF_TEAL),('TEXTCOLOR',(0,0),(-1,0),PDF_WHITE),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('GRID',(0,0),(-1,-1),0.5,PDF_MGRAY),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.append(Paragraph("SIGNATURES ET APPROBATION", S['h2'])); story.append(sg)
    doc.build(story, onFirstPage=hf, onLaterPages=hf)
    buf.seek(0); return buf

def pdf_fiche_equipement(eq_id):
    buf = io.BytesIO()
    S = get_pdf_styles()
    eq = st.session_state.equipements
    bt = st.session_state.bons_travaux
    di = st.session_state.demandes_intervention
    row = eq[eq["ID"]==eq_id].iloc[0]
    bt_eq = bt[bt["Équipement"]==eq_id]
    di_eq = di[di["Équipement"]==eq_id]
    def hf(c,d): make_hf(c,d,"FICHE ÉQUIPEMENT",eq_id)
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=35*mm, bottomMargin=16*mm, leftMargin=14*mm, rightMargin=14*mm)
    story = []; w = 167*mm
    sc = s_color(row["Statut"])
    tt = Table([[f"{row['ID']} — {row['Nom']}", row["Statut"].upper()]], colWidths=[118*mm,49*mm])
    tt.setStyle(TableStyle([('BACKGROUND',(0,0),(0,0),PDF_NAVY),('BACKGROUND',(1,0),(1,0),sc),
        ('TEXTCOLOR',(0,0),(-1,-1),PDF_WHITE),('FONTNAME',(0,0),(-1,-1),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(0,0),13),('FONTSIZE',(1,0),(1,0),12),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),14),('BOTTOMPADDING',(0,0),(-1,-1),14)]))
    story.append(tt); story.append(Spacer(1,4*mm))
    info = [["ID:",row["ID"],"Catégorie:",row["Catégorie"]],
            ["Site:",row["Site"],"Localisation:",row["Localisation"]],
            ["Criticité:",row["Criticité"],"Responsable:",row["Resp."]],
            ["Installation:",row["Installation"],"Valeur actif:",f"{row['Valeur (€)']:,}€"],
            ["Heures marche:",f"{row['Heures']:,}h","Dernière MNT:",row["Dernière MNT"]],
            ["Prochaine MNT:",row["Prochaine MNT"],"Statut:",row["Statut"]]]
    it = Table(info, colWidths=[28*mm,52*mm,28*mm,59*mm])
    it.setStyle(TableStyle([('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),('FONTNAME',(2,0),(2,-1),'Helvetica-Bold'),
        ('TEXTCOLOR',(0,0),(0,-1),PDF_NAVY),('TEXTCOLOR',(2,0),(2,-1),PDF_NAVY),
        ('FONTSIZE',(0,0),(-1,-1),9.5),('ROWBACKGROUNDS',(0,0),(-1,-1),[PDF_LGRAY,PDF_WHITE]),
        ('GRID',(0,0),(-1,-1),0.3,PDF_MGRAY),('TOPPADDING',(0,0),(-1,-1),6),
        ('BOTTOMPADDING',(0,0),(-1,-1),6),('LEFTPADDING',(0,0),(-1,-1),8)]))
    story.append(Paragraph("DONNÉES TECHNIQUES", S['h2'])); story.append(it); story.append(Spacer(1,4*mm))
    n_i=len(bt_eq); c_t=bt_eq["Coût estimé (€)"].sum(); n_c=len(bt_eq[bt_eq["Type"]=="Correctif"]); n_p=len(bt_eq[bt_eq["Type"]=="Préventif"]); n_di=len(di_eq)
    kd = [[Paragraph(str(n_i),S['kpi_v']),Paragraph(f"{c_t:,.0f}€",S['kpi_v']),
           Paragraph(str(n_c),S['kpi_v']),Paragraph(str(n_p),S['kpi_v']),Paragraph(str(n_di),S['kpi_v'])],
          [Paragraph("Interventions",S['kpi_l']),Paragraph("Coût total",S['kpi_l']),
           Paragraph("Correctifs",S['kpi_l']),Paragraph("Préventifs",S['kpi_l']),Paragraph("DI reçues",S['kpi_l'])]]
    kt = Table(kd, colWidths=[w/5]*5)
    kt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor("#E3F2FD")),
        ('GRID',(0,0),(-1,-1),0.4,PDF_MGRAY),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.append(Paragraph("INDICATEURS", S['h2'])); story.append(kt); story.append(Spacer(1,4*mm))
    if len(bt_eq)>0:
        bt_d = [["BT","Titre","Type","Statut","Technicien","Date","Coût"]]
        for _,r in bt_eq.iterrows():
            bt_d.append([r["BT"],r["Titre"][:22],r["Type"],r["Statut"],r["Technicien"],r["Date prévue"],f"{r['Coût estimé (€)']:,.0f}€"])
        btt = Table(bt_d, colWidths=[22*mm,38*mm,20*mm,18*mm,24*mm,22*mm,20*mm])
        ts = tbl_style()
        for i,r in enumerate(bt_d[1:],1):
            ts.add('TEXTCOLOR',(3,i),(3,i),s_color(r[3]))
        btt.setStyle(ts)
        story.append(Paragraph("HISTORIQUE INTERVENTIONS", S['h2'])); story.append(btt); story.append(Spacer(1,3*mm))
    if len(di_eq)>0:
        di_d = [["DI","Titre","Urgence","Statut","Demandeur","Date"]]
        for _,r in di_eq.iterrows():
            di_d.append([r["DI"],r["Titre"][:28],r["Urgence"],r["Statut"],r["Demandeur"],r["Date demande"]])
        dit = Table(di_d, colWidths=[22*mm,50*mm,18*mm,18*mm,24*mm,22*mm])
        ts2 = tbl_style(PDF_TEAL)
        for i,r in enumerate(di_d[1:],1):
            ts2.add('TEXTCOLOR',(2,i),(2,i),p_color(r[2])); ts2.add('TEXTCOLOR',(3,i),(3,i),s_color(r[3]))
        dit.setStyle(ts2)
        story.append(Paragraph("DEMANDES D'INTERVENTION", S['h2'])); story.append(dit)
    doc.build(story, onFirstPage=hf, onLaterPages=hf)
    buf.seek(0); return buf

def pdf_rapport_di():
    buf = io.BytesIO()
    S = get_pdf_styles()
    di = st.session_state.demandes_intervention
    def hf(c,d): make_hf(c,d,"RAPPORT DEMANDES D'INTERVENTION","Analyse complète")
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=35*mm, bottomMargin=16*mm, leftMargin=14*mm, rightMargin=14*mm)
    story = []; w = 167*mm
    cover = Table([["RAPPORT DEMANDES D'INTERVENTION"]], colWidths=[w])
    cover.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PDF_TEAL),('TEXTCOLOR',(0,0),(-1,-1),PDF_WHITE),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),18),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('TOPPADDING',(0,0),(-1,-1),20),('BOTTOMPADDING',(0,0),(-1,-1),20)]))
    story.append(cover); story.append(Spacer(1,5*mm))
    n_tot=len(di); n_att=len(di[di["Statut"]=="En attente"]); n_app=len(di[di["Statut"]=="Approuvée"])
    n_rej=len(di[di["Statut"]=="Rejetée"]); n_clo=len(di[di["Statut"]=="Clôturée"])
    kd = [[Paragraph(str(n_tot),S['kpi_v']),Paragraph(str(n_att),S['kpi_v']),
           Paragraph(str(n_app),S['kpi_v']),Paragraph(str(n_rej),S['kpi_v'])],
          [Paragraph("Total DI",S['kpi_l']),Paragraph("En attente",S['kpi_l']),
           Paragraph("Approuvées",S['kpi_l']),Paragraph("Rejetées",S['kpi_l'])]]
    kt = Table(kd, colWidths=[w/4]*4)
    kt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor("#E0F2F1")),
        ('GRID',(0,0),(-1,-1),0.4,PDF_MGRAY),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.append(kt); story.append(Spacer(1,5*mm))
    story.append(Paragraph("LISTE COMPLÈTE DES DEMANDES", S['h1'])); story.append(Spacer(1,3*mm))
    di_d = [["DI","Titre","Équip.","Urgence","Statut","Demandeur","Date dem.","Date souh.","BT"]]
    for _,r in di.iterrows():
        di_d.append([r["DI"],r["Titre"][:22],r["Équipement"],r["Urgence"],r["Statut"],
                     r["Demandeur"][:12],r["Date demande"],r["Date souhaitée"],r.get("BT généré","—") or "—"])
    dit = Table(di_d, colWidths=[22*mm,36*mm,14*mm,16*mm,18*mm,22*mm,18*mm,18*mm,18*mm])
    ts = tbl_style(PDF_TEAL)
    for i,r in enumerate(di_d[1:],1):
        ts.add('TEXTCOLOR',(3,i),(3,i),p_color(r[3])); ts.add('TEXTCOLOR',(4,i),(4,i),s_color(r[4]))
        ts.add('FONTNAME',(3,i),(3,i),'Helvetica-Bold')
    dit.setStyle(ts); story.append(dit); story.append(Spacer(1,5*mm))
    story.append(Paragraph("DI EN ATTENTE DE TRAITEMENT", S['h1'])); story.append(Spacer(1,3*mm))
    di_att = di[di["Statut"]=="En attente"]
    if len(di_att)>0:
        for _,r in di_att.iterrows():
            row_d = [[r["DI"],r["Titre"]],[f"Équipement: {r['Équipement']}\nDemandeur: {r['Demandeur']}\nService: {r['Service']}",
                      f"Urgence: {r['Urgence']}\nDate souhaitée: {r['Date souhaitée']}\nDescription: {r['Description'][:80]}"]]
            rt = Table(row_d, colWidths=[30*mm,137*mm])
            rt.setStyle(TableStyle([('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),
                ('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.HexColor("#FFF8E1"),PDF_WHITE]),
                ('GRID',(0,0),(-1,-1),0.3,PDF_MGRAY),('TOPPADDING',(0,0),(-1,-1),5),
                ('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),8),
                ('SPAN',(0,0),(0,0)),('SPAN',(1,0),(1,0))]))
            story.append(rt); story.append(Spacer(1,2*mm))
    else:
        story.append(Paragraph("Aucune DI en attente.", S['body']))
    doc.build(story, onFirstPage=hf, onLaterPages=hf)
    buf.seek(0); return buf

def pdf_rapport_stock():
    buf = io.BytesIO()
    S = get_pdf_styles()
    pc = st.session_state.pieces
    def hf(c,d): make_hf(c,d,"RAPPORT STOCK & INVENTAIRE",datetime.now().strftime("%d/%m/%Y"))
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=35*mm, bottomMargin=16*mm, leftMargin=14*mm, rightMargin=14*mm)
    story = []; w = 167*mm
    val = (pc["Stock"]*pc["Prix (€)"]).sum(); n_r=len(pc[pc["Stock"]<pc["Min"]])
    kd = [[Paragraph(str(len(pc)),S['kpi_v']),Paragraph(f"{val:,.0f}€",S['kpi_v']),
           Paragraph(str(n_r),S['kpi_v']),Paragraph(str(len(pc[pc["Stock"]>=pc["Min"]])),S['kpi_v'])],
          [Paragraph("Références",S['kpi_l']),Paragraph("Valeur stock",S['kpi_l']),
           Paragraph("Ruptures",S['kpi_l']),Paragraph("OK",S['kpi_l'])]]
    kt = Table(kd, colWidths=[w/4]*4)
    kt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor("#EDE7F6")),
        ('GRID',(0,0),(-1,-1),0.4,PDF_MGRAY),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.append(Paragraph("INVENTAIRE STOCK", S['h1'])); story.append(Spacer(1,4*mm)); story.append(kt); story.append(Spacer(1,4*mm))
    pc_d = [["Réf.","Désignation","Catég.","Stock","Min","Max","Statut","Prix","Valeur","Empl."]]
    for _,r in pc.iterrows():
        v=r["Stock"]*r["Prix (€)"]; st_t="⚠ RUPTURE" if r["Stock"]<r["Min"] else "✓ OK"
        pc_d.append([r["Réf"],r["Désignation"][:22],r["Catégorie"][:10],str(r["Stock"]),
                     str(r["Min"]),str(r["Max"]),st_t,f"{r['Prix (€)']:.2f}€",f"{v:.0f}€",r["Emplacement"][:12]])
    t = Table(pc_d, colWidths=[13*mm,36*mm,16*mm,12*mm,10*mm,10*mm,18*mm,16*mm,16*mm,20*mm])
    ts = tbl_style(PDF_PURPLE)
    for i,r in enumerate(pc_d[1:],1):
        if "RUPTURE" in r[6]: ts.add('BACKGROUND',(0,i),(-1,i),colors.HexColor("#FFEBEE")); ts.add('TEXTCOLOR',(6,i),(6,i),PDF_ACCENT); ts.add('FONTNAME',(6,i),(6,i),'Helvetica-Bold')
        else: ts.add('TEXTCOLOR',(6,i),(6,i),PDF_GREEN); ts.add('FONTNAME',(6,i),(6,i),'Helvetica-Bold')
    t.setStyle(ts); story.append(t)
    rup = pc[pc["Stock"]<pc["Min"]]
    if len(rup)>0:
        story.append(Spacer(1,5*mm)); story.append(Paragraph("COMMANDES SUGGÉRÉES", S['h1'])); story.append(Spacer(1,3*mm))
        cmd = [["Référence","Désignation","Stock actuel","À commander","Fournisseur","Délai","Coût"]]
        for _,r in rup.iterrows():
            q=r["Max"]-r["Stock"]; c=q*r["Prix (€)"]
            cmd.append([r["Réf"],r["Désignation"][:28],str(r["Stock"]),str(q),r["Fournisseur"],f"{r['Délai (j)']}j",f"{c:.2f}€"])
        ct = Table(cmd, colWidths=[16*mm,48*mm,20*mm,20*mm,28*mm,14*mm,21*mm])
        ct.setStyle(tbl_style(PDF_ACCENT)); story.append(ct)
    doc.build(story, onFirstPage=hf, onLaterPages=hf)
    buf.seek(0); return buf

def pdf_planning():
    buf = io.BytesIO()
    S = get_pdf_styles()
    bt = st.session_state.bons_travaux
    tech = st.session_state.techniciens
    w_p, h_p = landscape(A4)
    def hf(c,d):
        c.saveState()
        c.setFillColor(PDF_NAVY); c.rect(0,h_p-20*mm,w_p,20*mm,fill=1,stroke=0)
        c.setFillColor(PDF_WHITE); c.setFont("Helvetica-Bold",14)
        c.drawString(12*mm,h_p-13*mm,"⚙ GMAO PRO+ — PLANNING DES INTERVENTIONS")
        c.setFont("Helvetica",9); c.drawRightString(w_p-12*mm,h_p-13*mm,datetime.now().strftime("%d/%m/%Y"))
        c.setFillColor(PDF_LGRAY); c.rect(0,0,w_p,10*mm,fill=1,stroke=0)
        c.setFillColor(PDF_GRAY); c.setFont("Helvetica",8); c.drawCentredString(w_p/2,3*mm,f"Page {d.page}")
        c.restoreState()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), topMargin=25*mm, bottomMargin=14*mm, leftMargin=12*mm, rightMargin=12*mm)
    story = []; w = 253*mm
    bt_a = bt[bt["Statut"]!="Terminé"].sort_values("Priorité", key=lambda x: x.map({"Critique":0,"Haute":1,"Moyenne":2,"Basse":3}))
    pd_data = [["BT N°","Titre","Équip.","Type","Priorité","Technicien","Date prévue","Durée","Coût estimé","Statut"]]
    for _,r in bt_a.iterrows():
        pd_data.append([r["BT"],r["Titre"][:30],r["Équipement"],r["Type"],r["Priorité"],
                         r["Technicien"],r["Date prévue"],f"{r['Durée (h)']}h",f"{r['Coût estimé (€)']:,.0f}€",r["Statut"]])
    t = Table(pd_data, colWidths=[22*mm,52*mm,16*mm,20*mm,18*mm,24*mm,22*mm,14*mm,20*mm,18*mm])
    ts = tbl_style()
    for i,r in enumerate(pd_data[1:],1):
        ts.add('TEXTCOLOR',(4,i),(4,i),p_color(r[4])); ts.add('FONTNAME',(4,i),(4,i),'Helvetica-Bold')
        ts.add('TEXTCOLOR',(9,i),(9,i),s_color(r[9]))
    t.setStyle(ts)
    story.append(Paragraph("PLANNING INTERVENTIONS ACTIVES",ParagraphStyle('h',fontSize=14,fontName='Helvetica-Bold',textColor=PDF_NAVY,spaceAfter=6)))
    story.append(t); story.append(Spacer(1,6*mm))
    td = [["Technicien","Spécialité","Disponible","BT actifs","Efficacité","Certifications"]]
    for _,r in tech.iterrows():
        td.append([r["Nom"],r["Spécialité"],"✓ Oui" if r["Disponible"] else "✗ Non",
                    str(r["BT en cours"]),f"{r['Efficacité (%)']}%",r["Certifications"]])
    tt = Table(td, colWidths=[30*mm,52*mm,22*mm,20*mm,20*mm,70*mm])
    ts2 = tbl_style(PDF_BLUE)
    for i,r in enumerate(td[1:],1):
        ts2.add('TEXTCOLOR',(2,i),(2,i),PDF_GREEN if "✓" in r[2] else PDF_ACCENT)
        ts2.add('FONTNAME',(2,i),(2,i),'Helvetica-Bold')
    tt.setStyle(ts2)
    story.append(Paragraph("RESSOURCES TECHNICIENS",ParagraphStyle('h2',fontSize=12,fontName='Helvetica-Bold',textColor=PDF_NAVY,spaceAfter=4)))
    story.append(tt)
    doc.build(story, onFirstPage=hf, onLaterPages=hf)
    buf.seek(0); return buf

def merge_pdfs(bufs):
    w = PdfWriter()
    for b in bufs:
        r = PdfReader(b)
        for p in r.pages: w.add_page(p)
    out = io.BytesIO(); w.write(out); out.seek(0); return out

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="brand-logo">⚙ GMAO PRO+</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-tagline">Maintenance Industrielle Avancée</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    menu = st.radio("", [
        "🏠  Dashboard",
        "📨  Demandes d'intervention",
        "🔧  Équipements",
        "📋  Bons de travaux",
        "📦  Stock & Pièces",
        "👷  Techniciens",
        "📅  Planning",
        "📊  KPIs & Analyses",
        "📄  Centre PDF",
        "⚙️  Paramètres",
    ], label_visibility="collapsed")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    eq = st.session_state.equipements
    bt = st.session_state.bons_travaux
    pc = st.session_state.pieces
    di = st.session_state.demandes_intervention

    n_pannes = len(eq[eq["Statut"]=="En panne"])
    n_stock  = len(pc[pc["Stock"]<pc["Min"]])
    n_crit   = len(bt[(bt["Priorité"]=="Critique")&(bt["Statut"]!="Terminé")])
    n_di_att = len(di[di["Statut"]=="En attente"])

    if n_pannes:  st.markdown(f'<div class="alert-danger">🔴 {n_pannes} équipement(s) en panne</div>', unsafe_allow_html=True)
    if n_di_att:  st.markdown(f'<div class="alert-warning">🟡 {n_di_att} DI en attente de traitement</div>', unsafe_allow_html=True)
    if n_stock:   st.markdown(f'<div class="alert-warning">🟠 {n_stock} rupture(s) de stock</div>', unsafe_allow_html=True)
    if n_crit:    st.markdown(f'<div class="alert-danger">🔴 {n_crit} BT critique(s)</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    taux_dispo = round(len(eq[eq["Statut"]=="Opérationnel"])/len(eq)*100,1)
    st.markdown(f"""
    <div style='display:flex;flex-direction:column;gap:6px;'>
        <div class='stat-pill'>📊 Dispo parc: <b style='color:#60b0ff;'>{taux_dispo}%</b></div>
        <div class='stat-pill'>📋 BT actifs: <b style='color:#60b0ff;'>{len(bt[bt["Statut"]!="Terminé"])}</b></div>
        <div class='stat-pill'>📨 DI total: <b style='color:#60b0ff;'>{len(di)}</b></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:10px;color:#0d2540;margin-top:12px;">🕐 {datetime.now().strftime("%d/%m/%Y %H:%M")}</div>', unsafe_allow_html=True)

page = menu.split("  ")[-1].strip()

# ═════════════════════════════════════════════════════════════════════════════
# PAGES
# ═════════════════════════════════════════════════════════════════════════════

# ─── DASHBOARD ───────────────────────────────────────────────────────────────
if page == "Dashboard":
    page_title("TABLEAU DE BORD", "Vue d'ensemble en temps réel")

    taux_dispo = round(len(eq[eq["Statut"]=="Opérationnel"])/len(eq)*100,1)
    bt_actifs  = len(bt[bt["Statut"].isin(["Ouvert","En cours","Planifié"])])
    cout_total = bt["Coût estimé (€)"].sum()
    val_stock  = (pc["Stock"]*pc["Prix (€)"]).sum()
    taux_prev  = round(len(bt[bt["Type"]=="Préventif"])/len(bt)*100,1)
    n_di_att   = len(di[di["Statut"]=="En attente"])

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: kpi_card("🏭","Disponibilité",f"{taux_dispo}%","↑ +2.1% vs N-1","success" if taux_dispo>=90 else "warn")
    with c2: kpi_card("📋","BT Actifs",str(bt_actifs),f"{n_crit} critique(s)","warn")
    with c3: kpi_card("📨","DI en attente",str(n_di_att),"À traiter","warn" if n_di_att>0 else "success")
    with c4: kpi_card("💰","Coût total",f"{cout_total:,.0f}€","Budget: 15 000€","")
    with c5: kpi_card("📦","Val. stock",f"{val_stock:,.0f}€",f"{n_stock} rupture(s)","danger" if n_stock>0 else "success")
    with c6: kpi_card("🛡️","Taux préventif",f"{taux_prev}%","Obj: ≥70%","success" if taux_prev>=70 else "warn")

    st.markdown("<br>", unsafe_allow_html=True)
    hist = st.session_state.historique
    col1, col2 = st.columns([3,2])

    with col1:
        section_header("📈","Évolution des interventions & DI")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Correctifs",x=hist["Mois"],y=hist["Correctifs"],marker_color="#C62828",opacity=0.85))
        fig.add_trace(go.Bar(name="Préventifs",x=hist["Mois"],y=hist["Préventifs"],marker_color="#1565C0",opacity=0.85))
        fig.add_trace(go.Scatter(name="DI reçues",x=hist["Mois"],y=hist["DI reçues"],mode="lines+markers",
            line=dict(color="#00ccaa",width=2),marker=dict(size=7,color="#00ccaa")))
        fig.add_trace(go.Scatter(name="Disponibilité %",x=hist["Mois"],y=hist["Dispo (%)"],mode="lines",
            yaxis="y2",line=dict(color="#ffcc00",width=2,dash="dot")))
        fig.update_layout(**PLOT_LAYOUT, barmode="group", height=280,
            yaxis2=dict(overlaying="y",side="right",range=[80,100],color="#ffcc00",gridcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("📊","Statut du parc")
        sc_counts = eq["Statut"].value_counts()
        fig2 = go.Figure(go.Pie(labels=sc_counts.index, values=sc_counts.values, hole=0.6,
            marker=dict(colors=["#00cc66","#ff3355","#aa44ff","#ff8833"]),
            textfont=dict(color="#ffffff",size=11)))
        fig2.update_layout(**{**PLOT_LAYOUT, "margin":dict(l=0,r=0,t=0,b=0)}, height=280,
            annotations=[dict(text=f"{len(eq)}<br>EQ", x=0.5, y=0.5, font_size=16, font_color="#60b0ff", showarrow=False)])
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        section_header("📨","DI récentes")
        pm={"Critique":"critical","Haute":"high","Moyenne":"medium","Basse":"low"}
        sm_di={"En attente":"pending","Approuvée":"approved","Rejetée":"rejected","Clôturée":"closed"}
        for _, r in di.head(4).iterrows():
            uc = pm.get(r["Urgence"],"low"); sc = sm_di.get(r["Statut"],"pending")
            st.markdown(f"""<div class="di-card {sc}">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span class="item-id">{r['DI']}</span>
                        <div class="item-title" style="margin-top:2px;font-size:13px;">{r['Titre']}</div>
                    </div>
                    <div style="text-align:right;">
                        <span class="badge badge-{uc}">{r['Urgence']}</span><br>
                        <span class="badge badge-{sc}" style="margin-top:3px;display:inline-block;">{r['Statut']}</span>
                    </div>
                </div>
                <div class="item-meta">
                    <span>👤 {r['Demandeur']}</span><span>📅 {r['Date demande']}</span><span>🔧 {r['Équipement']}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    with col4:
        section_header("📋","BT prioritaires")
        pmap={"Critique":"critical","Haute":"high","Moyenne":"medium","Basse":"low"}
        smap={"Ouvert":"open","En cours":"progress","Terminé":"closed","Planifié":"planned"}
        for _,r in bt[bt["Statut"]!="Terminé"].sort_values("Priorité",key=lambda x: x.map({"Critique":0,"Haute":1,"Moyenne":2,"Basse":3})).head(4).iterrows():
            pc_cls = pmap.get(r["Priorité"],"low")
            st.markdown(f"""<div class="item-card {pc_cls}">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span class="item-id">{r['BT']}</span>
                        <div class="item-title" style="margin-top:2px;font-size:13px;">{r['Titre']}</div>
                    </div>
                    <div style="text-align:right;">
                        <span class="badge badge-{pc_cls}">{r['Priorité']}</span><br>
                        <span class="badge badge-{smap.get(r['Statut'],'open')}" style="margin-top:3px;display:inline-block;">{r['Statut']}</span>
                    </div>
                </div>
                <div class="item-meta">
                    <span>👷 {r['Technicien']}</span><span>📅 {r['Date prévue']}</span><span>💰 {r['Coût estimé (€)']}€</span>
                </div>
            </div>""", unsafe_allow_html=True)

# ─── DEMANDES D'INTERVENTION ─────────────────────────────────────────────────
elif page == "Demandes d'intervention":
    page_title("DEMANDES D'INTERVENTION", "Gestion et suivi des demandes")

    tab1, tab2, tab3 = st.tabs(["📋 Liste des DI", "➕ Nouvelle demande", "✅ Traitement & Approbation"])

    with tab1:
        c1,c2,c3,c4 = st.columns(4)
        with c1: f_urg = st.selectbox("Urgence", ["Tous","Critique","Haute","Moyenne","Basse"], key="di_urg")
        with c2: f_stat = st.selectbox("Statut", ["Tous","En attente","Approuvée","Rejetée","Clôturée"], key="di_stat")
        with c3: f_site = st.selectbox("Site", ["Tous"]+list(di["Site"].unique()), key="di_site")
        with c4: search = st.text_input("🔍 Rechercher", "", key="di_search")

        df_di = di.copy()
        if f_urg != "Tous": df_di = df_di[df_di["Urgence"]==f_urg]
        if f_stat != "Tous": df_di = df_di[df_di["Statut"]==f_stat]
        if f_site != "Tous": df_di = df_di[df_di["Site"]==f_site]
        if search: df_di = df_di[df_di.apply(lambda r: search.lower() in r.to_string().lower(), axis=1)]

        df_di = df_di.sort_values("Urgence", key=lambda x: x.map({"Critique":0,"Haute":1,"Moyenne":2,"Basse":3}))

        # Stats rapides
        cs1,cs2,cs3,cs4 = st.columns(4)
        with cs1: kpi_card("📨","Total DI",str(len(df_di)),"dans la sélection","")
        with cs2: kpi_card("⏳","En attente",str(len(df_di[df_di["Statut"]=="En attente"])),"À traiter","warn")
        with cs3: kpi_card("✅","Approuvées",str(len(df_di[df_di["Statut"]=="Approuvée"])),"DI traitées","success")
        with cs4: kpi_card("❌","Rejetées",str(len(df_di[df_di["Statut"]=="Rejetée"])),"","danger")

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("📋","Demandes d'intervention")

        for _, r in df_di.iterrows():
            di_card(r)

        if len(df_di) == 0:
            st.markdown('<div class="alert-info">ℹ️ Aucune demande trouvée pour les filtres sélectionnés.</div>', unsafe_allow_html=True)

    with tab2:
        section_header("➕","Créer une nouvelle demande d'intervention")
        st.markdown('<div class="form-section">', unsafe_allow_html=True)

        with st.form("form_di", clear_on_submit=True):
            c1,c2 = st.columns(2)
            with c1:
                titre_di = st.text_input("Titre de la demande *", placeholder="Ex: Bruit anormal moteur...")
                eq_options = eq["ID"] + " — " + eq["Nom"]
                eq_sel = st.selectbox("Équipement concerné *", eq_options)
                type_di = st.selectbox("Type d'intervention demandé", ["Correctif","Préventif","Prédictif","Inspection","Amélioration"])
                urgence = st.selectbox("Niveau d'urgence *", ["Critique","Haute","Moyenne","Basse"])
            with c2:
                demandeur = st.text_input("Nom du demandeur *", placeholder="Prénom Nom")
                service = st.selectbox("Service", ["Production","Maintenance","Qualité","Sécurité","Direction","Électricité","Logistique","Autre"])
                site_di = st.selectbox("Site", ["Usine A","Usine B","Usine C","Entrepôt"])
                date_souh = st.date_input("Date d'intervention souhaitée", value=date.today()+timedelta(days=2))
            desc_di = st.text_area("Description détaillée du problème *",
                placeholder="Décrivez précisément le problème observé, les symptômes, la fréquence...",
                height=120)
            impact = st.selectbox("Impact sur la production", ["Arrêt total de production","Production dégradée","Sans impact immédiat","Risque sécurité"])

            col_s1, col_s2 = st.columns([2,1])
            with col_s1:
                submitted = st.form_submit_button("📨 Soumettre la demande d'intervention", use_container_width=True)
            if submitted:
                if titre_di and demandeur and desc_di:
                    new_id = f"DI-{datetime.now().year}-{len(di)+1:03d}"
                    eq_id = eq_sel.split(" — ")[0]
                    new_row = {
                        "DI": new_id, "Titre": titre_di, "Équipement": eq_id, "Site": site_di,
                        "Urgence": urgence, "Statut": "En attente", "Demandeur": demandeur,
                        "Service": service, "Date demande": str(date.today()),
                        "Date souhaitée": str(date_souh), "Description": desc_di,
                        "BT généré": "", "Commentaire resp.": "", "Type demandé": type_di
                    }
                    st.session_state.demandes_intervention = pd.concat(
                        [di, pd.DataFrame([new_row])], ignore_index=True)
                    st.markdown(f'<div class="alert-success">✅ Demande <b>{new_id}</b> soumise avec succès! En attente de validation du responsable maintenance.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-danger">⚠️ Veuillez remplir tous les champs obligatoires (*).</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        section_header("✅","Traitement des demandes en attente")
        di_att = di[di["Statut"]=="En attente"]

        if len(di_att) == 0:
            st.markdown('<div class="alert-success">✅ Aucune demande en attente de traitement.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-warning">⚠️ {len(di_att)} demande(s) en attente de traitement.</div>', unsafe_allow_html=True)

            for _, r in di_att.iterrows():
                with st.expander(f"📨 {r['DI']} — {r['Titre']} [{r['Urgence']}]", expanded=(r["Urgence"]=="Critique")):
                    co1, co2 = st.columns([2,1])
                    with co1:
                        st.markdown(f"""
                        <div class='info-box'>
                        <b>Équipement:</b> {r['Équipement']} · <b>Demandeur:</b> {r['Demandeur']} · <b>Service:</b> {r['Service']}<br>
                        <b>Date demande:</b> {r['Date demande']} · <b>Date souhaitée:</b> {r['Date souhaitée']}<br>
                        <b>Description:</b> {r['Description']}
                        </div>""", unsafe_allow_html=True)
                    with co2:
                        action = st.selectbox("Décision", ["Approuver → Créer BT","Approuver → Planifier","Rejeter","Demander info complémentaire"], key=f"action_{r['DI']}")
                        commentaire = st.text_area("Commentaire", height=80, key=f"com_{r['DI']}", placeholder="Motif de la décision...")
                        if st.button(f"✅ Valider {r['DI']}", key=f"val_{r['DI']}"):
                            idx = st.session_state.demandes_intervention[st.session_state.demandes_intervention["DI"]==r["DI"]].index[0]
                            if "Approuver" in action:
                                st.session_state.demandes_intervention.at[idx,"Statut"] = "Approuvée"
                                if "Créer BT" in action:
                                    new_bt = f"BT-{datetime.now().year}-{len(bt)+1:03d}"
                                    new_row = {
                                        "BT":new_bt,"Équipement":r["Équipement"],"DI origine":r["DI"],
                                        "Titre":r["Titre"],"Type":r["Type demandé"],"Priorité":r["Urgence"],
                                        "Statut":"Ouvert","Demandeur":r["Demandeur"],"Technicien":"Non assigné",
                                        "Date création":str(date.today()),"Date prévue":r["Date souhaitée"],
                                        "Durée (h)":4,"Coût estimé (€)":500,"Coût réel (€)":0,
                                        "Description":r["Description"]
                                    }
                                    st.session_state.bons_travaux = pd.concat([bt, pd.DataFrame([new_row])], ignore_index=True)
                                    st.session_state.demandes_intervention.at[idx,"BT généré"] = new_bt
                                    st.markdown(f'<div class="alert-success">✅ DI approuvée · BT <b>{new_bt}</b> créé automatiquement!</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="alert-success">✅ DI {r["DI"]} approuvée pour planification.</div>', unsafe_allow_html=True)
                            elif "Rejeter" in action:
                                st.session_state.demandes_intervention.at[idx,"Statut"] = "Rejetée"
                                st.markdown(f'<div class="alert-danger">❌ DI {r["DI"]} rejetée.</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="alert-warning">ℹ️ Informations complémentaires demandées.</div>', unsafe_allow_html=True)
                            if commentaire:
                                st.session_state.demandes_intervention.at[idx,"Commentaire resp."] = commentaire

# ─── ÉQUIPEMENTS ──────────────────────────────────────────────────────────────
elif page == "Équipements":
    page_title("ÉQUIPEMENTS", "Gestion du parc machines")

    tab1, tab2, tab3 = st.tabs(["📋 Inventaire", "➕ Ajouter", "✏️ Modifier"])

    with tab1:
        c1,c2,c3,c4 = st.columns(4)
        with c1: f_site = st.selectbox("Site", ["Tous"]+list(eq["Site"].unique()))
        with c2: f_stat = st.selectbox("Statut", ["Tous"]+list(eq["Statut"].unique()))
        with c3: f_crit = st.selectbox("Criticité", ["Tous"]+list(eq["Criticité"].unique()))
        with c4: srch = st.text_input("🔍","")

        dfe = eq.copy()
        if f_site!="Tous": dfe=dfe[dfe["Site"]==f_site]
        if f_stat!="Tous": dfe=dfe[dfe["Statut"]==f_stat]
        if f_crit!="Tous": dfe=dfe[dfe["Criticité"]==f_crit]
        if srch: dfe=dfe[dfe.apply(lambda r: srch.lower() in r.to_string().lower(),axis=1)]

        pmap={"Critique":"critical","Haute":"high","Moyenne":"medium","Basse":"low"}
        smap_eq={"Opérationnel":"closed","En panne":"critical","En maintenance":"progress"}
        for _,r in dfe.iterrows():
            pc_cls = pmap.get(r["Criticité"],"low"); sc_cls = smap_eq.get(r["Statut"],"open")
            bt_count = len(bt[bt["Équipement"]==r["ID"]]); di_count = len(di[di["Équipement"]==r["ID"]])
            st.markdown(f"""<div class="item-card {pc_cls}">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span class="item-id">{r['ID']} · {r['Localisation']}</span>
                        <div class="item-title" style="margin-top:2px;">{r['Nom']}</div>
                    </div>
                    <div style="text-align:right;">
                        <span class="badge badge-{sc_cls}">{r['Statut']}</span>
                        <span class="badge badge-{pc_cls}" style="margin-left:4px;">{r['Criticité']}</span>
                    </div>
                </div>
                <div class="item-meta" style="margin-top:6px;">
                    <span>🏭 {r['Site']}</span><span>⚙️ {r['Catégorie']}</span>
                    <span>⏱ {r['Heures']:,}h</span><span>💰 {r['Valeur (€)']:,}€</span>
                    <span>👤 {r['Resp.']}</span><span>📋 {bt_count} BT</span><span>📨 {di_count} DI</span>
                    <span>📅 MNT: {r['Prochaine MNT']}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        col1,col2 = st.columns(2)
        with col1:
            fig3 = px.bar(eq.groupby(["Site","Statut"]).size().reset_index(name="n"),
                x="Site",y="n",color="Statut",title="Statut par site",
                color_discrete_map={"Opérationnel":"#00cc66","En panne":"#ff3355","En maintenance":"#aa44ff"})
            fig3.update_layout(**PLOT_LAYOUT, height=260)
            st.plotly_chart(fig3, use_container_width=True)
        with col2:
            fig4 = px.sunburst(eq, path=["Catégorie","Site"], title="Répartition par catégorie")
            fig4.update_layout(**{**PLOT_LAYOUT, "margin":dict(l=0,r=0,t=40,b=0)}, height=260)
            st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        with st.form("form_eq"):
            c1,c2 = st.columns(2)
            with c1:
                nom = st.text_input("Nom équipement *")
                cat = st.selectbox("Catégorie", ["Pneumatique","Hydraulique","Électrique","Mécanique","Robotique","Usinage","Manutention","Levage","Thermique","Autre"])
                site_eq = st.selectbox("Site", ["Usine A","Usine B","Usine C","Entrepôt"])
                crit = st.selectbox("Criticité", ["Critique","Haute","Moyenne","Basse"])
                localisation = st.text_input("Localisation", placeholder="Hall 1 - Zone A")
            with c2:
                stat_eq = st.selectbox("Statut", ["Opérationnel","En maintenance","En panne"])
                d_install = st.date_input("Date installation", value=date.today())
                heures = st.number_input("Heures initiales", min_value=0, value=0)
                valeur = st.number_input("Valeur actif (€)", min_value=0, value=10000)
                resp = st.text_input("Responsable")
            if st.form_submit_button("✅ Enregistrer l'équipement", use_container_width=True):
                if nom:
                    nid = f"EQ-{len(eq)+1:03d}"
                    new = {"ID":nid,"Nom":nom,"Catégorie":cat,"Site":site_eq,"Statut":stat_eq,"Criticité":crit,
                           "Installation":str(d_install),"Dernière MNT":"N/A","Prochaine MNT":"N/A",
                           "Heures":heures,"Valeur (€)":valeur,"Resp.":resp,"Localisation":localisation}
                    st.session_state.equipements = pd.concat([eq, pd.DataFrame([new])], ignore_index=True)
                    st.markdown(f'<div class="alert-success">✅ {nid} — {nom} ajouté!</div>', unsafe_allow_html=True)

    with tab3:
        eq_s = st.selectbox("Sélectionner", eq["ID"]+" — "+eq["Nom"])
        eid = eq_s.split(" — ")[0]
        row = eq[eq["ID"]==eid].iloc[0]
        with st.form("form_edit_eq"):
            c1,c2 = st.columns(2)
            with c1:
                ns = st.selectbox("Statut", ["Opérationnel","En maintenance","En panne"],
                    index=["Opérationnel","En maintenance","En panne"].index(row["Statut"]))
                nh = st.number_input("Heures", min_value=0, value=int(row["Heures"]))
            with c2:
                ndm = st.text_input("Dernière MNT", value=row["Dernière MNT"])
                npm = st.text_input("Prochaine MNT", value=row["Prochaine MNT"])
            if st.form_submit_button("💾 Mettre à jour", use_container_width=True):
                idx = st.session_state.equipements[st.session_state.equipements["ID"]==eid].index[0]
                st.session_state.equipements.at[idx,"Statut"]=ns
                st.session_state.equipements.at[idx,"Heures"]=nh
                st.session_state.equipements.at[idx,"Dernière MNT"]=ndm
                st.session_state.equipements.at[idx,"Prochaine MNT"]=npm
                st.markdown(f'<div class="alert-success">✅ {eid} mis à jour!</div>', unsafe_allow_html=True)

# ─── BONS DE TRAVAUX ──────────────────────────────────────────────────────────
elif page == "Bons de travaux":
    page_title("BONS DE TRAVAUX", "Suivi des interventions")

    tab1, tab2, tab3 = st.tabs(["📋 Liste", "➕ Créer BT", "✏️ Mise à jour"])

    with tab1:
        c1,c2,c3 = st.columns(3)
        with c1: ft = st.selectbox("Type", ["Tous","Correctif","Préventif","Prédictif"])
        with c2: fp = st.selectbox("Priorité", ["Tous","Critique","Haute","Moyenne","Basse"])
        with c3: fs = st.selectbox("Statut", ["Tous","Ouvert","En cours","Planifié","Terminé"])

        dfbt = bt.copy()
        if ft!="Tous": dfbt=dfbt[dfbt["Type"]==ft]
        if fp!="Tous": dfbt=dfbt[dfbt["Priorité"]==fp]
        if fs!="Tous": dfbt=dfbt[dfbt["Statut"]==fs]

        pmap={"Critique":"critical","Haute":"high","Moyenne":"medium","Basse":"low"}
        smap={"Ouvert":"open","En cours":"progress","Terminé":"closed","Planifié":"planned"}
        for _,r in dfbt.iterrows():
            di_ref = f"🔗 {r['DI origine']}" if r.get("DI origine") else ""
            st.markdown(f"""<div class="item-card {pmap.get(r['Priorité'],'low')}">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <div>
                        <span class="item-id">{r['BT']}</span>
                        {f'<span class="item-id" style="color:#00aaaa;margin-left:8px;">{di_ref}</span>' if di_ref else ''}
                        <div class="item-title" style="margin-top:2px;">{r['Titre']}</div>
                    </div>
                    <div style="text-align:right;">
                        <span class="badge badge-{pmap.get(r['Priorité'],'low')}">{r['Priorité']}</span>
                        <span class="badge badge-{smap.get(r['Statut'],'open')}" style="margin-left:4px;">{r['Statut']}</span>
                        <span class="badge" style="background:rgba(0,80,180,0.1);color:#4499ff;border:1px solid rgba(0,120,255,0.2);margin-left:4px;">{r['Type']}</span>
                    </div>
                </div>
                <div class="item-meta">
                    <span>🔧 {r['Équipement']}</span><span>👷 {r['Technicien']}</span>
                    <span>📅 {r['Date prévue']}</span><span>⏱ {r['Durée (h)']}h</span>
                    <span>💰 {r['Coût estimé (€)']}€</span><span>👤 {r['Demandeur']}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    with tab2:
        with st.form("form_bt"):
            c1,c2 = st.columns(2)
            with c1:
                titre_bt = st.text_input("Titre *")
                eq_s2 = st.selectbox("Équipement", eq["ID"]+" — "+eq["Nom"])
                type_bt = st.selectbox("Type", ["Correctif","Préventif","Prédictif","Amélioratif"])
                prio = st.selectbox("Priorité", ["Critique","Haute","Moyenne","Basse"])
                di_origine = st.selectbox("DI origine (optionnel)", ["Aucune"]+list(di["DI"]+" — "+di["Titre"]))
            with c2:
                tech_s = st.selectbox("Technicien", ["Non assigné"]+list(st.session_state.techniciens["Nom"]))
                d_prev = st.date_input("Date prévue", value=date.today()+timedelta(days=3))
                duree = st.number_input("Durée (h)", min_value=0.5, value=2.0, step=0.5)
                cout_bt = st.number_input("Coût estimé (€)", min_value=0, value=200)
                dem = st.text_input("Demandeur", value="Utilisateur")
            desc_bt = st.text_area("Description", height=70)
            if st.form_submit_button("✅ Créer BT", use_container_width=True):
                if titre_bt:
                    nbt = f"BT-{datetime.now().year}-{len(bt)+1:03d}"
                    di_ref = di_origine.split(" — ")[0] if di_origine != "Aucune" else ""
                    new = {"BT":nbt,"Équipement":eq_s2.split(" — ")[0],"DI origine":di_ref,
                           "Titre":titre_bt,"Type":type_bt,"Priorité":prio,"Statut":"Ouvert",
                           "Demandeur":dem,"Technicien":tech_s,"Date création":str(date.today()),
                           "Date prévue":str(d_prev),"Durée (h)":duree,"Coût estimé (€)":cout_bt,
                           "Coût réel (€)":0,"Description":desc_bt}
                    st.session_state.bons_travaux = pd.concat([bt, pd.DataFrame([new])], ignore_index=True)
                    st.markdown(f'<div class="alert-success">✅ {nbt} créé!</div>', unsafe_allow_html=True)

    with tab3:
        bt_s = st.selectbox("Sélectionner BT", bt["BT"]+" — "+bt["Titre"])
        bid = bt_s.split(" — ")[0]
        brow = bt[bt["BT"]==bid].iloc[0]
        with st.form("form_upd_bt"):
            c1,c2 = st.columns(2)
            with c1:
                slist = ["Ouvert","En cours","Planifié","Terminé"]
                cur_s = brow["Statut"] if brow["Statut"] in slist else "Ouvert"
                ns_bt = st.selectbox("Statut", slist, index=slist.index(cur_s))
                nt_bt = st.selectbox("Technicien", ["Non assigné"]+list(st.session_state.techniciens["Nom"]))
            with c2:
                cr_bt = st.number_input("Coût réel (€)", min_value=0, value=int(brow["Coût réel (€)"]))
                notes_bt = st.text_area("Notes intervention", height=70)
            if st.form_submit_button("💾 Mettre à jour", use_container_width=True):
                idx = st.session_state.bons_travaux[st.session_state.bons_travaux["BT"]==bid].index[0]
                st.session_state.bons_travaux.at[idx,"Statut"]=ns_bt
                st.session_state.bons_travaux.at[idx,"Technicien"]=nt_bt
                st.session_state.bons_travaux.at[idx,"Coût réel (€)"]=cr_bt
                st.markdown(f'<div class="alert-success">✅ {bid} → {ns_bt}</div>', unsafe_allow_html=True)

# ─── STOCK & PIÈCES ───────────────────────────────────────────────────────────
elif page == "Stock & Pièces":
    page_title("STOCK & PIÈCES", "Gestion de l'inventaire")

    for _,r in pc[pc["Stock"]<pc["Min"]].iterrows():
        st.markdown(f'<div class="alert-danger">⚠️ <b>{r["Désignation"]}</b> ({r["Réf"]}) — Stock: <b>{r["Stock"]}</b> / Min: {r["Min"]} · {r["Fournisseur"]} · Délai: {r["Délai (j)"]}j</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📦 Inventaire", "➕ Ajouter", "📊 Analyse"])

    with tab1:
        srch_p = st.text_input("🔍 Rechercher une pièce", "")
        dfp = pc.copy()
        if srch_p: dfp = dfp[dfp.apply(lambda r: srch_p.lower() in r.to_string().lower(), axis=1)]

        for _,r in dfp.iterrows():
            ratio = r["Stock"]/r["Max"] if r["Max"]>0 else 0
            if r["Stock"]<r["Min"]: st_cls="danger"; st_txt="RUPTURE"; bar_c="#ff3355"
            elif r["Stock"]<r["Min"]*1.5: st_cls="warning"; st_txt="FAIBLE"; bar_c="#ff8833"
            else: st_cls="success"; st_txt="OK"; bar_c="#00cc66"
            val = r["Stock"]*r["Prix (€)"]
            st.markdown(f"""<div class="item-card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <div>
                        <span class="item-id">{r['Réf']} · {r['Emplacement']}</span>
                        <div class="item-title" style="margin-top:2px;">{r['Désignation']}</div>
                    </div>
                    <div style="text-align:right;">
                        <span class="badge badge-{st_cls}">{st_txt}</span>
                        <span class="badge" style="background:rgba(0,80,180,0.1);color:#4499ff;border:1px solid rgba(0,120,255,0.2);margin-left:4px;">{r['Catégorie']}</span>
                    </div>
                </div>
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                    <div style="flex:1;background:#0a1525;border-radius:4px;height:5px;overflow:hidden;">
                        <div style="width:{min(ratio*100,100):.0f}%;height:100%;background:{bar_c};border-radius:4px;"></div>
                    </div>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#4a7aaa;">
                        {r['Stock']} / Min:{r['Min']} / Max:{r['Max']}
                    </span>
                </div>
                <div class="item-meta">
                    <span>💰 {r['Prix (€)']:.2f}€/u</span>
                    <span>📦 Valeur: {val:.0f}€</span>
                    <span>🏭 {r['Fournisseur']}</span>
                    <span>⏱ Délai: {r['Délai (j)']}j</span>
                </div>
            </div>""", unsafe_allow_html=True)

    with tab2:
        with st.form("form_piece"):
            c1,c2 = st.columns(2)
            with c1:
                ref = st.text_input("Référence *")
                desig = st.text_input("Désignation *")
                cat_p = st.selectbox("Catégorie", ["Roulements","Transmission","Étanchéité","Lubrifiants","Filtration","Électrique","Instrumentation","Visserie","Autre"])
                fourn = st.text_input("Fournisseur")
                empl = st.text_input("Emplacement")
            with c2:
                si = st.number_input("Stock initial", min_value=0, value=10)
                smin = st.number_input("Stock minimum", min_value=0, value=3)
                smax = st.number_input("Stock maximum", min_value=0, value=30)
                prix_p = st.number_input("Prix unitaire (€)", min_value=0.0, value=10.0, step=0.5)
                delai_p = st.number_input("Délai livraison (j)", min_value=1, value=5)
            if st.form_submit_button("✅ Ajouter la pièce", use_container_width=True):
                if ref and desig:
                    new = {"Réf":ref,"Désignation":desig,"Catégorie":cat_p,"Stock":si,"Min":smin,"Max":smax,
                           "Prix (€)":prix_p,"Fournisseur":fourn,"Délai (j)":delai_p,"Emplacement":empl}
                    st.session_state.pieces = pd.concat([pc, pd.DataFrame([new])], ignore_index=True)
                    st.markdown(f'<div class="alert-success">✅ {ref} — {desig} ajouté!</div>', unsafe_allow_html=True)

    with tab3:
        col1,col2 = st.columns(2)
        with col1:
            fig_s = go.Figure()
            fig_s.add_trace(go.Bar(name="Stock", x=pc["Désignation"].str[:15], y=pc["Stock"], marker_color="#1565C0"))
            fig_s.add_trace(go.Scatter(name="Min", x=pc["Désignation"].str[:15], y=pc["Min"], mode="lines+markers", line=dict(color="#ff3355",dash="dash")))
            fig_s.add_trace(go.Scatter(name="Max", x=pc["Désignation"].str[:15], y=pc["Max"], mode="lines", line=dict(color="#00cc66",dash="dot")))
            fig_s.update_layout(**PLOT_LAYOUT, title="Niveaux de stock", height=280)
            st.plotly_chart(fig_s, use_container_width=True)
        with col2:
            val_cat = (pc["Stock"]*pc["Prix (€)"]).groupby(pc["Catégorie"]).sum().reset_index()
            val_cat.columns=["Catégorie","Valeur (€)"]
            fig_v = px.pie(val_cat, names="Catégorie", values="Valeur (€)", hole=0.45, title="Valeur par catégorie")
            fig_v.update_layout(**{**PLOT_LAYOUT, "margin":dict(l=0,r=0,t=40,b=0)}, height=280)
            st.plotly_chart(fig_v, use_container_width=True)

# ─── TECHNICIENS ──────────────────────────────────────────────────────────────
elif page == "Techniciens":
    page_title("TECHNICIENS", "Gestion des ressources humaines")

    tech = st.session_state.techniciens
    cols = st.columns(len(tech))
    for i,(_, r) in enumerate(tech.iterrows()):
        with cols[i]:
            dc = "#00cc66" if r["Disponible"] else "#ff3355"
            ec = "#00cc66" if r["Efficacité (%)"]>=90 else "#ff8833" if r["Efficacité (%)"]>=80 else "#ff3355"
            charge_pct = min(r["BT en cours"]*25, 100)
            st.markdown(f"""<div style='background:linear-gradient(135deg,#0a1525,#08111e);border:1px solid #0d2540;border-radius:14px;padding:18px;text-align:center;'>
                <div style='width:58px;height:58px;border-radius:50%;background:linear-gradient(135deg,#0a3060,#1565C0);display:flex;align-items:center;justify-content:center;margin:0 auto 10px;font-size:24px;border:2px solid #1565C0;'>👷</div>
                <div style='font-family:Exo 2,sans-serif;font-size:15px;font-weight:800;color:#c8e0ff;letter-spacing:1px;'>{r['Nom']}</div>
                <div style='font-size:10px;color:#1a4a7a;margin:4px 0;letter-spacing:1px;'>{r['Spécialité']}</div>
                <div style='margin:8px 0;'>
                    <span style='display:inline-block;width:7px;height:7px;border-radius:50%;background:{dc};margin-right:4px;box-shadow:0 0 6px {dc};'></span>
                    <span style='font-size:11px;color:{dc};font-weight:700;'>{'DISPO' if r['Disponible'] else 'OCCUPÉ'}</span>
                </div>
                <div style='background:#0a1525;border-radius:4px;height:4px;margin:8px 0;overflow:hidden;'>
                    <div style='width:{charge_pct}%;height:100%;background:linear-gradient(90deg,#1565C0,#00aaff);border-radius:4px;'></div>
                </div>
                <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin-top:8px;'>
                    <div style='background:#0a1525;border-radius:6px;padding:6px 2px;'>
                        <div style='font-family:Exo 2;font-size:18px;font-weight:800;color:#4499ff;'>{r['BT en cours']}</div>
                        <div style='font-size:8px;color:#0d2540;letter-spacing:1px;'>BT</div>
                    </div>
                    <div style='background:#0a1525;border-radius:6px;padding:6px 2px;'>
                        <div style='font-family:Exo 2;font-size:18px;font-weight:800;color:{ec};'>{r['Efficacité (%)']}%</div>
                        <div style='font-size:8px;color:#0d2540;letter-spacing:1px;'>EFF.</div>
                    </div>
                    <div style='background:#0a1525;border-radius:6px;padding:6px 2px;'>
                        <div style='font-family:Exo 2;font-size:18px;font-weight:800;color:#00cc66;'>{r['€/h']}€</div>
                        <div style='font-size:8px;color:#0d2540;letter-spacing:1px;'>/H</div>
                    </div>
                </div>
                <div style='font-size:9px;color:#0d2540;margin-top:8px;'>{r['Certifications']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        tech_bt = bt[bt["Statut"].isin(["En cours","Planifié"])].groupby("Technicien").agg({"BT":"count","Durée (h)":"sum"}).reset_index()
        if len(tech_bt)>0:
            fig_t = go.Figure()
            fig_t.add_trace(go.Bar(name="Nb BT",x=tech_bt["Technicien"],y=tech_bt["BT"],marker_color="#1565C0"))
            fig_t.add_trace(go.Bar(name="Heures estimées",x=tech_bt["Technicien"],y=tech_bt["Durée (h)"],marker_color="#00cc66"))
            fig_t.update_layout(**PLOT_LAYOUT, barmode="group", title="Charge de travail", height=280)
            st.plotly_chart(fig_t, use_container_width=True)
    with col2:
        fig_eff = go.Figure(go.Bar(
            x=tech["Efficacité (%)"], y=tech["Nom"], orientation="h",
            marker=dict(color=tech["Efficacité (%)"],colorscale=[[0,"#ff3355"],[0.5,"#ff8833"],[1,"#00cc66"]]),
            text=tech["Efficacité (%)"].astype(str)+"%", textposition="outside",
            textfont=dict(color="#4a7aaa")
        ))
        fig_eff.update_layout(**PLOT_LAYOUT, title="Taux d'efficacité", height=280,
            xaxis=dict(**PLOT_LAYOUT["xaxis"], range=[0,105]))
        st.plotly_chart(fig_eff, use_container_width=True)

# ─── PLANNING ─────────────────────────────────────────────────────────────────
elif page == "Planning":
    page_title("PLANNING MAINTENANCE", "Calendrier et Gantt des interventions")

    bt_p = bt[bt["Statut"]!="Terminé"].copy()
    bt_p["Date prévue"] = pd.to_datetime(bt_p["Date prévue"])
    bt_p["Fin prévue"] = bt_p["Date prévue"] + pd.to_timedelta(bt_p["Durée (h)"], unit="h")

    color_map = {"Critique":"#ff3355","Haute":"#ff8833","Moyenne":"#1565C0","Basse":"#00cc66"}
    
    # Create Gantt chart
    fig_g = go.Figure()
    
    # Sort by date for better visualization
    bt_p_sorted = bt_p.sort_values("Date prévue")
    
    for _, r in bt_p_sorted.iterrows():
        color = color_map.get(r["Priorité"], "#4488ff")
        
        # Calculate duration in days (minimum 0.3 days for visibility)
        if pd.notna(r["Date prévue"]) and pd.notna(r["Fin prévue"]):
            duration = max((r["Fin prévue"] - r["Date prévue"]).total_seconds() / 86400, 0.3)
            
            fig_g.add_trace(go.Bar(
                name=r['BT'],
                y=[f"{r['BT']} | {r['Titre'][:26]}"],
                x=[duration],
                base=r["Date prévue"],
                orientation="h",
                marker=dict(color=color, opacity=0.85, line=dict(color=color, width=1)),
                hovertemplate=f"<b>{r['BT']}</b><br>{r['Titre']}<br>👷 {r['Technicien']}<br>Priorité: {r['Priorité']}<br>Début: %{{base|%d %b %Y}}<br>Durée: {r['Durée (h)']}h<extra></extra>",
                showlegend=False
            ))
    
    # Update layout with proper configuration
    fig_g.update_layout(
        barmode="overlay",
        height=400,
        title=dict(
            text="Diagramme de Gantt — Interventions planifiées",
            font=dict(color="#4a7aaa", family="Exo 2")
        ),
        xaxis=dict(
            title="Date",
            tickformat="%d %b",
            gridcolor="#0d2540",
            color="#2a4a6a",
            linecolor="#0d2540",
            zerolinecolor="#0d2540"
        ),
        yaxis=dict(
            title="",
            gridcolor="#0d2540",
            color="#2a4a6a",
            linecolor="#0d2540"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#4a7aaa", family="Exo 2"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig_g, use_container_width=True)

    section_header("📅","Prochaines maintenances préventives")
    for _,r in eq.sort_values("Prochaine MNT").head(8).iterrows():
        try:
            nd = pd.to_datetime(r["Prochaine MNT"])
            delta = (nd - pd.Timestamp.now()).days
            color_class = "#ff3355" if delta<7 else "#ff8833" if delta<30 else "#00cc66"
            icon = "🚨" if delta<7 else "⚠️" if delta<30 else "📅"
            st.markdown(f"""<div class="item-card" style="padding:10px 14px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span>{icon} <span class="item-title" style="font-size:13px;">{r['Nom']}</span>
                    <span class="item-id" style="margin-left:8px;">· {r['ID']} · {r['Site']}</span></span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:13px;color:{color_class};font-weight:700;">
                        {r['Prochaine MNT']} <span style="font-size:11px;font-weight:400;color:#1a3555;">({delta}j)</span>
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)
        except:
            pass

# ─── KPIs & ANALYSES ──────────────────────────────────────────────────────────
elif page == "KPIs & Analyses":
    page_title("KPIs & ANALYSES", "Indicateurs de performance")

    hist = st.session_state.historique
    taux_dispo = round(len(eq[eq["Statut"]=="Opérationnel"])/len(eq)*100,1)
    taux_prev = round(len(bt[bt["Type"]=="Préventif"])/len(bt)*100,1)
    taux_di_app = round(len(di[di["Statut"]=="Approuvée"])/len(di)*100,1)

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi_card("📊","Disponibilité",f"{taux_dispo}%","Obj: ≥95%","success" if taux_dispo>=95 else "warn")
    with c2: kpi_card("🛡️","Taux préventif",f"{taux_prev}%","Obj: ≥70%","success" if taux_prev>=70 else "warn")
    with c3: kpi_card("📨","Taux approbation DI",f"{taux_di_app}%","DI approuvées","success" if taux_di_app>=70 else "warn")
    with c4: kpi_card("⚡","MTBF moyen","720h","Obj: ≥700h","success")

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        fig=px.area(hist,x="Mois",y="Dispo (%)",title="Disponibilité (%)",color_discrete_sequence=["#00cc66"])
        fig.update_layout(**PLOT_LAYOUT,height=240); st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig2=px.bar(hist,x="Mois",y="Coût (€)",title="Coût mensuel (€)",color_discrete_sequence=["#1565C0"])
        fig2.update_layout(**PLOT_LAYOUT,height=240); st.plotly_chart(fig2,use_container_width=True)
    with c3:
        fig3=go.Figure()
        fig3.add_trace(go.Scatter(name="DI reçues",x=hist["Mois"],y=hist["DI reçues"],mode="lines+markers",line=dict(color="#00ccaa",width=2)))
        fig3.add_trace(go.Scatter(name="DI approuvées",x=hist["Mois"],y=hist["DI approuvées"],mode="lines+markers",line=dict(color="#4499ff",width=2,dash="dot")))
        fig3.update_layout(**PLOT_LAYOUT,title="Évolution DI",height=240); st.plotly_chart(fig3,use_container_width=True)

    section_header("📋","Synthèse mensuelle")
    synth = pd.DataFrame({
        "Mois":["Août","Sept","Oct","Nov","Déc","Jan 2025"],
        "BT Ouverts":[22,27,23,18,20,7],"BT Terminés":[20,24,22,17,19,3],
        "DI reçues":[12,15,9,8,11,8],"DI approuvées":[10,13,8,7,10,6],
        "Coût (€)":[8400,12500,7800,6200,9100,7145],
        "Taux réalisation (%)":[91,89,96,94,95,43],"Disponibilité (%)":[91,88,93,95,92,87.5],
    })
    st.dataframe(synth, use_container_width=True, hide_index=True)

    c_e1,c_e2,c_e3 = st.columns(3)
    with c_e1: st.download_button("⬇️ Export BT CSV", bt.to_csv(index=False).encode(), "bons_travaux.csv", "text/csv")
    with c_e2: st.download_button("⬇️ Export DI CSV", di.to_csv(index=False).encode(), "demandes_intervention.csv", "text/csv")
    with c_e3: st.download_button("⬇️ Export Équip. CSV", eq.to_csv(index=False).encode(), "equipements.csv", "text/csv")

# ─── CENTRE PDF ───────────────────────────────────────────────────────────────
elif page == "Centre PDF":
    page_title("CENTRE PDF", "Génération et export de documents professionnels")

    tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
        "📊 Rapport mensuel","📨 Demandes DI","📋 Bon de travaux",
        "🔧 Fiche équipement","📦 Rapport stock","📅 Planning PDF","🔗 Pack & Fusion"
    ])

    with tab1:
        section_header("📊","Rapport mensuel complet")
        st.markdown('<div class="alert-info">Rapport multi-pages: KPIs · Équipements · BT · DI · Stock · Évolution historique</div>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            mois_s = st.selectbox("Période", ["Janvier 2025","Décembre 2024","Novembre 2024"])
            sites_s = st.multiselect("Sites", ["Usine A","Usine B","Usine C"], default=["Usine A","Usine B","Usine C"])
        with c2:
            st.markdown('<div class="alert-info" style="margin-top:26px;">Contenu: KPIs exécutifs · État équipements · Suivi BT · Demandes DI · Stock · Recommandations</div>', unsafe_allow_html=True)
        if st.button("🔴 Générer rapport mensuel", use_container_width=True, key="btn_rapport"):
            with st.spinner("Génération en cours..."):
                try:
                    buf = pdf_rapport_mensuel()
                    pdf_button(buf, f"rapport_mensuel_{mois_s.replace(' ','_')}.pdf", f"Télécharger rapport {mois_s}")
                    st.markdown('<div class="alert-success">✅ Rapport généré! Cliquez pour télécharger.</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="alert-danger">Erreur: {e}</div>', unsafe_allow_html=True)

    with tab2:
        section_header("📨","Rapport Demandes d'Intervention")
        st.markdown('<div class="alert-info">Rapport DI: KPIs approbation · Liste complète · Détail DI en attente · Analyse urgence</div>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            di_pdf_sel = st.selectbox("Ou DI individuelle:", ["Toutes les DI"]+list(di["DI"]+" — "+di["Titre"]))
        with c2:
            st.markdown(f'<div class="stat-pill" style="margin-top:28px;">📨 {len(di)} DI · {len(di[di["Statut"]=="En attente"])} en attente</div>', unsafe_allow_html=True)

        c_b1, c_b2 = st.columns(2)
        with c_b1:
            if st.button("🔴 Rapport toutes les DI", use_container_width=True, key="btn_di_all"):
                with st.spinner("Génération..."):
                    try:
                        buf = pdf_rapport_di()
                        pdf_button(buf, "rapport_demandes_intervention.pdf", "Télécharger rapport DI")
                        st.markdown('<div class="alert-success">✅ Rapport DI généré!</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="alert-danger">Erreur: {e}</div>', unsafe_allow_html=True)
        with c_b2:
            if st.button("🔴 DI individuelle PDF", use_container_width=True, key="btn_di_ind"):
                if di_pdf_sel != "Toutes les DI":
                    di_id_sel = di_pdf_sel.split(" — ")[0]
                    with st.spinner("Génération..."):
                        try:
                            buf = pdf_demande_intervention(di_id_sel)
                            pdf_button(buf, f"DI_{di_id_sel}.pdf", f"Télécharger {di_id_sel}")
                            st.markdown('<div class="alert-success">✅ DI générée!</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.markdown(f'<div class="alert-danger">Erreur: {e}</div>', unsafe_allow_html=True)

        if st.button("🔴 Générer TOUTES les DI individuelles (fusionnées)", use_container_width=True, key="btn_di_all_ind"):
            with st.spinner("Génération de toutes les DI..."):
                try:
                    bufs = [pdf_demande_intervention(r["DI"]) for _, r in di.iterrows()]
                    merged = merge_pdfs(bufs)
                    pdf_button(merged, "toutes_DI.pdf", f"Télécharger {len(bufs)} DI fusionnées")
                    st.markdown(f'<div class="alert-success">✅ {len(bufs)} DI générées et fusionnées!</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="alert-danger">Erreur: {e}</div>', unsafe_allow_html=True)

    with tab3:
        section_header("📋","Bon de travaux")
        st.markdown('<div class="alert-info">PDF complet: En-tête priorité · Checklist sécurité · Rapport intervention · Signatures</div>', unsafe_allow_html=True)
        bt_opts = bt["BT"]+" — "+bt["Titre"]+" ["+bt["Priorité"]+"]"
        bt_pdf_sel = st.selectbox("Sélectionner le BT", bt_opts)
        bt_id_sel = bt_pdf_sel.split(" — ")[0]
        c_b1,c_b2 = st.columns(2)
        with c_b1:
            if st.button("🔴 Générer ce BT", use_container_width=True, key="btn_bt_ind"):
                with st.spinner("Génération..."):
                    try:
                        buf = pdf_bon_travaux(bt_id_sel)
                        pdf_button(buf, f"BT_{bt_id_sel}.pdf", f"Télécharger {bt_id_sel}")
                        st.markdown('<div class="alert-success">✅ BT généré!</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="alert-danger">Erreur: {e}</div>', unsafe_allow_html=True)
        with c_b2:
            if st.button("🔴 Tous les BT actifs (fusionnés)", use_container_width=True, key="btn_bt_all"):
                with st.spinner("Génération..."):
                    try:
                        bufs = [pdf_bon_travaux(r["BT"]) for _,r in bt[bt["Statut"]!="Terminé"].iterrows()]
                        merged = merge_pdfs(bufs)
                        pdf_button(merged, "tous_BT_actifs.pdf", f"Télécharger {len(bufs)} BT")
                        st.markdown(f'<div class="alert-success">✅ {len(bufs)} BT générés!</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="alert-danger">Erreur: {e}</div>', unsafe_allow_html=True)

    with tab4:
        section_header("🔧","Fiche équipement")
        st.markdown('<div class="alert-info">Fiche complète: Données techniques · KPIs · Historique BT & DI · Plan maintenance préventive</div>', unsafe_allow_html=True)
        eq_pdf_opts = eq["ID"]+" — "+eq["Nom"]+" ["+eq["Statut"]+"]"
        eq_pdf_sel = st.selectbox("Sélectionner équipement", eq_pdf_opts)
        eq_id_sel = eq_pdf_sel.split(" — ")[0]
        c_b1,c_b2 = st.columns(2)
        with c_b1:
            if st.button("🔴 Générer cette fiche", use_container_width=True, key="btn_eq_ind"):
                with st.spinner("Génération..."):
                    try:
                        buf = pdf_fiche_equipement(eq_id_sel)
                        pdf_button(buf, f"fiche_{eq_id_sel}.pdf", f"Télécharger fiche {eq_id_sel}")
                        st.markdown('<div class="alert-success">✅ Fiche générée!</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="alert-danger">Erreur: {e}</div>', unsafe_allow_html=True)
        with c_b2:
            if st.button("🔴 Toutes les fiches (fusionnées)", use_container_width=True, key="btn_eq_all"):
                with st.spinner("Génération..."):
                    try:
                        bufs = [pdf_fiche_equipement(r["ID"]) for _,r in eq.iterrows()]
                        merged = merge_pdfs(bufs)
                        pdf_button(merged, "toutes_fiches_equipements.pdf", f"Télécharger {len(bufs)} fiches")
                        st.markdown(f'<div class="alert-success">✅ {len(bufs)} fiches générées!</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="alert-danger">Erreur: {e}</div>', unsafe_allow_html=True)

    with tab5:
        section_header("📦","Rapport stock & inventaire")
        st.markdown('<div class="alert-info">Inventaire complet · Alertes ruptures · Bons de commande suggérés · Valeur stock</div>', unsafe_allow_html=True)
        v = (pc["Stock"]*pc["Prix (€)"]).sum(); nr = len(pc[pc["Stock"]<pc["Min"]])
        st.markdown(f'<div class="stat-pill">📦 {len(pc)} références · 💰 {v:,.2f}€ · ⚠️ {nr} rupture(s)</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔴 Générer rapport stock", use_container_width=True, key="btn_stock"):
            with st.spinner("Génération..."):
                try:
                    buf = pdf_rapport_stock()
                    pdf_button(buf, "rapport_stock.pdf", "Télécharger rapport stock")
                    st.markdown('<div class="alert-success">✅ Rapport stock généré!</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="alert-danger">Erreur: {e}</div>', unsafe_allow_html=True)

    with tab6:
        section_header("📅","Planning (A4 paysage)")
        st.markdown('<div class="alert-info">Format paysage · Planning interventions triées par priorité · Charge techniciens</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-pill">📋 {len(bt[bt["Statut"]!="Terminé"])} interventions actives · 👷 {len(st.session_state.techniciens)} techniciens</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔴 Générer planning PDF", use_container_width=True, key="btn_planning"):
            with st.spinner("Génération..."):
                try:
                    buf = pdf_planning()
                    pdf_button(buf, "planning_interventions.pdf", "Télécharger planning")
                    st.markdown('<div class="alert-success">✅ Planning généré!</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="alert-danger">Erreur: {e}</div>', unsafe_allow_html=True)

    with tab7:
        section_header("🔗","Pack complet & Fusion personnalisée")

        col_pack, col_custom = st.columns(2)

        with col_pack:
            st.markdown('<div class="alert-info"><b>Pack complet:</b> Rapport mensuel + Rapport DI + Planning + Stock + Toutes fiches + Tous BT actifs + Toutes DI</div>', unsafe_allow_html=True)
            if st.button("🔴 GÉNÉRER PACK COMPLET", use_container_width=True, key="btn_pack"):
                progress = st.progress(0, "Démarrage...")
                try:
                    all_bufs = []
                    progress.progress(10,"Rapport mensuel...")
                    all_bufs.append(pdf_rapport_mensuel())
                    progress.progress(22,"Rapport DI...")
                    all_bufs.append(pdf_rapport_di())
                    progress.progress(34,"Planning...")
                    all_bufs.append(pdf_planning())
                    progress.progress(44,"Rapport stock...")
                    all_bufs.append(pdf_rapport_stock())
                    progress.progress(55,"Fiches équipements...")
                    for _,r in eq.iterrows(): all_bufs.append(pdf_fiche_equipement(r["ID"]))
                    progress.progress(72,"Bons de travaux actifs...")
                    for _,r in bt[bt["Statut"]!="Terminé"].iterrows(): all_bufs.append(pdf_bon_travaux(r["BT"]))
                    progress.progress(86,"Demandes DI...")
                    for _,r in di.iterrows(): all_bufs.append(pdf_demande_intervention(r["DI"]))
                    progress.progress(96,"Fusion PDF...")
                    merged = merge_pdfs(all_bufs)
                    progress.progress(100,"✅ Terminé!")
                    pdf_button(merged, f"GMAO_Pack_Complet_{datetime.now().strftime('%Y%m%d')}.pdf",
                               f"Télécharger Pack ({len(all_bufs)} documents)")
                    st.markdown(f'<div class="alert-success">✅ Pack complet: {len(all_bufs)} documents fusionnés!</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="alert-danger">Erreur: {e}</div>', unsafe_allow_html=True)

        with col_custom:
            st.markdown('<div class="alert-info"><b>Sélection personnalisée:</b> Choisissez les documents à fusionner</div>', unsafe_allow_html=True)
            docs_custom = st.multiselect("Documents standards", ["Rapport mensuel","Rapport DI","Planning","Rapport stock"], default=["Rapport mensuel"])
            eq_custom = st.multiselect("Fiches équipements", list(eq["ID"]+" — "+eq["Nom"].str[:25]), default=[])
            bt_custom = st.multiselect("Bons de travaux", list(bt["BT"]+" — "+bt["Titre"].str[:25]), default=[])
            di_custom = st.multiselect("Demandes DI", list(di["DI"]+" — "+di["Titre"].str[:25]), default=[])

            if st.button("🔴 Fusionner la sélection", use_container_width=True, key="btn_custom"):
                with st.spinner("Génération..."):
                    try:
                        bufs = []
                        if "Rapport mensuel" in docs_custom: bufs.append(pdf_rapport_mensuel())
                        if "Rapport DI" in docs_custom: bufs.append(pdf_rapport_di())
                        if "Planning" in docs_custom: bufs.append(pdf_planning())
                        if "Rapport stock" in docs_custom: bufs.append(pdf_rapport_stock())
                        for e_s in eq_custom: bufs.append(pdf_fiche_equipement(e_s.split(" — ")[0]))
                        for b_s in bt_custom: bufs.append(pdf_bon_travaux(b_s.split(" — ")[0]))
                        for d_s in di_custom: bufs.append(pdf_demande_intervention(d_s.split(" — ")[0]))
                        if bufs:
                            merged = merge_pdfs(bufs)
                            pdf_button(merged, f"GMAO_selection_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                       f"Télécharger sélection ({len(bufs)} docs)")
                            st.markdown(f'<div class="alert-success">✅ {len(bufs)} documents fusionnés!</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="alert-warning">⚠️ Sélectionnez au moins un document.</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="alert-danger">Erreur: {e}</div>', unsafe_allow_html=True)

# ─── PARAMÈTRES ───────────────────────────────────────────────────────────────
elif page == "Paramètres":
    page_title("PARAMÈTRES", "Configuration de l'application")

    tab1, tab2 = st.tabs(["⚙️ Configuration", "🗄️ Données & Export"])

    with tab1:
        section_header("🏢","Configuration entreprise")
        c1,c2 = st.columns(2)
        with c1:
            st.text_input("Nom de l'entreprise", value="Industrie Maroc SA")
            st.text_input("Email notifications", value="maintenance@industrie.ma")
            st.selectbox("Devise", ["MAD (Dirham)","EUR (Euro)","USD (Dollar)"])
            st.selectbox("Langue", ["Français","Arabe","Anglais"])
        with c2:
            st.number_input("Seuil alerte stock (%)", value=20, min_value=5, max_value=50)
            st.number_input("Rappel maintenance (jours avant)", value=7, min_value=1, max_value=30)
            st.number_input("MTBF cible (heures)", value=700, min_value=100)
            st.number_input("Budget maintenance mensuel (€)", value=15000, min_value=1000)

        section_header("👥","Utilisateurs & Accès")
        users = pd.DataFrame([
            {"Utilisateur":"admin@industrie.ma","Rôle":"Administrateur","Accès":"Complet"},
            {"Utilisateur":"chef.maint@industrie.ma","Rôle":"Responsable Maintenance","Accès":"BT + DI + PDF + Rapports"},
            {"Utilisateur":"tech@industrie.ma","Rôle":"Technicien","Accès":"BT + DI lecture"},
            {"Utilisateur":"prod@industrie.ma","Rôle":"Production","Accès":"Saisie DI uniquement"},
        ])
        st.dataframe(users, use_container_width=True, hide_index=True)

    with tab2:
        section_header("🗄️","Gestion des données")
        c1,c2,c3 = st.columns(3)
        with c1:
            if st.button("🔄 Réinitialiser données démo", use_container_width=True):
                for k in ["equipements","bons_travaux","pieces","techniciens","demandes_intervention","historique"]:
                    if k in st.session_state: del st.session_state[k]
                init_data()
                st.markdown('<div class="alert-success">✅ Données réinitialisées!</div>', unsafe_allow_html=True)
        with c2:
            all_data = {"equipements":eq.to_dict(),"bons_travaux":bt.to_dict(),"pieces":pc.to_dict(),"demandes":di.to_dict()}
            st.download_button("⬇️ Export JSON complet", json.dumps(all_data,ensure_ascii=False,indent=2).encode(), "gmao_export.json","application/json", use_container_width=True)
        with c3:
            st.download_button("⬇️ Export DI (CSV)", di.to_csv(index=False).encode(), "demandes_intervention.csv","text/csv", use_container_width=True)

        st.markdown("""<br><div class="alert-info">
            <b style='font-family:Exo 2;letter-spacing:1px;'>GMAO Pro+ v3.0</b> — Application industrielle avancée de gestion de maintenance<br>
            <span style='font-size:11px;'>Python 3.9+ · Streamlit · Plotly · ReportLab · pypdf · Exo 2 Font · JetBrains Mono</span>
        </div>""", unsafe_allow_html=True)
