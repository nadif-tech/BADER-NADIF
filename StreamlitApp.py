import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import io
import base64

# ─── PDF IMPORTS ────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, HRFlowable, KeepTogether)
from reportlab.platypus import Frame, PageTemplate, BaseDocTemplate
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from pypdf import PdfReader, PdfWriter
import tempfile
import os

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(page_title="GMAO Pro+", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#080d1a;color:#dde8f5;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1220 0%,#0e1828 100%);border-right:1px solid #1a2d48;}
h1,h2,h3{font-family:'Rajdhani',sans-serif!important;font-weight:700!important;letter-spacing:1.5px;}
.kpi-card{background:linear-gradient(135deg,#0e1828,#0a1420);border:1px solid #1a3355;border-radius:14px;padding:22px 26px;position:relative;overflow:hidden;}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#00b4ff,#0055ff);}
.kpi-card.warn::before{background:linear-gradient(90deg,#ff9500,#ff4400);}
.kpi-card.danger::before{background:linear-gradient(90deg,#ff2244,#aa0033);}
.kpi-card.success::before{background:linear-gradient(90deg,#00e67a,#00aa55);}
.kpi-card.purple::before{background:linear-gradient(90deg,#aa55ff,#6633cc);}
.kpi-label{font-family:'Rajdhani',sans-serif;font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#4a7aaa;margin-bottom:8px;}
.kpi-value{font-family:'Rajdhani',sans-serif;font-size:38px;font-weight:700;color:#e8f2ff;line-height:1;}
.kpi-sub{font-size:11px;color:#3a5a7a;margin-top:6px;}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;font-family:'Rajdhani',sans-serif;letter-spacing:1px;text-transform:uppercase;}
.badge-critical{background:#3a0010;color:#ff4060;border:1px solid #ff406040;}
.badge-high{background:#3a1a00;color:#ff8800;border:1px solid #ff880040;}
.badge-medium{background:#2a2a00;color:#ffcc00;border:1px solid #ffcc0040;}
.badge-low{background:#003a1a;color:#00cc66;border:1px solid #00cc6640;}
.badge-open{background:#003060;color:#00aaff;border:1px solid #00aaff40;}
.badge-progress{background:#2a1a3a;color:#aa66ff;border:1px solid #aa66ff40;}
.badge-closed{background:#0a1a0a;color:#44aa44;border:1px solid #44aa4440;}
.badge-planned{background:#001a3a;color:#4488ff;border:1px solid #4488ff40;}
.section-title{font-family:'Rajdhani',sans-serif;font-size:20px;font-weight:700;color:#90b8e0;letter-spacing:2px;text-transform:uppercase;border-bottom:1px solid #1a2d48;padding-bottom:8px;margin-bottom:16px;}
.alert-box{background:#1a080e;border:1px solid #ff406040;border-left:4px solid #ff4060;border-radius:8px;padding:10px 14px;margin:6px 0;font-size:13px;}
.info-box{background:#001525;border:1px solid #00aaff40;border-left:4px solid #00aaff;border-radius:8px;padding:10px 14px;margin:6px 0;font-size:13px;}
.success-box{background:#001510;border:1px solid #00cc6640;border-left:4px solid #00cc66;border-radius:8px;padding:10px 14px;margin:6px 0;font-size:13px;}
.pdf-section{background:linear-gradient(135deg,#0a1525,#0d1c30);border:1px solid #1a3a5a;border-radius:14px;padding:20px;margin:10px 0;}
.pdf-btn{background:linear-gradient(135deg,#cc2200,#ff4400)!important;}
.brand{font-family:'Rajdhani',sans-serif;font-size:26px;font-weight:700;letter-spacing:3px;background:linear-gradient(135deg,#00c3ff,#0055ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.stTabs [data-baseweb="tab"]{font-family:'Rajdhani',sans-serif;font-weight:600;letter-spacing:1px;}
</style>
""", unsafe_allow_html=True)

# ─── PDF COLOR PALETTE ──────────────────────────────────────────────────────
PDF_NAVY   = colors.HexColor("#1B2A4A")
PDF_BLUE   = colors.HexColor("#1565C0")
PDF_LBLUE  = colors.HexColor("#E3F2FD")
PDF_ACCENT = colors.HexColor("#D32F2F")
PDF_GREEN  = colors.HexColor("#388E3C")
PDF_ORANGE = colors.HexColor("#F57C00")
PDF_GRAY   = colors.HexColor("#455A64")
PDF_LGRAY  = colors.HexColor("#F5F7FA")
PDF_WHITE  = colors.white
PDF_BLACK  = colors.HexColor("#1A1A1A")
PDF_MGRAY  = colors.HexColor("#B0BEC5")

# ─── DATA INIT ──────────────────────────────────────────────────────────────
def init_data():
    if "equipements" not in st.session_state:
        st.session_state.equipements = pd.DataFrame([
            {"ID":"EQ-001","Nom":"Compresseur Atlas Copco GA55","Catégorie":"Pneumatique","Site":"Usine A","Statut":"Opérationnel","Criticité":"Critique","Installation":"2019-03-15","Dernière maintenance":"2024-11-01","Prochaine maintenance":"2025-03-01","Heures":12450,"Valeur (€)":45000,"Responsable":"A. Martin"},
            {"ID":"EQ-002","Nom":"Convoyeur à bande CB-12","Catégorie":"Manutention","Site":"Usine A","Statut":"En panne","Criticité":"Haute","Installation":"2020-07-10","Dernière maintenance":"2024-09-15","Prochaine maintenance":"2025-01-15","Heures":8320,"Valeur (€)":28000,"Responsable":"B. Lefebvre"},
            {"ID":"EQ-003","Nom":"Pompe centrifuge PMP-3","Catégorie":"Hydraulique","Site":"Usine B","Statut":"Opérationnel","Criticité":"Haute","Installation":"2018-11-22","Dernière maintenance":"2024-10-20","Prochaine maintenance":"2025-04-20","Heures":18900,"Valeur (€)":12000,"Responsable":"C. Bernard"},
            {"ID":"EQ-004","Nom":"Robot soudure RS-7","Catégorie":"Robotique","Site":"Usine A","Statut":"En maintenance","Criticité":"Critique","Installation":"2021-02-01","Dernière maintenance":"2025-01-10","Prochaine maintenance":"2025-07-10","Heures":5600,"Valeur (€)":95000,"Responsable":"A. Martin"},
            {"ID":"EQ-005","Nom":"Tour CNC Mazak QT250","Catégorie":"Usinage","Site":"Usine C","Statut":"Opérationnel","Criticité":"Haute","Installation":"2017-06-18","Dernière maintenance":"2024-12-01","Prochaine maintenance":"2025-06-01","Heures":24300,"Valeur (€)":65000,"Responsable":"D. Rousseau"},
            {"ID":"EQ-006","Nom":"Groupe électrogène GE-100","Catégorie":"Électrique","Site":"Usine B","Statut":"Opérationnel","Criticité":"Critique","Installation":"2022-01-05","Dernière maintenance":"2024-08-10","Prochaine maintenance":"2025-02-10","Heures":3200,"Valeur (€)":38000,"Responsable":"B. Lefebvre"},
            {"ID":"EQ-007","Nom":"Chaudière vapeur CV-50","Catégorie":"Thermique","Site":"Usine C","Statut":"Opérationnel","Criticité":"Critique","Installation":"2016-09-30","Dernière maintenance":"2024-07-15","Prochaine maintenance":"2025-01-15","Heures":31500,"Valeur (€)":72000,"Responsable":"E. Petit"},
            {"ID":"EQ-008","Nom":"Pont roulant PR-10T","Catégorie":"Levage","Site":"Usine A","Statut":"Opérationnel","Criticité":"Haute","Installation":"2015-04-12","Dernière maintenance":"2024-11-30","Prochaine maintenance":"2025-05-30","Heures":41200,"Valeur (€)":55000,"Responsable":"C. Bernard"},
        ])
    if "bons_travaux" not in st.session_state:
        st.session_state.bons_travaux = pd.DataFrame([
            {"BT":"BT-2025-001","Équipement":"EQ-002","Titre":"Remplacement courroie convoyeur","Type":"Correctif","Priorité":"Haute","Statut":"En cours","Demandeur":"M. Dupont","Technicien":"A. Martin","Date création":"2025-01-15","Date prévue":"2025-01-20","Durée (h)":4,"Coût estimé (€)":850,"Coût réel (€)":0,"Description":"Courroie principale usée, remplacement urgent"},
            {"BT":"BT-2025-002","Équipement":"EQ-007","Titre":"Inspection annuelle chaudière","Type":"Préventif","Priorité":"Critique","Statut":"Planifié","Demandeur":"Système auto","Technicien":"B. Lefebvre","Date création":"2025-01-10","Date prévue":"2025-01-25","Durée (h)":8,"Coût estimé (€)":2400,"Coût réel (€)":0,"Description":"Inspection réglementaire annuelle obligatoire"},
            {"BT":"BT-2025-003","Équipement":"EQ-001","Titre":"Vidange huile compresseur","Type":"Préventif","Priorité":"Moyenne","Statut":"Terminé","Demandeur":"Système auto","Technicien":"C. Bernard","Date création":"2025-01-05","Date prévue":"2025-01-08","Durée (h)":2,"Coût estimé (€)":180,"Coût réel (€)":165,"Description":"Vidange périodique et remplacement filtre"},
            {"BT":"BT-2025-004","Équipement":"EQ-004","Titre":"Calibration robot soudure","Type":"Correctif","Priorité":"Critique","Statut":"En cours","Demandeur":"Production","Technicien":"A. Martin","Date création":"2025-01-12","Date prévue":"2025-01-18","Durée (h)":12,"Coût estimé (€)":3200,"Coût réel (€)":0,"Description":"Dérive de précision détectée, recalibrage nécessaire"},
            {"BT":"BT-2025-005","Équipement":"EQ-005","Titre":"Remplacement outil de coupe","Type":"Préventif","Priorité":"Basse","Statut":"Planifié","Demandeur":"Opérateur","Technicien":"D. Rousseau","Date création":"2025-01-14","Date prévue":"2025-02-01","Durée (h)":1,"Coût estimé (€)":95,"Coût réel (€)":0,"Description":"Usure normale de l'outil selon programme"},
            {"BT":"BT-2025-006","Équipement":"EQ-003","Titre":"Vérification étanchéité pompe","Type":"Préventif","Priorité":"Haute","Statut":"Ouvert","Demandeur":"Contrôle qualité","Technicien":"Non assigné","Date création":"2025-01-16","Date prévue":"2025-01-22","Durée (h)":3,"Coût estimé (€)":420,"Coût réel (€)":0,"Description":"Légère fuite constatée, diagnostic nécessaire"},
            {"BT":"BT-2024-089","Équipement":"EQ-008","Titre":"Lubrification pont roulant","Type":"Préventif","Priorité":"Moyenne","Statut":"Terminé","Demandeur":"Système auto","Technicien":"B. Lefebvre","Date création":"2024-12-20","Date prévue":"2024-12-22","Durée (h)":2,"Coût estimé (€)":240,"Coût réel (€)":220,"Description":"Lubrification trimestrielle tous points"},
            {"BT":"BT-2024-088","Équipement":"EQ-006","Titre":"Test groupe électrogène","Type":"Préventif","Priorité":"Haute","Statut":"Terminé","Demandeur":"Système auto","Technicien":"E. Petit","Date création":"2024-12-10","Date prévue":"2024-12-12","Durée (h)":3,"Coût estimé (€)":380,"Coût réel (€)":350,"Description":"Test mensuel de démarrage et charge"},
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
            {"Nom":"A. Martin","Spécialité":"Mécanique / Robotique","Disponible":True,"BT en cours":2,"Efficacité (%)":92,"Heure/mois":160,"Coût/h (€)":45},
            {"Nom":"B. Lefebvre","Spécialité":"Électrique / Pneumatique","Disponible":True,"BT en cours":1,"Efficacité (%)":88,"Heure/mois":160,"Coût/h (€)":42},
            {"Nom":"C. Bernard","Spécialité":"Hydraulique","Disponible":False,"BT en cours":0,"Efficacité (%)":95,"Heure/mois":160,"Coût/h (€)":48},
            {"Nom":"D. Rousseau","Spécialité":"Usinage / CNC","Disponible":True,"BT en cours":1,"Efficacité (%)":84,"Heure/mois":160,"Coût/h (€)":40},
            {"Nom":"E. Petit","Spécialité":"Chaudronnerie / Thermique","Disponible":True,"BT en cours":0,"Efficacité (%)":90,"Heure/mois":160,"Coût/h (€)":44},
        ])
    if "interventions_hist" not in st.session_state:
        months = ["Juil","Août","Sept","Oct","Nov","Déc","Jan"]
        st.session_state.interventions_hist = pd.DataFrame({
            "Mois": months,
            "Correctifs": [8,8,12,7,5,9,6],
            "Préventifs": [12,14,12,16,13,11,15],
            "Coût total (€)": [7200,8400,12500,7800,6200,9100,7145],
            "Disponibilité (%)": [89,91,88,93,95,92,87.5],
        })

init_data()

# ═══════════════════════════════════════════════════════════════════════════
# PDF GENERATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_pdf_styles():
    styles = getSampleStyleSheet()
    custom = {
        'title_main': ParagraphStyle('title_main', fontSize=28, textColor=PDF_WHITE, fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=6),
        'title_sub':  ParagraphStyle('title_sub', fontSize=13, textColor=PDF_LBLUE, fontName='Helvetica', alignment=TA_CENTER, spaceAfter=4),
        'h1':         ParagraphStyle('h1', fontSize=16, textColor=PDF_WHITE, fontName='Helvetica-Bold', spaceBefore=14, spaceAfter=8, backColor=PDF_NAVY, leftIndent=-10, rightIndent=-10, borderPad=6),
        'h2':         ParagraphStyle('h2', fontSize=13, textColor=PDF_NAVY, fontName='Helvetica-Bold', spaceBefore=12, spaceAfter=6, borderColor=PDF_BLUE, borderWidth=0, borderPad=0),
        'h3':         ParagraphStyle('h3', fontSize=11, textColor=PDF_BLUE, fontName='Helvetica-Bold', spaceBefore=8, spaceAfter=4),
        'body':       ParagraphStyle('body', fontSize=10, textColor=PDF_GRAY, fontName='Helvetica', spaceBefore=3, spaceAfter=3, leading=14),
        'body_bold':  ParagraphStyle('body_bold', fontSize=10, textColor=PDF_BLACK, fontName='Helvetica-Bold', spaceBefore=2, spaceAfter=2),
        'small':      ParagraphStyle('small', fontSize=8, textColor=PDF_MGRAY, fontName='Helvetica', alignment=TA_CENTER),
        'label':      ParagraphStyle('label', fontSize=9, textColor=PDF_BLUE, fontName='Helvetica-Bold', spaceBefore=2, spaceAfter=1),
        'value':      ParagraphStyle('value', fontSize=10, textColor=PDF_BLACK, fontName='Helvetica', spaceBefore=1, spaceAfter=2),
        'center':     ParagraphStyle('center', fontSize=10, alignment=TA_CENTER, fontName='Helvetica'),
        'right':      ParagraphStyle('right', fontSize=9, alignment=TA_RIGHT, textColor=PDF_MGRAY, fontName='Helvetica'),
        'kpi_val':    ParagraphStyle('kpi_val', fontSize=22, textColor=PDF_BLUE, fontName='Helvetica-Bold', alignment=TA_CENTER),
        'kpi_lbl':    ParagraphStyle('kpi_lbl', fontSize=8, textColor=PDF_GRAY, fontName='Helvetica', alignment=TA_CENTER),
        'alert':      ParagraphStyle('alert', fontSize=10, textColor=PDF_ACCENT, fontName='Helvetica-Bold', spaceBefore=3, spaceAfter=3),
        'footer':     ParagraphStyle('footer', fontSize=8, textColor=PDF_MGRAY, fontName='Helvetica', alignment=TA_CENTER),
    }
    return custom

def make_header_footer(canvas_obj, doc, title, subtitle=""):
    canvas_obj.saveState()
    w, h = A4
    # Header bar
    canvas_obj.setFillColor(PDF_NAVY)
    canvas_obj.rect(0, h - 28*mm, w, 28*mm, fill=1, stroke=0)
    # Logo text
    canvas_obj.setFillColor(PDF_WHITE)
    canvas_obj.setFont("Helvetica-Bold", 16)
    canvas_obj.drawString(15*mm, h - 16*mm, "⚙ GMAO PRO+")
    canvas_obj.setFont("Helvetica", 10)
    canvas_obj.setFillColor(colors.HexColor("#90CAF9"))
    canvas_obj.drawString(15*mm, h - 23*mm, title)
    # Date right
    canvas_obj.setFillColor(PDF_WHITE)
    canvas_obj.setFont("Helvetica", 9)
    canvas_obj.drawRightString(w - 15*mm, h - 16*mm, datetime.now().strftime("%d/%m/%Y %H:%M"))
    if subtitle:
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.setFillColor(colors.HexColor("#90CAF9"))
        canvas_obj.drawRightString(w - 15*mm, h - 23*mm, subtitle)
    # Blue accent line
    canvas_obj.setStrokeColor(colors.HexColor("#1565C0"))
    canvas_obj.setLineWidth(2)
    canvas_obj.line(0, h - 29*mm, w, h - 29*mm)
    # Footer
    canvas_obj.setFillColor(PDF_LGRAY)
    canvas_obj.rect(0, 0, w, 12*mm, fill=1, stroke=0)
    canvas_obj.setStrokeColor(PDF_MGRAY)
    canvas_obj.setLineWidth(0.5)
    canvas_obj.line(0, 12*mm, w, 12*mm)
    canvas_obj.setFillColor(PDF_GRAY)
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.drawString(15*mm, 4*mm, "Confidentiel — Usage interne uniquement")
    canvas_obj.drawCentredString(w/2, 4*mm, f"Page {doc.page}")
    canvas_obj.drawRightString(w - 15*mm, 4*mm, "Industrie Maroc SA")
    canvas_obj.restoreState()

def table_style_default(header_color=None):
    hc = header_color or PDF_NAVY
    return TableStyle([
        ('BACKGROUND', (0,0), (-1,0), hc),
        ('TEXTCOLOR', (0,0), (-1,0), PDF_WHITE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [PDF_WHITE, PDF_LGRAY]),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
        ('GRID', (0,0), (-1,-1), 0.4, PDF_MGRAY),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('ROWBACKGROUNDS', (0,0), (-1,0), [hc]),
    ])

def priority_color(p):
    return {"Critique": PDF_ACCENT, "Haute": PDF_ORANGE, "Moyenne": PDF_BLUE, "Basse": PDF_GREEN}.get(p, PDF_GRAY)

def status_color(s):
    return {"Opérationnel": PDF_GREEN, "En panne": PDF_ACCENT, "En maintenance": colors.HexColor("#7B1FA2"), "Terminé": PDF_GREEN, "En cours": PDF_BLUE, "Planifié": colors.HexColor("#F57C00"), "Ouvert": PDF_BLUE}.get(s, PDF_GRAY)

# ── PDF 1: Rapport mensuel complet ──────────────────────────────────────────
def generate_rapport_mensuel():
    buf = io.BytesIO()
    S = get_pdf_styles()
    eq = st.session_state.equipements
    bt = st.session_state.bons_travaux
    pc = st.session_state.pieces
    hist = st.session_state.interventions_hist

    def header_footer(c, d): make_header_footer(c, d, "RAPPORT MENSUEL DE MAINTENANCE", "Janvier 2025")

    doc = SimpleDocTemplate(buf, pagesize=A4,
        topMargin=35*mm, bottomMargin=18*mm, leftMargin=15*mm, rightMargin=15*mm)

    story = []
    w = 165*mm  # content width

    # ── Page de garde ──
    story.append(Spacer(1, 10*mm))
    # Cover box
    cover_data = [["RAPPORT MENSUEL DE MAINTENANCE\nJANVIER 2025"]]
    cover_tbl = Table(cover_data, colWidths=[w])
    cover_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), PDF_NAVY),
        ('TEXTCOLOR', (0,0), (-1,-1), PDF_WHITE),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 20),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 20),
        ('BOTTOMPADDING', (0,0), (-1,-1), 20),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [PDF_NAVY]),
    ]))
    story.append(cover_tbl)
    story.append(Spacer(1, 6*mm))

    # KPI summary row
    taux_dispo = round(len(eq[eq["Statut"]=="Opérationnel"])/len(eq)*100,1)
    bt_termines = len(bt[bt["Statut"]=="Terminé"])
    cout_total = bt["Coût estimé (€)"].sum()
    n_pannes = len(eq[eq["Statut"]=="En panne"])
    taux_prev = round(len(bt[bt["Type"]=="Préventif"])/len(bt)*100,1)

    kpi_data = [
        [Paragraph(f"{taux_dispo}%", S['kpi_val']), Paragraph(f"{bt_termines}/{len(bt)}", S['kpi_val']),
         Paragraph(f"{cout_total:,.0f}€", S['kpi_val']), Paragraph(f"{taux_prev}%", S['kpi_val'])],
        [Paragraph("Disponibilité parc", S['kpi_lbl']), Paragraph("BT réalisés", S['kpi_lbl']),
         Paragraph("Coût total estimé", S['kpi_lbl']), Paragraph("Taux préventif", S['kpi_lbl'])],
    ]
    kpi_tbl = Table(kpi_data, colWidths=[w/4]*4)
    kpi_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), PDF_LBLUE),
        ('GRID', (0,0), (-1,-1), 0.5, PDF_MGRAY),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 6*mm))

    # ── Section 1: Synthèse ──
    story.append(Paragraph("1. SYNTHÈSE EXÉCUTIVE", S['h1']))
    story.append(Spacer(1, 3*mm))

    synth_data = [
        ["Indicateur","Valeur","Objectif","Écart","Statut"],
        ["Disponibilité équipements", f"{taux_dispo}%", "95%", f"{taux_dispo-95:+.1f}%", "⚠" if taux_dispo<95 else "✓"],
        ["Taux maintenance préventive", f"{taux_prev}%", "70%", f"{taux_prev-70:+.1f}%", "✓" if taux_prev>=70 else "✗"],
        ["BT ouverts critiques", str(len(bt[(bt["Priorité"]=="Critique")&(bt["Statut"]!="Terminé")])), "0", "-", "⚠"],
        ["Coût moyen/intervention", f"{cout_total/len(bt):,.0f}€", "1 200€", "-", "-"],
        ["Pièces en rupture stock", str(len(pc[pc["Stock"]<pc["Min"]])), "0", "-", "⚠" if len(pc[pc["Stock"]<pc["Min"]])>0 else "✓"],
        ["Pannes équipements", str(n_pannes), "0", "-", "⚠" if n_pannes>0 else "✓"],
    ]
    tbl = Table(synth_data, colWidths=[55*mm, 28*mm, 25*mm, 25*mm, 22*mm])
    st_obj = table_style_default()
    for i, row in enumerate(synth_data[1:], 1):
        status = row[4]
        if status == "✓":
            st_obj.add('TEXTCOLOR', (4,i), (4,i), PDF_GREEN)
        elif status in ("⚠", "✗"):
            st_obj.add('TEXTCOLOR', (4,i), (4,i), PDF_ACCENT)
        st_obj.add('FONTNAME', (4,i), (4,i), 'Helvetica-Bold')
    tbl.setStyle(st_obj)
    story.append(tbl)
    story.append(Spacer(1, 5*mm))

    # ── Section 2: Équipements ──
    story.append(Paragraph("2. ÉTAT DU PARC ÉQUIPEMENTS", S['h1']))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("2.1 Inventaire complet", S['h2']))

    eq_data = [["ID","Nom","Site","Statut","Criticité","Heures","Proch. maintenance"]]
    for _, r in eq.iterrows():
        eq_data.append([r["ID"], r["Nom"][:30], r["Site"], r["Statut"], r["Criticité"], f"{r['Heures']:,}", r["Prochaine maintenance"]])
    tbl2 = Table(eq_data, colWidths=[16*mm,52*mm,18*mm,22*mm,18*mm,16*mm,28*mm])
    st2 = table_style_default()
    for i, r in enumerate(eq_data[1:], 1):
        sc = status_color(r[3])
        st2.add('TEXTCOLOR', (3,i), (3,i), sc)
        st2.add('FONTNAME', (3,i), (3,i), 'Helvetica-Bold')
        pc_col = priority_color(r[4])
        st2.add('TEXTCOLOR', (4,i), (4,i), pc_col)
    tbl2.setStyle(st2)
    story.append(tbl2)
    story.append(Spacer(1, 5*mm))

    # ── Section 3: Bons de travaux ──
    story.append(Paragraph("3. BONS DE TRAVAUX", S['h1']))
    story.append(Spacer(1, 3*mm))

    bt_data = [["BT","Titre","Type","Priorité","Statut","Technicien","Coût estimé"]]
    for _, r in bt.iterrows():
        bt_data.append([r["BT"], r["Titre"][:28], r["Type"], r["Priorité"], r["Statut"], r["Technicien"], f"{r['Coût estimé (€)']:,.0f}€"])
    tbl3 = Table(bt_data, colWidths=[22*mm,45*mm,20*mm,18*mm,18*mm,24*mm,18*mm])
    st3 = table_style_default()
    for i, r in enumerate(bt_data[1:], 1):
        st3.add('TEXTCOLOR', (3,i), (3,i), priority_color(r[3]))
        st3.add('TEXTCOLOR', (4,i), (4,i), status_color(r[4]))
        st3.add('FONTNAME', (3,i), (3,i), 'Helvetica-Bold')
    tbl3.setStyle(st3)
    story.append(tbl3)
    story.append(Spacer(1, 5*mm))

    # ── Section 4: Stock ──
    story.append(Paragraph("4. GESTION DES STOCKS", S['h1']))
    story.append(Spacer(1, 3*mm))

    pc_data = [["Référence","Désignation","Catégorie","Stock","Min","Max","Prix unit.","Valeur stock"]]
    for _, r in pc.iterrows():
        val = r["Stock"] * r["Prix (€)"]
        pc_data.append([r["Réf"], r["Désignation"][:28], r["Catégorie"], str(r["Stock"]), str(r["Min"]), str(r["Max"]), f"{r['Prix (€)']:.2f}€", f"{val:.2f}€"])
    tbl4 = Table(pc_data, colWidths=[16*mm,44*mm,22*mm,14*mm,11*mm,11*mm,18*mm,20*mm])
    st4 = table_style_default()
    for i, r in enumerate(pc_data[1:], 1):
        stock_val = int(r[3])
        min_val = int(r[4])
        if stock_val < min_val:
            st4.add('BACKGROUND', (0,i), (-1,i), colors.HexColor("#FFEBEE"))
            st4.add('TEXTCOLOR', (3,i), (3,i), PDF_ACCENT)
            st4.add('FONTNAME', (3,i), (3,i), 'Helvetica-Bold')
    tbl4.setStyle(st4)
    story.append(tbl4)
    story.append(Spacer(1, 3*mm))
    ruptures_text = f"⚠ {len(pc[pc['Stock']<pc['Min']])} référence(s) sous le seuil minimum — Commander immédiatement."
    story.append(Paragraph(ruptures_text, S['alert']))

    # ── Section 5: Évolution historique ──
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph("5. ÉVOLUTION MENSUELLE", S['h1']))
    story.append(Spacer(1, 3*mm))

    hist_data = [["Mois","Correctifs","Préventifs","Total","Coût (€)","Disponibilité"]]
    for _, r in hist.iterrows():
        hist_data.append([r["Mois"], str(r["Correctifs"]), str(r["Préventifs"]),
                          str(r["Correctifs"]+r["Préventifs"]), f"{r['Coût total (€)']:,.0f}€",
                          f"{r['Disponibilité (%)']:.1f}%"])
    tbl5 = Table(hist_data, colWidths=[22*mm,25*mm,25*mm,20*mm,32*mm,30*mm])
    tbl5.setStyle(table_style_default(PDF_BLUE))
    story.append(tbl5)

    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width=w, color=PDF_BLUE, thickness=1))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("Document généré automatiquement par GMAO Pro+ · Confidentiel", S['footer']))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    buf.seek(0)
    return buf

# ── PDF 2: Bon de travaux individuel ──────────────────────────────────────────
def generate_bon_travaux(bt_id):
    buf = io.BytesIO()
    S = get_pdf_styles()
    bt = st.session_state.bons_travaux
    eq = st.session_state.equipements
    row = bt[bt["BT"] == bt_id].iloc[0]

    def hf(c, d): make_header_footer(c, d, "BON DE TRAVAUX", bt_id)

    doc = SimpleDocTemplate(buf, pagesize=A4,
        topMargin=35*mm, bottomMargin=18*mm, leftMargin=15*mm, rightMargin=15*mm)
    story = []
    w = 165*mm

    # Title block
    prio_c = priority_color(row["Priorité"])
    title_data = [[f"BON DE TRAVAUX — {bt_id}",  f"PRIORITÉ: {row['Priorité'].upper()}"]]
    tt = Table(title_data, colWidths=[110*mm, 55*mm])
    tt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), PDF_NAVY),
        ('BACKGROUND', (1,0), (1,0), prio_c),
        ('TEXTCOLOR', (0,0), (-1,-1), PDF_WHITE),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (0,0), 14),
        ('FONTSIZE', (1,0), (1,0), 13),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(tt)
    story.append(Spacer(1, 5*mm))

    # Statut badge
    st_c = status_color(row["Statut"])
    status_data = [[f"Statut: {row['Statut'].upper()}  |  Type: {row['Type']}  |  Équipement: {row['Équipement']}"]]
    st_tbl = Table(status_data, colWidths=[w])
    st_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), st_c),
        ('TEXTCOLOR', (0,0), (-1,-1), PDF_WHITE),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(st_tbl)
    story.append(Spacer(1, 5*mm))

    # Info principale
    story.append(Paragraph("INFORMATIONS GÉNÉRALES", S['h2']))
    eq_row = eq[eq["ID"]==row["Équipement"]]
    eq_nom = eq_row["Nom"].values[0] if len(eq_row)>0 else "N/A"
    info_data = [
        ["Numéro BT:", row["BT"], "Date création:", row["Date création"]],
        ["Titre:", row["Titre"], "Date prévue:", row["Date prévue"]],
        ["Type:", row["Type"], "Durée estimée:", f"{row['Durée (h)']} heures"],
        ["Équipement:", f"{row['Équipement']} — {eq_nom}", "Coût estimé:", f"{row['Coût estimé (€)']:,.0f} €"],
        ["Demandeur:", row["Demandeur"], "Coût réel:", f"{row['Coût réel (€)']:,.0f} €" if row["Coût réel (€)"]>0 else "N/A"],
        ["Technicien assigné:", row["Technicien"], "Priorité:", row["Priorité"]],
    ]
    it = Table(info_data, colWidths=[32*mm, 52*mm, 32*mm, 49*mm])
    it.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,0), (0,-1), PDF_NAVY),
        ('TEXTCOLOR', (2,0), (2,-1), PDF_NAVY),
        ('FONTSIZE', (0,0), (-1,-1), 9.5),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [PDF_LGRAY, PDF_WHITE]),
        ('GRID', (0,0), (-1,-1), 0.3, PDF_MGRAY),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(it)
    story.append(Spacer(1, 5*mm))

    # Description
    story.append(Paragraph("DESCRIPTION DES TRAVAUX", S['h2']))
    desc_data = [[row["Description"] or "Aucune description fournie."]]
    dt = Table(desc_data, colWidths=[w])
    dt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), PDF_LBLUE),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('GRID', (0,0), (-1,-1), 0.5, PDF_BLUE),
    ]))
    story.append(dt)
    story.append(Spacer(1, 5*mm))

    # Checklist interventions
    story.append(Paragraph("CHECKLIST D'INTERVENTION", S['h2']))
    checklist = [
        ["☐", "Vérification EPI avant intervention"],
        ["☐", "Consignation / déconsignation équipement"],
        ["☐", "Diagnostic et identification du problème"],
        ["☐", "Préparation des outils et pièces nécessaires"],
        ["☐", "Réalisation des travaux"],
        ["☐", "Tests de fonctionnement après intervention"],
        ["☐", "Nettoyage du poste de travail"],
        ["☐", "Rapport d'intervention complété"],
        ["☐", "Validation responsable maintenance"],
    ]
    cl_tbl = Table(checklist, colWidths=[10*mm, 155*mm])
    cl_tbl.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [PDF_WHITE, PDF_LGRAY]),
        ('GRID', (0,0), (-1,-1), 0.3, PDF_MGRAY),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('ALIGN', (0,0), (0,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (0,-1), 14),
    ]))
    story.append(cl_tbl)
    story.append(Spacer(1, 5*mm))

    # Rapport de travaux
    story.append(Paragraph("RAPPORT D'INTERVENTION (à compléter)", S['h2']))
    rapport_data = [
        ["Heure début:", "_____________", "Heure fin:", "_____________"],
        ["Durée réelle:", "_____________", "Coût réel (€):", "_____________"],
        ["Pièces utilisées:", "", "", ""],
        ["", "", "", ""],
        ["Observations:", "", "", ""],
        ["", "", "", ""],
        ["", "", "", ""],
    ]
    rt = Table(rapport_data, colWidths=[32*mm, 52*mm, 32*mm, 49*mm])
    rt.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (2,0), (2,1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,0), (0,-1), PDF_NAVY),
        ('TEXTCOLOR', (2,0), (2,1), PDF_NAVY),
        ('FONTSIZE', (0,0), (-1,-1), 9.5),
        ('SPAN', (1,2), (-1,2)),
        ('SPAN', (0,3), (-1,3)),
        ('SPAN', (1,4), (-1,4)),
        ('SPAN', (0,5), (-1,5)),
        ('SPAN', (0,6), (-1,6)),
        ('GRID', (0,0), (-1,-1), 0.3, PDF_MGRAY),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [PDF_LGRAY, PDF_WHITE]),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(rt)
    story.append(Spacer(1, 6*mm))

    # Signatures
    story.append(Paragraph("SIGNATURES", S['h2']))
    sig_data = [
        ["Technicien intervenant", "Responsable maintenance", "Opérateur / Demandeur"],
        ["\n\n\n_________________________", "\n\n\n_________________________", "\n\n\n_________________________"],
        ["Nom: ___________________\nDate: __________________", "Nom: ___________________\nDate: __________________", "Nom: ___________________\nDate: __________________"],
    ]
    sg = Table(sig_data, colWidths=[w/3]*3)
    sg.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PDF_NAVY),
        ('TEXTCOLOR', (0,0), (-1,0), PDF_WHITE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, PDF_MGRAY),
        ('TOPPADDING', (0,0), (-1,0), 7),
        ('BOTTOMPADDING', (0,0), (-1,0), 7),
        ('TOPPADDING', (0,1), (-1,-1), 5),
        ('BOTTOMPADDING', (0,1), (-1,-1), 5),
    ]))
    story.append(sg)

    doc.build(story, onFirstPage=hf, onLaterPages=hf)
    buf.seek(0)
    return buf

# ── PDF 3: Fiche équipement ───────────────────────────────────────────────────
def generate_fiche_equipement(eq_id):
    buf = io.BytesIO()
    S = get_pdf_styles()
    eq = st.session_state.equipements
    bt = st.session_state.bons_travaux
    row = eq[eq["ID"]==eq_id].iloc[0]
    bt_eq = bt[bt["Équipement"]==eq_id]

    def hf(c, d): make_header_footer(c, d, "FICHE ÉQUIPEMENT", eq_id)
    doc = SimpleDocTemplate(buf, pagesize=A4,
        topMargin=35*mm, bottomMargin=18*mm, leftMargin=15*mm, rightMargin=15*mm)
    story = []
    w = 165*mm

    sc = status_color(row["Statut"])
    title_data = [[f"{row['ID']} — {row['Nom']}", row["Statut"].upper()]]
    tt = Table(title_data, colWidths=[120*mm, 45*mm])
    tt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), PDF_NAVY),
        ('BACKGROUND', (1,0), (1,0), sc),
        ('TEXTCOLOR', (0,0), (-1,-1), PDF_WHITE),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (0,0), 13),
        ('FONTSIZE', (1,0), (1,0), 12),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
    ]))
    story.append(tt)
    story.append(Spacer(1, 5*mm))

    story.append(Paragraph("INFORMATIONS TECHNIQUES", S['h2']))
    info = [
        ["Identifiant:", row["ID"], "Catégorie:", row["Catégorie"]],
        ["Site / Localisation:", row["Site"], "Criticité:", row["Criticité"]],
        ["Date installation:", row["Installation"], "Responsable:", row["Responsable"]],
        ["Heures de marche:", f"{row['Heures']:,} h", "Valeur actif:", f"{row['Valeur (€)']:,} €"],
        ["Dernière maintenance:", row["Dernière maintenance"], "Prochaine maintenance:", row["Prochaine maintenance"]],
    ]
    it = Table(info, colWidths=[38*mm,46*mm,38*mm,43*mm])
    it.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,0), (0,-1), PDF_NAVY),
        ('TEXTCOLOR', (2,0), (2,-1), PDF_NAVY),
        ('FONTSIZE', (0,0), (-1,-1), 9.5),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [PDF_LGRAY, PDF_WHITE]),
        ('GRID', (0,0), (-1,-1), 0.3, PDF_MGRAY),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(it)
    story.append(Spacer(1, 5*mm))

    # KPIs équipement
    story.append(Paragraph("INDICATEURS DE PERFORMANCE", S['h2']))
    n_interv = len(bt_eq)
    cout_total_eq = bt_eq["Coût estimé (€)"].sum()
    n_correctifs = len(bt_eq[bt_eq["Type"]=="Correctif"])
    n_preventifs = len(bt_eq[bt_eq["Type"]=="Préventif"])
    kpi_data = [
        [Paragraph(str(n_interv), S['kpi_val']), Paragraph(f"{cout_total_eq:,.0f}€", S['kpi_val']),
         Paragraph(str(n_correctifs), S['kpi_val']), Paragraph(str(n_preventifs), S['kpi_val'])],
        [Paragraph("Total interventions", S['kpi_lbl']), Paragraph("Coût total (€)", S['kpi_lbl']),
         Paragraph("Correctifs", S['kpi_lbl']), Paragraph("Préventifs", S['kpi_lbl'])],
    ]
    kt = Table(kpi_data, colWidths=[w/4]*4)
    kt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), PDF_LBLUE),
        ('GRID', (0,0), (-1,-1), 0.5, PDF_MGRAY),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(kt)
    story.append(Spacer(1, 5*mm))

    # Historique BT
    story.append(Paragraph("HISTORIQUE DES INTERVENTIONS", S['h2']))
    if len(bt_eq) > 0:
        bt_data = [["BT","Titre","Type","Priorité","Statut","Date","Coût"]]
        for _, r in bt_eq.iterrows():
            bt_data.append([r["BT"], r["Titre"][:25], r["Type"], r["Priorité"], r["Statut"], r["Date prévue"], f"{r['Coût estimé (€)']:,.0f}€"])
        bt_tbl = Table(bt_data, colWidths=[22*mm,42*mm,20*mm,18*mm,18*mm,20*mm,20*mm])
        st_bt = table_style_default()
        for i, r in enumerate(bt_data[1:], 1):
            st_bt.add('TEXTCOLOR', (3,i), (3,i), priority_color(r[3]))
            st_bt.add('TEXTCOLOR', (4,i), (4,i), status_color(r[4]))
        bt_tbl.setStyle(st_bt)
        story.append(bt_tbl)
    else:
        story.append(Paragraph("Aucune intervention enregistrée pour cet équipement.", S['body']))

    story.append(Spacer(1, 5*mm))

    # Plan de maintenance
    story.append(Paragraph("PLAN DE MAINTENANCE PRÉVENTIVE", S['h2']))
    plan_data = [
        ["Fréquence","Opération","Durée","Intervenant"],
        ["Hebdomadaire","Inspection visuelle et niveaux","0.5 h","Opérateur"],
        ["Mensuelle","Vérification sécurités et connexions","1 h","Technicien"],
        ["Trimestrielle","Lubrification et nettoyage complet","2 h","Technicien"],
        ["Semestrielle","Remplacement filtres et consommables","3 h","Technicien Senior"],
        ["Annuelle","Révision générale complète","8 h","Équipe maintenance"],
    ]
    pt = Table(plan_data, colWidths=[30*mm,75*mm,20*mm,40*mm])
    pt.setStyle(table_style_default(PDF_BLUE))
    story.append(pt)
    story.append(Spacer(1, 5*mm))

    # Pièces associées
    story.append(Paragraph("PIÈCES DE RECHANGE ASSOCIÉES", S['h2']))
    pieces_data = [
        ["Référence","Désignation","Stock actuel","Criticité pièce"],
        ["P-001","Courroie V-Belt A60","8 unités","Haute"],
        ["P-002","Roulement SKF 6205-2RS","2 unités","Critique"],
        ["P-005","Filtre air compresseur","3 unités","Haute"],
    ]
    pct = Table(pieces_data, colWidths=[25*mm,75*mm,30*mm,35*mm])
    pct.setStyle(table_style_default())
    story.append(pct)

    doc.build(story, onFirstPage=hf, onLaterPages=hf)
    buf.seek(0)
    return buf

# ── PDF 4: Rapport stock ──────────────────────────────────────────────────────
def generate_rapport_stock():
    buf = io.BytesIO()
    S = get_pdf_styles()
    pc = st.session_state.pieces

    def hf(c, d): make_header_footer(c, d, "RAPPORT D'INVENTAIRE STOCK", datetime.now().strftime("%d/%m/%Y"))
    doc = SimpleDocTemplate(buf, pagesize=A4,
        topMargin=35*mm, bottomMargin=18*mm, leftMargin=15*mm, rightMargin=15*mm)
    story = []
    w = 165*mm

    story.append(Paragraph("RAPPORT D'INVENTAIRE — PIÈCES & CONSOMMABLES", S['h1']))
    story.append(Spacer(1, 4*mm))

    val_totale = (pc["Stock"] * pc["Prix (€)"]).sum()
    n_ruptures = len(pc[pc["Stock"] < pc["Min"]])
    n_ok = len(pc[pc["Stock"] >= pc["Min"]])

    kpi_data = [
        [Paragraph(f"{len(pc)}", S['kpi_val']), Paragraph(f"{val_totale:,.0f}€", S['kpi_val']),
         Paragraph(str(n_ruptures), S['kpi_val']), Paragraph(str(n_ok), S['kpi_val'])],
        [Paragraph("Références totales", S['kpi_lbl']), Paragraph("Valeur stock total", S['kpi_lbl']),
         Paragraph("Ruptures / alertes", S['kpi_lbl']), Paragraph("Stocks OK", S['kpi_lbl'])],
    ]
    kt = Table(kpi_data, colWidths=[w/4]*4)
    kt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), PDF_LBLUE),
        ('GRID', (0,0), (-1,-1), 0.5, PDF_MGRAY),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(kt)
    story.append(Spacer(1, 5*mm))

    if n_ruptures > 0:
        story.append(Paragraph(f"⚠ ALERTE: {n_ruptures} référence(s) nécessitent un réapprovisionnement urgent!", S['alert']))
        story.append(Spacer(1, 3*mm))

    # Table complète
    story.append(Paragraph("INVENTAIRE COMPLET", S['h2']))
    pc_data = [["Réf.","Désignation","Catég.","Stock","Min","Max","Statut","Prix","Valeur","Fourn.","Délai"]]
    for _, r in pc.iterrows():
        val = r["Stock"] * r["Prix (€)"]
        statut = "⚠ RUPTURE" if r["Stock"] < r["Min"] else ("⚡ FAIBLE" if r["Stock"] < r["Min"]*1.5 else "✓ OK")
        pc_data.append([r["Réf"], r["Désignation"][:22], r["Catégorie"][:10],
                         str(r["Stock"]), str(r["Min"]), str(r["Max"]),
                         statut, f"{r['Prix (€)']:.2f}€", f"{val:.0f}€",
                         r["Fournisseur"][:12], f"{r['Délai (j)']}j"])
    pt = Table(pc_data, colWidths=[13*mm,35*mm,17*mm,12*mm,10*mm,10*mm,18*mm,16*mm,16*mm,20*mm,9*mm])
    st_pc = table_style_default()
    for i, r in enumerate(pc_data[1:], 1):
        if "RUPTURE" in r[6]:
            st_pc.add('BACKGROUND', (0,i), (-1,i), colors.HexColor("#FFEBEE"))
            st_pc.add('TEXTCOLOR', (6,i), (6,i), PDF_ACCENT)
            st_pc.add('FONTNAME', (6,i), (6,i), 'Helvetica-Bold')
        elif "FAIBLE" in r[6]:
            st_pc.add('TEXTCOLOR', (6,i), (6,i), PDF_ORANGE)
        else:
            st_pc.add('TEXTCOLOR', (6,i), (6,i), PDF_GREEN)
        st_pc.add('FONTNAME', (6,i), (6,i), 'Helvetica-Bold')
    pt.setStyle(st_pc)
    story.append(pt)
    story.append(Spacer(1, 5*mm))

    # Commandes suggérées
    ruptures = pc[pc["Stock"] < pc["Min"]]
    if len(ruptures) > 0:
        story.append(Paragraph("BONS DE COMMANDE SUGGÉRÉS", S['h2']))
        cmd_data = [["Référence","Désignation","Stock actuel","Qté à commander","Fournisseur","Délai","Coût estimé"]]
        for _, r in ruptures.iterrows():
            qte = r["Max"] - r["Stock"]
            cout = qte * r["Prix (€)"]
            cmd_data.append([r["Réf"], r["Désignation"][:28], str(r["Stock"]),
                              str(qte), r["Fournisseur"], f"{r['Délai (j)']}j", f"{cout:.2f}€"])
        ct = Table(cmd_data, colWidths=[16*mm,45*mm,20*mm,22*mm,28*mm,14*mm,20*mm])
        ct.setStyle(table_style_default(PDF_ACCENT))
        story.append(ct)

    doc.build(story, onFirstPage=hf, onLaterPages=hf)
    buf.seek(0)
    return buf

# ── PDF 5: Planning interventions (paysage) ───────────────────────────────────
def generate_planning_pdf():
    buf = io.BytesIO()
    S = get_pdf_styles()
    bt = st.session_state.bons_travaux
    tech = st.session_state.techniciens

    w_page, h_page = landscape(A4)

    def hf(c, d):
        c.saveState()
        c.setFillColor(PDF_NAVY)
        c.rect(0, h_page-20*mm, w_page, 20*mm, fill=1, stroke=0)
        c.setFillColor(PDF_WHITE)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(12*mm, h_page-13*mm, "⚙ GMAO PRO+ — PLANNING DES INTERVENTIONS")
        c.setFont("Helvetica", 9)
        c.drawRightString(w_page-12*mm, h_page-13*mm, datetime.now().strftime("%d/%m/%Y"))
        c.setFillColor(PDF_LGRAY)
        c.rect(0, 0, w_page, 10*mm, fill=1, stroke=0)
        c.setFillColor(PDF_GRAY)
        c.setFont("Helvetica", 8)
        c.drawCentredString(w_page/2, 3*mm, f"Page {d.page} — Confidentiel")
        c.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=landscape(A4),
        topMargin=25*mm, bottomMargin=15*mm, leftMargin=12*mm, rightMargin=12*mm)
    story = []
    cw = 254*mm

    story.append(Spacer(1, 3*mm))

    # BT actifs
    bt_actif = bt[bt["Statut"]!="Terminé"].sort_values("Priorité", key=lambda x: x.map({"Critique":0,"Haute":1,"Moyenne":2,"Basse":3}))

    # Tableau principal
    plan_data = [["BT N°","Titre","Équipement","Type","Priorité","Technicien","Date prévue","Durée","Coût estimé","Statut"]]
    for _, r in bt_actif.iterrows():
        plan_data.append([
            r["BT"], r["Titre"][:32], r["Équipement"], r["Type"],
            r["Priorité"], r["Technicien"], r["Date prévue"],
            f"{r['Durée (h)']}h", f"{r['Coût estimé (€)']:,.0f}€", r["Statut"]
        ])
    pt = Table(plan_data, colWidths=[22*mm,55*mm,18*mm,20*mm,18*mm,24*mm,22*mm,14*mm,20*mm,18*mm])
    st_plan = table_style_default()
    for i, r in enumerate(plan_data[1:], 1):
        st_plan.add('TEXTCOLOR', (4,i), (4,i), priority_color(r[4]))
        st_plan.add('TEXTCOLOR', (9,i), (9,i), status_color(r[9]))
        st_plan.add('FONTNAME', (4,i), (4,i), 'Helvetica-Bold')
    pt.setStyle(st_plan)
    story.append(Paragraph("INTERVENTIONS PLANIFIÉES", ParagraphStyle('h', fontSize=14, fontName='Helvetica-Bold', textColor=PDF_NAVY, spaceAfter=6)))
    story.append(pt)
    story.append(Spacer(1, 6*mm))

    # Charge techniciens
    story.append(Paragraph("CHARGE DE TRAVAIL PAR TECHNICIEN", ParagraphStyle('h2', fontSize=12, fontName='Helvetica-Bold', textColor=PDF_NAVY, spaceAfter=4)))
    tech_data = [["Technicien","Spécialité","Disponible","BT en cours","Efficacité","Heures/mois","Coût/h"]]
    for _, r in tech.iterrows():
        tech_data.append([r["Nom"], r["Spécialité"], "✓ Oui" if r["Disponible"] else "✗ Non",
                           str(r["BT en cours"]), f"{r['Efficacité (%)']}%",
                           f"{r['Heure/mois']}h", f"{r['Coût/h (€)']}€"])
    tt = Table(tech_data, colWidths=[32*mm,55*mm,22*mm,22*mm,20*mm,24*mm,18*mm])
    st_tech = table_style_default(PDF_BLUE)
    for i, r in enumerate(tech_data[1:], 1):
        if "✓" in r[2]:
            st_tech.add('TEXTCOLOR', (2,i), (2,i), PDF_GREEN)
        else:
            st_tech.add('TEXTCOLOR', (2,i), (2,i), PDF_ACCENT)
        st_tech.add('FONTNAME', (2,i), (2,i), 'Helvetica-Bold')
    tt.setStyle(st_tech)
    story.append(tt)

    doc.build(story, onFirstPage=hf, onLaterPages=hf)
    buf.seek(0)
    return buf

# ── PDF 6: Rapport KPI ────────────────────────────────────────────────────────
def generate_rapport_kpi():
    buf = io.BytesIO()
    S = get_pdf_styles()
    eq = st.session_state.equipements
    bt = st.session_state.bons_travaux
    hist = st.session_state.interventions_hist

    def hf(c, d): make_header_footer(c, d, "RAPPORT KPI & PERFORMANCE", "Analyse mensuelle")
    doc = SimpleDocTemplate(buf, pagesize=A4,
        topMargin=35*mm, bottomMargin=18*mm, leftMargin=15*mm, rightMargin=15*mm)
    story = []
    w = 165*mm

    story.append(Paragraph("TABLEAU DE BORD KPI — PERFORMANCE MAINTENANCE", S['h1']))
    story.append(Spacer(1, 4*mm))

    # KPIs principaux
    taux_dispo = round(len(eq[eq["Statut"]=="Opérationnel"])/len(eq)*100,1)
    taux_prev = round(len(bt[bt["Type"]=="Préventif"])/len(bt)*100,1)
    mtbf = 720
    mttr = 3.8

    kpi_rows = [
        ["KPI","Valeur actuelle","Objectif","Tendance","Évaluation"],
        ["Taux de disponibilité", f"{taux_dispo}%", "≥ 95%", "↑ +2.1%", "⚠ En dessous" if taux_dispo<95 else "✓ Atteint"],
        ["MTBF (Temps moyen entre pannes)", f"{mtbf} h", "≥ 700 h", "↑ Stable", "✓ Atteint"],
        ["MTTR (Temps moyen de réparation)", f"{mttr} h", "≤ 4 h", "↓ -0.3h", "✓ Atteint"],
        ["Taux maintenance préventive", f"{taux_prev}%", "≥ 70%", "↑ +3%", "✓ Atteint" if taux_prev>=70 else "⚠ En dessous"],
        ["Taux de réalisation des BT", f"{round(len(bt[bt['Statut']=='Terminé'])/len(bt)*100,1)}%", "≥ 85%", "→ Stable", "⚠ En dessous"],
        ["Coût maintenance / valeur actifs", f"{round(bt['Coût estimé (€)'].sum()/eq['Valeur (€)'].sum()*100,2)}%", "≤ 3%", "↓ Bon", "✓ Atteint"],
        ["Pièces en rupture de stock", f"{len(st.session_state.pieces[st.session_state.pieces['Stock']<st.session_state.pieces['Min']])}", "= 0", "→", "⚠ À corriger"],
    ]
    kt = Table(kpi_rows, colWidths=[55*mm,28*mm,22*mm,22*mm,32*mm])
    skt = table_style_default()
    for i, r in enumerate(kpi_rows[1:], 1):
        if "✓" in r[4]:
            skt.add('TEXTCOLOR', (4,i), (4,i), PDF_GREEN)
            skt.add('BACKGROUND', (4,i), (4,i), colors.HexColor("#E8F5E9"))
        elif "⚠" in r[4]:
            skt.add('TEXTCOLOR', (4,i), (4,i), PDF_ORANGE)
            skt.add('BACKGROUND', (4,i), (4,i), colors.HexColor("#FFF8E1"))
        skt.add('FONTNAME', (4,i), (4,i), 'Helvetica-Bold')
    kt.setStyle(skt)
    story.append(kt)
    story.append(Spacer(1, 5*mm))

    # Évolution
    story.append(Paragraph("ÉVOLUTION DES INDICATEURS (7 DERNIERS MOIS)", S['h2']))
    hist_data = [["Mois","Correctifs","Préventifs","Total","Coût (€)","Dispo (%)"]]
    for _, r in hist.iterrows():
        hist_data.append([r["Mois"], str(r["Correctifs"]), str(r["Préventifs"]),
                          str(r["Correctifs"]+r["Préventifs"]), f"{r['Coût total (€)']:,.0f}€",
                          f"{r['Disponibilité (%)']:.1f}%"])
    ht = Table(hist_data, colWidths=[22*mm,25*mm,25*mm,20*mm,35*mm,32*mm])
    ht.setStyle(table_style_default(PDF_BLUE))
    story.append(ht)
    story.append(Spacer(1, 5*mm))

    # Analyse par site
    story.append(Paragraph("ANALYSE PAR SITE", S['h2']))
    sites = eq["Site"].unique()
    site_data = [["Site","Total équipements","Opérationnels","En panne","En maintenance","Taux dispo"]]
    for site in sites:
        seq = eq[eq["Site"]==site]
        op = len(seq[seq["Statut"]=="Opérationnel"])
        ep = len(seq[seq["Statut"]=="En panne"])
        em = len(seq[seq["Statut"]=="En maintenance"])
        td = round(op/len(seq)*100,0)
        site_data.append([site, str(len(seq)), str(op), str(ep), str(em), f"{td}%"])
    st_tbl = Table(site_data, colWidths=[28*mm,32*mm,28*mm,22*mm,28*mm,25*mm])
    sst = table_style_default()
    for i in range(1, len(site_data)):
        sst.add('TEXTCOLOR', (5,i), (5,i), PDF_GREEN if float(site_data[i][5].replace("%",""))>=90 else PDF_ORANGE)
        sst.add('FONTNAME', (5,i), (5,i), 'Helvetica-Bold')
    st_tbl.setStyle(sst)
    story.append(st_tbl)
    story.append(Spacer(1, 5*mm))

    # Recommandations
    story.append(Paragraph("RECOMMANDATIONS", S['h2']))
    reco_data = [
        ["N°","Recommandation","Priorité","Échéance"],
        ["1","Augmenter le stock de roulement SKF 6205-2RS (rupture)","Critique","Immédiat"],
        ["2","Planifier l'inspection annuelle de la chaudière CV-50","Critique","< 7 jours"],
        ["3","Réparer le convoyeur CB-12 pour restaurer la disponibilité","Haute","< 3 jours"],
        ["4","Mettre à jour le plan de maintenance préventive robot RS-7","Haute","< 15 jours"],
        ["5","Commander filtre air compresseur avant rupture totale","Moyenne","< 30 jours"],
    ]
    rt = Table(reco_data, colWidths=[10*mm,95*mm,22*mm,28*mm])
    srt = table_style_default()
    prio_map = {"Critique": PDF_ACCENT, "Haute": PDF_ORANGE, "Moyenne": PDF_BLUE}
    for i, r in enumerate(reco_data[1:], 1):
        c = prio_map.get(r[2], PDF_GRAY)
        srt.add('TEXTCOLOR', (2,i), (2,i), c)
        srt.add('FONTNAME', (2,i), (2,i), 'Helvetica-Bold')
    rt.setStyle(srt)
    story.append(rt)

    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width=w, color=PDF_BLUE, thickness=1))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(f"Rapport généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} — GMAO Pro+", S['footer']))

    doc.build(story, onFirstPage=hf, onLaterPages=hf)
    buf.seek(0)
    return buf

# ── PDF 7: Merge multiple PDFs ────────────────────────────────────────────────
def merge_pdfs(pdf_buffers, titles):
    writer = PdfWriter()
    for buf in pdf_buffers:
        reader = PdfReader(buf)
        for page in reader.pages:
            writer.add_page(page)
    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out

def pdf_download_button(buf, filename, label, key):
    data = buf.getvalue() if hasattr(buf, 'getvalue') else buf.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="text-decoration:none;"><button style="background:linear-gradient(135deg,#c62828,#ef5350);color:white;border:none;border-radius:8px;padding:10px 20px;font-family:Rajdhani,sans-serif;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;cursor:pointer;font-size:13px;width:100%;">📄 {label}</button></a>'
    st.markdown(href, unsafe_allow_html=True)

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">⚙ GMAO PRO+</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;letter-spacing:3px;color:#2a4a6a;text-transform:uppercase;margin-bottom:16px;">Maintenance Avancée</div>', unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1a2d48;margin:12px 0;'>", unsafe_allow_html=True)

    menu = st.radio("", [
        "🏠  Dashboard", "🔧  Équipements", "📋  Bons de travaux",
        "📦  Stock & Pièces", "👷  Techniciens", "📅  Planning",
        "📊  KPIs & Rapports", "📄  Centre PDF", "⚙️  Paramètres"
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1a2d48;margin:12px 0;'>", unsafe_allow_html=True)

    eq = st.session_state.equipements
    bt = st.session_state.bons_travaux
    pc = st.session_state.pieces
    n_pannes = len(eq[eq["Statut"]=="En panne"])
    n_stock = len(pc[pc["Stock"]<pc["Min"]])
    n_crit = len(bt[(bt["Priorité"]=="Critique")&(bt["Statut"]!="Terminé")])

    st.markdown(f"""
    <div style='font-size:10px;letter-spacing:2px;color:#2a4a6a;text-transform:uppercase;margin-bottom:8px;'>Alertes</div>
    <div class='alert-box'>🔴 {n_pannes} équipement(s) en panne</div>
    <div class='alert-box'>🟡 {n_stock} pièce(s) rupture stock</div>
    <div class='alert-box'>🟠 {n_crit} BT critique(s) ouverts</div>
    """, unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:10px;color:#1a3a5a;margin-top:12px;'>🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>", unsafe_allow_html=True)

page = menu.split("  ")[-1].strip()

# ═══════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════

# ── DASHBOARD ────────────────────────────────────────────────────────────────
if page == "Dashboard":
    st.markdown("<div style='font-family:Rajdhani;font-size:30px;font-weight:700;letter-spacing:3px;color:#90c0e8;'>TABLEAU DE BORD</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:12px;color:#2a4a6a;margin-bottom:20px;'>{datetime.now().strftime('%A %d %B %Y')}</div>", unsafe_allow_html=True)

    taux_dispo = round(len(eq[eq["Statut"]=="Opérationnel"])/len(eq)*100,1)
    bt_actifs = len(bt[bt["Statut"].isin(["Ouvert","En cours","Planifié"])])
    cout_mois = bt["Coût estimé (€)"].sum()
    val_stock = (pc["Stock"]*pc["Prix (€)"]).sum()
    mtbf_val = 720
    taux_prev = round(len(bt[bt["Type"]=="Préventif"])/len(bt)*100,1)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpis = [
        (c1, f"{taux_dispo}%","Disponibilité","↑ +2.1% vs N-1","success" if taux_dispo>=90 else "warn"),
        (c2, f"{bt_actifs}","BT Actifs",f"{n_crit} critiques","warn"),
        (c3, f"{cout_mois:,.0f}€","Coût total","Budget: 15 000€",""),
        (c4, f"{val_stock:,.0f}€","Valeur stock",f"{n_stock} rupture(s)","danger" if n_stock>0 else "success"),
        (c5, f"{mtbf_val}h","MTBF moyen","↑ Stable","success"),
        (c6, f"{taux_prev}%","Taux préventif","Obj: ≥70%","success" if taux_prev>=70 else "warn"),
    ]
    for col, val, label, sub, cls in kpis:
        with col:
            st.markdown(f"""<div class='kpi-card {cls}'>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-value'>{val}</div>
                <div class='kpi-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([3,2])

    with col1:
        st.markdown("<div class='section-title'>Évolution interventions (7 mois)</div>", unsafe_allow_html=True)
        hist = st.session_state.interventions_hist
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Correctifs", x=hist["Mois"], y=hist["Correctifs"], marker_color="#ef5350", opacity=0.85))
        fig.add_trace(go.Bar(name="Préventifs", x=hist["Mois"], y=hist["Préventifs"], marker_color="#1565C0", opacity=0.85))
        fig.add_trace(go.Scatter(name="Disponibilité %", x=hist["Mois"], y=hist["Disponibilité (%)"], mode="lines+markers", yaxis="y2", marker=dict(color="#00e676",size=7), line=dict(color="#00e676",width=2)))
        fig.update_layout(barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#7090a0"),
            yaxis2=dict(overlaying="y", side="right", range=[80,100], color="#00e676"),
            legend=dict(font=dict(color="#7090a0"), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#152030"), yaxis=dict(gridcolor="#152030"),
            margin=dict(l=0,r=0,t=10,b=0), height=260)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Répartition statuts</div>", unsafe_allow_html=True)
        sc = eq["Statut"].value_counts()
        fig2 = go.Figure(go.Pie(labels=sc.index, values=sc.values, hole=0.55,
            marker=dict(colors=["#00cc66","#ff4060","#aa66ff","#ff8800"])))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#7090a0"),
            legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=0,r=0,t=0,b=0), height=260)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-title'>BT prioritaires</div>", unsafe_allow_html=True)
        pm = {"Critique":"critical","Haute":"high","Moyenne":"medium","Basse":"low"}
        sm = {"Ouvert":"open","En cours":"progress","Terminé":"closed","Planifié":"planned"}
        for _, r in bt[bt["Statut"]!="Terminé"].sort_values("Priorité", key=lambda x: x.map({"Critique":0,"Haute":1,"Moyenne":2,"Basse":3})).head(4).iterrows():
            st.markdown(f"""<div style='background:#0e1828;border:1px solid #1a3355;border-radius:8px;padding:10px 14px;margin:5px 0;'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <span style='font-family:Rajdhani;font-size:13px;color:#90c0e8;font-weight:700;'>{r['BT']}</span>
                    <div><span class='badge badge-{pm.get(r["Priorité"],"low")}'>{r['Priorité']}</span>&nbsp;<span class='badge badge-{sm.get(r["Statut"],"open")}'>{r['Statut']}</span></div>
                </div>
                <div style='font-size:12px;color:#c0d8f0;margin-top:3px;'>{r['Titre']}</div>
                <div style='font-size:11px;color:#2a4a6a;margin-top:2px;'>👷 {r['Technicien']} · 📅 {r['Date prévue']} · 💰 {r['Coût estimé (€)']}€</div>
            </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='section-title'>Équipements hors service</div>", unsafe_allow_html=True)
        for _, r in eq[eq["Statut"]!="Opérationnel"].iterrows():
            dc = {"En panne":"#ff4060","En maintenance":"#aa66ff"}.get(r["Statut"],"#ff8800")
            st.markdown(f"""<div style='background:#0e1828;border:1px solid #1a3355;border-radius:8px;padding:10px 14px;margin:5px 0;display:flex;align-items:center;gap:10px;'>
                <div style='width:10px;height:10px;border-radius:50%;background:{dc};box-shadow:0 0 8px {dc};flex-shrink:0;'></div>
                <div>
                    <div style='font-family:Rajdhani;font-size:14px;color:#c0d8f0;font-weight:600;'>{r['Nom']}</div>
                    <div style='font-size:11px;color:#2a4a6a;'>{r['ID']} · {r['Site']} · <span style='color:{dc};'>{r['Statut']}</span></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Radar chart criticité
        cats = ["Pneumatique","Hydraulique","Électrique","Mécanique","Thermique"]
        vals = [3,2,2,1,1]
        fig3 = go.Figure(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]], fill='toself', fillcolor='rgba(21,101,192,0.2)', line=dict(color='#1565C0')))
        fig3.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", angularaxis=dict(color="#7090a0"), radialaxis=dict(color="#7090a0", gridcolor="#152030")),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#7090a0"), margin=dict(l=20,r=20,t=20,b=20), height=200, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

# ── ÉQUIPEMENTS ──────────────────────────────────────────────────────────────
elif page == "Équipements":
    st.markdown("<div style='font-family:Rajdhani;font-size:30px;font-weight:700;letter-spacing:3px;color:#90c0e8;'>ÉQUIPEMENTS</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Liste", "➕ Ajouter", "✏️ Modifier"])

    with tab1:
        c1,c2,c3,c4 = st.columns(4)
        with c1: f_site = st.selectbox("Site", ["Tous"]+list(eq["Site"].unique()))
        with c2: f_stat = st.selectbox("Statut", ["Tous"]+list(eq["Statut"].unique()))
        with c3: f_crit = st.selectbox("Criticité", ["Tous"]+list(eq["Criticité"].unique()))
        with c4: search = st.text_input("🔍 Recherche", "")

        df = eq.copy()
        if f_site!="Tous": df=df[df["Site"]==f_site]
        if f_stat!="Tous": df=df[df["Statut"]==f_stat]
        if f_crit!="Tous": df=df[df["Criticité"]==f_crit]
        if search: df=df[df.apply(lambda r: search.lower() in r.to_string().lower(),axis=1)]

        def style_eq(row):
            styles = [""]*len(row)
            idx = row.index.tolist()
            if "Statut" in idx:
                sc = {"Opérationnel":"color:#00cc66;font-weight:700","En panne":"color:#ff4060;font-weight:700","En maintenance":"color:#aa66ff;font-weight:700"}.get(row["Statut"],"")
                styles[idx.index("Statut")] = sc
            if "Criticité" in idx:
                cc = {"Critique":"color:#ff4060","Haute":"color:#ff8800","Moyenne":"color:#ffcc00","Basse":"color:#00cc66"}.get(row["Criticité"],"")
                styles[idx.index("Criticité")] = cc
            return styles

        st.dataframe(df.style.apply(style_eq, axis=1), use_container_width=True, hide_index=True, height=380)
        st.markdown(f"<div style='font-size:12px;color:#2a4a6a;'>{len(df)} équipement(s)</div>", unsafe_allow_html=True)

        # Charts
        col1,col2 = st.columns(2)
        with col1:
            fig = px.bar(eq.groupby(["Site","Statut"]).size().reset_index(name="n"), x="Site", y="n", color="Statut",
                color_discrete_map={"Opérationnel":"#00cc66","En panne":"#ff4060","En maintenance":"#aa66ff"}, title="Statut par site")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#7090a0"),height=280,margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.sunburst(eq, path=["Site","Catégorie"], title="Répartition parc")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#7090a0"),height=280,margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        with st.form("form_eq"):
            c1,c2 = st.columns(2)
            with c1:
                nom = st.text_input("Nom équipement *")
                cat = st.selectbox("Catégorie", ["Pneumatique","Hydraulique","Électrique","Mécanique","Robotique","Usinage","Manutention","Levage","Thermique","Autre"])
                site = st.selectbox("Site", ["Usine A","Usine B","Usine C","Entrepôt"])
                crit = st.selectbox("Criticité", ["Critique","Haute","Moyenne","Basse"])
                resp = st.text_input("Responsable")
            with c2:
                stat = st.selectbox("Statut", ["Opérationnel","En maintenance","En panne"])
                d_install = st.date_input("Date installation", value=date.today())
                heures = st.number_input("Heures initiales", min_value=0, value=0)
                valeur = st.number_input("Valeur actif (€)", min_value=0, value=10000)
                notes = st.text_area("Notes", height=80)
            if st.form_submit_button("✅ Enregistrer"):
                if nom:
                    nid = f"EQ-{len(eq)+1:03d}"
                    new = {"ID":nid,"Nom":nom,"Catégorie":cat,"Site":site,"Statut":stat,"Criticité":crit,
                           "Installation":str(d_install),"Dernière maintenance":"N/A","Prochaine maintenance":"N/A",
                           "Heures":heures,"Valeur (€)":valeur,"Responsable":resp}
                    st.session_state.equipements = pd.concat([eq, pd.DataFrame([new])], ignore_index=True)
                    st.markdown(f"<div class='success-box'>✅ {nid} — {nom} ajouté!</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='alert-box'>⚠️ Nom obligatoire.</div>", unsafe_allow_html=True)

    with tab3:
        eq_sel = st.selectbox("Sélectionner équipement", eq["ID"]+" — "+eq["Nom"])
        eid = eq_sel.split(" — ")[0]
        row = eq[eq["ID"]==eid].iloc[0]
        with st.form("form_edit_eq"):
            c1,c2 = st.columns(2)
            with c1:
                new_stat = st.selectbox("Statut", ["Opérationnel","En maintenance","En panne"], index=["Opérationnel","En maintenance","En panne"].index(row["Statut"]))
                new_heures = st.number_input("Heures de marche", min_value=0, value=int(row["Heures"]))
            with c2:
                new_dm = st.text_input("Dernière maintenance", value=row["Dernière maintenance"])
                new_pm = st.text_input("Prochaine maintenance", value=row["Prochaine maintenance"])
            if st.form_submit_button("💾 Mettre à jour"):
                idx = st.session_state.equipements[st.session_state.equipements["ID"]==eid].index[0]
                st.session_state.equipements.at[idx,"Statut"] = new_stat
                st.session_state.equipements.at[idx,"Heures"] = new_heures
                st.session_state.equipements.at[idx,"Dernière maintenance"] = new_dm
                st.session_state.equipements.at[idx,"Prochaine maintenance"] = new_pm
                st.markdown(f"<div class='success-box'>✅ {eid} mis à jour!</div>", unsafe_allow_html=True)

# ── BONS DE TRAVAUX ──────────────────────────────────────────────────────────
elif page == "Bons de travaux":
    st.markdown("<div style='font-family:Rajdhani;font-size:30px;font-weight:700;letter-spacing:3px;color:#90c0e8;'>BONS DE TRAVAUX</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Liste", "➕ Créer BT", "📝 Mise à jour"])

    with tab1:
        c1,c2,c3 = st.columns(3)
        with c1: ft = st.selectbox("Type", ["Tous","Correctif","Préventif","Prédictif"])
        with c2: fp = st.selectbox("Priorité", ["Tous","Critique","Haute","Moyenne","Basse"])
        with c3: fs = st.selectbox("Statut", ["Tous","Ouvert","En cours","Planifié","Terminé"])

        dbt = bt.copy()
        if ft!="Tous": dbt=dbt[dbt["Type"]==ft]
        if fp!="Tous": dbt=dbt[dbt["Priorité"]==fp]
        if fs!="Tous": dbt=dbt[dbt["Statut"]==fs]

        pm={"Critique":"critical","Haute":"high","Moyenne":"medium","Basse":"low"}
        sm={"Ouvert":"open","En cours":"progress","Terminé":"closed","Planifié":"planned"}

        for _, r in dbt.iterrows():
            st.markdown(f"""<div style='background:#0e1828;border:1px solid #1a3355;border-radius:10px;padding:12px 16px;margin:6px 0;'>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>
                    <div style='display:flex;align-items:center;gap:10px;'>
                        <span style='font-family:Rajdhani;font-size:13px;color:#4a88cd;font-weight:700;'>{r['BT']}</span>
                        <span style='font-family:Rajdhani;font-size:15px;color:#c8e0f8;font-weight:600;'>{r['Titre']}</span>
                        <span class='badge' style='background:#0a1a2a;color:#4488aa;border:1px solid #4488aa30;font-size:10px;'>{r['Type']}</span>
                    </div>
                    <div><span class='badge badge-{pm.get(r["Priorité"],"low")}'>{r['Priorité']}</span>&nbsp;<span class='badge badge-{sm.get(r["Statut"],"open")}'>{r['Statut']}</span></div>
                </div>
                <div style='font-size:11px;color:#2a4a6a;display:flex;gap:18px;flex-wrap:wrap;'>
                    <span>🔧 {r['Équipement']}</span><span>👷 {r['Technicien']}</span>
                    <span>📅 {r['Date prévue']}</span><span>⏱ {r['Durée (h)']}h</span>
                    <span>💰 {r['Coût estimé (€)']}€</span><span>👤 {r['Demandeur']}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    with tab2:
        with st.form("form_bt"):
            c1,c2 = st.columns(2)
            with c1:
                titre = st.text_input("Titre *")
                eq_sel = st.selectbox("Équipement", eq["ID"]+" — "+eq["Nom"])
                type_bt = st.selectbox("Type", ["Correctif","Préventif","Prédictif","Amélioratif"])
                prio = st.selectbox("Priorité", ["Critique","Haute","Moyenne","Basse"])
            with c2:
                tech_sel = st.selectbox("Technicien", ["Non assigné"]+list(st.session_state.techniciens["Nom"]))
                d_prevue = st.date_input("Date prévue", value=date.today()+timedelta(days=3))
                duree = st.number_input("Durée estimée (h)", min_value=0.5, value=2.0, step=0.5)
                cout = st.number_input("Coût estimé (€)", min_value=0, value=200)
            demandeur = st.text_input("Demandeur", value="Utilisateur")
            desc = st.text_area("Description détaillée", height=80)
            if st.form_submit_button("✅ Créer BT"):
                if titre:
                    nbt = f"BT-{datetime.now().year}-{len(bt)+1:03d}"
                    eq_id = eq_sel.split(" — ")[0]
                    new = {"BT":nbt,"Équipement":eq_id,"Titre":titre,"Type":type_bt,"Priorité":prio,"Statut":"Ouvert",
                           "Demandeur":demandeur,"Technicien":tech_sel,"Date création":str(date.today()),
                           "Date prévue":str(d_prevue),"Durée (h)":duree,"Coût estimé (€)":cout,"Coût réel (€)":0,"Description":desc}
                    st.session_state.bons_travaux = pd.concat([bt, pd.DataFrame([new])], ignore_index=True)
                    st.markdown(f"<div class='success-box'>✅ {nbt} créé!</div>", unsafe_allow_html=True)

    with tab3:
        bt_sel = st.selectbox("Sélectionner BT", bt["BT"]+" — "+bt["Titre"])
        bid = bt_sel.split(" — ")[0]
        brow = bt[bt["BT"]==bid].iloc[0]
        with st.form("form_update_bt"):
            c1,c2 = st.columns(2)
            with c1:
                new_stat = st.selectbox("Statut", ["Ouvert","En cours","Planifié","Terminé"], index=["Ouvert","En cours","Planifié","Terminé"].index(brow["Statut"]) if brow["Statut"] in ["Ouvert","En cours","Planifié","Terminé"] else 0)
                new_tech = st.selectbox("Technicien", ["Non assigné"]+list(st.session_state.techniciens["Nom"]))
            with c2:
                cout_reel = st.number_input("Coût réel (€)", min_value=0, value=int(brow["Coût réel (€)"]))
                notes_interv = st.text_area("Notes intervention", height=80)
            if st.form_submit_button("💾 Mettre à jour"):
                idx = st.session_state.bons_travaux[st.session_state.bons_travaux["BT"]==bid].index[0]
                st.session_state.bons_travaux.at[idx,"Statut"] = new_stat
                st.session_state.bons_travaux.at[idx,"Technicien"] = new_tech
                st.session_state.bons_travaux.at[idx,"Coût réel (€)"] = cout_reel
                st.markdown(f"<div class='success-box'>✅ {bid} mis à jour → {new_stat}</div>", unsafe_allow_html=True)

# ── STOCK & PIÈCES ────────────────────────────────────────────────────────────
elif page == "Stock & Pièces":
    st.markdown("<div style='font-family:Rajdhani;font-size:30px;font-weight:700;letter-spacing:3px;color:#90c0e8;'>STOCK & PIÈCES</div>", unsafe_allow_html=True)

    ruptures = pc[pc["Stock"]<pc["Min"]]
    for _, r in ruptures.iterrows():
        st.markdown(f"<div class='alert-box'>⚠️ <b>{r['Désignation']}</b> ({r['Réf']}) — Stock: <b>{r['Stock']}</b> / Min: {r['Min']} · {r['Fournisseur']} · Délai: {r['Délai (j)']}j · Commande suggérée: {r['Max']-r['Stock']} unités</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📦 Inventaire", "➕ Ajouter pièce", "📊 Analyse stock"])

    with tab1:
        search_p = st.text_input("🔍 Rechercher", "")
        df = pc.copy()
        if search_p: df=df[df.apply(lambda r: search_p.lower() in r.to_string().lower(),axis=1)]

        def style_stock(row):
            styles=[""]*len(row)
            idx=row.index.tolist()
            if "Stock" in idx:
                si=idx.index("Stock")
                if row["Stock"]<row["Min"]: styles[si]="color:#ff4060;font-weight:700;background-color:#2a0808"
                elif row["Stock"]<row["Min"]*1.5: styles[si]="color:#ff8800;font-weight:600"
                else: styles[si]="color:#00cc66"
            return styles

        st.dataframe(df.style.apply(style_stock,axis=1), use_container_width=True, hide_index=True, height=350)

        csv = pc.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export CSV stock", csv, "stock.csv", "text/csv")

    with tab2:
        with st.form("form_piece"):
            c1,c2=st.columns(2)
            with c1:
                ref=st.text_input("Référence *")
                desig=st.text_input("Désignation *")
                cat=st.selectbox("Catégorie",["Roulements","Transmission","Étanchéité","Lubrifiants","Filtration","Électrique","Instrumentation","Visserie","Autre"])
                fourn=st.text_input("Fournisseur")
                empl=st.text_input("Emplacement")
            with c2:
                stock_i=st.number_input("Stock initial",min_value=0,value=10)
                stock_min=st.number_input("Stock minimum",min_value=0,value=3)
                stock_max=st.number_input("Stock maximum",min_value=0,value=30)
                prix=st.number_input("Prix unitaire (€)",min_value=0.0,value=10.0,step=0.5)
                delai=st.number_input("Délai livraison (j)",min_value=1,value=5)
            if st.form_submit_button("✅ Ajouter"):
                if ref and desig:
                    new={"Réf":ref,"Désignation":desig,"Catégorie":cat,"Stock":stock_i,"Min":stock_min,"Max":stock_max,"Prix (€)":prix,"Fournisseur":fourn,"Délai (j)":delai,"Emplacement":empl}
                    st.session_state.pieces=pd.concat([pc,pd.DataFrame([new])],ignore_index=True)
                    st.markdown(f"<div class='success-box'>✅ {ref} — {desig} ajouté!</div>",unsafe_allow_html=True)

    with tab3:
        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(name="Stock actuel", x=pc["Désignation"].str[:18], y=pc["Stock"], marker_color="#1565C0"))
        fig_s.add_trace(go.Scatter(name="Min", x=pc["Désignation"].str[:18], y=pc["Min"], mode="lines+markers", line=dict(color="#ff4060",dash="dash")))
        fig_s.add_trace(go.Scatter(name="Max", x=pc["Désignation"].str[:18], y=pc["Max"], mode="lines", line=dict(color="#00cc66",dash="dot")))
        fig_s.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#7090a0"),height=320,margin=dict(l=0,r=0,t=10,b=0),xaxis=dict(gridcolor="#152030"),yaxis=dict(gridcolor="#152030"))
        st.plotly_chart(fig_s, use_container_width=True)

        val_cat = (pc["Stock"]*pc["Prix (€)"]).groupby(pc["Catégorie"]).sum().reset_index()
        val_cat.columns=["Catégorie","Valeur (€)"]
        fig_v = px.pie(val_cat, names="Catégorie", values="Valeur (€)", title="Valeur stock par catégorie", hole=0.4)
        fig_v.update_layout(paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#7090a0"),height=300,margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_v, use_container_width=True)

# ── TECHNICIENS ────────────────────────────────────────────────────────────────
elif page == "Techniciens":
    st.markdown("<div style='font-family:Rajdhani;font-size:30px;font-weight:700;letter-spacing:3px;color:#90c0e8;'>TECHNICIENS</div>", unsafe_allow_html=True)

    tech = st.session_state.techniciens
    cols = st.columns(len(tech))
    for i,(_, r) in enumerate(tech.iterrows()):
        with cols[i]:
            dc = "#00cc66" if r["Disponible"] else "#ff4060"
            ec = "#00cc66" if r["Efficacité (%)"]>=90 else "#ff8800" if r["Efficacité (%)"]>=80 else "#ff4060"
            st.markdown(f"""<div style='background:#0e1828;border:1px solid #1a3355;border-radius:12px;padding:18px;text-align:center;'>
                <div style='width:55px;height:55px;border-radius:50%;background:linear-gradient(135deg,#0d47a1,#1976D2);display:flex;align-items:center;justify-content:center;margin:0 auto 10px;font-size:22px;'>👷</div>
                <div style='font-family:Rajdhani;font-size:15px;font-weight:700;color:#c8e0f8;'>{r['Nom']}</div>
                <div style='font-size:10px;color:#3a6a9a;margin:3px 0;'>{r['Spécialité']}</div>
                <div style='margin:8px 0;'><span style='display:inline-block;width:7px;height:7px;border-radius:50%;background:{dc};margin-right:4px;'></span><span style='font-size:11px;color:{dc};'>{"Disponible" if r["Disponible"] else "Occupé"}</span></div>
                <div style='display:flex;justify-content:space-around;margin-top:10px;padding-top:10px;border-top:1px solid #1a2d48;'>
                    <div><div style='font-family:Rajdhani;font-size:22px;font-weight:700;color:#1565C0;'>{r['BT en cours']}</div><div style='font-size:9px;color:#2a4a6a;'>BT actifs</div></div>
                    <div><div style='font-family:Rajdhani;font-size:22px;font-weight:700;color:{ec};'>{r['Efficacité (%)']}%</div><div style='font-size:9px;color:#2a4a6a;'>Efficacité</div></div>
                    <div><div style='font-family:Rajdhani;font-size:22px;font-weight:700;color:#00cc66;'>{r['Coût/h (€)']}€</div><div style='font-size:9px;color:#2a4a6a;'>/heure</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        tech_bt = bt[bt["Statut"].isin(["En cours","Planifié"])].groupby("Technicien").agg({"BT":"count","Durée (h)":"sum"}).reset_index()
        if len(tech_bt)>0:
            fig_t = go.Figure()
            fig_t.add_trace(go.Bar(name="Nb BT", x=tech_bt["Technicien"], y=tech_bt["BT"], marker_color="#1565C0"))
            fig_t.add_trace(go.Bar(name="Heures", x=tech_bt["Technicien"], y=tech_bt["Durée (h)"], marker_color="#00cc66"))
            fig_t.update_layout(barmode="group",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#7090a0"),height=280,margin=dict(l=0,r=0,t=10,b=0),xaxis=dict(gridcolor="#152030"),yaxis=dict(gridcolor="#152030"))
            st.plotly_chart(fig_t, use_container_width=True)
    with col2:
        fig_eff = go.Figure(go.Bar(
            x=tech["Efficacité (%)"], y=tech["Nom"], orientation="h",
            marker=dict(color=tech["Efficacité (%)"], colorscale=[[0,"#ff4060"],[0.5,"#ff8800"],[1,"#00cc66"]]),
            text=tech["Efficacité (%)"].astype(str)+"%", textposition="outside"
        ))
        fig_eff.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#7090a0"),height=280,margin=dict(l=0,r=40,t=10,b=0),xaxis=dict(range=[0,105],gridcolor="#152030"),yaxis=dict(color="#7090a0"))
        st.plotly_chart(fig_eff, use_container_width=True)

# ── PLANNING ──────────────────────────────────────────────────────────────────
elif page == "Planning":
    st.markdown("<div style='font-family:Rajdhani;font-size:30px;font-weight:700;letter-spacing:3px;color:#90c0e8;'>PLANNING MAINTENANCE</div>", unsafe_allow_html=True)

    bt_p = bt[bt["Statut"]!="Terminé"].copy()
    bt_p["Date prévue"] = pd.to_datetime(bt_p["Date prévue"])
    bt_p["Fin prévue"] = bt_p["Date prévue"] + pd.to_timedelta(bt_p["Durée (h)"], unit="h")

    color_map = {"Critique":"#ff4060","Haute":"#ff8800","Moyenne":"#1565C0","Basse":"#00cc66"}

    fig_g = go.Figure()
    for i,(_, r) in enumerate(bt_p.iterrows()):
        c = color_map.get(r["Priorité"],"#4488ff")
        dur = max((r["Fin prévue"]-r["Date prévue"]).total_seconds()/86400, 0.2)
        base = (r["Date prévue"]-pd.Timestamp("2025-01-01")).total_seconds()/86400
        fig_g.add_trace(go.Bar(
            y=[f"{r['BT']} | {r['Titre'][:28]}"], x=[dur], base=[base],
            orientation="h", marker=dict(color=c, opacity=0.85),
            hovertemplate=f"<b>{r['BT']}</b><br>{r['Titre']}<br>👷 {r['Technicien']}<br>Priorité: {r['Priorité']}<extra></extra>",
            showlegend=False
        ))

    fig_g.update_layout(barmode="overlay", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7090a0"), height=420, margin=dict(l=0,r=0,t=20,b=0),
        xaxis=dict(tickvals=list(range(0,50,7)), ticktext=[(pd.Timestamp("2025-01-01")+timedelta(days=d)).strftime("%d %b") for d in range(0,50,7)], gridcolor="#152030", color="#7090a0"),
        yaxis=dict(color="#7090a0", gridcolor="#152030"),
        title="Diagramme de Gantt — Interventions planifiées")
    st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("<div class='section-title'>Prochaines maintenances préventives</div>", unsafe_allow_html=True)
    eq_s = eq.copy()
    for _, r in eq_s.sort_values("Prochaine maintenance").head(8).iterrows():
        try:
            nd = pd.to_datetime(r["Prochaine maintenance"])
            delta = (nd - pd.Timestamp.now()).days
            c = "#ff4060" if delta<7 else "#ff8800" if delta<30 else "#00cc66"
            ic = "⚠️" if delta<7 else "🔔" if delta<30 else "📅"
            st.markdown(f"""<div style='background:#0e1828;border:1px solid #1a3355;border-radius:8px;padding:9px 14px;margin:4px 0;display:flex;justify-content:space-between;align-items:center;'>
                <span style='font-family:Rajdhani;font-size:13px;color:#c8e0f8;'>{ic} {r['Nom']} <span style='font-size:11px;color:#2a4a6a;'>· {r['ID']} · {r['Site']}</span></span>
                <span style='font-family:Rajdhani;font-size:14px;color:{c};font-weight:700;'>{r['Prochaine maintenance']} <span style='font-size:12px;font-weight:400;'>({delta}j)</span></span>
            </div>""", unsafe_allow_html=True)
        except: pass

# ── KPIs & RAPPORTS ───────────────────────────────────────────────────────────
elif page == "KPIs & Rapports":
    st.markdown("<div style='font-family:Rajdhani;font-size:30px;font-weight:700;letter-spacing:3px;color:#90c0e8;'>KPIs & RAPPORTS</div>", unsafe_allow_html=True)

    taux_dispo = round(len(eq[eq["Statut"]=="Opérationnel"])/len(eq)*100,1)
    taux_prev = round(len(bt[bt["Type"]=="Préventif"])/len(bt)*100,1)
    cout_total = bt["Coût estimé (€)"].sum()
    val_actifs = eq["Valeur (€)"].sum()

    c1,c2,c3,c4 = st.columns(4)
    for col,(label,val,sub,cls) in zip([c1,c2,c3,c4],[
        ("Disponibilité",f"{taux_dispo}%","Obj: ≥95%","success" if taux_dispo>=95 else "warn"),
        ("Taux préventif",f"{taux_prev}%","Obj: ≥70%","success" if taux_prev>=70 else "warn"),
        ("MTBF moyen","720 h","Obj: ≥700h","success"),
        ("Coût / valeur actifs",f"{round(cout_total/val_actifs*100,2)}%","Obj: ≤3%","success"),
    ]):
        with col: st.markdown(f"""<div class='kpi-card {cls}'><div class='kpi-label'>{label}</div><div class='kpi-value'>{val}</div><div class='kpi-sub'>{sub}</div></div>""",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    col1,col2,col3 = st.columns(3)
    hist = st.session_state.interventions_hist
    with col1:
        fig=px.line(hist,x="Mois",y="Disponibilité (%)",markers=True,title="Disponibilité (%)",color_discrete_sequence=["#00cc66"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#7090a0"),height=250,margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        fig2=px.bar(hist,x="Mois",y="Coût total (€)",title="Coût mensuel (€)",color_discrete_sequence=["#1565C0"])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#7090a0"),height=250,margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig2,use_container_width=True)
    with col3:
        tc=bt["Type"].value_counts()
        fig3=go.Figure(go.Pie(labels=tc.index,values=tc.values,hole=0.5,marker=dict(colors=["#1565C0","#00cc66","#ff8800","#aa66ff"])))
        fig3.update_layout(title="Type interventions",paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#7090a0"),height=250,margin=dict(l=0,r=0,t=40,b=0),legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig3,use_container_width=True)

    # Export CSV
    st.markdown("<div class='section-title'>Exports de données</div>",unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: st.download_button("⬇️ Export BT (CSV)", bt.to_csv(index=False).encode(), "bons_travaux.csv", "text/csv")
    with c2: st.download_button("⬇️ Export Équipements (CSV)", eq.to_csv(index=False).encode(), "equipements.csv", "text/csv")
    with c3: st.download_button("⬇️ Export Stock (CSV)", pc.to_csv(index=False).encode(), "stock.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# ── CENTRE PDF ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Centre PDF":
    st.markdown("<div style='font-family:Rajdhani;font-size:30px;font-weight:700;letter-spacing:3px;color:#90c0e8;'>📄 CENTRE DE GÉNÉRATION PDF</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;color:#2a4a6a;margin-bottom:20px;'>Générez, exportez et fusionnez tous vos documents GMAO en PDF professionnel.</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Rapport mensuel", "📋 Bon de travaux", "🔧 Fiche équipement",
        "📦 Rapport stock", "📅 Planning PDF", "🔗 Fusion PDF"
    ])

    # ── TAB 1: Rapport mensuel
    with tab1:
        st.markdown("<div class='pdf-section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Rapport mensuel complet</div>", unsafe_allow_html=True)
        st.markdown("""<div class='info-box'>
            Ce rapport contient : KPIs exécutifs · État parc équipements · Bons de travaux ·
            Gestion stocks · Évolution mensuelle · Synthèse multi-pages avec en-têtes/pieds de page professionnels.
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            mois_sel = st.selectbox("Période", ["Janvier 2025","Décembre 2024","Novembre 2024"])
            site_sel = st.multiselect("Sites inclus", ["Usine A","Usine B","Usine C"], default=["Usine A","Usine B","Usine C"])
        with col2:
            inclure_stock = st.checkbox("Inclure rapport stock", value=True)
            inclure_hist = st.checkbox("Inclure historique mensuel", value=True)
            inclure_reco = st.checkbox("Inclure recommandations", value=True)

        if st.button("🔴 Générer rapport mensuel PDF", key="btn_rapport"):
            with st.spinner("Génération du rapport en cours..."):
                try:
                    buf = generate_rapport_mensuel()
                    pdf_download_button(buf, f"rapport_mensuel_{mois_sel.replace(' ','_')}.pdf",
                                        f"Télécharger rapport {mois_sel}", "dl_rapport")
                    st.markdown("<div class='success-box'>✅ Rapport généré avec succès! Cliquez pour télécharger.</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"<div class='alert-box'>⚠️ Erreur: {str(e)}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 2: Bon de travaux
    with tab2:
        st.markdown("<div class='pdf-section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Bon de travaux individuel</div>", unsafe_allow_html=True)
        st.markdown("""<div class='info-box'>
            PDF d'intervention avec : En-tête coloré par priorité · Informations complètes ·
            Checklist sécurité · Zone rapport technicien · Signatures officielles.
        </div>""", unsafe_allow_html=True)

        bt_options = bt["BT"] + " — " + bt["Titre"] + " [" + bt["Priorité"] + "]"
        bt_sel = st.selectbox("Sélectionner le bon de travaux", bt_options)
        bt_id = bt_sel.split(" — ")[0]
        brow = bt[bt["BT"]==bt_id].iloc[0]

        pm={"Critique":"critical","Haute":"high","Moyenne":"medium","Basse":"low"}
        sm={"Ouvert":"open","En cours":"progress","Terminé":"closed","Planifié":"planned"}
        st.markdown(f"""<div style='background:#0e1828;border:1px solid #1a3355;border-radius:8px;padding:12px 16px;margin:10px 0;'>
            <div style='display:flex;gap:10px;align-items:center;'>
                <span class='badge badge-{pm.get(brow["Priorité"],"low")}'>{brow['Priorité']}</span>
                <span class='badge badge-{sm.get(brow["Statut"],"open")}'>{brow['Statut']}</span>
                <span style='color:#c0d8f0;font-family:Rajdhani;font-size:14px;'>{brow['Titre']}</span>
            </div>
            <div style='font-size:11px;color:#2a4a6a;margin-top:6px;'>Équipement: {brow['Équipement']} · Technicien: {brow['Technicien']} · Prévu: {brow['Date prévue']} · {brow['Durée (h)']}h · {brow['Coût estimé (€)']}€</div>
        </div>""", unsafe_allow_html=True)

        col1,col2 = st.columns(2)
        with col1:
            if st.button("🔴 Générer BT PDF", key="btn_bt"):
                with st.spinner("Génération..."):
                    try:
                        buf = generate_bon_travaux(bt_id)
                        pdf_download_button(buf, f"BT_{bt_id}.pdf", f"Télécharger {bt_id}", "dl_bt")
                        st.markdown("<div class='success-box'>✅ BT généré!</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f"<div class='alert-box'>Erreur: {e}</div>", unsafe_allow_html=True)
        with col2:
            if st.button("🔴 Générer TOUS les BT actifs", key="btn_allbt"):
                with st.spinner("Génération de tous les BT..."):
                    try:
                        bufs = []
                        bt_actifs = bt[bt["Statut"]!="Terminé"]
                        for _, r in bt_actifs.iterrows():
                            bufs.append(generate_bon_travaux(r["BT"]))
                        merged = merge_pdfs(bufs, [])
                        pdf_download_button(merged, "tous_BT_actifs.pdf", f"Télécharger {len(bufs)} BT", "dl_allbt")
                        st.markdown(f"<div class='success-box'>✅ {len(bufs)} BT générés et fusionnés!</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f"<div class='alert-box'>Erreur: {e}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 3: Fiche équipement
    with tab3:
        st.markdown("<div class='pdf-section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Fiche technique équipement</div>", unsafe_allow_html=True)
        st.markdown("""<div class='info-box'>
            Fiche complète avec : Données techniques · KPIs de performance · Historique interventions ·
            Plan de maintenance préventive · Pièces de rechange associées.
        </div>""", unsafe_allow_html=True)

        eq_options = eq["ID"] + " — " + eq["Nom"] + " [" + eq["Statut"] + "]"
        eq_sel = st.selectbox("Sélectionner équipement", eq_options)
        eq_id = eq_sel.split(" — ")[0]
        erow = eq[eq["ID"]==eq_id].iloc[0]

        sc_css = {"Opérationnel":"low","En panne":"critical","En maintenance":"progress"}.get(erow["Statut"],"open")
        st.markdown(f"""<div style='background:#0e1828;border:1px solid #1a3355;border-radius:8px;padding:10px 14px;margin:8px 0;'>
            <span class='badge badge-{sc_css}'>{erow['Statut']}</span>&nbsp;
            <span style='color:#c0d8f0;font-family:Rajdhani;font-size:14px;font-weight:600;'>{erow['Nom']}</span>
            <div style='font-size:11px;color:#2a4a6a;margin-top:4px;'>Site: {erow['Site']} · Criticité: {erow['Criticité']} · Heures: {erow['Heures']:,}h · Valeur: {erow['Valeur (€)']:,}€</div>
        </div>""", unsafe_allow_html=True)

        col1,col2 = st.columns(2)
        with col1:
            if st.button("🔴 Générer fiche équipement", key="btn_eq"):
                with st.spinner("Génération..."):
                    try:
                        buf = generate_fiche_equipement(eq_id)
                        pdf_download_button(buf, f"fiche_{eq_id}.pdf", f"Télécharger fiche {eq_id}", "dl_eq")
                        st.markdown("<div class='success-box'>✅ Fiche générée!</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f"<div class='alert-box'>Erreur: {e}</div>", unsafe_allow_html=True)
        with col2:
            if st.button("🔴 Générer TOUTES les fiches", key="btn_alleq"):
                with st.spinner("Génération..."):
                    try:
                        bufs = [generate_fiche_equipement(r["ID"]) for _, r in eq.iterrows()]
                        merged = merge_pdfs(bufs, [])
                        pdf_download_button(merged, "toutes_fiches_equipements.pdf", f"Télécharger {len(bufs)} fiches", "dl_alleq")
                        st.markdown(f"<div class='success-box'>✅ {len(bufs)} fiches générées!</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f"<div class='alert-box'>Erreur: {e}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 4: Rapport stock
    with tab4:
        st.markdown("<div class='pdf-section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Rapport d'inventaire & stock</div>", unsafe_allow_html=True)
        st.markdown("""<div class='info-box'>
            Rapport stock avec : KPIs inventaire · Tableau complet avec alertes colorées ·
            Bons de commande suggérés pour ruptures · Valeur totale du stock.
        </div>""", unsafe_allow_html=True)

        n_rupt = len(pc[pc["Stock"]<pc["Min"]])
        val_stock = (pc["Stock"]*pc["Prix (€)"]).sum()
        st.markdown(f"""<div style='background:#0e1828;border:1px solid #1a3355;border-radius:8px;padding:10px 14px;margin:8px 0;font-size:13px;color:#c0d8f0;'>
            📦 {len(pc)} références · 💰 Valeur totale: {val_stock:,.2f}€ · ⚠️ {n_rupt} rupture(s) détectée(s)
        </div>""", unsafe_allow_html=True)

        if st.button("🔴 Générer rapport stock PDF", key="btn_stock"):
            with st.spinner("Génération..."):
                try:
                    buf = generate_rapport_stock()
                    pdf_download_button(buf, "rapport_stock.pdf", "Télécharger rapport stock", "dl_stock")
                    st.markdown("<div class='success-box'>✅ Rapport stock généré!</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"<div class='alert-box'>Erreur: {e}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 5: Planning PDF
    with tab5:
        st.markdown("<div class='pdf-section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Planning des interventions (A4 paysage)</div>", unsafe_allow_html=True)
        st.markdown("""<div class='info-box'>
            Planning format paysage avec : Toutes interventions planifiées par priorité ·
            Charge de travail par technicien · Statuts colorés · Format imprimable.
        </div>""", unsafe_allow_html=True)

        n_planifies = len(bt[bt["Statut"]!="Terminé"])
        st.markdown(f"<div style='color:#c0d8f0;font-size:13px;margin:8px 0;'>📅 {n_planifies} interventions actives à planifier · {len(st.session_state.techniciens)} techniciens</div>", unsafe_allow_html=True)

        if st.button("🔴 Générer planning PDF", key="btn_planning"):
            with st.spinner("Génération..."):
                try:
                    buf = generate_planning_pdf()
                    pdf_download_button(buf, "planning_interventions.pdf", "Télécharger planning", "dl_planning")
                    st.markdown("<div class='success-box'>✅ Planning généré!</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"<div class='alert-box'>Erreur: {e}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 6: Fusion PDF
    with tab6:
        st.markdown("<div class='pdf-section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Fusion & consolidation PDF</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title' style='font-size:14px;'>Pack PDF complet — Tous les documents en 1 clic</div>", unsafe_allow_html=True)
        st.markdown("""<div class='info-box'>
            Génère et fusionne automatiquement : Rapport mensuel + Planning + Rapport stock + Toutes fiches équipements + Tous BT actifs en un seul PDF consolidé.
        </div>""", unsafe_allow_html=True)

        if st.button("🔴 Générer PACK COMPLET PDF", key="btn_pack"):
            progress = st.progress(0, text="Démarrage...")
            try:
                all_bufs = []
                progress.progress(10, text="Rapport mensuel...")
                all_bufs.append(generate_rapport_mensuel())
                progress.progress(25, text="Planning...")
                all_bufs.append(generate_planning_pdf())
                progress.progress(40, text="Rapport stock...")
                all_bufs.append(generate_rapport_stock())
                progress.progress(55, text="Fiches équipements...")
                for _, r in eq.iterrows():
                    all_bufs.append(generate_fiche_equipement(r["ID"]))
                progress.progress(75, text="Bons de travaux actifs...")
                for _, r in bt[bt["Statut"]!="Terminé"].iterrows():
                    all_bufs.append(generate_bon_travaux(r["BT"]))
                progress.progress(90, text="Fusion PDF...")
                merged = merge_pdfs(all_bufs, [])
                progress.progress(100, text="Terminé!")
                pdf_download_button(merged, f"GMAO_Pack_Complet_{datetime.now().strftime('%Y%m%d')}.pdf",
                                    f"Télécharger Pack complet ({len(all_bufs)} documents)", "dl_pack")
                st.markdown(f"<div class='success-box'>✅ Pack complet: {len(all_bufs)} documents fusionnés avec succès!</div>", unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"<div class='alert-box'>⚠️ Erreur: {str(e)}</div>", unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1a2d48;margin:20px 0;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title' style='font-size:14px;'>Sélection personnalisée</div>", unsafe_allow_html=True)

        col1,col2 = st.columns(2)
        with col1:
            docs_sel = st.multiselect("Documents à inclure", [
                "Rapport mensuel", "Planning interventions",
                "Rapport stock", "Rapport KPI"
            ], default=["Rapport mensuel","Rapport stock"])
        with col2:
            eq_sel_m = st.multiselect("Fiches équipements", list(eq["ID"]+" — "+eq["Nom"]), default=[])
            bt_sel_m = st.multiselect("Bons de travaux", list(bt["BT"]+" — "+bt["Titre"]), default=[])

        if st.button("🔴 Fusionner sélection", key="btn_custom"):
            with st.spinner("Génération et fusion..."):
                try:
                    bufs = []
                    if "Rapport mensuel" in docs_sel: bufs.append(generate_rapport_mensuel())
                    if "Planning interventions" in docs_sel: bufs.append(generate_planning_pdf())
                    if "Rapport stock" in docs_sel: bufs.append(generate_rapport_stock())
                    if "Rapport KPI" in docs_sel: bufs.append(generate_rapport_kpi())
                    for e_sel in eq_sel_m:
                        eid = e_sel.split(" — ")[0]
                        bufs.append(generate_fiche_equipement(eid))
                    for b_sel in bt_sel_m:
                        bid = b_sel.split(" — ")[0]
                        bufs.append(generate_bon_travaux(bid))
                    if bufs:
                        merged = merge_pdfs(bufs, [])
                        pdf_download_button(merged, f"GMAO_selection_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                            f"Télécharger sélection ({len(bufs)} docs)", "dl_custom")
                        st.markdown(f"<div class='success-box'>✅ {len(bufs)} document(s) fusionné(s)!</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='alert-box'>⚠️ Sélectionnez au moins un document.</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"<div class='alert-box'>Erreur: {e}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ── PARAMÈTRES ────────────────────────────────────────────────────────────────
elif page == "Paramètres":
    st.markdown("<div style='font-family:Rajdhani;font-size:30px;font-weight:700;letter-spacing:3px;color:#90c0e8;'>PARAMÈTRES</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["⚙️ Configuration", "🗄️ Données"])

    with tab1:
        st.markdown("<div class='section-title'>Configuration générale</div>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            st.text_input("Nom de l'entreprise", value="Industrie Maroc SA")
            st.text_input("Email notifications", value="maintenance@industrie.ma")
            st.selectbox("Devise", ["MAD (Dirham)","EUR (Euro)","USD (Dollar)"])
            st.selectbox("Langue", ["Français","Arabe","Anglais"])
        with c2:
            st.number_input("Seuil alerte stock (%)", value=20, min_value=5, max_value=50)
            st.number_input("Rappel maintenance (jours)", value=7, min_value=1, max_value=30)
            st.number_input("MTBF cible (heures)", value=700, min_value=100)
            st.number_input("Budget mensuel maintenance (€)", value=15000, min_value=1000)

        st.markdown("<div class='section-title'>Utilisateurs</div>", unsafe_allow_html=True)
        users = pd.DataFrame([
            {"Utilisateur":"admin@industrie.ma","Rôle":"Administrateur","Accès":"Complet"},
            {"Utilisateur":"chef.maint@industrie.ma","Rôle":"Responsable maintenance","Accès":"BT + Rapports + PDF"},
            {"Utilisateur":"tech@industrie.ma","Rôle":"Technicien","Accès":"BT uniquement"},
        ])
        st.dataframe(users, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("<div class='section-title'>Gestion des données</div>", unsafe_allow_html=True)
        col1,col2,col3 = st.columns(3)
        with col1:
            if st.button("🔄 Réinitialiser données demo"):
                for k in ["equipements","bons_travaux","pieces","techniciens","interventions_hist"]:
                    if k in st.session_state: del st.session_state[k]
                init_data()
                st.markdown("<div class='success-box'>✅ Données réinitialisées!</div>", unsafe_allow_html=True)
        with col2:
            all_data = {
                "equipements": eq.to_dict(),
                "bons_travaux": bt.to_dict(),
                "pieces": pc.to_dict(),
            }
            import json
            st.download_button("⬇️ Export JSON complet", json.dumps(all_data, ensure_ascii=False, indent=2).encode(), "gmao_export.json", "application/json")
        with col3:
            st.download_button("⬇️ Export Excel (BT)", bt.to_csv(index=False).encode(), "bons_travaux.csv", "text/csv")

        st.markdown("""<br><div class='info-box'>
            <b style='font-family:Rajdhani;'>GMAO Pro+ v2.0</b> — Application avancée de Gestion de Maintenance<br>
            <span style='font-size:12px;color:#4a7aaa;'>Python 3.9+ · Streamlit · Plotly · ReportLab · pypdf · Génération PDF complète</span>
        </div>""", unsafe_allow_html=True)
