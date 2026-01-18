import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import time
from fpdf import FPDF
import base64
from datetime import datetime

st.set_page_config(
    page_title="Gage R&R - Étendues",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé avec animations et effets visuels
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.1);
        animation: fadeIn 1s ease-out;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-title {
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .main-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f5f7fa);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
        box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.05), 
                    -5px -5px 15px rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 15px 15px 30px rgba(0, 0, 0, 0.1), 
                    -15px -15px 30px rgba(255, 255, 255, 0.9);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.6s ease;
    }
    
    .metric-card:hover::before {
        transform: scaleX(1);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2c3e50, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #7f8c8d;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .result-indicator {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        backdrop-filter: blur(10px);
        transition: all 0.4s ease;
        border: 2px solid transparent;
        animation: pulse 2s infinite;
    }
    
    .result-indicator:hover {
        transform: scale(1.03);
    }
    
    .good {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.15), rgba(39, 174, 96, 0.25));
        color: #27ae60;
        border-color: #2ecc71;
    }
    
    .warning {
        background: linear-gradient(135deg, rgba(241, 196, 15, 0.15), rgba(243, 156, 18, 0.25));
        color: #f39c12;
        border-color: #f1c40f;
    }
    
    .bad {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.15), rgba(192, 57, 43, 0.25));
        color: #c0392b;
        border-color: #e74c3c;
        animation: shake 0.5s ease-in-out;
    }
    
    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .plot-container {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
        margin: 1.5rem 0;
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }
    
    .plot-container:hover {
        transform: translateY(-5px);
    }
    
    .dataframe-container {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #e0e6ed;
    }
    
    .upload-area {
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
        margin: 2rem 0;
    }
    
    .upload-area:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: #764ba2;
    }
    
    .download-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        display: inline-flex;
        align-items: center;
        gap: 10px;
        margin: 1rem 0;
    }
    
    .download-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .download-btn-pdf {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
        display: inline-flex;
        align-items: center;
        gap: 10px;
        margin: 1rem 0;
    }
    
    .download-btn-pdf:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(231, 76, 60, 0.4);
    }
    
    .sidebar-content {
        padding: 1.5rem;
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 0 20px 20px 0;
        height: 100%;
    }
    
    .report-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-left: 5px solid;
        transition: transform 0.3s ease;
    }
    
    .report-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }
    
    .floating-badge {
        position: absolute;
        top: -10px;
        right: -10px;
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(255, 107, 107, 0.3);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
        20%, 40%, 60%, 80% { transform: translateX(5px); }
    }
    
    .progress-container {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 3px;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 10px;
        border-radius: 8px;
        background: linear-gradient(90deg, #2ecc71, #f1c40f, #e74c3c);
        transition: width 1.5s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #3498db;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        border-left-color: #667eea;
        transform: translateX(5px);
    }
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
</style>
""", unsafe_allow_html=True)

# Header principal avec animation
st.markdown("""
<div class="main-header">
    <div class="main-title">📊 Gage R&R - Méthode des Étendues</div>
    <div class="main-subtitle">Analyse avancée de la capacité du système de mesure avec rapport PDF</div>
</div>
""", unsafe_allow_html=True)

# ---------------- d2 FUNCTION ----------------
def get_d2(z, w):
    if z > 15 and w == 3:
        return 1.693
    if z == 1 and w == 3:
        return 1.91
    if z == 1 and w == 10:
        return 3.18
    return 1.0

# ---------------- CLASS PDF REPORT ----------------
class GageRRPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        # Logo
        self.image("https://img.icons8.com/color/48/000000/statistics.png", 10, 8, 15)
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'RAPPORT GAGE R&R - MÉTHODE DES ÉTENDUES', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 5, f'Date du rapport: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
        
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)
        
    def add_section_title(self, title, level=1):
        if level == 1:
            self.set_font('Arial', 'B', 12)
            self.set_text_color(0, 0, 139)  # Dark blue
            self.cell(0, 8, title, 0, 1)
            self.ln(2)
        else:
            self.set_font('Arial', 'B', 11)
            self.set_text_color(0, 0, 0)
            self.cell(0, 7, title, 0, 1)
            self.ln(1)
        
    def add_text(self, text):
        self.set_font('Arial', '', 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, text)
        self.ln(2)
        
    def add_bullet(self, text):
        self.set_font('Arial', '', 10)
        self.set_text_color(0, 0, 0)
        self.cell(5)
        self.cell(5, 5, '•')
        self.multi_cell(0, 5, text[1:])
        
    def add_table(self, headers, data):
        self.set_font('Arial', 'B', 10)
        col_width = 190 / len(headers)
        
        # Headers
        self.set_fill_color(220, 220, 220)
        for header in headers:
            self.cell(col_width, 7, header, 1, 0, 'C', True)
        self.ln()
        
        # Data
        self.set_font('Arial', '', 9)
        fill = False
        for row in data:
            for item in row:
                self.cell(col_width, 6, str(item), 1, 0, 'C', fill)
            self.ln()
            fill = not fill
            
    def add_metric_box(self, title, value, status, color_rgb):
        self.set_fill_color(*color_rgb)
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 10)
        self.cell(47, 10, title, 1, 0, 'C', True)
        self.set_text_color(0, 0, 0)
        self.set_font('Arial', '', 9)
        self.cell(47, 10, value, 1, 0, 'C')
        self.set_font('Arial', 'B', 9)
        self.cell(47, 10, status, 1, 0, 'C')
        self.ln()

# ---------------- FONCTION DE GÉNÉRATION DE RAPPORT ----------------
def generate_report(p_grr, ev, av, grr, vp, vt, n_pieces, n_operateurs, n_essais, r_double_bar, 
                   confidence_factor, operators_data, df, filename):
    """Génère un rapport PDF complet"""
    
    # Calcul des pourcentages
    ev_percent = (ev / vt) * 100 if vt > 0 else 0
    av_percent = (av / vt) * 100 if vt > 0 else 0
    vp_percent = (vp / vt) * 100 if vt > 0 else 0
    ratio_vp_grr = vp / grr if grr > 0 else 0
    
    # Évaluation générale
    if p_grr < 10:
        overall_status = "EXCELLENT"
        overall_color = (46, 204, 113)  # Vert
        overall_message = "Le système de mesure est optimal et fiable pour les analyses critiques."
    elif p_grr <= 30:
        overall_status = "ACCEPTABLE"
        overall_color = (241, 196, 15)  # Orange
        overall_message = "Le système est acceptable mais des améliorations sont recommandées."
    else:
        overall_status = "INACCEPTABLE"
        overall_color = (231, 76, 60)   # Rouge
        overall_message = "Le système nécessite des actions correctives urgentes."
    
    # Création du PDF
    pdf = GageRRPDF()
    pdf.add_page()
    
    # Page de titre
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 40, '', 0, 1, 'C')
    pdf.cell(0, 10, 'RAPPORT D\'ANALYSE GAGE R&R', 0, 1, 'C')
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Méthode des Étendues', 0, 1, 'C')
    pdf.ln(20)
    
    # Résumé exécutif
    pdf.add_section_title("1. RÉSUMÉ EXÉCUTIF")
    
    # Carte de statut
    pdf.set_fill_color(*overall_color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 12, f'STATUT: {overall_status} - {p_grr:.1f}%', 0, 1, 'C', True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', 'I', 10)
    pdf.multi_cell(0, 5, overall_message)
    pdf.ln(5)
    
    # Métriques clés
    pdf.add_section_title("2. MÉTRIQUES CLÉS", level=2)
    
    metrics = [
        ("%GRR Total", f"{p_grr:.2f}%", overall_status, overall_color),
        ("Répétabilité (EV)", f"{ev:.4f} ({ev_percent:.1f}%)", "✓" if ev_percent < 20 else "⚠", (52, 152, 219)),
        ("Reproductibilité (AV)", f"{av:.4f} ({av_percent:.1f}%)", "✓" if av_percent < 20 else "⚠", (46, 204, 113)),
        ("Variation Pièces (VP)", f"{vp:.4f} ({vp_percent:.1f}%)", "✓" if vp_percent > 50 else "⚠", (231, 76, 60)),
        ("Variation Totale (VT)", f"{vt:.4f}", "-", (155, 89, 182)),
        ("Ratio VP/GRR", f"{ratio_vp_grr:.2f}", "✓" if ratio_vp_grr > 4 else "⚠", (243, 156, 18))
    ]
    
    for i in range(0, len(metrics), 2):
        row1 = metrics[i]
        row2 = metrics[i+1] if i+1 < len(metrics) else None
        
        pdf.add_metric_box(row1[0], row1[1], row1[2], row1[3])
        if row2:
            pdf.add_metric_box(row2[0], row2[1], row2[2], row2[3])
        pdf.ln(2)
    
    pdf.add_page()
    
    # Informations sur l'étude
    pdf.add_section_title("3. INFORMATIONS SUR L'ÉTUDE")
    
    study_info = [
        ["Paramètre", "Valeur"],
        ["Date d'analyse", datetime.now().strftime("%d/%m/%Y %H:%M")],
        ["Nombre de pièces", str(n_pieces)],
        ["Nombre d'opérateurs", str(n_operateurs)],
        ["Nombre d'essais par pièce", str(n_essais)],
        ["Facteur de confiance (k)", f"{confidence_factor:.2f}"],
        ["Étendue moyenne (R̄)", f"{r_double_bar:.4f}"],
        ["Méthode utilisée", "Méthode des Étendues"],
        ["Critère d'acceptation", "%GRR < 30%"]
    ]
    
    pdf.add_table([study_info[0][0], study_info[0][1]], [row[1:] for row in study_info[1:]])
    pdf.ln(10)
    
    # Performance des opérateurs
    pdf.add_section_title("4. PERFORMANCE DES OPÉRATEURS")
    
    op_headers = ["Opérateur", "Moyenne", "Étendue Moyenne", "Écart-Type"]
    op_data = []
    for i, op in enumerate(operators_data):
        op_data.append([
            f"Opérateur {i+1}",
            f"{np.mean(df[[f'OP{i+1}-1', f'OP{i+1}-2', f'OP{i+1}-3']].values.flatten()):.4f}",
            f"{[r_bar_op1, r_bar_op2, r_bar_op3][i]:.4f}",
            f"{np.std(df[[f'OP{i+1}-1', f'OP{i+1}-2', f'OP{i+1}-3']].values.flatten()):.4f}"
        ])
    
    pdf.add_table(op_headers, op_data)
    pdf.ln(10)
    
    # Données brutes (premières 10 lignes)
    pdf.add_section_title("5. DONNÉES BRUTES (EXTRAIT)")
    
    # Préparer les données pour le tableau
    data_headers = ["Pièce"] + [f"OP{i+1}-{j+1}" for i in range(3) for j in range(3)]
    sample_data = []
    for idx in range(min(10, len(df))):
        row = [f"Pièce {idx+1}"]
        for i in range(3):
            for j in range(3):
                row.append(f"{df.iloc[idx][f'OP{i+1}-{j+1}']:.4f}")
        sample_data.append(row)
    
    pdf.set_font('Arial', '', 7)
    col_width = 190 / len(data_headers)
    
    # Headers
    pdf.set_font('Arial', 'B', 7)
    pdf.set_fill_color(240, 240, 240)
    for header in data_headers:
        pdf.cell(col_width, 5, header, 1, 0, 'C', True)
    pdf.ln()
    
    # Data
    pdf.set_font('Arial', '', 7)
    fill = False
    for row in sample_data:
        for item in row:
            pdf.cell(col_width, 4, item, 1, 0, 'C', fill)
        pdf.ln()
        fill = not fill
    
    pdf.add_page()
    
    # Interprétation des résultats
    pdf.add_section_title("6. INTERPRÉTATION DES RÉSULTATS")
    
    pdf.add_section_title("6.1 Évaluation Globale", level=2)
    pdf.add_text(f"Le système de mesure présente un %GRR de {p_grr:.1f}%, ce qui le classe comme '{overall_status}' selon les critères de l'industrie automobile (AIAG).")
    
    pdf.add_section_title("6.2 Analyse des Composantes", level=2)
    
    if ev_percent > av_percent:
        pdf.add_text("La source principale de variation est la RÉPÉTABILITÉ (EV), ce qui indique que:")
        pdf.add_bullet(" La variabilité intra-opérateur est dominante")
        pdf.add_bullet(" Causes possibles: instrument instable, procédure non standardisée")
        pdf.add_bullet(" Recommandation: vérifier l'étalonnage et la stabilité des équipements")
    else:
        pdf.add_text("La source principale de variation est la REPRODUCTIBILITÉ (AV), ce qui indique que:")
        pdf.add_bullet(" Les différences entre opérateurs sont dominantes")
        pdf.add_bullet(" Causes possibles: méthodes de mesure divergentes, formation insuffisante")
        pdf.add_bullet(" Recommandation: harmoniser les procédures et former les opérateurs")
    
    pdf.add_section_title("6.3 Capacité de Discrimination", level=2)
    if ratio_vp_grr > 4:
        pdf.add_text(f"Le ratio VP/GRR de {ratio_vp_grr:.2f} indique une EXCELLENTE capacité à distinguer les différences entre pièces.")
    elif ratio_vp_grr > 2:
        pdf.add_text(f"Le ratio VP/GRR de {ratio_vp_grr:.2f} indique une capacité ACCEPTABLE à distinguer les différences entre pièces.")
    else:
        pdf.add_text(f"Le ratio VP/GRR de {ratio_vp_grr:.2f} indique une FAIBLE capacité à distinguer les différences entre pièces.")
    
    pdf.add_page()
    
    # Recommandations et plan d'action
    pdf.add_section_title("7. RECOMMANDATIONS ET PLAN D'ACTION")
    
    if p_grr > 30:
        pdf.add_section_title("7.1 Actions Correctives Immédiates (Critique)", level=2)
        actions = [
            "Suspension temporaire du système pour les mesures critiques",
            "Réétalonnage complet des instruments de mesure",
            "Formation standardisée pour tous les opérateurs",
            "Contrôle des conditions environnementales",
            "Mise à jour des procédures de mesure"
        ]
    elif p_grr > 15:
        pdf.add_section_title("7.2 Améliorations Recommandées (Amélioration)", level=2)
        actions = [
            "Amélioration de la documentation des méthodes",
            "Implémentation d'aides à la mesure (gabarits)",
            "Audits croisés entre opérateurs",
            "Surveillance régulière des performances",
            "Identification des causes racines"
        ]
    else:
        pdf.add_section_title("7.3 Actions de Maintenance (Optimisation)", level=2)
        actions = [
            "Maintenance de la documentation à jour",
            "Étalonnage préventif programmé",
            "Surveillance statistique continue",
            "Formation des nouveaux opérateurs",
            "Réévaluation annuelle du système"
        ]
    
    for action in actions:
        pdf.add_bullet(f" {action}")
    
    pdf.ln(5)
    
    pdf.add_section_title("7.4 Priorités d'Action", level=2)
    if ev_percent > av_percent:
        pdf.add_text("PRIORITÉ 1: Améliorer la répétabilité")
        pdf.add_bullet(" Standardiser la méthode de prise de mesure")
        pdf.add_bullet(" Vérifier l'état et l'étalonnage des instruments")
        pdf.add_bullet(" Minimiser les variations environnementales")
    else:
        pdf.add_text("PRIORITÉ 1: Améliorer la reproductibilité")
        pdf.add_bullet(" Formation commune à tous les opérateurs")
        pdf.add_bullet(" Création d'aides visuelles pour les décisions")
        pdf.add_bullet(" Audits croisés réguliers")
    
    pdf.add_text("PRIORITÉ 2: Valider les améliorations")
    pdf.add_bullet(" Refaire l'étude Gage R&R après corrections")
    pdf.add_bullet(" Suivre les indicateurs clés sur tableau de bord")
    pdf.add_bullet(" Documenter toutes les actions correctives")
    
    pdf.add_page()
    
    # Critères d'acceptation et références
    pdf.add_section_title("8. CRITÈRES D'ACCEPTATION ET RÉFÉRENCES")
    
    pdf.add_section_title("8.1 Échelle d'Évaluation", level=2)
    criteria = [
        ["%GRR", "Évaluation", "Recommandation"],
        ["< 10%", "EXCELLENT", "Système optimal, utilisation sans restriction"],
        ["10% - 30%", "ACCEPTABLE", "Système acceptable, améliorations possibles"],
        ["> 30%", "INACCEPTABLE", "Action corrective requise avant utilisation"]
    ]
    
    pdf.add_table(criteria[0], criteria[1:])
    pdf.ln(10)
    
    pdf.add_section_title("8.2 Références Normatives", level=2)
    references = [
        "AIAG MSA Manual 4th Edition - Automotive Industry Action Group",
        "ISO 22514-7:2012 - Statistical methods in process management",
        "ISO 5725:1994 - Accuracy (trueness and precision) of measurement methods",
        "ASTM E691 - Standard Practice for Conducting an Interlaboratory Study"
    ]
    
    for ref in references:
        pdf.add_bullet(f" {ref}")
    
    # Signature
    pdf.ln(20)
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 5, "_________________________________________", 0, 1, 'C')
    pdf.cell(0, 5, "Responsable Qualité / Ingénieur Méthodes", 0, 1, 'C')
    pdf.cell(0, 5, "Date: ___________________________", 0, 1, 'C')
    
    # Sauvegarde du PDF
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    
    return pdf_bytes

# ---------------- SIDEBAR STYLÉE ----------------
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    st.markdown('<div style="font-size: 1.5rem; font-weight: 700; color: #2c3e50; margin-bottom: 2rem;">⚙️ Configuration</div>', unsafe_allow_html=True)
    
    confidence_factor = st.slider(
        "**Facteur de Confiance (k)**",
        min_value=4.0,
        max_value=6.0,
        value=5.15,
        step=0.05,
        help="Facteur pour le niveau de confiance des calculs"
    )
    
    # Configuration PDF
    st.markdown("---")
    st.markdown('<div style="font-size: 1.2rem; font-weight: 600; color: #2c3e50; margin: 1.5rem 0 1rem 0;">📄 Options PDF</div>', unsafe_allow_html=True)
    
    include_raw_data = st.checkbox("Inclure données brutes", value=True)
    include_charts = st.checkbox("Inclure graphiques", value=True)
    company_name = st.text_input("Nom de l'entreprise", "Votre Entreprise")
    study_name = st.text_input("Nom de l'étude", "Analyse Gage R&R")
    
    st.markdown("---")
    
    st.markdown('<div style="font-size: 1.2rem; font-weight: 600; color: #2c3e50; margin: 1.5rem 0 1rem 0;">📈 Guide de Lecture</div>', unsafe_allow_html=True)
    
    with st.expander("🔍 Comprendre les indicateurs", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class="stat-card">
                <div style="color: #3498db; font-weight: 600;">EV</div>
                <div style="color: #7f8c8d; font-size: 0.85rem;">Répétabilité</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="stat-card">
                <div style="color: #2ecc71; font-weight: 600;">AV</div>
                <div style="color: #7f8c8d; font-size: 0.85rem;">Reproductibilité</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown("""
            <div class="stat-card">
                <div style="color: #9b59b6; font-weight: 600;">GRR</div>
                <div style="color: #7f8c8d; font-size: 0.85rem;">Variation système</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="stat-card">
                <div style="color: #e74c3c; font-weight: 600;">%GRR</div>
                <div style="color: #7f8c8d; font-size: 0.85rem;">Pourcentage total</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div style="font-size: 1.2rem; font-weight: 600; color: #2c3e50; margin: 1.5rem 0 1rem 0;">🎯 Critères</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(46, 204, 113, 0.1), rgba(39, 174, 96, 0.2)); 
                padding: 1rem; border-radius: 10px; border-left: 4px solid #2ecc71; margin-bottom: 0.5rem;">
        <div style="font-weight: 600; color: #27ae60;">✓ EXCELLENT</div>
        <div style="color: #7f8c8d; font-size: 0.9rem;">&lt; 10% - Système optimal</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(241, 196, 15, 0.1), rgba(243, 156, 18, 0.2)); 
                padding: 1rem; border-radius: 10px; border-left: 4px solid #f1c40f; margin-bottom: 0.5rem;">
        <div style="font-weight: 600; color: #f39c12;">⚠ ACCEPTABLE</div>
        <div style="color: #7f8c8d; font-size: 0.9rem;">10-30% - Amélioration souhaitée</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(231, 76, 60, 0.1), rgba(192, 57, 43, 0.2)); 
                padding: 1rem; border-radius: 10px; border-left: 4px solid #e74c3c;">
        <div style="font-weight: 600; color: #c0392b;">✗ INACCEPTABLE</div>
        <div style="color: #7f8c8d; font-size: 0.9rem;">&gt; 30% - Action corrective requise</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ZONE D'UPLOAD STYLÉE ----------------
st.markdown('<div class="section-header"><span>📥 Importation des Données</span></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["xlsx"],
    help="Téléversez votre fichier Excel contenant les mesures",
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown("""
    <div class="upload-area">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📁</div>
        <div style="font-size: 1.5rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">
            Glissez-déposez votre fichier Excel
        </div>
        <div style="color: #7f8c8d; margin-bottom: 2rem;">
            ou cliquez pour parcourir
        </div>
        <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; display: inline-block;">
            <div style="font-weight: 600; color: #667eea;">Format requis :</div>
            <div style="color: #7f8c8d; font-size: 0.9rem; text-align: left; margin-top: 0.5rem;">
                • Colonnes : OP1-1, OP1-2, OP1-3, OP2-1, OP2-2, OP2-3, OP3-1, OP3-2, OP3-3<br>
                • Lignes : Pièces mesurées<br>
                • 3 opérateurs × 3 essais
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file:
    # Animation de chargement
    with st.spinner('🔄 Traitement des données en cours...'):
        time.sleep(0.5)
        df = pd.read_excel(uploaded_file)

    # ---------------- APERÇU DES DONNÉES ----------------
    st.markdown('<div class="section-header"><span>📄 Aperçu des Données</span></div>', unsafe_allow_html=True)
    
    with st.expander("Voir les données détaillées", expanded=True):
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        
        # Style amélioré pour le DataFrame
        def color_gradient(val):
            if isinstance(val, (int, float)):
                intensity = min(0.8, abs(val - df.values.mean()) / df.values.std() * 0.3)
                if val > df.values.mean():
                    return f'background: linear-gradient(90deg, rgba(46, 204, 113, {intensity}), rgba(39, 174, 96, {intensity/2}))'
                else:
                    return f'background: linear-gradient(90deg, rgba(52, 152, 219, {intensity}), rgba(41, 128, 185, {intensity/2}))'
            return ''
        
        styled_df = df.style.applymap(color_gradient)
        st.dataframe(styled_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Colonnes opérateurs
    op1_cols = ["OP1-1", "OP1-2", "OP1-3"]
    op2_cols = ["OP2-1", "OP2-2", "OP2-3"]
    op3_cols = ["OP3-1", "OP3-2", "OP3-3"]

    n_pieces = df.shape[0]
    n_operateurs = 3
    n_essais = 3

    # ---------------- CALCULS ----------------
    df["R_OP1"] = df[op1_cols].max(axis=1) - df[op1_cols].min(axis=1)
    df["R_OP2"] = df[op2_cols].max(axis=1) - df[op2_cols].min(axis=1)
    df["R_OP3"] = df[op3_cols].max(axis=1) - df[op3_cols].min(axis=1)

    r_bar_op1 = df["R_OP1"].mean()
    r_bar_op2 = df["R_OP2"].mean()
    r_bar_op3 = df["R_OP3"].mean()

    x_bar_op1 = df[op1_cols].values.mean()
    x_bar_op2 = df[op2_cols].values.mean()
    x_bar_op3 = df[op3_cols].values.mean()

    # Calculs GRR
    r_double_bar = (r_bar_op1 + r_bar_op2 + r_bar_op3) / n_operateurs
    d2_ev = get_d2(n_pieces * n_operateurs, n_essais)
    ev = (confidence_factor * r_double_bar) / d2_ev

    means_ops = [x_bar_op1, x_bar_op2, x_bar_op3]
    x_range = max(means_ops) - min(means_ops)
    d2_av = get_d2(1, n_operateurs)

    av_term = (confidence_factor * x_range / d2_av) ** 2
    ev_corr = (ev ** 2) / (n_pieces * n_essais)
    av = np.sqrt(max(0, av_term - ev_corr))

    grr = np.sqrt(ev ** 2 + av ** 2)

    # Variabilité pièces
    df["Moy_Piece"] = df[op1_cols + op2_cols + op3_cols].mean(axis=1)
    rp = df["Moy_Piece"].max() - df["Moy_Piece"].min()

    d2_vp = get_d2(1, n_pieces)
    vp = (confidence_factor * rp) / d2_vp

    vt = np.sqrt(grr ** 2 + vp ** 2)
    p_grr = (grr / vt) * 100

    # Préparation des données opérateurs pour le PDF
    operators_data = []
    for i in range(3):
        op_cols = [f"OP{i+1}-1", f"OP{i+1}-2", f"OP{i+1}-3"]
        op_data = df[op_cols].values.flatten()
        operators_data.append({
            'moyenne': np.mean(op_data),
            'etendue': [r_bar_op1, r_bar_op2, r_bar_op3][i],
            'ecart_type': np.std(op_data)
        })

    # ---------------- GÉNÉRATION DU PDF ----------------
    st.markdown('<div class="section-header"><span>📄 Génération du Rapport PDF</span></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Aperçu du Rapport")
        st.markdown(f"""
        <div class="report-card" style="border-left-color: {'#2ecc71' if p_grr < 10 else '#f1c40f' if p_grr <= 30 else '#e74c3c'};">
            <div style="font-size: 1.2rem; font-weight: 600; color: #2c3e50;">
                Rapport Gage R&R - {study_name}
            </div>
            <div style="margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #7f8c8d;">Statut:</span>
                    <span style="font-weight: 600; color: {'#27ae60' if p_grr < 10 else '#f39c12' if p_grr <= 30 else '#c0392b'}">
                        {'EXCELLENT' if p_grr < 10 else 'ACCEPTABLE' if p_grr <= 30 else 'INACCEPTABLE'}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #7f8c8d;">%GRR:</span>
                    <span style="font-weight: 600;">{p_grr:.2f}%</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #7f8c8d;">Pages estimées:</span>
                    <span style="font-weight: 600;">5-6 pages</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📋 Sections Incluses")
        sections = [
            "✅ Résumé exécutif et statut",
            "✅ Métriques clés et évaluation",
            "✅ Informations sur l'étude",
            "✅ Performance des opérateurs",
            "✅ Données brutes (extrait)",
            "✅ Interprétation détaillée",
            "✅ Recommandations et plan d'action",
            "✅ Critères d'acceptation",
            "✅ Références normatives",
            "✅ Zone de signature"
        ]
        
        for section in sections:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 10px; margin: 0.3rem 0;">
                <div style="color: #2ecc71;">✓</div>
                <div style="color: #2c3e50;">{section}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Bouton pour générer le PDF
    st.markdown("---")
    
    if st.button("📄 Générer le Rapport PDF", type="primary", use_container_width=True):
        with st.spinner("🔄 Génération du rapport PDF en cours..."):
            try:
                # Générer le nom du fichier
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"Rapport_Gage_RR_{study_name.replace(' ', '_')}_{timestamp}.pdf"
                
                # Générer le PDF
                pdf_bytes = generate_report(
                    p_grr=p_grr,
                    ev=ev,
                    av=av,
                    grr=grr,
                    vp=vp,
                    vt=vt,
                    n_pieces=n_pieces,
                    n_operateurs=n_operateurs,
                    n_essais=n_essais,
                    r_double_bar=r_double_bar,
                    confidence_factor=confidence_factor,
                    operators_data=operators_data,
                    df=df,
                    filename=filename
                )
                
                # Encoder le PDF en base64
                b64_pdf = base64.b64encode(pdf_bytes).decode()
                
                # Afficher le bouton de téléchargement
                st.markdown(f"""
                <div style="text-align: center; margin: 2rem 0;">
                    <a href="data:application/pdf;base64,{b64_pdf}" 
                       download="{filename}"
                       style="text-decoration: none;">
                        <div class="download-btn-pdf">
                            📥 Télécharger le Rapport PDF Complet
                        </div>
                    </a>
                    <div style="color: #7f8c8d; font-size: 0.9rem; margin-top: 1rem;">
                        Fichier: {filename} • {len(pdf_bytes)//1024} Ko
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Aperçu du PDF (optionnel)
                with st.expander("👁️ Aperçu du rapport (première page)"):
                    st.markdown("""
                    <div style="background: #f8f9fa; padding: 2rem; border-radius: 10px; text-align: center;">
                        <div style="font-size: 3rem;">📄</div>
                        <div style="font-weight: 600; color: #2c3e50; margin: 1rem 0;">
                            Rapport PDF Généré avec Succès!
                        </div>
                        <div style="color: #7f8c8d;">
                            Le rapport contient toutes les analyses, interprétations et recommandations.
                        </div>
                        <div style="margin-top: 1.5rem; padding: 1rem; background: white; border-radius: 8px; text-align: left;">
                            <div style="font-weight: 600; color: #2c3e50;">Contenu inclus:</div>
                            <div style="color: #7f8c8d; font-size: 0.9rem; margin-top: 0.5rem;">
                                • Page de titre professionnelle<br>
                                • Résumé exécutif avec statut<br>
                                • Tableaux de résultats détaillés<br>
                                • Analyse complète des composantes<br>
                                • Plan d'action personnalisé<br>
                                • Références normatives<br>
                                • Zone de signature
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success("✅ Rapport PDF généré avec succès!")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de la génération du PDF: {str(e)}")
    
    # ---------------- VISUALISATIONS AVANCÉES ----------------
    st.markdown('<div class="section-header"><span>📈 Visualisations</span></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique 1 : Composantes de variation (3D style)
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        components = ['EV', 'AV', 'GRR', 'VP', 'VT']
        values = [ev, av, grr, vp, vt]
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']
        
        # Barres avec effet 3D
        bars = ax1.bar(components, values, color=colors, edgecolor='white', 
                      linewidth=2, alpha=0.9, zorder=3)
        
        # Style amélioré
        ax1.grid(True, alpha=0.3, zorder=0)
        ax1.set_facecolor('#f8fafc')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Ajouter les valeurs avec animation visuelle
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{value:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10, color='#2c3e50')
        
        ax1.set_title('📊 Composantes de Variation', fontsize=14, fontweight=600, pad=20)
        plt.tight_layout()
        st.pyplot(fig1)
        st.markdown('</div>', unsafe_allow_html=True)
        plt.close()
    
    with col2:
        # Graphique 3 : Camembert amélioré
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        
        labels = ['Variation Système\n(GRR)', 'Variation Pièces\n(VP)']
        sizes = [grr**2, vp**2]
        colors = ['#9b59b6', '#e74c3c']
        explode = (0.1, 0)
        
        wedges, texts, autotexts = ax3.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        centre_circle = plt.Circle((0,0), 0.70, fc='white', edgecolor='white', linewidth=2)
        fig3.gca().add_artist(centre_circle)
        
        ax3.axis('equal')
        ax3.set_title('🥧 Répartition des Variations', fontsize=14, fontweight=600, pad=20)
        plt.tight_layout()
        st.pyplot(fig3)
        st.markdown('</div>', unsafe_allow_html=True)
        plt.close()

    # ---------------- RÉSULTATS PRINCIPAUX ----------------
    st.markdown('<div class="section-header"><span>📊 Résultats Principaux</span></div>', unsafe_allow_html=True)
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_data = [
        ("EV", ev, "#3498db", "Répétabilité"),
        ("AV", av, "#2ecc71", "Reproductibilité"),
        ("GRR", grr, "#9b59b6", "Variation Système"),
        ("%GRR", p_grr, "#e74c3c", "Pourcentage Total")
    ]
    
    for col, (label, value, color, desc) in zip([col1, col2, col3, col4], metrics_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{desc}</div>
                <div class="metric-value" style="background: linear-gradient(135deg, {color}, #2c3e50); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {value:.3f}{'%' if label == '%GRR' else ''}
                </div>
                <div style="color: #95a5a6; font-size: 0.9rem; margin-top: 0.5rem;">
                    <strong>{label}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Barre de progression avec animation
    progress_html = f"""
    <div style="margin: 2rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <div style="font-weight: 600; color: #2c3e50;">Progression du %GRR</div>
            <div style="font-weight: 600; color: #e74c3c;">{p_grr:.1f}%</div>
        </div>
        <div class="progress-container">
            <div class="progress-bar" style="width: {min(p_grr, 100)}%"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.85rem; color: #7f8c8d;">
            <div>0%</div>
            <div>10%</div>
            <div>30%</div>
            <div>100%</div>
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)
    
    # Indicateur de résultat
    if p_grr < 10:
        status = ("good", "✅", "SYSTÈME EXCELLENT", "Le système de mesure est optimal")
        st.balloons()
    elif p_grr <= 30:
        status = ("warning", "⚠️", "SYSTÈME ACCEPTABLE", "Améliorations possibles")
    else:
        status = ("bad", "❌", "SYSTÈME INACCEPTABLE", "Action corrective requise")
    
    st.markdown(f"""
    <div class="result-indicator {status[0]}">
        <div style="font-size: 1.3rem; margin-bottom: 0.5rem;">{status[1]} {status[2]}</div>
        <div style="font-size: 0.95rem; opacity: 0.9;">{status[3]}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- STATISTIQUES DÉTAILLÉES ----------------
    st.markdown('<div class="section-header"><span>📋 Statistiques Détaillées</span></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tableau des opérateurs
        st.markdown("**👥 Performance par Opérateur**")
        operators_display = []
        for i, (op_cols, op_name, r_bar) in enumerate(zip([op1_cols, op2_cols, op3_cols], 
                                                         ['Opérateur 1', 'Opérateur 2', 'Opérateur 3'],
                                                         [r_bar_op1, r_bar_op2, r_bar_op3]), 1):
            op_data = df[op_cols].values.flatten()
            operators_display.append({
                'Opérateur': f'👤 {op_name}',
                'Moyenne': f'{np.mean(op_data):.4f}',
                'Étendue': f'{r_bar:.4f}',
                'σ': f'{np.std(op_data):.4f}'
            })
        
        operators_df = pd.DataFrame(operators_display)
        st.dataframe(
            operators_df.style
            .background_gradient(subset=['Moyenne', 'Étendue', 'σ'], cmap='YlOrRd')
            .set_properties(**{'text-align': 'center'}),
            use_container_width=True
        )
    
    with col2:
        # Indicateurs secondaires
        st.markdown("**📈 Indicateurs Complémentaires**")
        
        secondary_metrics = [
            ("VP (Pièces)", f"{vp:.4f}", "#e74c3c"),
            ("VT (Totale)", f"{vt:.4f}", "#f39c12"),
            ("R̄ (Étendue)", f"{r_double_bar:.4f}", "#3498db"),
            ("Pièces (n)", str(n_pieces), "#95a5a6"),
            ("Essais (r)", str(n_essais), "#95a5a6"),
            ("Opérateurs (o)", str(n_operateurs), "#95a5a6")
        ]
        
        for label, value, color in secondary_metrics:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; 
                        border-left: 4px solid {color}; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 600; color: #2c3e50;">{label}</div>
                    <div style="font-weight: 700; color: {color}; font-size: 1.1rem;">{value}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---------------- INTERPRÉTATION ET CONSEILS ----------------
    st.markdown('<div class="section-header"><span>💡 Interprétation & Conseils</span></div>', unsafe_allow_html=True)
    
    # Calcul des pourcentages pour l'interprétation
    ev_percent = (ev / vt) * 100 if vt > 0 else 0
    av_percent = (av / vt) * 100 if vt > 0 else 0
    ratio_vp_grr = vp / grr if grr > 0 else 0
    
    # Interprétation détaillée
    with st.expander("🔍 Analyse Détaillée des Résultats", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Analyse des Composantes")
            
            if ev_percent > av_percent:
                st.markdown("""
                <div style="background: #e3f2fd; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #2196f3;">
                    <div style="font-weight: 600; color: #1565c0; margin-bottom: 0.5rem;">
                        🔍 Source Principale: RÉPÉTABILITÉ (EV)
                    </div>
                    <div style="color: #424242;">
                        La variabilité intra-opérateur domine. Cela suggère que:
                    </div>
                    <div style="margin-top: 1rem;">
                        <div style="color: #424242;">• L'instrument peut être instable</div>
                        <div style="color: #424242;">• La procédure n'est pas suffisamment standardisée</div>
                        <div style="color: #424242;">• Les conditions environnementales varient</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #e8f5e9; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #4caf50;">
                    <div style="font-weight: 600; color: #2e7d32; margin-bottom: 0.5rem;">
                        🔍 Source Principale: REPRODUCTIBILITÉ (AV)
                    </div>
                    <div style="color: #424242;">
                        Les différences entre opérateurs dominent. Cela suggère que:
                    </div>
                    <div style="margin-top: 1rem;">
                        <div style="color: #424242;">• Les méthodes de mesure divergent</div>
                        <div style="color: #424242;">• La formation est insuffisante ou inégale</div>
                        <div style="color: #424242;">• L'interprétation des résultats est subjective</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🎯 Capacité de Discrimination")
            
            if ratio_vp_grr > 4:
                st.markdown(f"""
                <div style="background: #e8f5e9; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #4caf50;">
                    <div style="font-weight: 600; color: #2e7d32; margin-bottom: 0.5rem;">
                        ✅ EXCELLENTE CAPACITÉ
                    </div>
                    <div style="font-size: 2rem; font-weight: 700; color: #2e7d32; text-align: center; margin: 1rem 0;">
                        {ratio_vp_grr:.1f}:1
                    </div>
                    <div style="color: #424242; text-align: center;">
                        Le système distingue clairement les différences entre pièces
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif ratio_vp_grr > 2:
                st.markdown(f"""
                <div style="background: #fff3e0; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #ff9800;">
                    <div style="font-weight: 600; color: #ef6c00; margin-bottom: 0.5rem;">
                        ⚠ CAPACITÉ ACCEPTABLE
                    </div>
                    <div style="font-size: 2rem; font-weight: 700; color: #ef6c00; text-align: center; margin: 1rem 0;">
                        {ratio_vp_grr:.1f}:1
                    </div>
                    <div style="color: #424242; text-align: center;">
                        Le système distingue raisonnablement les différences entre pièces
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #ffebee; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #f44336;">
                    <div style="font-weight: 600; color: #c62828; margin-bottom: 0.5rem;">
                        ❌ FAIBLE CAPACITÉ
                    </div>
                    <div style="font-size: 2rem; font-weight: 700; color: #c62828; text-align: center; margin: 1rem 0;">
                        {ratio_vp_grr:.1f}:1
                    </div>
                    <div style="color: #424242; text-align: center;">
                        Le système a du mal à distinguer les différences entre pièces
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Conseils personnalisés
    with st.expander("🎯 Plan d'Action Recommandé", expanded=True):
        if p_grr > 30:
            st.markdown("""
            <div style="background: #ffebee; padding: 1.5rem; border-radius: 10px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
                    <div style="font-size: 1.5rem;">🔴</div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: #c62828;">
                        ACTIONS CORRECTIVES IMMÉDIATES
                    </div>
                </div>
                <div style="color: #424242;">
                    1. **Suspendre temporairement** l'utilisation du système pour les mesures critiques<br>
                    2. **Réétalonner** tous les instruments de mesure<br>
                    3. **Former/reformer** les opérateurs avec méthode standardisée<br>
                    4. **Vérifier** la stabilité des conditions environnementales<br>
                    5. **Revoir** le plan d'échantillonnage des pièces<br>
                    6. **Documenter** toutes les actions correctives
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif p_grr > 15:
            st.markdown("""
            <div style="background: #fff3e0; padding: 1.5rem; border-radius: 10px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
                    <div style="font-size: 1.5rem;">🟡</div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: #ef6c00;">
                        AMÉLIORATIONS RECOMMANDÉES
                    </div>
                </div>
                <div style="color: #424242;">
                    1. **Améliorer** la procédure écrite de mesure<br>
                    2. **Implémenter** des gabarits ou dispositifs d'aide<br>
                    3. **Organiser** des audits croisés entre opérateurs<br>
                    4. **Augmenter** le nombre d'essais pour réduire l'incertitude<br>
                    5. **Surveiller régulièrement** la performance du système<br>
                    6. **Planifier** une réévaluation dans 6 mois
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #e8f5e9; padding: 1.5rem; border-radius: 10px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
                    <div style="font-size: 1.5rem;">🟢</div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: #2e7d32;">
                        ACTIONS DE MAINTENANCE
                    </div>
                </div>
                <div style="color: #424242;">
                    1. **Maintenir** la documentation à jour<br>
                    2. **Programmer** des étalonnages réguliers<br>
                    3. **Surveiller** les tendances dans le temps<br>
                    4. **Former** les nouveaux opérateurs avec méthode validée<br>
                    5. **Réaliser** des vérifications périodiques<br>
                    6. **Capitaliser** sur les bonnes pratiques
                </div>
            </div>
            """, unsafe_allow_html=True)

# Pied de page élégant
st.markdown("""
<div style="margin-top: 4rem; padding: 2rem; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); 
            border-radius: 20px; text-align: center; border-top: 1px solid #e0e6ed;">
    <div style="font-size: 0.9rem; color: #7f8c8d;">
        <div style="display: flex; justify-content: center; align-items: center; gap: 10px; margin-bottom: 0.5rem;">
            <div>📊</div>
            <div><strong>Gage R&R - Méthode des Étendues avec Rapport PDF</strong></div>
            <div>⚡</div>
        </div>
        <div>Analyse avancée de la capacité du système de mesure • Génération automatique de rapports PDF professionnels</div>
        <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.7;">
            Développé avec Streamlit • FPDF • Optimisé pour la qualité industrielle • Conforme aux normes AIAG
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
