import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import time
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
    <div class="main-subtitle">Analyse avancée de la capacité du système de mesure avec rapport complet</div>
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

# ---------------- FONCTION POUR GÉNÉRER RAPPORT HTML ----------------
def generate_html_report(report_data, df_sample):
    """Génère un rapport HTML complet"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rapport Gage R&R - {report_data['study_name']}</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            * {{
                font-family: 'Inter', sans-serif;
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                background: #f8fafc;
                color: #333;
                line-height: 1.6;
                padding: 20px;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                padding: 30px;
            }}
            
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 12px;
                text-align: center;
                margin-bottom: 30px;
            }}
            
            .header h1 {{
                font-size: 28px;
                font-weight: 700;
                margin-bottom: 10px;
            }}
            
            .header .date {{
                font-size: 14px;
                opacity: 0.9;
            }}
            
            .section {{
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid #eef2f7;
            }}
            
            .section-title {{
                background: linear-gradient(90deg, #667eea, #764ba2);
                color: white;
                padding: 12px 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                font-size: 18px;
                font-weight: 600;
            }}
            
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }}
            
            .metric-card {{
                background: linear-gradient(145deg, #ffffff, #f5f7fa);
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                border-left: 4px solid;
                transition: transform 0.3s;
            }}
            
            .metric-card:hover {{
                transform: translateY(-5px);
            }}
            
            .metric-value {{
                font-size: 24px;
                font-weight: 700;
                margin: 10px 0;
            }}
            
            .metric-label {{
                color: #666;
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .status-badge {{
                display: inline-block;
                padding: 6px 15px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 14px;
                margin-bottom: 15px;
            }}
            
            .status-excellent {{
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }}
            
            .status-acceptable {{
                background: #fff3cd;
                color: #856404;
                border: 1px solid #ffeaa7;
            }}
            
            .status-unacceptable {{
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                font-size: 14px;
            }}
            
            th {{
                background: #667eea;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
            }}
            
            td {{
                padding: 10px;
                border-bottom: 1px solid #eef2f7;
            }}
            
            tr:nth-child(even) {{
                background: #f8fafc;
            }}
            
            .recommendations {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
            }}
            
            .recommendation-item {{
                margin: 10px 0;
                padding-left: 20px;
                position: relative;
            }}
            
            .recommendation-item:before {{
                content: "•";
                position: absolute;
                left: 0;
                color: #667eea;
                font-size: 20px;
            }}
            
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 2px solid #eef2f7;
                text-align: center;
                color: #666;
                font-size: 12px;
            }}
            
            .signature {{
                margin-top: 40px;
                text-align: center;
                padding: 20px;
            }}
            
            .signature-line {{
                width: 300px;
                height: 1px;
                background: #333;
                margin: 40px auto 10px;
            }}
            
            .page-break {{
                page-break-before: always;
            }}
            
            @media print {{
                body {{
                    padding: 0;
                }}
                
                .container {{
                    box-shadow: none;
                    padding: 20px;
                }}
                
                .metric-card {{
                    page-break-inside: avoid;
                }}
                
                .page-break {{
                    page-break-before: always;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- En-tête -->
            <div class="header">
                <h1>📊 RAPPORT GAGE R&R</h1>
                <p>Méthode des Étendues - Analyse du Système de Mesure</p>
                <p class="date">Date du rapport : {report_data['date']}</p>
                <p class="date">Nom de l'étude : {report_data['study_name']}</p>
            </div>
            
            <!-- Section 1: Résumé Exécutif -->
            <div class="section">
                <div class="section-title">1. RÉSUMÉ EXÉCUTIF</div>
                
                <div class="status-badge {report_data['status_class']}">
                    STATUT : {report_data['overall_status']} - {report_data['p_grr']:.2f}%
                </div>
                
                <p style="margin: 15px 0;">{report_data['overall_message']}</p>
                
                <div class="metrics-grid">
                    <div class="metric-card" style="border-left-color: #3498db;">
                        <div class="metric-label">Répétabilité (EV)</div>
                        <div class="metric-value">{report_data['ev']:.4f}</div>
                        <div style="color: #666; font-size: 12px;">{report_data['ev_percent']:.1f}% de la variation totale</div>
                    </div>
                    
                    <div class="metric-card" style="border-left-color: #2ecc71;">
                        <div class="metric-label">Reproductibilité (AV)</div>
                        <div class="metric-value">{report_data['av']:.4f}</div>
                        <div style="color: #666; font-size: 12px;">{report_data['av_percent']:.1f}% de la variation totale</div>
                    </div>
                    
                    <div class="metric-card" style="border-left-color: #9b59b6;">
                        <div class="metric-label">Variation Système (GRR)</div>
                        <div class="metric-value">{report_data['grr']:.4f}</div>
                        <div style="color: #666; font-size: 12px;">{report_data['p_grr']:.2f}% de la variation totale</div>
                    </div>
                    
                    <div class="metric-card" style="border-left-color: #e74c3c;">
                        <div class="metric-label">Variation Pièces (VP)</div>
                        <div class="metric-value">{report_data['vp']:.4f}</div>
                        <div style="color: #666; font-size: 12px;">{report_data['vp_percent']:.1f}% de la variation totale</div>
                    </div>
                </div>
            </div>
            
            <!-- Section 2: Informations de l'Étude -->
            <div class="section">
                <div class="section-title">2. INFORMATIONS DE L'ÉTUDE</div>
                
                <table>
                    <tr>
                        <th style="width: 40%;">Paramètre</th>
                        <th>Valeur</th>
                    </tr>
                    <tr>
                        <td>Date d'analyse</td>
                        <td>{report_data['date']}</td>
                    </tr>
                    <tr>
                        <td>Nombre de pièces</td>
                        <td>{report_data['n_pieces']}</td>
                    </tr>
                    <tr>
                        <td>Nombre d'opérateurs</td>
                        <td>{report_data['n_operateurs']}</td>
                    </tr>
                    <tr>
                        <td>Nombre d'essais par pièce</td>
                        <td>{report_data['n_essais']}</td>
                    </tr>
                    <tr>
                        <td>Facteur de confiance (k)</td>
                        <td>{report_data['confidence_factor']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Étendue moyenne (R̄)</td>
                        <td>{report_data['r_double_bar']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Ratio Signal/Bruit (VP/GRR)</td>
                        <td>{report_data['ratio_vp_grr']:.2f}</td>
                    </tr>
                </table>
            </div>
            
            <div class="page-break"></div>
            
            <!-- Section 3: Performance des Opérateurs -->
            <div class="section">
                <div class="section-title">3. PERFORMANCE DES OPÉRATEURS</div>
                
                <table>
                    <tr>
                        <th>Opérateur</th>
                        <th>Moyenne</th>
                        <th>Étendue Moyenne</th>
                        <th>Écart-Type</th>
                    </tr>
                    {''.join([f'''
                    <tr>
                        <td>👤 {op['name']}</td>
                        <td>{op['moyenne']:.4f}</td>
                        <td>{op['etendue']:.4f}</td>
                        <td>{op['ecart_type']:.4f}</td>
                    </tr>
                    ''' for op in report_data['operators']])}
                </table>
            </div>
            
            <!-- Section 4: Données Brutes (Extrait) -->
            <div class="section">
                <div class="section-title">4. DONNÉES BRUTES (EXTRAIT)</div>
                
                <table>
                    <tr>
                        <th>Pièce</th>
                        <th>OP1-1</th>
                        <th>OP1-2</th>
                        <th>OP1-3</th>
                        <th>OP2-1</th>
                        <th>OP2-2</th>
                        <th>OP2-3</th>
                        <th>OP3-1</th>
                        <th>OP3-2</th>
                        <th>OP3-3</th>
                    </tr>
                    {''.join([f'''
                    <tr>
                        <td>Pièce {i+1}</td>
                        <td>{df_sample.iloc[i]['OP1-1']:.4f}</td>
                        <td>{df_sample.iloc[i]['OP1-2']:.4f}</td>
                        <td>{df_sample.iloc[i]['OP1-3']:.4f}</td>
                        <td>{df_sample.iloc[i]['OP2-1']:.4f}</td>
                        <td>{df_sample.iloc[i]['OP2-2']:.4f}</td>
                        <td>{df_sample.iloc[i]['OP2-3']:.4f}</td>
                        <td>{df_sample.iloc[i]['OP3-1']:.4f}</td>
                        <td>{df_sample.iloc[i]['OP3-2']:.4f}</td>
                        <td>{df_sample.iloc[i]['OP3-3']:.4f}</td>
                    </tr>
                    ''' for i in range(min(10, len(df_sample)))])}
                </table>
                <p style="font-size: 12px; color: #666; margin-top: 10px;">
                    * Affichage des 10 premières pièces seulement
                </p>
            </div>
            
            <div class="page-break"></div>
            
            <!-- Section 5: Interprétation des Résultats -->
            <div class="section">
                <div class="section-title">5. INTERPRÉTATION DES RÉSULTATS</div>
                
                <div style="margin: 20px 0;">
                    <h3 style="color: #667eea; margin-bottom: 10px;">Source Principale de Variation</h3>
                    <p>{report_data['source_principale']}</p>
                    
                    <h3 style="color: #667eea; margin: 20px 0 10px 0;">Capacité de Discrimination</h3>
                    <p>{report_data['capacite_discrimination']}</p>
                    
                    <h3 style="color: #667eea; margin: 20px 0 10px 0;">Évaluation des Composantes</h3>
                    <ul style="margin-left: 20px;">
                        <li style="margin: 8px 0;">Répétabilité (EV): {report_data['diagnostic_ev']}</li>
                        <li style="margin: 8px 0;">Reproductibilité (AV): {report_data['diagnostic_av']}</li>
                        <li style="margin: 8px 0;">Ratio VP/GRR: {report_data['diagnostic_ratio']}</li>
                    </ul>
                </div>
            </div>
            
            <!-- Section 6: Recommandations -->
            <div class="section">
                <div class="section-title">6. RECOMMANDATIONS ET PLAN D'ACTION</div>
                
                <div class="recommendations">
                    <h3 style="color: #{'c62828' if report_data['p_grr'] > 30 else 'ef6c00' if report_data['p_grr'] > 15 else '2e7d32'}; 
                        margin-bottom: 15px;">
                        {report_data['recommandations_titre']}
                    </h3>
                    
                    {''.join([f'''
                    <div class="recommendation-item">{rec}</div>
                    ''' for rec in report_data['recommandations_liste']])}
                    
                    <h3 style="color: #667eea; margin: 25px 0 15px 0;">Priorités d'Action</h3>
                    {''.join([f'''
                    <div class="recommendation-item">{action}</div>
                    ''' for action in report_data['actions_prioritaires']])}
                </div>
            </div>
            
            <!-- Section 7: Critères d'Acceptation -->
            <div class="section">
                <div class="section-title">7. CRITÈRES D'ACCEPTATION</div>
                
                <table>
                    <tr>
                        <th>%GRR</th>
                        <th>Évaluation</th>
                        <th>Recommandation</th>
                    </tr>
                    <tr>
                        <td>&lt; 10%</td>
                        <td><span style="color: #27ae60;">✓ EXCELLENT</span></td>
                        <td>Système optimal, utilisation sans restriction</td>
                    </tr>
                    <tr>
                        <td>10% - 30%</td>
                        <td><span style="color: #f39c12;">⚠ ACCEPTABLE</span></td>
                        <td>Système acceptable, améliorations possibles</td>
                    </tr>
                    <tr>
                        <td>&gt; 30%</td>
                        <td><span style="color: #c0392b;">✗ INACCEPTABLE</span></td>
                        <td>Action corrective requise avant utilisation</td>
                    </tr>
                </table>
            </div>
            
            <!-- Signature -->
            <div class="signature">
                <div class="signature-line"></div>
                <p style="margin-top: 10px; color: #666;">
                    Responsable Qualité / Ingénieur Méthodes
                </p>
                <p style="color: #666; margin-top: 5px;">
                    Date : _________________________
                </p>
            </div>
            
            <!-- Pied de page -->
            <div class="footer">
                <p>Rapport généré automatiquement par l'application Gage R&R - Méthode des Étendues</p>
                <p>Conforme aux normes AIAG MSA 4th Edition</p>
                <p>© {datetime.now().year} - Système d'Analyse de la Capacité de Mesure</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

# ---------------- FONCTION POUR GÉNÉRER RAPPORT EXCEL AVEC INTERPRÉTATION ----------------
def generate_excel_report_with_interpretation(df, results, operators_data, report_data):
    """Génère un rapport Excel complet avec interprétation"""
    
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Feuille 1: Résultats Principaux
        results_df = pd.DataFrame({
            "Paramètre": ["%GRR Total", "Répétabilité (EV)", "Reproductibilité (AV)", 
                         "Variation Système (GRR)", "Variation Pièces (VP)", 
                         "Variation Totale (VT)", "Ratio VP/GRR", "Étendue moyenne (R̄)"],
            "Valeur": [
                f"{results['p_grr']:.2f}%",
                f"{results['ev']:.4f} ({results['ev_percent']:.1f}%)",
                f"{results['av']:.4f} ({results['av_percent']:.1f}%)",
                f"{results['grr']:.4f}",
                f"{results['vp']:.4f} ({results['vp_percent']:.1f}%)",
                f"{results['vt']:.4f}",
                f"{results['ratio_vp_grr']:.2f}",
                f"{results['r_double_bar']:.4f}"
            ],
            "Statut": [
                report_data['overall_status'],
                "✓ Acceptable" if results['ev_percent'] < 20 else "⚠ Conditionnel",
                "✓ Acceptable" if results['av_percent'] < 20 else "⚠ Conditionnel",
                "✓ Excellent" if results['p_grr'] < 10 else ("⚠ Acceptable" if results['p_grr'] <= 30 else "✗ Inacceptable"),
                "✓ Bonne discrimination" if results['ratio_vp_grr'] > 4 else "⚠ Discrimination limitée",
                "-",
                "✓ Bon" if results['ratio_vp_grr'] > 4 else ("⚠ Moyen" if results['ratio_vp_grr'] > 2 else "✗ Faible"),
                "-"
            ]
        })
        results_df.to_excel(writer, sheet_name='Résultats Principaux', index=False)
        
        # Feuille 2: Données Brutes
        df.to_excel(writer, sheet_name='Données Brutes', index=False)
        
        # Feuille 3: Performance Opérateurs
        op_df = pd.DataFrame(operators_data)
        op_df.to_excel(writer, sheet_name='Performance Opérateurs', index=False)
        
        # Feuille 4: Rapport d'Interprétation
        interpretation_data = {
            "Section": [
                "ÉVALUATION GLOBALE",
                "Statut",
                "Score %GRR",
                "Message",
                "",
                "ANALYSE DES COMPOSANTES",
                "Source principale",
                "Diagnostic Répétabilité",
                "Diagnostic Reproductibilité",
                "Diagnostic Discrimination",
                "",
                "RECOMMANDATIONS",
                "Titre",
                "Actions prioritaires",
                "",
                "INFORMATIONS DE L'ÉTUDE",
                "Date",
                "Pièces (n)",
                "Opérateurs (o)",
                "Essais (r)",
                "Facteur k"
            ],
            "Contenu": [
                "",
                report_data['overall_status'],
                f"{results['p_grr']:.2f}%",
                report_data['overall_message'],
                "",
                "",
                report_data['source_principale'],
                report_data['diagnostic_ev'],
                report_data['diagnostic_av'],
                report_data['diagnostic_ratio'],
                "",
                "",
                report_data['recommandations_titre'],
                "; ".join(report_data['recommandations_liste'][:3]),
                "",
                "",
                report_data['date'],
                str(results['n_pieces']),
                str(results['n_operateurs']),
                str(results['n_essais']),
                f"{results['confidence_factor']:.2f}"
            ]
        }
        interpretation_df = pd.DataFrame(interpretation_data)
        interpretation_df.to_excel(writer, sheet_name='Rapport Interprétation', index=False)
        
        # Feuille 5: Plan d'Action
        action_data = {
            "Priorité": ["P1", "P1", "P1", "P2", "P2", "P3"],
            "Action": report_data['actions_prioritaires'] + ["Suivre les indicateurs clés", "Documenter les actions", "Planifier réévaluation"],
            "Responsable": ["Technicien", "Qualité", "Formation", "Qualité", "Qualité", "Management"],
            "Échéance": ["Immédiate", "1 semaine", "2 semaines", "1 mois", "Continu", "6 mois"],
            "Statut": ["À faire", "À faire", "À faire", "À planifier", "En cours", "À planifier"]
        }
        action_df = pd.DataFrame(action_data)
        action_df.to_excel(writer, sheet_name='Plan Action', index=False)
    
    output.seek(0)
    return output

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
    
    # Options de rapport
    st.markdown("---")
    st.markdown('<div style="font-size: 1.2rem; font-weight: 600; color: #2c3e50; margin: 1.5rem 0 1rem 0;">📄 Options de Rapport</div>', unsafe_allow_html=True)
    
    study_name = st.text_input("Nom de l'étude", "Analyse Gage R&R")
    company_name = st.text_input("Nom de l'entreprise", "")
    
    export_format = st.radio(
        "Format d'export",
        ["HTML (pour impression/PDF)", "Excel complet"],
        index=0
    )
    
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

    # Calculs pour l'interprétation
    ev_percent = (ev / vt) * 100 if vt > 0 else 0
    av_percent = (av / vt) * 100 if vt > 0 else 0
    vp_percent = (vp / vt) * 100 if vt > 0 else 0
    ratio_vp_grr = vp / grr if grr > 0 else 0

    # Données des opérateurs
    operators = []
    for i in range(3):
        op_cols = [f"OP{i+1}-1", f"OP{i+1}-2", f"OP{i+1}-3"]
        op_data = df[op_cols].values.flatten()
        operators.append({
            'name': f'Opérateur {i+1}',
            'moyenne': np.mean(op_data),
            'etendue': [r_bar_op1, r_bar_op2, r_bar_op3][i],
            'ecart_type': np.std(op_data)
        })

    # ---------------- PRÉPARATION DES DONNÉES POUR RAPPORT ----------------
    
    # Évaluation générale
    if p_grr < 10:
        overall_status = "EXCELLENT"
        overall_color = "#27ae60"
        status_class = "status-excellent"
        overall_message = "Le système de mesure est optimal et fiable pour les analyses critiques."
    elif p_grr <= 30:
        overall_status = "ACCEPTABLE"
        overall_color = "#f39c12"
        status_class = "status-acceptable"
        overall_message = "Le système est acceptable mais des améliorations sont recommandées pour une meilleure fiabilité."
    else:
        overall_status = "INACCEPTABLE"
        overall_color = "#c0392b"
        status_class = "status-unacceptable"
        overall_message = "Le système nécessite des actions correctives urgentes avant toute utilisation."
    
    # Analyse par composante
    if ev_percent > av_percent:
        source_principale = "La RÉPÉTABILITÉ (EV) est la principale source de variation, indiquant une variabilité intra-opérateur élevée."
    else:
        source_principale = "La REPRODUCTIBILITÉ (AV) est la principale source de variation, indiquant des différences significatives entre opérateurs."
    
    # Diagnostic
    diagnostic_ev = f"{ev_percent:.1f}% - {'Excellente' if ev_percent < 10 else 'Bonne' if ev_percent < 20 else 'À améliorer'}"
    diagnostic_av = f"{av_percent:.1f}% - {'Excellente' if av_percent < 10 else 'Bonne' if av_percent < 20 else 'À améliorer'}"
    
    if ratio_vp_grr > 4:
        diagnostic_ratio = f"{ratio_vp_grr:.2f}:1 - Excellente capacité à distinguer les pièces"
        capacite_discrimination = f"Avec un ratio de {ratio_vp_grr:.2f}:1, le système possède une excellente capacité à distinguer les différences entre pièces."
    elif ratio_vp_grr > 2:
        diagnostic_ratio = f"{ratio_vp_grr:.2f}:1 - Capacité acceptable"
        capacite_discrimination = f"Avec un ratio de {ratio_vp_grr:.2f}:1, le système possède une capacité acceptable à distinguer les différences entre pièces."
    else:
        diagnostic_ratio = f"{ratio_vp_grr:.2f}:1 - Faible capacité"
        capacite_discrimination = f"Avec un ratio de {ratio_vp_grr:.2f}:1, le système a une faible capacité à distinguer les différences entre pièces."
    
    # Recommandations
    if p_grr > 30:
        recommandations_titre = "ACTIONS CORRECTIVES IMMÉDIATES"
        recommandations_liste = [
            "Suspendre temporairement l'utilisation du système pour les mesures critiques",
            "Réétalonner tous les instruments de mesure",
            "Former/reformer les opérateurs avec méthode standardisée",
            "Vérifier la stabilité des conditions environnementales",
            "Revoir le plan d'échantillonnage des pièces"
        ]
        actions_prioritaires = [
            "Standardiser la méthode de prise de mesure",
            "Vérifier l'état et l'étalonnage des instruments",
            "Minimiser les variations environnementales"
        ]
    elif p_grr > 15:
        recommandations_titre = "AMÉLIORATIONS RECOMMANDÉES"
        recommandations_liste = [
            "Améliorer la procédure écrite de mesure",
            "Implémenter des gabarits ou dispositifs d'aide",
            "Organiser des audits croisés entre opérateurs",
            "Augmenter le nombre d'essais pour réduire l'incertitude",
            "Surveiller régulièrement la performance du système"
        ]
        if ev_percent > av_percent:
            actions_prioritaires = [
                "Standardiser la méthode de prise de mesure",
                "Vérifier l'état et l'étalonnage des instruments",
                "Minimiser les variations environnementales"
            ]
        else:
            actions_prioritaires = [
                "Organiser une formation commune à tous les opérateurs",
                "Créer des aides visuelles pour les décisions limites",
                "Implémenter des audits croisés réguliers"
            ]
    else:
        recommandations_titre = "ACTIONS DE MAINTENANCE"
        recommandations_liste = [
            "Maintenir la documentation à jour",
            "Programmer des étalonnages réguliers",
            "Surveiller les tendances dans le temps",
            "Former les nouveaux opérateurs avec méthode validée",
            "Réaliser des vérifications périodiques du système"
        ]
        actions_prioritaires = [
            "Maintenir la documentation à jour",
            "Programmer des étalonnages préventifs",
            "Surveiller statistiquement les performances"
        ]
    
    # Données pour le rapport
    report_data = {
        'date': datetime.now().strftime("%d/%m/%Y %H:%M"),
        'study_name': study_name,
        'company_name': company_name,
        'p_grr': p_grr,
        'ev': ev,
        'av': av,
        'grr': grr,
        'vp': vp,
        'vt': vt,
        'ev_percent': ev_percent,
        'av_percent': av_percent,
        'vp_percent': vp_percent,
        'ratio_vp_grr': ratio_vp_grr,
        'r_double_bar': r_double_bar,
        'n_pieces': n_pieces,
        'n_operateurs': n_operateurs,
        'n_essais': n_essais,
        'confidence_factor': confidence_factor,
        'overall_status': overall_status,
        'status_class': status_class,
        'overall_message': overall_message,
        'source_principale': source_principale,
        'capacite_discrimination': capacite_discrimination,
        'diagnostic_ev': diagnostic_ev,
        'diagnostic_av': diagnostic_av,
        'diagnostic_ratio': diagnostic_ratio,
        'recommandations_titre': recommandations_titre,
        'recommandations_liste': recommandations_liste,
        'actions_prioritaires': actions_prioritaires,
        'operators': operators
    }
    
    # Données pour les résultats
    results_data = {
        'p_grr': p_grr,
        'ev': ev,
        'av': av,
        'grr': grr,
        'vp': vp,
        'vt': vt,
        'ev_percent': ev_percent,
        'av_percent': av_percent,
        'vp_percent': vp_percent,
        'ratio_vp_grr': ratio_vp_grr,
        'r_double_bar': r_double_bar,
        'n_pieces': n_pieces,
        'n_operateurs': n_operateurs,
        'n_essais': n_essais,
        'confidence_factor': confidence_factor
    }
    
    operators_display_data = [
        {
            'Opérateur': op['name'],
            'Moyenne': f"{op['moyenne']:.4f}",
            'Étendue Moyenne': f"{op['etendue']:.4f}",
            'Écart-Type': f"{op['ecart_type']:.4f}"
        }
        for op in operators
    ]

    # ---------------- GÉNÉRATION DE RAPPORT ----------------
    st.markdown('<div class="section-header"><span>📄 Génération de Rapport</span></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Aperçu du Rapport")
        st.markdown(f"""
        <div class="report-card" style="border-left-color: {'#2ecc71' if p_grr < 10 else '#f1c40f' if p_grr <= 30 else '#e74c3c'};">
            <div style="font-size: 1.2rem; font-weight: 600; color: #2c3e50;">
                {study_name}
            </div>
            <div style="margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #7f8c8d;">Statut:</span>
                    <span style="font-weight: 600; color: {'#27ae60' if p_grr < 10 else '#f39c12' if p_grr <= 30 else '#c0392b'}">
                        {overall_status}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #7f8c8d;">%GRR:</span>
                    <span style="font-weight: 600;">{p_grr:.2f}%</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #7f8c8d;">Format:</span>
                    <span style="font-weight: 600;">{export_format}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Boutons de génération
        st.markdown("---")
        
        if export_format == "HTML (pour impression/PDF)":
            if st.button("📄 Générer Rapport HTML", type="primary", use_container_width=True):
                with st.spinner("🔄 Génération du rapport HTML en cours..."):
                    # Générer le HTML
                    html_content = generate_html_report(report_data, df)
                    
                    # Créer un fichier HTML téléchargeable
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"Rapport_Gage_RR_{study_name.replace(' ', '_')}_{timestamp}.html"
                    
                    # Encoder en base64
                    b64 = base64.b64encode(html_content.encode()).decode()
                    
                    # Afficher le bouton de téléchargement
                    st.markdown(f"""
                    <div style="text-align: center; margin: 2rem 0;">
                        <a href="data:text/html;base64,{b64}" 
                           download="{filename}"
                           style="text-decoration: none;">
                            <div class="download-btn-pdf">
                                📥 Télécharger le Rapport HTML
                            </div>
                        </a>
                        <div style="color: #7f8c8d; font-size: 0.9rem; margin-top: 1rem;">
                            Fichier: {filename}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Instructions pour conversion PDF
                    with st.expander("ℹ️ Comment convertir en PDF"):
                        st.markdown("""
                        ### Instructions pour conversion HTML vers PDF:
                        
                        1. **Téléchargez** le fichier HTML ci-dessus
                        2. **Ouvrez-le** dans votre navigateur (Chrome, Edge, Firefox)
                        3. **Imprimez** la page (Ctrl+P ou Cmd+P)
                        4. **Choisissez** "Enregistrer au format PDF" comme imprimante
                        5. **Ajustez** les marges si nécessaire
                        6. **Enregistrez** le fichier PDF
                        
                        ### Paramètres d'impression recommandés:
                        - Orientation: Portrait
                        - Marges: Minimales
                        - Mise à l'échelle: 100%
                        - En-têtes et pieds de page: Désactivés
                        - Arrière-plan: Inclure
                        """)
                    
                    st.success("✅ Rapport HTML généré avec succès!")
        
        else:  # Excel format
            if st.button("📊 Générer Rapport Excel", type="primary", use_container_width=True):
                with st.spinner("🔄 Génération du rapport Excel en cours..."):
                    # Générer le rapport Excel
                    excel_output = generate_excel_report_with_interpretation(
                        df, results_data, operators_display_data, report_data
                    )
                    
                    # Nom du fichier
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"Rapport_Gage_RR_{study_name.replace(' ', '_')}_{timestamp}.xlsx"
                    
                    # Encoder en base64
                    b64 = base64.b64encode(excel_output.getvalue()).decode()
                    
                    # Afficher le bouton de téléchargement
                    st.markdown(f"""
                    <div style="text-align: center; margin: 2rem 0;">
                        <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" 
                           download="{filename}"
                           style="text-decoration: none;">
                            <div class="download-btn">
                                📥 Télécharger le Rapport Excel
                            </div>
                        </a>
                        <div style="color: #7f8c8d; font-size: 0.9rem; margin-top: 1rem;">
                            Fichier: {filename} • {len(excel_output.getvalue())//1024} Ko
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("✅ Rapport Excel généré avec succès!")
    
    with col2:
        st.markdown("### 📋 Contenu du Rapport")
        
        sections = [
            "✅ Page de titre professionnelle",
            "✅ Résumé exécutif avec évaluation",
            "✅ Métriques complètes détaillées",
            "✅ Informations sur l'étude",
            "✅ Performance des opérateurs",
            "✅ Données brutes (extrait)",
            "✅ Interprétation détaillée des résultats",
            "✅ Recommandations personnalisées",
            "✅ Plan d'action prioritaire",
            "✅ Critères d'acceptation AIAG",
            "✅ Zone de signature",
            "✅ Références normatives"
        ]
        
        for section in sections:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 10px; margin: 0.3rem 0;">
                <div style="color: #2ecc71;">✓</div>
                <div style="color: #2c3e50;">{section}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Résumé des Résultats")
        
        summary_data = {
            "Indicateur": ["%GRR Total", "Répétabilité", "Reproductibilité", "Ratio VP/GRR"],
            "Valeur": [f"{p_grr:.2f}%", f"{ev_percent:.1f}%", f"{av_percent:.1f}%", f"{ratio_vp_grr:.2f}"],
            "Évaluation": [overall_status, 
                          "✓" if ev_percent < 20 else "⚠", 
                          "✓" if av_percent < 20 else "⚠",
                          "✓" if ratio_vp_grr > 4 else ("⚠" if ratio_vp_grr > 2 else "✗")]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(
            summary_df.style
            .apply(lambda x: ['background: #d4edda' if v in ['✓', 'EXCELLENT', 'ACCEPTABLE'] else 
                             'background: #fff3cd' if v == '⚠' else 
                             'background: #f8d7da' if v in ['✗', 'INACCEPTABLE'] else '' for v in x], 
                   subset=['Évaluation'])
            .set_properties(**{'text-align': 'center'}),
            use_container_width=True,
            hide_index=True
        )

    # ---------------- VISUALISATIONS ----------------
    st.markdown('<div class="section-header"><span>📈 Visualisations des Résultats</span></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique 1 : Composantes de variation
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        components = ['EV', 'AV', 'GRR', 'VP', 'VT']
        values = [ev, av, grr, vp, vt]
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']
        
        bars = ax1.bar(components, values, color=colors, edgecolor='white', 
                      linewidth=2, alpha=0.9, zorder=3)
        
        ax1.grid(True, alpha=0.3, zorder=0)
        ax1.set_facecolor('#f8fafc')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
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
        # Graphique 2 : Répartition en pourcentage
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        
        labels = ['Variation Système\n(GRR)', 'Variation Pièces\n(VP)']
        sizes = [grr**2, vp**2]
        colors = ['#9b59b6', '#e74c3c']
        explode = (0.1, 0)
        
        wedges, texts, autotexts = ax2.pie(
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
        fig2.gca().add_artist(centre_circle)
        
        ax2.axis('equal')
        ax2.set_title('🥧 Répartition des Variations', fontsize=14, fontweight=600, pad=20)
        plt.tight_layout()
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)
        plt.close()

    # ---------------- RÉSULTATS PRINCIPAUX ----------------
    st.markdown('<div class="section-header"><span>📊 Résultats Principaux</span></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_data = [
        ("EV", ev, "#3498db", f"Répétabilité\n({ev_percent:.1f}%)"),
        ("AV", av, "#2ecc71", f"Reproductibilité\n({av_percent:.1f}%)"),
        ("GRR", grr, "#9b59b6", f"Variation Système\n({p_grr:.1f}%)"),
        ("VP", vp, "#e74c3c", f"Variation Pièces\n({vp_percent:.1f}%)")
    ]
    
    for col, (label, value, color, desc) in zip([col1, col2, col3, col4], metrics_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{desc}</div>
                <div class="metric-value" style="background: linear-gradient(135deg, {color}, #2c3e50); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {value:.3f}
                </div>
                <div style="color: #95a5a6; font-size: 0.9rem; margin-top: 0.5rem;">
                    <strong>{label}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Barre de progression
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

    # ---------------- INTERPRÉTATION DÉTAILLÉE ----------------
    st.markdown('<div class="section-header"><span>💡 Interprétation Détaillée</span></div>', unsafe_allow_html=True)
    
    with st.expander("🔍 Analyse Complète", expanded=True):
        tab1, tab2, tab3 = st.tabs(["📊 Évaluation", "🎯 Recommandations", "📈 Performance"])
        
        with tab1:
            st.markdown(f"""
            <div style="background: {'#d4edda' if p_grr < 10 else '#fff3cd' if p_grr <= 30 else '#f8d7da'}; 
                        padding: 1.5rem; border-radius: 10px; border-left: 5px solid {'#28a745' if p_grr < 10 else '#ffc107' if p_grr <= 30 else '#dc3545'};">
                <div style="font-size: 1.2rem; font-weight: 600; color: {'#155724' if p_grr < 10 else '#856404' if p_grr <= 30 else '#721c24'}; 
                            margin-bottom: 1rem;">
                    Évaluation: {overall_status}
                </div>
                <div style="color: {'#155724' if p_grr < 10 else '#856404' if p_grr <= 30 else '#721c24'};">
                    {overall_message}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Analyse des Composantes")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div style="background: #e3f2fd; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <div style="font-weight: 600; color: #1565c0;">Répétabilité (EV)</div>
                    <div style="color: #424242; margin-top: 0.5rem;">
                        {diagnostic_ev}
                        <div style="font-size: 0.9rem; margin-top: 0.3rem;">
                            Variabilité des mesures répétées par le même opérateur
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div style="background: #e8f5e9; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <div style="font-weight: 600; color: #2e7d32;">Reproductibilité (AV)</div>
                    <div style="color: #424242; margin-top: 0.5rem;">
                        {diagnostic_av}
                        <div style="font-size: 0.9rem; margin-top: 0.3rem;">
                            Différences entre les opérateurs
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: #fff3e0; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <div style="font-weight: 600; color: #ef6c00;">Capacité de Discrimination</div>
                <div style="color: #424242; margin-top: 0.5rem;">
                    {capacite_discrimination}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown(f"""
            <div style="background: {'#ffebee' if p_grr > 30 else '#fff3e0' if p_grr > 15 else '#e8f5e9'}; 
                        padding: 1.5rem; border-radius: 10px;">
                <div style="font-size: 1.2rem; font-weight: 600; color: {'#c62828' if p_grr > 30 else '#ef6c00' if p_grr > 15 else '#2e7d32'}; 
                            margin-bottom: 1rem;">
                    {recommandations_titre}
                </div>
                <div style="color: {'#721c24' if p_grr > 30 else '#856404' if p_grr > 15 else '#2e7d32'};">
                    {''.join([f'<div style="margin: 0.5rem 0;">• {rec}</div>' for rec in recommandations_liste])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Plan d'Action Prioritaire")
            for i, action in enumerate(actions_prioritaires, 1):
                st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #667eea;">
                    <div style="font-weight: 600; color: #2c3e50;">P{i}: {action}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("#### Performance des Opérateurs")
            operators_df = pd.DataFrame(operators_display_data)
            st.dataframe(
                operators_df.style
                .background_gradient(subset=['Moyenne', 'Étendue Moyenne', 'Écart-Type'], cmap='YlOrRd')
                .set_properties(**{'text-align': 'center'}),
                use_container_width=True
            )
            
            st.markdown("#### Indicateurs Secondaires")
            
            col_x, col_y = st.columns(2)
            with col_x:
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <div style="font-weight: 600; color: #2c3e50;">Ratio Signal/Bruit</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {'#2ecc71' if ratio_vp_grr > 4 else '#f1c40f' if ratio_vp_grr > 2 else '#e74c3c'}">
                        {ratio_vp_grr:.2f}:1
                    </div>
                    <div style="font-size: 0.9rem; color: #7f8c8d; margin-top: 0.3rem;">
                        {'Excellente' if ratio_vp_grr > 4 else 'Acceptable' if ratio_vp_grr > 2 else 'Faible'} discrimination
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_y:
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <div style="font-weight: 600; color: #2c3e50;">Étendue Moyenne (R̄)</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #3498db">
                        {r_double_bar:.4f}
                    </div>
                    <div style="font-size: 0.9rem; color: #7f8c8d; margin-top: 0.3rem;">
                        Moyenne des étendues par opérateur
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Pied de page
st.markdown("""
<div style="margin-top: 4rem; padding: 2rem; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); 
            border-radius: 20px; text-align: center; border-top: 1px solid #e0e6ed;">
    <div style="font-size: 0.9rem; color: #7f8c8d;">
        <div style="display: flex; justify-content: center; align-items: center; gap: 10px; margin-bottom: 0.5rem;">
            <div>📊</div>
            <div><strong>Gage R&R - Méthode des Étendues avec Rapports Complets</strong></div>
            <div>⚡</div>
        </div>
        <div>Analyse avancée de la capacité du système de mesure • Rapports HTML/Excel professionnels</div>
        <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.7;">
            Conforme aux normes AIAG MSA • Utilise uniquement: Streamlit, Pandas, NumPy, Matplotlib, OpenPyXL
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
