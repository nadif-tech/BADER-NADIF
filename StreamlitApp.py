import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import time

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Gage R&R - Méthode des Étendues",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= STYLE PROFESSIONNEL =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

body { background-color: #f4f6f9; }

.main-header {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    padding: 2rem;
    border-radius: 14px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 12px 30px rgba(0,0,0,0.12);
}
.main-title {
    color: #ffffff;
    font-size: 2.4rem;
    font-weight: 700;
}
.main-subtitle {
    color: #cbd5e1;
    font-size: 1rem;
}

.section-header {
    background: #ffffff;
    padding: 0.9rem 1.2rem;
    border-radius: 10px;
    margin: 2rem 0 1rem 0;
    font-weight: 600;
    border-left: 5px solid #2563eb;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}

.sidebar-content {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 0 16px 16px 0;
}

.metric-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
}
.metric-label {
    color: #64748b;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #0f172a;
}

.result-indicator {
    padding: 1.2rem;
    border-radius: 12px;
    margin: 1.5rem 0;
    text-align: center;
    font-weight: 600;
}
.good {
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #34d399;
}
.warning {
    background: #fffbeb;
    color: #b45309;
    border: 1px solid #facc15;
}
.bad {
    background: #fef2f2;
    color: #b91c1c;
    border: 1px solid #f87171;
}

.upload-area {
    border: 2px dashed #2563eb;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    background: #f8fafc;
}

.plot-container {
    background: #ffffff;
    padding: 1.2rem;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
}

.download-btn {
    background: #2563eb;
    color: white;
    padding: 0.9rem 1.8rem;
    border-radius: 10px;
    font-weight: 600;
    text-decoration: none;
}
.download-btn:hover { background: #1d4ed8; }
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="main-header">
    <div class="main-title">📊 Gage R&R – Méthode des Étendues</div>
    <div class="main-subtitle">Analyse de la capacité du système de mesure</div>
</div>
""", unsafe_allow_html=True)

# ================= FONCTION D2 =================
def get_d2(z, w):
    if z > 15 and w == 3: return 1.693
    if z == 1 and w == 3: return 1.91
    if z == 1 and w == 10: return 3.18
    return 1.0

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    confidence_factor = st.slider("Facteur de confiance (k)", 4.0, 6.0, 5.15, 0.05)
    st.markdown("</div>", unsafe_allow_html=True)

# ================= UPLOAD =================
st.markdown('<div class="section-header">📥 Importation des données</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["xlsx"], label_visibility="collapsed")

if uploaded_file is None:
    st.markdown("""
    <div class="upload-area">
        <h3>Glissez votre fichier Excel ici</h3>
        <p>Colonnes : OP1-1 … OP3-3 (3 opérateurs × 3 essais)</p>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file:
    with st.spinner("Analyse en cours..."):
        time.sleep(0.5)
        df = pd.read_excel(uploaded_file)

    op1 = ["OP1-1","OP1-2","OP1-3"]
    op2 = ["OP2-1","OP2-2","OP2-3"]
    op3 = ["OP3-1","OP3-2","OP3-3"]

    n_pieces, n_operateurs, n_essais = df.shape[0], 3, 3

    df["R1"] = df[op1].max(axis=1) - df[op1].min(axis=1)
    df["R2"] = df[op2].max(axis=1) - df[op2].min(axis=1)
    df["R3"] = df[op3].max(axis=1) - df[op3].min(axis=1)

    r_bar = (df["R1"].mean()+df["R2"].mean()+df["R3"].mean())/3
    ev = confidence_factor * r_bar / get_d2(n_pieces*n_operateurs,n_essais)

    means = [df[op1].values.mean(), df[op2].values.mean(), df[op3].values.mean()]
    av = np.sqrt(max(0,(confidence_factor*(max(means)-min(means))/get_d2(1,3))**2 - ev**2/(n_pieces*n_essais)))

    grr = np.sqrt(ev**2 + av**2)

    df["Mean"] = df[op1+op2+op3].mean(axis=1)
    vp = confidence_factor*(df["Mean"].max()-df["Mean"].min())/get_d2(1,n_pieces)

    vt = np.sqrt(grr**2 + vp**2)
    p_grr = grr/vt*100

    # ================= METRICS =================
    st.markdown('<div class="section-header">📊 Résultats</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col,label,val in zip(cols,["EV","AV","GRR","%GRR"],[ev,av,grr,p_grr]):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val:.3f}{'%' if label=='%GRR' else ''}</div>
        </div>
        """, unsafe_allow_html=True)

    # ================= STATUS =================
    if p_grr < 10:
        cls,msg="good","✅ Système excellent"
    elif p_grr <= 30:
        cls,msg="warning","⚠ Système acceptable"
    else:
        cls,msg="bad","❌ Système non acceptable"

    st.markdown(f"""
    <div class="result-indicator {cls}">{msg}</div>
    """, unsafe_allow_html=True)

    # ================= EXPORT =================
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame({
            "EV":[ev],"AV":[av],"GRR":[grr],"%GRR":[p_grr]
        }).to_excel(writer,index=False)

    output.seek(0)
    st.markdown(
        f"<a class='download-btn' download='gage_rr.xlsx' href='data:application/octet-stream;base64,{output.getvalue().hex()}'>📥 Télécharger le rapport</a>",
        unsafe_allow_html=True
    )

# ================= FOOTER =================
st.markdown("""
<div style="text-align:center; color:#64748b; margin-top:3rem;">
    Gage R&R • Streamlit • Qualité Industrielle
</div>
""", unsafe_allow_html=True)
