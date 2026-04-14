import streamlit as st
import pandas as pd
import re
import io
import time
from datetime import datetime
from PIL import Image
import numpy as np

# ==============================================
# CONFIGURATION DE LA PAGE
# ==============================================
st.set_page_config(
    page_title="Extracteur OCR Pro",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================
# INITIALISATION DE L'ÉTAT DE SESSION
# ==============================================
def init_session_state():
    if 'photos_traitees' not in st.session_state:
        st.session_state.photos_traitees = []
    if 'compteur_photos' not in st.session_state:
        st.session_state.compteur_photos = 0
    if 'lecteur_ocr' not in st.session_state:
        st.session_state.lecteur_ocr = None

init_session_state()

# ==============================================
# CHARGEMENT DU MODÈLE OCR (PaddleOCR - Léger)
# ==============================================
@st.cache_resource
def charger_modele_ocr():
    """Charge PaddleOCR (plus léger qu'EasyOCR)"""
    try:
        from paddleocr import PaddleOCR
        # Configuration minimale pour le cloud
        return PaddleOCR(
            lang='fr', 
            use_angle_cls=True, 
            show_log=False,
            use_gpu=False
        )
    except ImportError as e:
        st.error(f"❌ PaddleOCR n'est pas installé : {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de PaddleOCR : {str(e)}")
        return None

# ==============================================
# PRÉTRAITEMENT D'IMAGE
# ==============================================
def pretraiter_image(image_pil):
    """Améliore la qualité pour l'OCR"""
    try:
        import cv2
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        return Image.fromarray(enhanced)
    except:
        return image_pil

# ==============================================
# EXTRACTION DES NUMÉROS
# ==============================================
def extraire_numeros(texte):
    """Extrait tous les nombres du texte"""
    patterns = [
        r'\d+[.,]?\d*',
        r'\d+[\s]?\d*[.,]?\d*',
        r'[\d]+[.,]?\d*\s?[€$£]',
    ]
    numeros = []
    for pattern in patterns:
        numeros.extend(re.findall(pattern, texte))
    
    return list(set([n.strip() for n in numeros if n.strip()]))

# ==============================================
# TRAITEMENT D'UNE PHOTO (ADAPTÉ POUR PADDLEOCR)
# ==============================================
def traiter_photo(image, nom_fichier, lecteur, pretraitement=True):
    """Traite une photo avec PaddleOCR"""
    debut = time.time()
    
    if pretraitement:
        image_traitee = pretraiter_image(image)
    else:
        image_traitee = image
    
    # Sauvegarde temporaire pour PaddleOCR
    img_bytes = io.BytesIO()
    image_traitee.save(img_bytes, format='PNG')
    img_path = img_bytes.getvalue()
    
    try:
        # PaddleOCR nécessite un fichier ou des bytes
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(img_path)
            tmp_path = tmp.name
        
        resultats_ocr = lecteur.ocr(tmp_path, cls=False)
        
        # Extraction du texte
        texte_complet = ""
        if resultats_ocr and resultats_ocr[0]:
            textes = [line[1][0] for line in resultats_ocr[0]]
            texte_complet = " ".join(textes)
        
        # Nettoyage du fichier temporaire
        import os
        os.unlink(tmp_path)
        
    except Exception as e:
        texte_complet = ""
        st.error(f"Erreur OCR : {str(e)}")
    
    numeros = extraire_numeros(texte_complet)
    temps_traitement = round(time.time() - debut, 2)
    
    return {
        "nom_fichier": nom_fichier,
        "numeros": ", ".join(numeros) if numeros else "Aucun numéro détecté",
        "texte_brut": texte_complet[:500] + "..." if len(texte_complet) > 500 else texte_complet,
        "nombre_numeros": len(numeros),
        "temps_traitement": temps_traitement,
        "horodatage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image": image,
    }

# ==============================================
# EXPORT EXCEL
# ==============================================
def exporter_excel(donnees):
    """Crée un fichier Excel"""
    df = pd.DataFrame([{
        "Nom du fichier": d["nom_fichier"],
        "Numéros extraits": d["numeros"],
        "Quantité": d["nombre_numeros"],
        "Texte brut": d["texte_brut"],
        "Temps (sec)": d["temps_traitement"],
        "Horodatage": d["horodatage"]
    } for d in donnees])
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultats_OCR')
    
    return output.getvalue()

# ==============================================
# INTERFACE PRINCIPALE
# ==============================================
def main():
    st.title("📸 Extracteur de Numéros OCR")
    st.markdown("---")
    
    # Chargement du modèle
    if st.session_state.lecteur_ocr is None:
        with st.spinner("🔄 Chargement du modèle OCR (première fois ~30s)..."):
            st.session_state.lecteur_ocr = charger_modele_ocr()
    
    lecteur = st.session_state.lecteur_ocr
    
    if lecteur is None:
        st.error("❌ Impossible de charger le modèle OCR.")
        st.stop()
    
    # ===== BARRE LATÉRALE =====
    with st.sidebar:
        st.header("⚙️ Configuration")
        pretraitement = st.checkbox("Activer le prétraitement", value=True)
        
        st.divider()
        st.subheader("📤 Ajouter des photos")
        
        nouveaux_fichiers = st.file_uploader(
            "Sélectionnez des images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.compteur_photos}"
        )
        
        if st.button("🔄 Réinitialiser l'upload", use_container_width=True):
            st.session_state.compteur_photos += 1
            st.rerun()
        
        st.divider()
        
        # Statistiques
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Photos", len(st.session_state.photos_traitees))
        with col2:
            total = sum(p["nombre_numeros"] for p in st.session_state.photos_traitees)
            st.metric("🔢 Numéros", total)
        
        st.divider()
        
        # Export
        if st.session_state.photos_traitees:
            excel_data = exporter_excel(st.session_state.photos_traitees)
            st.download_button(
                label="📥 Télécharger Excel",
                data=excel_data,
                file_name=f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            if st.button("🗑️ Tout effacer", use_container_width=True):
                st.session_state.photos_traitees = []
                st.rerun()
    
    # ===== TRAITEMENT =====
    if nouveaux_fichiers:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, fichier in enumerate(nouveaux_fichiers):
            if not any(p["nom_fichier"] == fichier.name for p in st.session_state.photos_traitees):
                status_text.text(f"🔍 Traitement de {fichier.name}...")
                image = Image.open(fichier)
                resultat = traiter_photo(image, fichier.name, lecteur, pretraitement)
                st.session_state.photos_traitees.append(resultat)
            
            progress_bar.progress((i + 1) / len(nouveaux_fichiers))
        
        status_text.text("✅ Terminé !")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        st.rerun()
    
    # ===== AFFICHAGE =====
    if st.session_state.photos_traitees:
        st.subheader("📋 Résultats")
        
        df = pd.DataFrame([{
            "Fichier": p["nom_fichier"],
            "Numéros extraits": p["numeros"],
            "Quantité": p["nombre_numeros"],
            "Temps": f"{p['temps_traitement']}s"
        } for p in st.session_state.photos_traitees])
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("### 🖼️ Détail par photo")
        for i, photo in enumerate(st.session_state.photos_traitees):
            with st.expander(f"📷 {photo['nom_fichier']} - {photo['nombre_numeros']} numéro(s)"):
                st.image(photo["image"], caption="Image originale", use_container_width=True)
                st.markdown(f"**Numéros :** `{photo['numeros']}`")
                st.caption(f"Texte brut : {photo['texte_brut']}")
                
                if st.button("🗑️ Supprimer cette photo", key=f"del_{i}"):
                    st.session_state.photos_traitees.pop(i)
                    st.rerun()
    else:
        st.info("👈 Ajoutez des photos pour commencer l'extraction")

if __name__ == "__main__":
    main()
