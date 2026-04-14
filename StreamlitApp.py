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
    if 'historique' not in st.session_state:
        st.session_state.historique = []
    if 'compteur_photos' not in st.session_state:
        st.session_state.compteur_photos = 0
    if 'lecteur_ocr' not in st.session_state:
        st.session_state.lecteur_ocr = None

init_session_state()

# ==============================================
# CHARGEMENT DU MODÈLE OCR (version cloud)
# ==============================================
@st.cache_resource
def charger_modele_ocr():
    """Charge EasyOCR avec configuration optimisée pour le cloud"""
    try:
        import easyocr
        # Configuration pour CPU uniquement (cloud gratuit)
        return easyocr.Reader(['fr', 'en'], gpu=False, verbose=False)
    except ImportError:
        st.error("❌ EasyOCR n'est pas installé correctement")
        return None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement d'EasyOCR : {str(e)}")
        return None

# ==============================================
# PRÉTRAITEMENT D'IMAGE (OpenCV)
# ==============================================
def pretraiter_image(image_pil):
    """Améliore la qualité pour l'OCR"""
    try:
        import cv2
        # Convertir PIL en OpenCV
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # Niveaux de gris
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Amélioration du contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return Image.fromarray(enhanced)
    except ImportError:
        st.warning("⚠️ OpenCV non disponible, prétraitement désactivé")
        return image_pil
    except Exception as e:
        st.warning(f"⚠️ Erreur prétraitement : {str(e)}")
        return image_pil

# ==============================================
# EXTRACTION DES NUMÉROS
# ==============================================
def extraire_numeros(texte):
    """Extrait tous les nombres du texte"""
    patterns = [
        r'\d+[.,]?\d*',           # Nombres simples
        r'\d+[\s]?\d*[.,]?\d*',   # Nombres avec espaces
        r'[\d]+[.,]?\d*\s?[€$£]', # Montants
    ]
    numeros = []
    for pattern in patterns:
        numeros.extend(re.findall(pattern, texte))
    
    # Nettoyage
    numeros = list(set([n.strip() for n in numeros if n.strip()]))
    return numeros

# ==============================================
# TRAITEMENT D'UNE PHOTO
# ==============================================
def traiter_photo(image, nom_fichier, lecteur, pretraitement=True):
    """Traite une photo avec OCR"""
    debut = time.time()
    
    # Prétraitement
    if pretraitement:
        image_traitee = pretraiter_image(image)
    else:
        image_traitee = image
    
    # OCR
    img_bytes = io.BytesIO()
    image_traitee.save(img_bytes, format='PNG')
    
    try:
        resultats_ocr = lecteur.readtext(img_bytes.getvalue(), detail=0, paragraph=True)
        texte_complet = " ".join(resultats_ocr)
    except Exception as e:
        texte_complet = ""
        st.error(f"Erreur OCR : {str(e)}")
    
    # Extraction
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
        "image_traitee": image_traitee
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
    
    # Chargement du modèle OCR (une seule fois)
    if st.session_state.lecteur_ocr is None:
        with st.spinner("🔄 Chargement du modèle OCR (peut prendre 30-60s la première fois)..."):
            st.session_state.lecteur_ocr = charger_modele_ocr()
    
    lecteur = st.session_state.lecteur_ocr
    
    if lecteur is None:
        st.error("❌ Impossible de charger le modèle OCR. Vérifiez l'installation.")
        st.stop()
    
    # ===== BARRE LATÉRALE =====
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        pretraitement = st.checkbox("Activer le prétraitement d'image", value=True)
        
        st.divider()
        
        st.subheader("📤 Ajouter des photos")
        nouveaux_fichiers = st.file_uploader(
            "Sélectionnez des images",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.compteur_photos}",
            help="Ajoutez une ou plusieurs photos"
        )
        
        if st.button("🔄 Réinitialiser", use_container_width=True):
            st.session_state.compteur_photos += 1
            st.rerun()
        
        st.divider()
        
        # Statistiques
        st.metric("📊 Photos traitées", len(st.session_state.photos_traitees))
        total = sum(p["nombre_numeros"] for p in st.session_state.photos_traitees)
        st.metric("🔢 Total numéros", total)
        
        st.divider()
        
        # Export et reset
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
    
    # ===== TRAITEMENT DES NOUVEAUX FICHIERS =====
    if nouveaux_fichiers:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, fichier in enumerate(nouveaux_fichiers):
            # Vérifier si déjà traité
            if not any(p["nom_fichier"] == fichier.name for p in st.session_state.photos_traitees):
                status_text.text(f"🔍 Traitement de {fichier.name}...")
                
                image = Image.open(fichier)
                resultat = traiter_photo(image, fichier.name, lecteur, pretraitement)
                st.session_state.photos_traitees.append(resultat)
            
            progress_bar.progress((i + 1) / len(nouveaux_fichiers))
        
        status_text.text("✅ Traitement terminé !")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        st.rerun()
    
    # ===== AFFICHAGE DES RÉSULTATS =====
    if st.session_state.photos_traitees:
        st.subheader("📋 Résultats")
        
        # Tableau
        df = pd.DataFrame([{
            "Fichier": p["nom_fichier"],
            "Numéros": p["numeros"],
            "Qté": p["nombre_numeros"]
        } for p in st.session_state.photos_traitees])
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Détail par photo
        st.markdown("### 🖼️ Détail par photo")
        for i, photo in enumerate(st.session_state.photos_traitees):
            with st.expander(f"📷 {photo['nom_fichier']} - {photo['nombre_numeros']} numéro(s)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(photo["image"], caption="Original", use_container_width=True)
                with col2:
                    if photo["image_traitee"]:
                        st.image(photo["image_traitee"], caption="Prétraitée", use_container_width=True)
                
                st.markdown(f"**Numéros :** `{photo['numeros']}`")
                
                if st.button("🗑️ Supprimer", key=f"del_{i}"):
                    st.session_state.photos_traitees.pop(i)
                    st.rerun()
    else:
        st.info("👈 Ajoutez des photos via le panneau de gauche pour commencer")

# ==============================================
# LANCEMENT
# ==============================================
if __name__ == "__main__":
    main()
