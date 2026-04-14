import streamlit as st
import pandas as pd
import re
import io
import time
from datetime import datetime
from PIL import Image
import subprocess
import sys
import os

# ==============================================
# INSTALLATION AUTOMATIQUE DE TESSERACT
# ==============================================
@st.cache_resource
def installer_tesseract():
    """Installe Tesseract automatiquement dans l'environnement Streamlit Cloud"""
    try:
        # Vérifier si Tesseract est déjà installé
        subprocess.run(['tesseract', '--version'], capture_output=True, check=True)
        return True
    except:
        try:
            # Télécharger et installer Tesseract portable
            st.info("⏳ Installation de Tesseract OCR (première fois uniquement)...")
            
            # Créer un dossier local pour Tesseract
            os.makedirs('/tmp/tesseract', exist_ok=True)
            
            # Télécharger le binaire statique de Tesseract
            subprocess.run([
                'wget', '-q', 
                'https://github.com/tesseract-ocr/tesseract/releases/download/5.3.3/tesseract-5.3.3.tar.gz',
                '-O', '/tmp/tesseract.tar.gz'
            ], check=True)
            
            # Extraire
            subprocess.run(['tar', '-xzf', '/tmp/tesseract.tar.gz', '-C', '/tmp/tesseract'], check=True)
            
            # Ajouter au PATH
            os.environ['PATH'] += ':/tmp/tesseract'
            
            st.success("✅ Tesseract installé avec succès")
            return True
        except Exception as e:
            st.error(f"❌ Impossible d'installer Tesseract: {str(e)}")
            return False

# ==============================================
# CONFIGURATION DE LA PAGE
# ==============================================
st.set_page_config(
    page_title="Extracteur Numéros",
    page_icon="📸",
    layout="wide"
)

# ==============================================
# INITIALISATION
# ==============================================
def init_session():
    if 'photos_traitees' not in st.session_state:
        st.session_state.photos_traitees = []
    if 'compteur_photos' not in st.session_state:
        st.session_state.compteur_photos = 0

init_session()

# ==============================================
# OCR AVEC TESSERACT
# ==============================================
def ocr_image(image):
    """Effectue l'OCR sur une image avec pytesseract"""
    try:
        import pytesseract
        
        # Configuration pour les chiffres
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,'
        
        # Essayer d'abord avec la config système
        texte = pytesseract.image_to_string(image, lang='fra', config=custom_config)
        
        # Si vide, essayer sans langue spécifique
        if not texte.strip():
            texte = pytesseract.image_to_string(image, config=custom_config)
        
        return texte
    
    except Exception as e:
        # Fallback: utiliser Tesseract directement via subprocess
        try:
            import tempfile
            
            # Sauvegarder l'image temporairement
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name
            
            # Appeler Tesseract directement
            result = subprocess.run(
                ['tesseract', tmp_path, 'stdout', '--psm', '6', '-c', 'tessedit_char_whitelist=0123456789.,'],
                capture_output=True, text=True
            )
            
            # Nettoyer
            os.unlink(tmp_path)
            
            return result.stdout
        
        except Exception as e2:
            st.error(f"Erreur OCR: {str(e2)}")
            return ""

# ==============================================
# EXTRACTION DES NUMÉROS
# ==============================================
def extraire_numeros(texte):
    """Extrait tous les nombres du texte"""
    if not texte:
        return []
    
    # Patterns pour différents formats de nombres
    patterns = [
        r'\d+[.,]\d+',  # 12.34 ou 12,34
        r'\d+',         # 123
    ]
    
    numeros = []
    for pattern in patterns:
        numeros.extend(re.findall(pattern, texte))
    
    # Nettoyage et déduplication
    numeros_propres = []
    for n in numeros:
        n = n.strip()
        if n and n not in numeros_propres and len(n) > 0:
            numeros_propres.append(n)
    
    return numeros_propres

# ==============================================
# TRAITEMENT D'UNE PHOTO
# ==============================================
def traiter_photo(image, nom_fichier):
    """Traite une photo"""
    debut = time.time()
    
    # OCR
    texte_complet = ocr_image(image)
    
    # Extraction des numéros
    numeros = extraire_numeros(texte_complet)
    
    temps_traitement = round(time.time() - debut, 2)
    
    return {
        "nom_fichier": nom_fichier,
        "numeros": ", ".join(numeros) if numeros else "Aucun",
        "texte_brut": texte_complet[:300] + "..." if len(texte_complet) > 300 else texte_complet,
        "nombre_numeros": len(numeros),
        "temps": temps_traitement,
        "date": datetime.now().strftime("%H:%M:%S"),
        "image": image
    }

# ==============================================
# EXPORT EXCEL
# ==============================================
def exporter_excel(donnees):
    """Crée un fichier Excel"""
    df = pd.DataFrame([{
        "Fichier": d["nom_fichier"],
        "Numéros": d["numeros"],
        "Quantité": d["nombre_numeros"],
        "Texte brut": d["texte_brut"],
        "Temps (s)": d["temps"],
        "Heure": d["date"]
    } for d in donnees])
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultats')
    
    return output.getvalue()

# ==============================================
# INTERFACE PRINCIPALE
# ==============================================
def main():
    st.title("📸 Extracteur de Numéros")
    st.caption("Extraction des chiffres depuis vos photos")
    
    # Sidebar
    with st.sidebar:
        st.header("📤 Ajouter des photos")
        
        fichiers = st.file_uploader(
            "Choisir des images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key=f"upload_{st.session_state.compteur_photos}"
        )
        
        if st.button("🔄 Vider la sélection", use_container_width=True):
            st.session_state.compteur_photos += 1
            st.rerun()
        
        st.divider()
        
        # Stats
        n_photos = len(st.session_state.photos_traitees)
        n_numeros = sum(p["nombre_numeros"] for p in st.session_state.photos_traitees)
        
        col1, col2 = st.columns(2)
        col1.metric("📊 Photos", n_photos)
        col2.metric("🔢 Numéros", n_numeros)
        
        st.divider()
        
        # Actions
        if st.session_state.photos_traitees:
            excel_data = exporter_excel(st.session_state.photos_traitees)
            st.download_button(
                "📥 Télécharger Excel",
                data=excel_data,
                file_name=f"numeros_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            if st.button("🗑️ Tout effacer", use_container_width=True):
                st.session_state.photos_traitees = []
                st.rerun()
    
    # Traitement
    if fichiers:
        progress = st.progress(0)
        status = st.empty()
        
        for i, f in enumerate(fichiers):
            if not any(p["nom_fichier"] == f.name for p in st.session_state.photos_traitees):
                status.text(f"🔍 {f.name}...")
                img = Image.open(f)
                
                # Redimensionner si trop grande
                if img.width > 1500:
                    ratio = 1500 / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((1500, new_height), Image.Resampling.LANCZOS)
                
                resultat = traiter_photo(img, f.name)
                st.session_state.photos_traitees.append(resultat)
            
            progress.progress((i + 1) / len(fichiers))
        
        status.text("✅ Terminé !")
        time.sleep(1)
        status.empty()
        progress.empty()
        st.rerun()
    
    # Affichage résultats
    if st.session_state.photos_traitees:
        st.subheader("📋 Résultats")
        
        df = pd.DataFrame([{
            "Fichier": p["nom_fichier"],
            "Numéros": p["numeros"],
            "Qté": p["nombre_numeros"]
        } for p in st.session_state.photos_traitees])
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.divider()
        st.subheader("🖼️ Détail par photo")
        
        for i, photo in enumerate(st.session_state.photos_traitees):
            with st.expander(f"📷 {photo['nom_fichier']} - {photo['nombre_numeros']} numéro(s)"):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(photo["image"], use_container_width=True)
                with col2:
                    st.markdown("**Numéros extraits :**")
                    st.code(photo["numeros"] if photo["numeros"] != "Aucun" else "Aucun numéro trouvé")
                    st.caption(f"⏱️ {photo['temps']}s | 🕐 {photo['date']}")
                
                if st.button("🗑️ Supprimer", key=f"del_{i}"):
                    st.session_state.photos_traitees.pop(i)
                    st.rerun()
    else:
        st.info("👈 Ajoutez des photos dans le menu de gauche pour commencer")

if __name__ == "__main__":
    main()
