import streamlit as st
import easyocr
import pandas as pd
import re
import cv2
import numpy as np
from PIL import Image
import io
import time
from datetime import datetime

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
    """Initialise toutes les variables de session Streamlit"""
    if 'photos_traitees' not in st.session_state:
        st.session_state.photos_traitees = []  # Liste des résultats
    if 'historique' not in st.session_state:
        st.session_state.historique = []  # Historique des actions
    if 'compteur_photos' not in st.session_state:
        st.session_state.compteur_photos = 0

init_session_state()

# ==============================================
# CHARGEMENT DU MODÈLE OCR (avec cache)
# ==============================================
@st.cache_resource
def charger_modele_ocr(langues=['fr', 'en']):
    """Charge le modèle EasyOCR avec cache pour performance"""
    with st.spinner("🔄 Chargement du modèle OCR en cours..."):
        return easyocr.Reader(langues, gpu=False)

# ==============================================
# PRÉTRAITEMENT D'IMAGE (améliore la précision)
# ==============================================
def pretraiter_image(image_pil):
    """
    Applique des transformations pour améliorer la reconnaissance
    - Conversion en niveaux de gris
    - Réduction du bruit
    - Amélioration du contraste
    """
    # Convertir PIL en OpenCV
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # Niveaux de gris
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Réduction du bruit
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Amélioration du contraste (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Seuillage adaptatif pour les textes sombres sur fond clair
    binary = cv2.adaptiveThreshold(
        enhanced, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return Image.fromarray(binary)

# ==============================================
# EXTRACTION DES NUMÉROS
# ==============================================
def extraire_numeros(texte, pattern_personnalise=None):
    """
    Extrait les nombres du texte avec différents patterns
    """
    if pattern_personnalise:
        numeros = re.findall(pattern_personnalise, texte)
    else:
        # Pattern par défaut : nombres entiers, décimaux, avec séparateurs
        patterns = [
            r'\d+[\.,]?\d*',           # Nombres simples
            r'\d+[\s]?\d*[\.,]?\d*',   # Nombres avec espaces
            r'[\d]+[.,]?\d*\s?[€$£]',  # Montants avec devise
        ]
        numeros = []
        for p in patterns:
            numeros.extend(re.findall(p, texte))
    
    # Nettoyer et dédupliquer
    numeros = list(set([n.strip() for n in numeros if n.strip()]))
    return numeros

# ==============================================
# TRAITEMENT D'UNE PHOTO
# ==============================================
def traiter_photo(image, nom_fichier, lecteur, pretraitement=True):
    """
    Traite une photo : prétraitement optionnel + OCR + extraction
    """
    debut = time.time()
    
    # Prétraitement optionnel
    if pretraitement:
        image_traitee = pretraiter_image(image)
    else:
        image_traitee = image
    
    # Conversion en bytes pour OCR
    img_bytes = io.BytesIO()
    image_traitee.save(img_bytes, format='PNG')
    
    # Reconnaissance OCR
    resultats_ocr = lecteur.readtext(img_bytes.getvalue(), detail=0, paragraph=True)
    texte_complet = " ".join(resultats_ocr)
    
    # Extraction des numéros
    numeros = extraire_numeros(texte_complet)
    
    temps_traitement = round(time.time() - debut, 2)
    
    return {
        "nom_fichier": nom_fichier,
        "numeros": ", ".join(numeros) if numeros else "Aucun numéro détecté",
        "texte_brut": texte_complet,
        "nombre_numeros": len(numeros),
        "temps_traitement": temps_traitement,
        "horodatage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image": image,
        "image_traitee": image_traitee if pretraitement else None
    }

# ==============================================
# EXPORT EXCEL
# ==============================================
def exporter_excel(donnees):
    """Crée un fichier Excel à partir des résultats"""
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
        df.to_excel(writer, index=False, sheet_name='Résultats_OCR')
        
        # Ajuster la largeur des colonnes
        worksheet = writer.sheets['Résultats_OCR']
        for column in worksheet.columns:
            max_length = max(len(str(cell.value)) for cell in column)
            worksheet.column_dimensions[column[0].column_letter].width = min(max_length + 2, 50)
    
    return output.getvalue()

# ==============================================
# INTERFACE PRINCIPALE
# ==============================================
def main():
    # Chargement du modèle
    lecteur = charger_modele_ocr()
    
    # ===== BARRE LATÉRALE =====
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Options OCR
        st.subheader("Paramètres OCR")
        pretraitement = st.checkbox("Activer le prétraitement d'image", value=True, 
                                   help="Améliore la précision sur les images de mauvaise qualité")
        
        langue = st.selectbox("Langue principale", 
                              ["Français + Anglais", "Anglais uniquement", "Français uniquement"],
                              help="Langue du texte environnant")
        
        # Options d'affichage
        st.subheader("Affichage")
        afficher_images = st.checkbox("Afficher les images traitées", value=True)
        afficher_texte_brut = st.checkbox("Afficher le texte brut OCR", value=False)
        
        st.divider()
        
        # Zone d'ajout de photos
        st.subheader("📤 Ajouter des photos")
        nouveaux_fichiers = st.file_uploader(
            "Sélectionnez des images",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.compteur_photos}",
            help="Vous pouvez ajouter plusieurs photos à la fois"
        )
        
        # Bouton pour vider l'uploader (permet de réajouter les mêmes fichiers)
        if st.button("🔄 Réinitialiser l'upload", use_container_width=True):
            st.session_state.compteur_photos += 1
            st.rerun()
        
        st.divider()
        
        # Actions globales
        st.subheader("📊 Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Tout effacer", use_container_width=True):
                st.session_state.photos_traitees = []
                st.session_state.historique = []
                st.success("✅ Toutes les données ont été effacées")
                st.rerun()
        
        with col2:
            if st.session_state.photos_traitees:
                excel_data = exporter_excel(st.session_state.photos_traitees)
                st.download_button(
                    label="📥 Exporter Excel",
                    data=excel_data,
                    file_name=f"extraction_numeros_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        # Statistiques
        st.divider()
        st.subheader("📈 Statistiques")
        st.metric("Photos traitées", len(st.session_state.photos_traitees))
        total_numeros = sum(p["nombre_numeros"] for p in st.session_state.photos_traitees)
        st.metric("Total numéros extraits", total_numeros)
    
    # ===== ZONE PRINCIPALE =====
    st.title("📸 Extracteur de Numéros - Version Pro")
    st.markdown("---")
    
    # Traitement des nouveaux fichiers
    if nouveaux_fichiers:
        with st.spinner("🔍 Traitement des photos en cours..."):
            for fichier in nouveaux_fichiers:
                # Vérifier si déjà traité
                if not any(p["nom_fichier"] == fichier.name for p in st.session_state.photos_traitees):
                    image = Image.open(fichier)
                    resultat = traiter_photo(image, fichier.name, lecteur, pretraitement)
                    st.session_state.photos_traitees.append(resultat)
                    st.session_state.historique.append(f"✅ {fichier.name} traité - {resultat['nombre_numeros']} numéros trouvés")
        
        st.success(f"✅ {len(nouveaux_fichiers)} nouvelle(s) photo(s) traitée(s)")
        st.rerun()
    
    # ===== AFFICHAGE DES RÉSULTATS =====
    if st.session_state.photos_traitees:
        # Onglets pour différentes vues
        tab1, tab2, tab3 = st.tabs(["📋 Tableau des résultats", "🖼️ Galerie", "📜 Historique"])
        
        with tab1:
            # Tableau récapitulatif
            df_affichage = pd.DataFrame([{
                "Fichier": p["nom_fichier"],
                "Numéros extraits": p["numeros"],
                "Qté": p["nombre_numeros"],
                "Temps": f"{p['temps_traitement']}s"
            } for p in st.session_state.photos_traitees])
            
            st.dataframe(df_affichage, use_container_width=True, hide_index=True)
            
            # Détail par photo
            st.markdown("### 📝 Détail par photo")
            for i, photo in enumerate(st.session_state.photos_traitees):
                with st.expander(f"📷 {photo['nom_fichier']} - {photo['nombre_numeros']} numéro(s)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(photo["image"], caption="Original", use_container_width=True)
                    with col2:
                        if photo["image_traitee"] and afficher_images:
                            st.image(photo["image_traitee"], caption="Prétraitée", use_container_width=True)
                    
                    st.markdown(f"**Numéros extraits :** `{photo['numeros']}`")
                    if afficher_texte_brut:
                        st.markdown(f"**Texte brut :** _{photo['texte_brut']}_")
                    
                    # Bouton suppression individuelle
                    if st.button(f"🗑️ Supprimer", key=f"del_{i}"):
                        st.session_state.photos_traitees.pop(i)
                        st.rerun()
        
        with tab2:
            # Galerie d'images
            cols = st.columns(3)
            for i, photo in enumerate(st.session_state.photos_traitees):
                with cols[i % 3]:
                    st.image(photo["image"], caption=photo["nom_fichier"], use_container_width=True)
                    st.caption(f"📊 {photo['nombre_numeros']} numéro(s)")
        
        with tab3:
            # Historique
            for log in st.session_state.historique:
                st.text(log)
    
    else:
        # Message d'accueil quand aucune photo
        st.info("👈 Utilisez le panneau de gauche pour ajouter des photos à analyser")
        
        # Démonstration
        with st.expander("📖 Comment ça marche ?"):
            st.markdown("""
            1. **Ajoutez des photos** via l'uploader dans la barre latérale
            2. **L'OCR analyse** automatiquement chaque image
            3. **Les numéros sont extraits** et affichés dans le tableau
            4. **Exportez en Excel** avec un seul clic
            
            **Astuces :**
            - Activez le prétraitement pour les photos de mauvaise qualité
            - Vous pouvez ajouter plusieurs photos en une fois
            - L'historique garde une trace de toutes les actions
            """)

# ==============================================
# LANCEMENT DE L'APPLICATION
# ==============================================
if __name__ == "__main__":
    main()
