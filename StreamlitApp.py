import streamlit as st
import pandas as pd
import re
import io
import time
from datetime import datetime
from PIL import Image
import pytesseract

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
# EXTRACTION DES NUMÉROS
# ==============================================
def extraire_numeros(texte):
    """Extrait tous les nombres du texte"""
    if not texte:
        return []
    
    patterns = [
        r'\d+[.,]\d+',           # Nombres décimaux (12.34 ou 12,34)
        r'\d+',                  # Nombres entiers
    ]
    
    numeros = []
    for pattern in patterns:
        numeros.extend(re.findall(pattern, texte))
    
    # Nettoyage et déduplication
    numeros_propres = []
    for n in numeros:
        n = n.strip()
        if n and n not in numeros_propres:
            numeros_propres.append(n)
    
    return numeros_propres

# ==============================================
# TRAITEMENT D'UNE PHOTO
# ==============================================
def traiter_photo(image, nom_fichier):
    """Traite une photo avec Tesseract OCR"""
    debut = time.time()
    
    try:
        # Configuration Tesseract pour meilleure reconnaissance des chiffres
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,'
        
        # OCR
        texte_complet = pytesseract.image_to_string(image, lang='fra', config=custom_config)
        
        # Extraction des numéros
        numeros = extraire_numeros(texte_complet)
        
    except Exception as e:
        texte_complet = f"Erreur: {str(e)}"
        numeros = []
    
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
# INTERFACE
# ==============================================
def main():
    st.title("📸 Extracteur de Numéros")
    st.caption("Téléchargez des photos pour extraire les chiffres")
    
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
            # Éviter doublons
            if not any(p["nom_fichier"] == f.name for p in st.session_state.photos_traitees):
                status.text(f"🔍 {f.name}...")
                img = Image.open(f)
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
        
        # Tableau
        df = pd.DataFrame([{
            "Fichier": p["nom_fichier"],
            "Numéros": p["numeros"],
            "Qté": p["nombre_numeros"]
        } for p in st.session_state.photos_traitees])
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Détail
        st.divider()
        st.subheader("🖼️ Détail par photo")
        
        for i, photo in enumerate(st.session_state.photos_traitees):
            with st.expander(f"📷 {photo['nom_fichier']} - {photo['nombre_numeros']} numéro(s)"):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(photo["image"], use_container_width=True)
                with col2:
                    st.markdown(f"**Numéros extraits :**")
                    st.code(photo["numeros"] if photo["numeros"] != "Aucun" else "Aucun numéro trouvé")
                    st.caption(f"Temps: {photo['temps']}s | {photo['date']}")
                
                if st.button("🗑️ Supprimer", key=f"del_{i}"):
                    st.session_state.photos_traitees.pop(i)
                    st.rerun()
    else:
        st.info("👈 Ajoutez des photos dans le menu de gauche pour commencer")

if __name__ == "__main__":
    main()
