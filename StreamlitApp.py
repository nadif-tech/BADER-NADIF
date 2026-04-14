import streamlit as st
import pandas as pd
import re
import io
from PIL import Image
import pytesseract

st.set_page_config(page_title="Extracteur Numéros", page_icon="📸")
st.title("📸 Extraire les numéros des photos")

# Upload des photos
fichiers = st.file_uploader("Choisissez des photos", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if fichiers:
    resultats = []
    
    for fichier in fichiers:
        # Afficher la photo
        image = Image.open(fichier)
        st.image(image, caption=fichier.name, width=300)
        
        # OCR
        with st.spinner(f"Analyse de {fichier.name}..."):
            texte = pytesseract.image_to_string(image)
            
            # Extraire les numéros
            numeros = re.findall(r'\d+', texte)
            
            if numeros:
                st.success(f"✅ Numéros trouvés : {', '.join(numeros)}")
            else:
                st.warning(f"⚠️ Aucun numéro trouvé")
            
            resultats.append({
                "Fichier": fichier.name,
                "Numéros": ", ".join(numeros) if numeros else "Aucun",
                "Quantité": len(numeros)
            })
    
    # Tableau des résultats
    if resultats:
        st.subheader("📊 Résultats")
        df = pd.DataFrame(resultats)
        st.dataframe(df, use_container_width=True)
        
        # Export Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Numéros')
        
        st.download_button(
            label="📥 Télécharger Excel",
            data=output.getvalue(),
            file_name="numeros_extraits.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
