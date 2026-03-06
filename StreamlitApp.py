import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="GMAO Avancée",
    page_icon="🔧",
    layout="wide"
)

# --- Initialisation des données (Base de données simulée) ---
@st.cache_data
def load_data():
    # Exemple de données pour les équipements
    data_equip = {
        'ID': ['EQ-001', 'EQ-002', 'EQ-003', 'EQ-004'],
        'Nom': ['Compresseur A', 'Pompe Hydraulique', 'Tapis Roulant 1', 'Robot de peinture'],
        'Localisation': ['Atelier A', 'Atelier B', 'Ligne 2', 'Atelier C'],
        'Statut': ['Opérationnel', 'En maintenance', 'Opérationnel', 'Arrêté'],
        'Date Dernière Maintenance': ['2023-10-15', '2023-10-20', '2023-09-10', '2023-10-25']
    }
    df_equip = pd.DataFrame(data_equip)

    # Exemple de données pour les demandes d'intervention (Tickets)
    data_tickets = {
        'ID': ['TKT-101', 'TKT-102', 'TKT-103', 'TKT-104'],
        'Équipement': ['Compresseur A', 'Pompe Hydraulique', 'Tapis Roulant 1', 'Robot de peinture'],
        'Description': ['Bruit anormal', 'Fuite d\'huile', 'Usure des bandes', 'Erreur de calibration'],
        'Priorité': ['Haute', 'Moyenne', 'Basse', 'Haute'],
        'Statut': ['Ouvert', 'En cours', 'Fermé', 'Ouvert'],
        'Date Création': ['2023-10-26', '2023-10-27', '2023-10-20', '2023-10-28'],
        'Technicien Assigné': ['Jean Dupont', 'Marie Curie', 'Pierre Martin', None]
    }
    df_tickets = pd.DataFrame(data_tickets)

    # Exemple de données pour les techniciens
    data_techniciens = {
        'ID': ['TECH-01', 'TECH-02', 'TECH-03'],
        'Nom': ['Jean Dupont', 'Marie Curie', 'Pierre Martin'],
        'Spécialité': ['Mécanique', 'Hydraulique', 'Automatisme'],
        'Disponibilité': ['Occupé', 'Disponible', 'Disponible']
    }
    df_techniciens = pd.DataFrame(data_techniciens)

    return df_equip, df_tickets, df_techniciens

df_equip, df_tickets, df_techniciens = load_data()

# --- Fonctions utilitaires ---
def get_status_color(status):
    if status == 'Opérationnel' or status == 'Fermé' or status == 'Disponible':
        return 'normal'
    elif status == 'En maintenance' or status == 'En cours' or status == 'Occupé':
        return 'off'
    else:
        return 'gray'

# --- Barre latérale ---
st.sidebar.title("Navigation GMAO")
page = st.sidebar.radio("Aller à", ["Tableau de Bord", "Équipements", "Demandes d'Intervention", "Techniciens", "Rapports"])

# --- Page Tableau de Bord ---
if page == "Tableau de Bord":
    st.title("🔧 Tableau de Bord GMAO")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Équipements", len(df_equip))
    with col2:
        st.metric("Tickets Ouverts", len(df_tickets[df_tickets['Statut'] == 'Ouvert']))
    with col3:
        st.metric("Tickets en Cours", len(df_tickets[df_tickets['Statut'] == 'En cours']))
    with col4:
        st.metric("Techniciens Disponibles", len(df_techniciens[df_techniciens['Disponibilité'] == 'Disponible']))

    st.markdown("---")

    # Graphiques et tableaux récapitulatifs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tickets par Priorité")
        priority_counts = df_tickets['Priorité'].value_counts()
        st.bar_chart(priority_counts)

    with col2:
        st.subheader("Derniers Tickets Créés")
        recent_tickets = df_tickets.sort_values(by='Date Création', ascending=False).head(5)
        st.dataframe(recent_tickets, use_container_width=True)

# --- Page Équipements ---
elif page == "Équipements":
    st.title("📦 Gestion des Équipements")
    
    # Filtre
    status_filter = st.selectbox("Filtrer par statut", ["Tous", "Opérationnel", "En maintenance", "Arrêté"])
    if status_filter != "Tous":
        filtered_df = df_equip[df_equip['Statut'] == status_filter]
    else:
        filtered_df = df_equip

    # Affichage des données
    st.dataframe(filtered_df, use_container_width=True)

    # Formulaire d'ajout d'équipement
    with st.expander("Ajouter un nouvel équipement"):
        with st.form("add_equip_form"):
            new_id = st.text_input("ID de l'équipement")
            new_name = st.text_input("Nom de l'équipement")
            new_loc = st.text_input("Localisation")
            new_status = st.selectbox("Statut", ["Opérationnel", "En maintenance", "Arrêté"])
            submitted = st.form_submit_button("Ajouter")
            if submitted:
                new_data = pd.DataFrame({
                    'ID': [new_id], 'Nom': [new_name], 'Localisation': [new_loc],
                    'Statut': [new_status], 'Date Dernière Maintenance': [datetime.now().strftime("%Y-%m-%d")]
                })
                df_equip = pd.concat([df_equip, new_data], ignore_index=True)
                st.success("Équipement ajouté avec succès !")
                st.rerun()

# --- Page Demandes d'Intervention ---
elif page == "Demandes d'Intervention":
    st.title("🎫 Demandes d'Intervention")
    
    # Filtre
    status_filter = st.selectbox("Filtrer par statut", ["Tous", "Ouvert", "En cours", "Fermé"])
    if status_filter != "Tous":
        filtered_df = df_tickets[df_tickets['Statut'] == status_filter]
    else:
        filtered_df = df_tickets

    # Affichage des données
    st.dataframe(filtered_df, use_container_width=True)

    # Formulaire de création de ticket
    with st.expander("Créer une nouvelle demande d'intervention"):
        with st.form("add_ticket_form"):
            ticket_id = st.text_input("ID du Ticket")
            equip = st.selectbox("Équipement concerné", df_equip['Nom'].unique())
            desc = st.text_area("Description du problème")
            priority = st.selectbox("Priorité", ["Haute", "Moyenne", "Basse"])
            tech = st.selectbox("Technicien assigné", df_techniciens['Nom'].unique())
            submitted = st.form_submit_button("Créer le Ticket")
            if submitted:
                new_ticket = pd.DataFrame({
                    'ID': [ticket_id], 'Équipement': [equip], 'Description': [desc],
                    'Priorité': [priority], 'Statut': ['Ouvert'],
                    'Date Création': [datetime.now().strftime("%Y-%m-%d")],
                    'Technicien Assigné': [tech]
                })
                df_tickets = pd.concat([df_tickets, new_ticket], ignore_index=True)
                st.success("Ticket créé avec succès !")
                st.rerun()

# --- Page Techniciens ---
elif page == "Techniciens":
    st.title("👷 Gestion des Techniciens")
    
    st.dataframe(df_techniciens, use_container_width=True)

    # Formulaire d'ajout de technicien
    with st.expander("Ajouter un technicien"):
        with st.form("add_tech_form"):
            tech_id = st.text_input("ID du Technicien")
            tech_name = st.text_input("Nom du Technicien")
            specialty = st.text_input("Spécialité")
            availability = st.selectbox("Disponibilité", ["Disponible", "Occupé"])
            submitted = st.form_submit_button("Ajouter")
            if submitted:
                new_tech = pd.DataFrame({
                    'ID': [tech_id], 'Nom': [tech_name], 'Spécialité': [specialty], 'Disponibilité': [availability]
                })
                df_techniciens = pd.concat([df_techniciens, new_tech], ignore_index=True)
                st.success("Technicien ajouté avec succès !")
                st.rerun()

# --- Page Rapports ---
elif page == "Rapports":
    st.title("📊 Rapports et Analyses")
    
    # Sélection de l'équipement pour l'historique
    selected_equip = st.selectbox("Sélectionner un équipement pour l'historique", df_equip['Nom'].unique())
    
    # Filtre des tickets pour l'équipement sélectionné
    equip_tickets = df_tickets[df_tickets['Équipement'] == selected_equip]
    
    st.subheader(f"Historique des interventions pour : {selected_equip}")
    if not equip_tickets.empty:
        st.dataframe(equip_tickets, use_container_width=True)
    else:
        st.info("Aucune intervention enregistrée pour cet équipement.")

    # Statistiques globales
    st.markdown("---")
    st.subheader("Statistiques Globales")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Répartition des statuts des équipements")
        status_counts = df_equip['Statut'].value_counts()
        st.bar_chart(status_counts)
    
    with col2:
        st.write("Répartition des tickets par technicien")
        tech_counts = df_tickets['Technicien Assigné'].value_counts()
        st.bar_chart(tech_counts)

