"""
Page de gestion des équipements
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px

def show(managers):
    """Affiche la page de gestion des équipements"""
    
    st.title("🔧 Gestion des Équipements")
    
    # Onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Liste des équipements", 
        "➕ Nouvel équipement",
        "📊 Statistiques",
        "📈 Historique"
    ])
    
    with tab1:
        show_assets_list(managers)
    
    with tab2:
        show_add_asset(managers)
    
    with tab3:
        show_assets_stats(managers)
    
    with tab4:
        show_assets_history(managers)

def show_assets_list(managers):
    """Affiche la liste des équipements"""
    
    st.subheader("Liste des équipements")
    
    # Filtres
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_filter = st.selectbox(
            "Statut",
            ["Tous", "Actif", "En maintenance", "Hors service", "En réserve"]
        )
    
    with col2:
        type_filter = st.selectbox(
            "Type",
            ["Tous"] + Config.ASSET_TYPES
        )
    
    with col3:
        dept_filter = st.text_input("Département", placeholder="Filtrer par département")
    
    with col4:
        search = st.text_input("Recherche", placeholder="Nom, code, modèle...")
    
    # Construction des filtres
    filters = {}
    if status_filter != "Tous":
        filters['status'] = status_filter
    if type_filter != "Tous":
        filters['type'] = type_filter
    if dept_filter:
        filters['departement'] = dept_filter
    if search:
        filters['search'] = search
    
    # Récupération des données
    assets = managers['assets'].get_all_assets(filters)
    
    if not assets.empty:
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total", len(assets))
        with col2:
            actifs = len(assets[assets['status'] == 'Actif'])
            st.metric("Actifs", actifs)
        with col3:
            maintenance = len(assets[assets['status'] == 'En maintenance'])
            st.metric("En maintenance", maintenance)
        with col4:
            valeur_totale = assets['valeur_achat'].sum()
            st.metric("Valeur totale", f"{valeur_totale:,.0f} €")
        
        # Affichage du tableau
        display_cols = ['code', 'nom', 'type', 'modele', 'localisation', 
                       'status', 'prochain_entretien', 'responsable_nom']
        
        display_df = assets[display_cols].copy()
        display_df.columns = ['Code', 'Nom', 'Type', 'Modèle', 'Localisation',
                            'Statut', 'Prochain entretien', 'Responsable']
        
        # Coloration conditionnelle
        def color_status(val):
            colors = {
                'Actif': 'background-color: #90EE90',
                'En maintenance': 'background-color: #FFB6C1',
                'Hors service': 'background-color: #FFA07A',
                'En réserve': 'background-color: #87CEEB'
            }
            return colors.get(val, '')
        
        styled_df = display_df.style.applymap(color_status, subset=['Statut'])
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400,
            column_config={
                "Prochain entretien": st.column_config.DateColumn(format="DD/MM/YYYY")
            }
        )
        
        # Actions sur les équipements
        st.subheader("Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_code = st.selectbox(
                "Sélectionner un équipement",
                options=assets['code'].tolist(),
                format_func=lambda x: f"{x} - {assets[assets['code']==x]['nom'].iloc[0]}"
            )
        
        if selected_code:
            selected_asset = assets[assets['code'] == selected_code].iloc[0]
            
            with col2:
                if st.button("✏️ Modifier", use_container_width=True):
                    st.session_state['edit_asset'] = selected_asset.to_dict()
                    st.rerun()
            
            with col3:
                if st.button("🗑️ Supprimer", use_container_width=True):
                    success, message = managers['assets'].delete_asset(selected_asset['id'])
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
    
    else:
        st.info("Aucun équipement trouvé")
    
    # Formulaire de modification (si actif)
    if 'edit_asset' in st.session_state:
        with st.expander("Modifier l'équipement", expanded=True):
            edit_asset_form(managers, st.session_state['edit_asset'])
            if st.button("Annuler"):
                del st.session_state['edit_asset']
                st.rerun()

def show_add_asset(managers):
    """Affiche le formulaire d'ajout d'équipement"""
    
    st.subheader("Ajouter un nouvel équipement")
    
    with st.form("add_asset_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            nom = st.text_input("Nom *", placeholder="Nom de l'équipement")
            type_asset = st.selectbox("Type *", Config.ASSET_TYPES)
            modele = st.text_input("Modèle")
            fabricant = st.text_input("Fabricant")
            numero_serie = st.text_input("Numéro de série")
            date_acquisition = st.date_input("Date d'acquisition")
            date_mise_service = st.date_input("Date de mise en service")
            garantie = st.number_input("Garantie (jours)", min_value=0, step=30)
        
        with col2:
            localisation = st.text_input("Localisation")
            departement = st.text_input("Département")
            
            # Liste des responsables
            users = managers['auth'].get_all_users()
            responsables = {row['id']: f"{row['prenom']} {row['nom']}" 
                          for _, row in users.iterrows()}
            
            responsable_id = st.selectbox(
                "Responsable",
                options=list(responsables.keys()),
                format_func=lambda x: responsables.get(x, "")
            )
            
            valeur_achat = st.number_input("Valeur d'achat (€)", min_value=0.0, step=100.0)
            duree_vie = st.number_input("Durée de vie (années)", min_value=1, step=1)
            periodicite = st.number_input(
                "Périodicité entretien (jours)", 
                min_value=1, 
                step=30,
                help="Nombre de jours entre chaque entretien préventif"
            )
            
            notes = st.text_area("Notes", height=100)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col2:
            submitted = st.form_submit_button("✅ Ajouter l'équipement", use_container_width=True)
        
        if submitted:
            if not nom or not type_asset:
                st.error("Veuillez remplir tous les champs obligatoires (*)")
            else:
                # Préparation des données
                asset_data = {
                    'nom': nom,
                    'type': type_asset,
                    'modele': modele,
                    'fabricant': fabricant,
                    'numero_serie': numero_serie,
                    'date_acquisition': date_acquisition.strftime('%Y-%m-%d') if date_acquisition else None,
                    'date_mise_service': date_mise_service.strftime('%Y-%m-%d') if date_mise_service else None,
                    'garantie_jours': garantie,
                    'localisation': localisation,
                    'departement': departement,
                    'responsable_id': responsable_id if responsable_id else None,
                    'valeur_achat': valeur_achat,
                    'duree_vie_ans': duree_vie,
                    'periodicite_entretien_jours': periodicite,
                    'notes': notes,
                    'status': 'Actif'
                }
                
                success, result = managers['assets'].create_asset(asset_data)
                
                if success:
                    st.success("Équipement ajouté avec succès!")
                    st.balloons()
                else:
                    st.error(f"Erreur: {result}")

def edit_asset_form(managers, asset):
    """Formulaire de modification d'équipement"""
    
    with st.form("edit_asset_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            nom = st.text_input("Nom *", value=asset.get('nom', ''))
            type_asset = st.selectbox(
                "Type *", 
                Config.ASSET_TYPES,
                index=Config.ASSET_TYPES.index(asset.get('type', 'Machine')) 
                    if asset.get('type') in Config.ASSET_TYPES else 0
            )
            modele = st.text_input("Modèle", value=asset.get('modele', ''))
            fabricant = st.text_input("Fabricant", value=asset.get('fabricant', ''))
            numero_serie = st.text_input("Numéro de série", value=asset.get('numero_serie', ''))
            
            # Conversion des dates
            date_acquisition = None
            if asset.get('date_acquisition'):
                try:
                    date_acquisition = datetime.strptime(asset['date_acquisition'], '%Y-%m-%d').date()
                except:
                    pass
            
            date_acquisition = st.date_input(
                "Date d'acquisition", 
                value=date_acquisition if date_acquisition else datetime.now().date()
            )
            
            date_mise_service = None
            if asset.get('date_mise_service'):
                try:
                    date_mise_service = datetime.strptime(asset['date_mise_service'], '%Y-%m-%d').date()
                except:
                    pass
            
            date_mise_service = st.date_input(
                "Date de mise en service",
                value=date_mise_service if date_mise_service else datetime.now().date()
            )
            
            garantie = st.number_input(
                "Garantie (jours)", 
                min_value=0, 
                step=30,
                value=int(asset.get('garantie_jours', 0)) if asset.get('garantie_jours') else 0
            )
        
        with col2:
            localisation = st.text_input("Localisation", value=asset.get('localisation', ''))
            departement = st.text_input("Département", value=asset.get('departement', ''))
            
            # Statut
            status = st.selectbox(
                "Statut",
                Config.ASSET_STATUS,
                index=Config.ASSET_STATUS.index(asset.get('status', 'Actif'))
                    if asset.get('status') in Config.ASSET_STATUS else 0
            )
            
            # Responsable
            users = managers['auth'].get_all_users()
            responsables = {row['id']: f"{row['prenom']} {row['nom']}" 
                          for _, row in users.iterrows()}
            
            responsable_id = st.selectbox(
                "Responsable",
                options=list(responsables.keys()),
                format_func=lambda x: responsables.get(x, ""),
                index=list(responsables.keys()).index(asset.get('responsable_id')) 
                    if asset.get('responsable_id') in responsables.keys() else 0
            )
            
            valeur_achat = st.number_input(
                "Valeur d'achat (€)", 
                min_value=0.0, 
                step=100.0,
                value=float(asset.get('valeur_achat', 0)) if asset.get('valeur_achat') else 0.0
            )
            
            duree_vie = st.number_input(
                "Durée de vie (années)", 
                min_value=1, 
                step=1,
                value=int(asset.get('duree_vie_ans', 5)) if asset.get('duree_vie_ans') else 5
            )
            
            periodicite = st.number_input(
                "Périodicité entretien (jours)", 
                min_value=1, 
                step=30,
                value=int(asset.get('periodicite_entretien_jours', 90)) 
                    if asset.get('periodicite_entretien_jours') else 90
            )
            
            notes = st.text_area("Notes", value=asset.get('notes', ''), height=100)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col2:
            submitted = st.form_submit_button("✅ Mettre à jour", use_container_width=True)
        
        if submitted:
            asset_data = {
                'nom': nom,
                'type': type_asset,
                'modele': modele,
                'fabricant': fabricant,
                'numero_serie': numero_serie,
                'date_acquisition': date_acquisition.strftime('%Y-%m-%d'),
                'date_mise_service': date_mise_service.strftime('%Y-%m-%d'),
                'garantie_jours': garantie,
                'localisation': localisation,
                'departement': departement,
                'status': status,
                'responsable_id': responsable_id,
                'valeur_achat': valeur_achat,
                'duree_vie_ans': duree_vie,
                'periodicite_entretien_jours': periodicite,
                'notes': notes
            }
            
            success, message = managers['assets'].update_asset(asset['id'], asset_data)
            
            if success:
                st.success("Équipement mis à jour avec succès!")
                del st.session_state['edit_asset']
                st.rerun()
            else:
                st.error(f"Erreur: {message}")

def show_assets_stats(managers):
    """Affiche les statistiques des équipements"""
    
    st.subheader("Statistiques des équipements")
    
    # Récupération des statistiques
    stats = managers['assets'].get_asset_stats()
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total équipements", stats.get('total', 0))
    with col2:
        st.metric("Maintenances à venir", stats.get('maintenance_due', 0))
    with col3:
        st.metric("Valeur totale", f"{stats.get('total_value', 0):,.0f} €")
    with col4:
        actifs = stats.get('by_status', {}).get('Actif', 0)
        st.metric("Taux d'activité", f"{(actifs/stats['total']*100):.1f}%" if stats['total'] > 0 else "0%")
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        # Répartition par statut
        data = managers['assets'].get_assets_by_status()
        if not data.empty:
            fig = px.pie(
                data, 
                values='count', 
                names='status',
                title="Répartition par statut",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Répartition par type
        data = managers['assets'].get_assets_by_type()
        if not data.empty:
            fig = px.bar(
                data, 
                x='type', 
                y='count',
                title="Répartition par type",
                color='type'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Répartition par département
    data = managers['assets'].get_assets_by_department()
    if not data.empty:
        fig = px.pie(
            data, 
            values='count', 
            names='departement',
            title="Répartition par département"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Maintenances à venir
    st.subheader("Maintenances préventives à venir")
    
    due_assets = managers['assets'].get_assets_due_for_maintenance(days=30)
    if not due_assets.empty:
        display_df = due_assets[['code', 'nom', 'prochain_entretien', 'jours_restants']].copy()
        display_df['jours_restants'] = display_df['jours_restants'].round(0).astype(int)
        display_df.columns = ['Code', 'Nom', 'Date prévue', 'Jours restants']
        
        # Coloration conditionnelle
        def color_days(val):
            if val <= 0:
                return 'background-color: #FF6B6B'
            elif val <= 7:
                return 'background-color: #FFD93D'
            else:
                return ''
        
        styled_df = display_df.style.applymap(color_days, subset=['Jours restants'])
        
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("Aucune maintenance prévue dans les 30 prochains jours")

def show_assets_history(managers):
    """Affiche l'historique des équipements"""
    
    st.subheader("Historique des modifications")
    
    # Sélection d'un équipement
    assets = managers['assets'].get_all_assets()
    
    if not assets.empty:
        selected_code = st.selectbox(
            "Sélectionner un équipement",
            options=assets['code'].tolist(),
            format_func=lambda x: f"{x} - {assets[assets['code']==x]['nom'].iloc[0]}",
            key="history_asset"
        )
        
        if selected_code:
            asset = assets[assets['code'] == selected_code].iloc[0]
            history = managers['assets'].get_asset_history(asset['id'])
            
            if not history.empty:
                for _, event in history.iterrows():
                    with st.expander(f"{event['date_action']} - {event['action']}"):
                        st.json(event['modifications'] if event['modifications'] else {})
            else:
                st.info("Aucun historique pour cet équipement")
    else:
        st.info("Aucun équipement disponible")
