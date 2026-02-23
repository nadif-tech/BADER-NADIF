import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib
import json
from PIL import Image
import io

# Configuration de la page
st.set_page_config(
    page_title="GMAO - Solution de Maintenance",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .stat-label {
        color: #666;
        font-size: 0.9rem;
    }
    .alert-critical {
        background: #ff4444;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-warning {
        background: #ffbb33;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-normal {
        background: #00C851;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation de la session d'état
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'page' not in st.session_state:
    st.session_state.page = "Tableau de bord"

# Données de démonstration
@st.cache_data
def load_sample_data():
    # Équipements
    equipments = pd.DataFrame({
        'id': range(1, 11),
        'nom': ['Compresseur A1', 'Pompe B2', 'Moteur C3', 'Convoyeur D4', 'Tour E5',
                'Fraiseuse F6', 'Presse G7', 'Ventilateur H8', 'Générateur I9', 'Pompe J10'],
        'type': ['Compresseur', 'Pompe', 'Moteur', 'Convoyeur', 'Tour',
                'Fraiseuse', 'Presse', 'Ventilateur', 'Générateur', 'Pompe'],
        'localisation': ['Atelier A', 'Atelier B', 'Atelier A', 'Ligne 1', 'Atelier C',
                        'Atelier B', 'Ligne 2', 'Atelier A', 'Salle machine', 'Ligne 1'],
        'statut': ['En fonctionnement', 'En maintenance', 'Arrêt', 'En fonctionnement', 'En fonctionnement',
                  'En maintenance', 'En fonctionnement', 'Arrêt', 'En fonctionnement', 'En fonctionnement'],
        'derniere_maintenance': pd.date_range(start='2024-01-01', periods=10, freq='W'),
        'prochaine_maintenance': pd.date_range(start='2024-03-01', periods=10, freq='M'),
        'priorite': ['Haute', 'Moyenne', 'Basse', 'Haute', 'Moyenne', 'Basse', 'Haute', 'Moyenne', 'Haute', 'Basse']
    })
    
    # Interventions
    interventions = pd.DataFrame({
        'id': range(1, 21),
        'equipement_id': np.random.randint(1, 11, 20),
        'type': np.random.choice(['Préventive', 'Curative', 'Inspection'], 20),
        'description': [f'Intervention {i}' for i in range(1, 21)],
        'date_debut': pd.date_range(start='2024-02-01', periods=20, freq='D'),
        'date_fin': pd.date_range(start='2024-02-02', periods=20, freq='D'),
        'technicien': np.random.choice(['Jean', 'Marie', 'Pierre', 'Sophie', 'Luc'], 20),
        'statut': np.random.choice(['Planifiée', 'En cours', 'Terminée', 'Annulée'], 20),
        'cout': np.random.uniform(100, 1000, 20).round(2),
        'notes': ''
    })
    
    # Pièces détachées
    pieces = pd.DataFrame({
        'id': range(1, 16),
        'nom': ['Roulement 6204', 'Courroie A45', 'Filtre à huile', 'Joint spi', 'Boulon M8',
                'Roulement 6205', 'Courroie B50', 'Filtre à air', 'Joint torque', 'Boulon M10',
                'Pompe à eau', 'Capteur pression', 'Vanne 3/4', 'Flexible hydraulique', 'Fusible 10A'],
        'reference': [f'REF{100+i}' for i in range(1, 16)],
        'stock': np.random.randint(0, 50, 15),
        'stock_min': np.random.randint(5, 20, 15),
        'localisation': np.random.choice(['Magasin A', 'Magasin B', 'Armoire 1', 'Armoire 2'], 15),
        'prix_unitaire': np.random.uniform(5, 200, 15).round(2),
        'fournisseur': np.random.choice(['Fournisseur A', 'Fournisseur B', 'Fournisseur C'], 15)
    })
    
    # Utilisateurs
    users = pd.DataFrame({
        'username': ['admin', 'technicien1', 'technicien2', 'superviseur'],
        'password': [hashlib.md5('admin123'.encode()).hexdigest(),
                    hashlib.md5('tech123'.encode()).hexdigest(),
                    hashlib.md5('tech456'.encode()).hexdigest(),
                    hashlib.md5('super123'.encode()).hexdigest()],
        'role': ['Administrateur', 'Technicien', 'Technicien', 'Superviseur'],
        'email': ['admin@gmao.com', 'tech1@gmao.com', 'tech2@gmao.com', 'super@gmao.com']
    })
    
    return equipments, interventions, pieces, users

# Chargement des données
equipments, interventions, pieces, users = load_sample_data()

# Fonction d'authentification
def authenticate(username, password):
    hashed_password = hashlib.md5(password.encode()).hexdigest()
    user = users[users['username'] == username]
    if len(user) > 0 and user.iloc[0]['password'] == hashed_password:
        return True, user.iloc[0]['role']
    return False, None

# Page de connexion
def login_page():
    st.markdown("<div class='main-header'><h1>🔧 GMAO - Connexion</h1></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            st.markdown("### 👤 Connexion à l'application")
            username = st.text_input("Nom d'utilisateur")
            password = st.text_input("Mot de passe", type="password")
            submitted = st.form_submit_button("Se connecter", use_container_width=True)
            
            if submitted:
                authenticated, role = authenticate(username, password)
                if authenticated:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.role = role
                    st.success("Connexion réussie!")
                    st.rerun()
                else:
                    st.error("Nom d'utilisateur ou mot de passe incorrect")
        
        st.markdown("---")
        st.markdown("**Comptes de démonstration:**")
        st.markdown("- admin / admin123 (Administrateur)")
        st.markdown("- technicien1 / tech123 (Technicien)")
        st.markdown("- superviseur / super123 (Superviseur)")

# Tableau de bord principal
def dashboard_page():
    st.markdown("<div class='main-header'><h1>📊 Tableau de bord GMAO</h1></div>", unsafe_allow_html=True)
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_equipments = len(equipments)
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{total_equipments}</div>
            <div class='stat-label'>Équipements</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        interventions_en_cours = len(interventions[interventions['statut'] == 'En cours'])
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{interventions_en_cours}</div>
            <div class='stat-label'>Interventions en cours</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        pieces_alerte = len(pieces[pieces['stock'] <= pieces['stock_min']])
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{pieces_alerte}</div>
            <div class='stat-label'>Pièces sous stock min</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        taux_dispo = (len(equipments[equipments['statut'] == 'En fonctionnement']) / len(equipments)) * 100
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{taux_dispo:.1f}%</div>
            <div class='stat-label'>Disponibilité</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Répartition des équipements par statut")
        status_counts = equipments['statut'].value_counts()
        fig = px.pie(values=status_counts.values, names=status_counts.index, 
                     title="Statut des équipements",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Interventions par type")
        intervention_types = interventions['type'].value_counts()
        fig = px.bar(x=intervention_types.index, y=intervention_types.values,
                     title="Types d'interventions",
                     color=intervention_types.index,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    
    # Alertes
    st.subheader("⚠️ Alertes et notifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚨 Maintenances à venir")
        prochaines_maintenances = equipments[equipments['prochaine_maintenance'] <= datetime.now() + timedelta(days=7)]
        if len(prochaines_maintenances) > 0:
            for _, eq in prochaines_maintenances.iterrows():
                days_until = (eq['prochaine_maintenance'] - datetime.now()).days
                if days_until <= 2:
                    st.markdown(f"<div class='alert-critical'>⚠️ {eq['nom']} - Maintenance dans {days_until} jours</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='alert-warning'>⚠️ {eq['nom']} - Maintenance dans {days_until} jours</div>", unsafe_allow_html=True)
        else:
            st.info("Aucune maintenance prévue dans les 7 prochains jours")
    
    with col2:
        st.markdown("#### 📦 Stock critique")
        stock_critique = pieces[pieces['stock'] <= pieces['stock_min']]
        if len(stock_critique) > 0:
            for _, piece in stock_critique.iterrows():
                st.markdown(f"<div class='alert-critical'>📦 {piece['nom']} - Stock: {piece['stock']} / Min: {piece['stock_min']}</div>", unsafe_allow_html=True)
        else:
            st.info("Aucun stock critique")

# Gestion des équipements
def equipments_page():
    st.markdown("<div class='main-header'><h1>🏭 Gestion des équipements</h1></div>", unsafe_allow_html=True)
    
    # Formulaire d'ajout
    with st.expander("➕ Ajouter un nouvel équipement", expanded=False):
        with st.form("add_equipment"):
            col1, col2 = st.columns(2)
            with col1:
                nom = st.text_input("Nom de l'équipement")
                type_eq = st.selectbox("Type", ['Compresseur', 'Pompe', 'Moteur', 'Convoyeur', 'Tour', 'Fraiseuse', 'Presse', 'Ventilateur', 'Générateur'])
                localisation = st.text_input("Localisation")
            with col2:
                statut = st.selectbox("Statut", ['En fonctionnement', 'En maintenance', 'Arrêt'])
                priorite = st.selectbox("Priorité", ['Haute', 'Moyenne', 'Basse'])
                date_prochaine = st.date_input("Date prochaine maintenance")
            
            submitted = st.form_submit_button("Ajouter l'équipement")
            if submitted:
                st.success(f"Équipement {nom} ajouté avec succès!")
    
    # Filtres
    st.subheader("🔍 Liste des équipements")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_type = st.multiselect("Filtrer par type", options=equipments['type'].unique())
    with col2:
        filter_status = st.multiselect("Filtrer par statut", options=equipments['statut'].unique())
    with col3:
        filter_priority = st.multiselect("Filtrer par priorité", options=equipments['priorite'].unique())
    
    # Application des filtres
    filtered_df = equipments.copy()
    if filter_type:
        filtered_df = filtered_df[filtered_df['type'].isin(filter_type)]
    if filter_status:
        filtered_df = filtered_df[filtered_df['statut'].isin(filter_status)]
    if filter_priority:
        filtered_df = filtered_df[filtered_df['priorite'].isin(filter_priority)]
    
    # Affichage du tableau
    st.dataframe(
        filtered_df,
        column_config={
            "id": "ID",
            "nom": "Nom",
            "type": "Type",
            "localisation": "Localisation",
            "statut": "Statut",
            "derniere_maintenance": "Dernière maintenance",
            "prochaine_maintenance": "Prochaine maintenance",
            "priorite": "Priorité"
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Statistiques détaillées
    st.subheader("📊 Analyse des équipements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(equipments, x='type', color='statut',
                          title="Répartition des équipements par type et statut",
                          barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(equipments, x='derniere_maintenance', y='prochaine_maintenance',
                        color='priorite', hover_data=['nom'],
                        title="Planning des maintenances")
        st.plotly_chart(fig, use_container_width=True)

# Gestion des interventions
def interventions_page():
    st.markdown("<div class='main-header'><h1>🔨 Gestion des interventions</h1></div>", unsafe_allow_html=True)
    
    # Formulaire de planification
    with st.expander("📅 Planifier une nouvelle intervention", expanded=False):
        with st.form("add_intervention"):
            col1, col2 = st.columns(2)
            with col1:
                equipement = st.selectbox("Équipement", equipments['nom'].tolist())
                type_int = st.selectbox("Type d'intervention", ['Préventive', 'Curative', 'Inspection'])
                description = st.text_area("Description")
            with col2:
                technicien = st.selectbox("Technicien", ['Jean', 'Marie', 'Pierre', 'Sophie', 'Luc'])
                date_debut = st.date_input("Date de début")
                date_fin = st.date_input("Date de fin")
            
            submitted = st.form_submit_button("Planifier l'intervention")
            if submitted:
                st.success(f"Intervention planifiée pour {equipement}")
    
    # Tableau des interventions
    st.subheader("📋 Liste des interventions")
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_status = st.selectbox("Filtrer par statut", 
                                    options=['Tous'] + list(interventions['statut'].unique()))
    with col2:
        filter_technicien = st.selectbox("Filtrer par technicien",
                                        options=['Tous'] + list(interventions['technicien'].unique()))
    with col3:
        search = st.text_input("Rechercher par description")
    
    # Application des filtres
    filtered_int = interventions.copy()
    if filter_status != 'Tous':
        filtered_int = filtered_int[filtered_int['statut'] == filter_status]
    if filter_technicien != 'Tous':
        filtered_int = filtered_int[filtered_int['technicien'] == filter_technicien]
    if search:
        filtered_int = filtered_int[filtered_int['description'].str.contains(search, case=False)]
    
    # Ajout du nom de l'équipement
    filtered_int = filtered_int.merge(equipments[['id', 'nom']], left_on='equipement_id', right_on='id', suffixes=('', '_eq'))
    
    st.dataframe(
        filtered_int[['id', 'nom', 'type', 'description', 'date_debut', 'date_fin', 'technicien', 'statut', 'cout']],
        column_config={
            "id": "ID",
            "nom": "Équipement",
            "type": "Type",
            "description": "Description",
            "date_debut": "Début",
            "date_fin": "Fin",
            "technicien": "Technicien",
            "statut": "Statut",
            "cout": st.column_config.NumberColumn("Coût (€)", format="%.2f €")
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Graphiques d'analyse
    st.subheader("📈 Analyse des interventions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Évolution temporelle
        interventions['mois'] = pd.to_datetime(interventions['date_debut']).dt.to_period('M')
        monthly_stats = interventions.groupby('mois').size().reset_index(name='count')
        monthly_stats['mois'] = monthly_stats['mois'].astype(str)
        
        fig = px.line(monthly_stats, x='mois', y='count',
                     title="Évolution du nombre d'interventions",
                     markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Coûts par technicien
        costs_by_tech = interventions.groupby('technicien')['cout'].sum().reset_index()
        fig = px.pie(costs_by_tech, values='cout', names='technicien',
                    title="Répartition des coûts par technicien")
        st.plotly_chart(fig, use_container_width=True)

# Gestion des stocks
def stock_page():
    st.markdown("<div class='main-header'><h1>📦 Gestion des stocks</h1></div>", unsafe_allow_html=True)
    
    # Formulaire d'ajout
    with st.expander("➕ Ajouter une pièce détachée", expanded=False):
        with st.form("add_piece"):
            col1, col2 = st.columns(2)
            with col1:
                nom = st.text_input("Nom de la pièce")
                reference = st.text_input("Référence")
                fournisseur = st.text_input("Fournisseur")
            with col2:
                stock = st.number_input("Stock initial", min_value=0, value=0)
                stock_min = st.number_input("Stock minimum", min_value=0, value=10)
                prix = st.number_input("Prix unitaire (€)", min_value=0.0, value=0.0)
            
            localisation = st.selectbox("Localisation", ['Magasin A', 'Magasin B', 'Armoire 1', 'Armoire 2'])
            
            submitted = st.form_submit_button("Ajouter la pièce")
            if submitted:
                st.success(f"Pièce {nom} ajoutée au stock")
    
    # Vue du stock
    st.subheader("📊 État du stock")
    
    # Alertes stock
    stock_alert = pieces[pieces['stock'] <= pieces['stock_min']]
    if len(stock_alert) > 0:
        st.warning(f"⚠️ {len(stock_alert)} pièces sont en dessous du stock minimum")
    
    # Filtres
    col1, col2 = st.columns(2)
    with col1:
        search_piece = st.text_input("🔍 Rechercher une pièce")
    with col2:
        show_alert_only = st.checkbox("Afficher uniquement les alertes stock")
    
    # Application des filtres
    filtered_pieces = pieces.copy()
    if show_alert_only:
        filtered_pieces = filtered_pieces[filtered_pieces['stock'] <= filtered_pieces['stock_min']]
    if search_piece:
        filtered_pieces = filtered_pieces[filtered_pieces['nom'].str.contains(search_piece, case=False)]
    
    # Affichage du tableau
    st.dataframe(
        filtered_pieces,
        column_config={
            "id": "ID",
            "nom": "Pièce",
            "reference": "Référence",
            "stock": st.column_config.NumberColumn("Stock", format="%d"),
            "stock_min": "Stock min",
            "localisation": "Localisation",
            "prix_unitaire": st.column_config.NumberColumn("Prix unitaire", format="%.2f €"),
            "fournisseur": "Fournisseur"
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Visualisations
    col1, col2 = st.columns(2)
    
    with col1:
        # Stock par localisation
        stock_by_location = pieces.groupby('localisation')['stock'].sum().reset_index()
        fig = px.bar(stock_by_location, x='localisation', y='stock',
                    title="Stock par localisation",
                    color='localisation')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pièces sous seuil critique
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Stock actuel',
            x=pieces['nom'],
            y=pieces['stock'],
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            name='Stock minimum',
            x=pieces['nom'],
            y=pieces['stock_min'],
            marker_color='red'
        ))
        fig.update_layout(title="Stock vs Seuil minimum",
                         barmode='group',
                         xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# Rapports et analyses
def reports_page():
    st.markdown("<div class='main-header'><h1>📊 Rapports et analyses</h1></div>", unsafe_allow_html=True)
    
    # Sélection du type de rapport
    report_type = st.selectbox(
        "Type de rapport",
        ["Performance des équipements", "Analyse des interventions", "Gestion des stocks", "Coûts de maintenance"]
    )
    
    # Période d'analyse
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Date de début", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("Date de fin", datetime.now())
    
    if report_type == "Performance des équipements":
        st.subheader("📈 Performance des équipements")
        
        # Calcul des métriques de performance
        eq_stats = equipments.copy()
        eq_stats['taux_dispo'] = np.random.uniform(70, 100, len(eq_stats))
        eq_stats['mtbf'] = np.random.uniform(100, 500, len(eq_stats)).round()  # Mean Time Between Failures
        eq_stats['mttr'] = np.random.uniform(1, 24, len(eq_stats)).round(1)    # Mean Time To Repair
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MTBF Moyen", f"{eq_stats['mtbf'].mean():.0f} heures")
        with col2:
            st.metric("MTTR Moyen", f"{eq_stats['mttr'].mean():.1f} heures")
        with col3:
            st.metric("Taux de disponibilité moyen", f"{eq_stats['taux_dispo'].mean():.1f}%")
        
        # Graphique de performance
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Taux de disponibilité par équipement", "MTBF et MTTR par équipement")
        )
        
        fig.add_trace(
            go.Bar(x=eq_stats['nom'], y=eq_stats['taux_dispo'],
                  name="Disponibilité (%)", marker_color='green'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=eq_stats['nom'], y=eq_stats['mtbf'],
                      name="MTBF (heures)", mode='lines+markers', line=dict(color='blue')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=eq_stats['nom'], y=eq_stats['mttr'],
                      name="MTTR (heures)", mode='lines+markers', line=dict(color='red')),
            row=2, col=1
        )
        
        fig.update_layout(height=800, showlegend=True)
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    elif report_type == "Analyse des interventions":
        st.subheader("🔧 Analyse des interventions")
        
        # Filtrage par période
        mask = (pd.to_datetime(interventions['date_debut']).dt.date >= start_date) & \
               (pd.to_datetime(interventions['date_debut']).dt.date <= end_date)
        period_interventions = interventions[mask]
        
        # Métriques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre d'interventions", len(period_interventions))
        with col2:
            st.metric("Coût total", f"{period_interventions['cout'].sum():.2f} €")
        with col3:
            duree_moyenne = (pd.to_datetime(period_interventions['date_fin']) - 
                           pd.to_datetime(period_interventions['date_debut'])).dt.total_seconds() / 3600
            st.metric("Durée moyenne", f"{duree_moyenne.mean():.1f} heures")
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            # Répartition par type
            type_counts = period_interventions['type'].value_counts()
            fig = px.pie(values=type_counts.values, names=type_counts.index,
                        title="Répartition par type d'intervention")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Évolution temporelle
            daily_count = period_interventions.groupby(
                pd.to_datetime(period_interventions['date_debut']).dt.date
            ).size().reset_index(name='count')
            daily_count.columns = ['date', 'count']
            
            fig = px.line(daily_count, x='date', y='count',
                         title="Évolution quotidienne",
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
    
    elif report_type == "Gestion des stocks":
        st.subheader("📦 Analyse des stocks")
        
        # Valeur du stock
        pieces['valeur_stock'] = pieces['stock'] * pieces['prix_unitaire']
        valeur_totale = pieces['valeur_stock'].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Valeur totale du stock", f"{valeur_totale:.2f} €")
        with col2:
            st.metric("Nombre de références", len(pieces))
        with col3:
            stock_critique = len(pieces[pieces['stock'] <= pieces['stock_min']])
            st.metric("Stock critique", stock_critique)
        
        # Pareto du stock (20% des pièces représentent 80% de la valeur)
        pieces_sorted = pieces.sort_values('valeur_stock', ascending=False)
        pieces_sorted['cumul_pourcentage'] = pieces_sorted['valeur_stock'].cumsum() / valeur_totale * 100
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=pieces_sorted['nom'], y=pieces_sorted['valeur_stock'],
                  name="Valeur par pièce"),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=pieces_sorted['nom'], y=pieces_sorted['cumul_pourcentage'],
                      name="% Cumulé", line=dict(color='red', width=2)),
            secondary_y=True,
        )
        
        fig.update_layout(title="Analyse ABC du stock",
                         xaxis_tickangle=-45)
        fig.update_yaxes(title_text="Valeur (€)", secondary_y=False)
        fig.update_yaxes(title_text="Pourcentage cumulé (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif report_type == "Coûts de maintenance":
        st.subheader("💰 Analyse des coûts")
        
        # Coûts par catégorie
        period_interventions = interventions[
            (pd.to_datetime(interventions['date_debut']).dt.date >= start_date) &
            (pd.to_datetime(interventions['date_debut']).dt.date <= end_date)
        ]
        
        # Coûts par type d'intervention
        costs_by_type = period_interventions.groupby('type')['cout'].agg(['sum', 'mean', 'count']).round(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=costs_by_type['sum'], names=costs_by_type.index,
                        title="Répartition des coûts par type d'intervention")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Coûts par équipement
            costs_by_eq = period_interventions.merge(
                equipments[['id', 'nom']], left_on='equipement_id', right_on='id'
            ).groupby('nom')['cout'].sum().sort_values(ascending=False).head(10)
            
            fig = px.bar(x=costs_by_eq.values, y=costs_by_eq.index,
                        title="Top 10 équipements par coût de maintenance",
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
        
        # Tableau détaillé des coûts
        st.subheader("📊 Détail des coûts par intervention")
        detailed_costs = period_interventions.merge(
            equipments[['id', 'nom']], left_on='equipement_id', right_on='id'
        )[['id_x', 'nom', 'type', 'description', 'date_debut', 'technicien', 'cout']]
        detailed_costs.columns = ['ID', 'Équipement', 'Type', 'Description', 'Date', 'Technicien', 'Coût']
        
        st.dataframe(detailed_costs, use_container_width=True, hide_index=True)

# Configuration utilisateur
def settings_page():
    st.markdown("<div class='main-header'><h1>⚙️ Paramètres</h1></div>", unsafe_allow_html=True)
    
    # Onglets pour les différents paramètres
    tab1, tab2, tab3 = st.tabs(["Profil", "Préférences", "Administration"])
    
    with tab1:
        st.subheader("👤 Mon profil")
        
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Nom d'utilisateur", value=st.session_state.username, disabled=True)
            st.text_input("Rôle", value=st.session_state.role, disabled=True)
        with col2:
            email = st.text_input("Email", value="user@example.com")
            telephone = st.text_input("Téléphone", value="+33 6 12 34 56 78")
        
        if st.button("Mettre à jour le profil"):
            st.success("Profil mis à jour avec succès!")
    
    with tab2:
        st.subheader("🎨 Préférences d'affichage")
        
        theme = st.selectbox("Thème", ["Clair", "Sombre", "Automatique"])
        langue = st.selectbox("Langue", ["Français", "English", "Español"])
        notifications = st.multiselect(
            "Notifications",
            ["Email", "SMS", "Push", "In-app"],
            default=["Email", "In-app"]
        )
        
        if st.button("Sauvegarder les préférences"):
            st.success("Préférences sauvegardées!")
    
    with tab3:
        st.subheader("🔐 Administration")
        
        if st.session_state.role == "Administrateur":
            st.warning("⚠️ Zone réservée aux administrateurs")
            
            # Gestion des utilisateurs (simplifiée)
            st.markdown("#### Gestion des utilisateurs")
            
            # Liste des utilisateurs
            st.dataframe(
                users[['username', 'role', 'email']],
                use_container_width=True,
                hide_index=True
            )
            
            # Formulaire d'ajout d'utilisateur
            with st.expander("Ajouter un utilisateur"):
                with st.form("add_user"):
                    new_username = st.text_input("Nom d'utilisateur")
                    new_password = st.text_input("Mot de passe", type="password")
                    new_role = st.selectbox("Rôle", ["Technicien", "Superviseur", "Administrateur"])
                    new_email = st.text_input("Email")
                    
                    if st.form_submit_button("Créer l'utilisateur"):
                        st.success(f"Utilisateur {new_username} créé avec succès!")
            
            # Sauvegarde/Export des données
            st.markdown("#### Gestion des données")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Sauvegarder la base de données"):
                    # Simulation de sauvegarde
                    backup_data = {
                        "equipements": equipments.to_dict('records'),
                        "interventions": interventions.to_dict('records'),
                        "pieces": pieces.to_dict('records'),
                        "date": datetime.now().isoformat()
                    }
                    st.download_button(
                        label="Télécharger la sauvegarde",
                        data=json.dumps(backup_data, indent=2, default=str),
                        file_name=f"gmao_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("🔄 Réinitialiser les données"):
                    st.warning("Cette action est irréversible!")
                    if st.checkbox("Je confirme la réinitialisation"):
                        st.success("Données réinitialisées avec succès!")
        else:
            st.error("Vous n'avez pas les droits d'administration")

# Barre latérale de navigation
def sidebar_navigation():
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=GMAO", use_column_width=True)
        st.markdown(f"### 👋 Bienvenue, {st.session_state.username}")
        st.markdown(f"**Rôle:** {st.session_state.role}")
        st.markdown("---")
        
        # Menu de navigation
        menu_options = {
            "📊 Tableau de bord": "Tableau de bord",
            "🏭 Équipements": "Équipements",
            "🔨 Interventions": "Interventions",
            "📦 Stock": "Stock",
            "📊 Rapports": "Rapports",
            "⚙️ Paramètres": "Paramètres"
        }
        
        for icon_label, page_name in menu_options.items():
            if st.sidebar.button(icon_label, use_container_width=True):
                st.session_state.page = page_name
                st.rerun()
        
        st.markdown("---")
        
        # Affichage de l'heure et de la date
        st.markdown(f"**Date:** {datetime.now().strftime('%d/%m/%Y')}")
        st.markdown(f"**Heure:** {datetime.now().strftime('%H:%M')}")
        
        # Bouton de déconnexion
        if st.button("🚪 Déconnexion", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.role = None
            st.rerun()

# Main application
def main():
    if not st.session_state.authenticated:
        login_page()
    else:
        sidebar_navigation()
        
        # Navigation vers les pages
        if st.session_state.page == "Tableau de bord":
            dashboard_page()
        elif st.session_state.page == "Équipements":
            equipments_page()
        elif st.session_state.page == "Interventions":
            interventions_page()
        elif st.session_state.page == "Stock":
            stock_page()
        elif st.session_state.page == "Rapports":
            reports_page()
        elif st.session_state.page == "Paramètres":
            settings_page()

if __name__ == "__main__":
    main()
