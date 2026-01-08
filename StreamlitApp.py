import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from datetime import datetime, timedelta

# =====================================================
# CONFIGURATION GÉNÉRALE
# =====================================================
st.set_page_config(
    page_title="Optimisation VRP - Voyageurs Représentants Placiers",
    page_icon="🚚",
    layout="wide"
)

# =====================================================
# ALGORITHMES D'OPTIMISATION VRP
# =====================================================

def calculate_distance_matrix(coordinates):
    """Calcule la matrice des distances entre tous les points"""
    n = len(coordinates)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                lat1, lon1 = coordinates[i]
                lat2, lon2 = coordinates[j]
                
                # Distance euclidienne simplifiée
                dist_lat = (lat2 - lat1) * 111
                dist_lon = (lon2 - lon1) * 111 * math.cos(math.radians((lat1 + lat2) / 2))
                dist_matrix[i][j] = math.sqrt(dist_lat**2 + dist_lon**2)
            else:
                dist_matrix[i][j] = 0
    return dist_matrix

def nearest_neighbor_vrp(distance_matrix, depot_index=0, n_vehicles=3):
    """Algorithme du plus proche voisin pour VRP"""
    n_nodes = len(distance_matrix)
    unvisited = set(range(1, n_nodes))
    
    routes = []
    route_distances = []
    
    for v in range(n_vehicles):
        if not unvisited:
            break
            
        current = depot_index
        route = [current]
        route_distance = 0
        
        while unvisited:
            nearest = None
            min_dist = float('inf')
            
            for node in unvisited:
                if distance_matrix[current][node] < min_dist:
                    min_dist = distance_matrix[current][node]
                    nearest = node
            
            if nearest is None:
                break
                
            route.append(nearest)
            route_distance += min_dist
            current = nearest
            unvisited.remove(nearest)
        
        route.append(depot_index)
        route_distance += distance_matrix[current][depot_index]
        routes.append(route)
        route_distances.append(route_distance)
    
    return routes, route_distances

def create_route_visualization(coordinates, routes, depot_index=0):
    """Crée une visualisation des itinéraires"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    lats = [coord[0] for coord in coordinates]
    lons = [coord[1] for coord in coordinates]
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    
    # Points clients
    ax.scatter(lons, lats, c='gray', alpha=0.5, s=100)
    
    # Dépôt
    ax.scatter(lons[depot_index], lats[depot_index], 
               c='black', s=200, marker='s', edgecolors='white', linewidth=2)
    
    # Itinéraires
    for i, route in enumerate(routes):
        route_color = colors[i % len(colors)]
        route_lons = [coordinates[node][1] for node in route]
        route_lats = [coordinates[node][0] for node in route]
        
        ax.plot(route_lons, route_lats, color=route_color, 
                linewidth=2, marker='o', markersize=6,
                label=f'Véhicule {i+1}')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Visualisation des Itinéraires')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

# =====================================================
# INTERFACE UTILISATEUR
# =====================================================

st.sidebar.title("⚙️ Paramètres")
n_vehicles = st.sidebar.number_input("Nombre de véhicules", 1, 10, 3)
depot_index = st.sidebar.number_input("Index du dépôt", 0, 100, 0)

st.title("🚚 Optimisation VRP")
st.markdown("Optimisation des itinéraires pour représentants commerciaux")

# =====================================================
# IMPORTATION DES DONNÉES
# =====================================================
st.header("📥 Importation des données")

uploaded_file = st.file_uploader("Choisissez un fichier CSV ou Excel", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ Fichier importé: {df.shape[0]} lignes")
        
        # Afficher les colonnes
        st.write("Colonnes disponibles:", list(df.columns))
        
        # Sélection des colonnes
        col1, col2 = st.columns(2)
        with col1:
            lat_col = st.selectbox("Colonne Latitude", df.columns)
        with col2:
            lon_col = st.selectbox("Colonne Longitude", df.columns)
        
        # Vérifier les données
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
        df_clean = df.dropna(subset=[lat_col, lon_col])
        
        if len(df_clean) > 0:
            coordinates = list(zip(df_clean[lat_col], df_clean[lon_col]))
            clients_data = []
            
            for idx, row in df_clean.iterrows():
                clients_data.append({
                    'id': idx,
                    'name': f'Client {idx}',
                    'latitude': row[lat_col],
                    'longitude': row[lon_col]
                })
            
            st.success(f"✅ {len(coordinates)} clients valides")
            
            # =====================================================
            # OPTIMISATION
            # =====================================================
            if st.button("🚀 Lancer l'optimisation"):
                with st.spinner("Calcul en cours..."):
                    distance_matrix = calculate_distance_matrix(coordinates)
                    routes, distances = nearest_neighbor_vrp(distance_matrix, depot_index, n_vehicles)
                    
                    # Résultats
                    st.header("📊 Résultats")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Clients", len(coordinates))
                    with col2:
                        st.metric("Véhicules", len(routes))
                    with col3:
                        st.metric("Distance totale", f"{sum(distances):.1f} km")
                    
                    # Détails par véhicule
                    st.subheader("🚛 Itinéraires")
                    for i, (route, distance) in enumerate(zip(routes, distances)):
                        with st.expander(f"Véhicule {i+1} - {distance:.1f} km"):
                            route_str = " → ".join([f"Client {node}" for node in route])
                            st.write(f"**Séquence:** {route_str}")
                            st.write(f"**Distance:** {distance:.1f} km")
                    
                    # Visualisation
                    st.subheader("🗺️ Visualisation")
                    fig = create_route_visualization(coordinates, routes, depot_index)
                    st.pyplot(fig)
                    
                    # Export
                    st.subheader("💾 Export")
                    export_data = []
                    for i, route in enumerate(routes):
                        for node in route:
                            export_data.append({
                                'Véhicule': i+1,
                                'Client': node,
                                'Latitude': coordinates[node][0],
                                'Longitude': coordinates[node][1]
                            })
                    
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Télécharger CSV",
                        data=csv,
                        file_name="itineraires.csv",
                        mime="text/csv"
                    )
        else:
            st.error("❌ Aucune coordonnée valide trouvée")
            
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
else:
    st.info("📝 Veuillez importer un fichier pour commencer")

st.markdown("---")
st.caption("VRP Optimization Tool - Optimisation des itinéraires")
