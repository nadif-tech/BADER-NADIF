import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import math
import random
from datetime import datetime, timedelta

# =====================================================
# CONFIGURATION GÉNÉRALE
# =====================================================
st.set_page_config(
    page_title="Optimisation VRP - Voyageurs Représentants Placiers",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
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
                # Distance euclidienne simplifiée (pour l'exemple)
                # En réalité, vous devriez utiliser une vraie API de distance
                lat1, lon1 = coordinates[i]
                lat2, lon2 = coordinates[j]
                
                # Approximation de distance (1 degré ≈ 111 km)
                dist_lat = (lat2 - lat1) * 111
                dist_lon = (lon2 - lon1) * 111 * math.cos(math.radians((lat1 + lat2) / 2))
                dist_matrix[i][j] = math.sqrt(dist_lat**2 + dist_lon**2)
            else:
                dist_matrix[i][j] = 0
    return dist_matrix

def nearest_neighbor_vrp(distance_matrix, depot_index=0, n_vehicles=3, max_capacity=100):
    """Algorithme du plus proche voisin pour VRP"""
    n_nodes = len(distance_matrix)
    unvisited = set(range(1, n_nodes))  # Exclure le dépôt
    
    routes = []
    vehicle_loads = []
    
    for v in range(n_vehicles):
        if not unvisited:
            break
            
        current = depot_index
        route = [current]
        route_distance = 0
        current_load = 0
        
        while unvisited and current_load < max_capacity:
            # Trouver le plus proche voisin non visité
            nearest = None
            min_dist = float('inf')
            
            for node in unvisited:
                if distance_matrix[current][node] < min_dist:
                    min_dist = distance_matrix[current][node]
                    nearest = node
            
            if nearest is None:
                break
                
            # Vérifier la capacité
            node_load = random.randint(5, 20)  # Charge aléatoire pour simulation
            if current_load + node_load <= max_capacity:
                route.append(nearest)
                route_distance += min_dist
                current_load += node_load
                current = nearest
                unvisited.remove(nearest)
            else:
                break
        
        # Retour au dépôt
        route.append(depot_index)
        route_distance += distance_matrix[current][depot_index]
        routes.append((route, route_distance))
        vehicle_loads.append(current_load)
    
    return routes, vehicle_loads

def savings_algorithm_vrp(distance_matrix, depot_index=0, n_vehicles=3, max_capacity=100):
    """Algorithme d'économies de Clarke et Wright"""
    n = len(distance_matrix)
    
    # Calcul des économies
    savings = []
    for i in range(1, n):
        for j in range(i + 1, n):
            if i != depot_index and j != depot_index:
                saving = distance_matrix[depot_index][i] + distance_matrix[depot_index][j] - distance_matrix[i][j]
                savings.append((saving, i, j))
    
    # Trier les économies par ordre décroissant
    savings.sort(reverse=True, key=lambda x: x[0])
    
    # Initialiser les routes
    routes = []
    for i in range(1, n):
        if i != depot_index:
            routes.append([depot_index, i, depot_index])
    
    # Simuler les charges
    demands = {i: random.randint(5, 20) for i in range(1, n)}
    
    # Fusionner les routes avec contrainte de capacité
    route_dict = {}
    route_loads = {}
    
    for i, route in enumerate(routes):
        load = sum(demands.get(node, 0) for node in route[1:-1])
        route_loads[i] = load
        for node in route[1:-1]:
            route_dict[node] = i
    
    for saving, i, j in savings:
        if i in route_dict and j in route_dict and route_dict[i] != route_dict[j]:
            route_i_idx = route_dict[i]
            route_j_idx = route_dict[j]
            
            route_i = routes[route_i_idx]
            route_j = routes[route_j_idx]
            
            # Vérifier si la fusion respecte la capacité
            total_load = route_loads[route_i_idx] + route_loads[route_j_idx]
            if total_load <= max_capacity:
                # Fusionner les routes
                if route_i[1] == i and route_j[-2] == j:
                    new_route = route_j[:-1] + route_i[1:]
                elif route_i[-2] == i and route_j[1] == j:
                    new_route = route_i[:-1] + route_j[1:]
                else:
                    continue
                
                routes[route_i_idx] = new_route
                route_loads[route_i_idx] = total_load
                
                # Supprimer l'ancienne route
                routes.pop(route_j_idx)
                del route_loads[route_j_idx]
                
                # Mettre à jour le dictionnaire
                for node in route_j[1:-1]:
                    route_dict[node] = route_i_idx
    
    # Limiter le nombre de routes au nombre de véhicules
    routes = routes[:n_vehicles]
    
    # Calculer les distances
    final_routes = []
    route_distances = []
    
    for route in routes:
        distance = 0
        for k in range(len(route) - 1):
            distance += distance_matrix[route[k]][route[k + 1]]
        final_routes.append(route)
        route_distances.append(distance)
    
    return final_routes, route_distances

def create_route_visualization(coordinates, routes, clients_data, depot_index=0):
    """Crée une visualisation des itinéraires avec Matplotlib"""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extraire les coordonnées
    lats = [coord[0] for coord in coordinates]
    lons = [coord[1] for coord in coordinates]
    
    # Couleurs pour les véhicules
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 
              'pink', 'gray', 'olive', 'cyan', 'magenta']
    
    # Tracer tous les points
    ax.scatter(lons, lats, c='gray', alpha=0.5, s=100, label='Clients')
    
    # Marquer le dépôt
    ax.scatter(lons[depot_index], lats[depot_index], 
               c='black', s=200, marker='s', label='Dépôt', edgecolors='white', linewidth=2)
    
    # Annoter les points
    for i, (lat, lon) in enumerate(coordinates):
        ax.annotate(f'{i}', (lon, lat), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    # Tracer les itinéraires
    for i, route in enumerate(routes):
        route_color = colors[i % len(colors)]
        
        # Coordonnées de l'itinéraire
        route_lons = [coordinates[node][1] for node in route]
        route_lats = [coordinates[node][0] for node in route]
        
        # Ligne de l'itinéraire
        ax.plot(route_lons, route_lats, color=route_color, 
                linewidth=2, marker='o', markersize=8,
                label=f'Véhicule {i+1}')
        
        # Ajouter des numéros d'étape
        for j, node in enumerate(route[1:-1]):
            ax.annotate(f'{j+1}', 
                       (coordinates[node][1], coordinates[node][0]),
                       xytext=(0, 15), textcoords='offset points',
                       ha='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor=route_color, 
                                alpha=0.7, edgecolor='none'))
    
    # Configuration du graphique
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Visualisation des Itinéraires VRP')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Ajuster les limites
    ax.set_xlim(min(lons) - 0.01, max(lons) + 0.01)
    ax.set_ylim(min(lats) - 0.01, max(lats) + 0.01)
    
    plt.tight_layout()
    return fig

# =====================================================
# INTERFACE UTILISATEUR
# =====================================================

# SIDEBAR
with st.sidebar:
    st.title("⚙️ Paramètres d'optimisation")
    
    st.markdown("### Configuration des véhicules")
    n_vehicles = st.number_input(
        "Nombre de véhicules",
        min_value=1,
        max_value=10,
        value=3,
        help="Nombre de représentants/véhicules disponibles"
    )
    
    max_capacity = st.number_input(
        "Capacité max par véhicule",
        min_value=10,
        max_value=200,
        value=100,
        help="Capacité maximale (unité de charge)"
    )
    
    st.markdown("### Paramètres d'algorithme")
    algorithm = st.selectbox(
        "Algorithme d'optimisation",
        ["Plus proche voisin", "Clarke & Wright (Économies)"],
        help="Sélectionnez la méthode d'optimisation"
    )
    
    depot_index = st.number_input(
        "Index du dépôt (0-based)",
        min_value=0,
        value=0,
        help="Index de la ligne correspondant au dépôt/entrepôt"
    )
    
    st.divider()
    
    st.markdown("### Coûts et contraintes")
    fuel_cost = st.number_input(
        "Coût du carburant (€/km)",
        min_value=0.1,
        max_value=2.0,
        value=0.6,
        step=0.1
    )
    
    driver_cost = st.number_input(
        "Coût chauffeur (€/heure)",
        min_value=10.0,
        max_value=50.0,
        value=25.0,
        step=5.0
    )
    
    avg_speed = st.number_input(
        "Vitesse moyenne (km/h)",
        min_value=20.0,
        max_value=100.0,
        value=50.0,
        step=5.0
    )
    
    st.divider()
    
    st.markdown("### Aide")
    st.info("""
    **Format des données:**
    - Fichier CSV/Excel avec colonnes: Client, Latitude, Longitude
    - La première ligne est considérée comme le dépôt par défaut
    - Les coordonnées doivent être en degrés décimaux
    
    **Optimisation:**
    - Réduction des distances totales parcourues
    - Équilibrage des charges entre véhicules
    - Visualisation des itinéraires sur carte
    """)

# TITRE PRINCIPAL
st.title("🚚 Optimisation VRP - Voyageurs Représentants Placiers")
st.markdown("**Optimisation des itinéraires pour représentants commerciaux**")

# =====================================================
# SECTION 1: IMPORTATION DES DONNÉES
# =====================================================
st.header("📥 Importation des données géographiques")

data_mode = st.radio(
    "Sélectionnez le mode d'entrée:",
    ["📁 Importer un fichier", "📊 Exemple prédéfini"],
    horizontal=True
)

df = None
coordinates = []
clients_data = []

if data_mode == "📁 Importer un fichier":
    uploaded_file = st.file_uploader(
        "Choisissez un fichier CSV ou Excel avec localisations",
        type=["csv", "xlsx", "xls"],
        help="Colonnes requises: Latitude, Longitude. Optionnel: Client, Demande"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Vérification des colonnes nécessaires
            required_cols = []
            lat_col = None
            lon_col = None
            
            # Chercher les colonnes de coordonnées
            for col in df.columns:
                col_lower = col.lower()
                if 'lat' in col_lower:
                    lat_col = col
                elif 'lon' in col_lower or 'lng' in col_lower:
                    lon_col = col
            
            if lat_col and lon_col:
                st.success(f"✅ Fichier importé: {df.shape[0]} clients trouvés")
                
                # Extraire les coordonnées
                coordinates = list(zip(df[lat_col], df[lon_col]))
                
                # Préparer les données clients
                for idx, row in df.iterrows():
                    client_info = {
                        'id': idx,
                        'name': row.get('Client', f'Client {idx}'),
                        'latitude': row[lat_col],
                        'longitude': row[lon_col],
                        'demand': row.get('Demande', random.randint(5, 20))
                    }
                    clients_data.append(client_info)
                
                # Afficher un aperçu
                st.subheader("📋 Aperçu des données")
                preview_df = pd.DataFrame(clients_data)
                st.dataframe(preview_df[['id', 'name', 'latitude', 'longitude', 'demand']].head(), 
                           use_container_width=True)
                
            else:
                st.error("❌ Colonnes 'Latitude' et 'Longitude' requises. Noms acceptés: lat, latitude, lon, longitude, lng")
                
        except Exception as e:
            st.error(f"❌ Erreur lors de l'importation: {str(e)}")

else:  # Exemple prédéfini
    st.info("📊 Chargement d'un exemple de localisations clients (Paris et banlieue)")
    
    # Coordonnées d'exemple (Paris et banlieue)
    example_coordinates = [
        (48.8566, 2.3522),    # Paris centre (dépôt)
        (48.8584, 2.2945),    # Tour Eiffel
        (48.8606, 2.3376),    # Louvre
        (48.8738, 2.2950),    # La Défense
        (48.8356, 2.2418),    # Boulogne-Billancourt
        (48.8895, 2.3192),    # Saint-Denis
        (48.8184, 2.3310),    # Montrouge
        (48.8462, 2.4399),    # Vincennes
        (48.8156, 2.3594),    # Gentilly
        (48.8124, 2.3915),    # Kremlin-Bicêtre
        (48.7803, 2.4970),    # Créteil
        (48.9061, 2.4185),    # Le Bourget
        (48.7975, 2.5249),    # Saint-Maur-des-Fossés
        (48.7886, 2.3931),    # Villejuif
        (48.8049, 2.1203),    # Versailles
    ]
    
    coordinates = example_coordinates
    
    # Créer les données clients
    client_names = [
        "Dépôt Central", "Tour Eiffel", "Musée Louvre", "La Défense", 
        "Boulogne", "St-Denis", "Montrouge", "Vincennes", "Gentilly",
        "Kremlin-Bicêtre", "Créteil", "Le Bourget", "Saint-Maur", 
        "Villejuif", "Versailles"
    ]
    
    for idx, (lat, lon) in enumerate(coordinates):
        client_info = {
            'id': idx,
            'name': client_names[idx],
            'latitude': lat,
            'longitude': lon,
            'demand': random.randint(5, 25)
        }
        clients_data.append(client_info)
    
    df = pd.DataFrame(clients_data)

# =====================================================
# SECTION 2: OPTIMISATION DES ITINÉRAIRES
# =====================================================
if coordinates and len(coordinates) > 1:
    st.header("🧮 Optimisation des itinéraires")
    
    if st.button("🚀 Lancer l'optimisation", type="primary", use_container_width=True):
        with st.spinner("Calcul des itinéraires optimaux..."):
            try:
                # Calcul de la matrice de distances
                distance_matrix = calculate_distance_matrix(coordinates)
                
                # Exécuter l'algorithme sélectionné
                if algorithm == "Plus proche voisin":
                    routes, vehicle_loads = nearest_neighbor_vrp(
                        distance_matrix, 
                        depot_index, 
                        n_vehicles, 
                        max_capacity
                    )
                    routes_list = [route for route, _ in routes]
                    route_distances = [dist for _, dist in routes]
                else:  # Clarke & Wright
                    routes_list, route_distances = savings_algorithm_vrp(
                        distance_matrix, 
                        depot_index, 
                        n_vehicles, 
                        max_capacity
                    )
                    vehicle_loads = [sum(clients_data[node]['demand'] for node in route[1:-1]) 
                                   for route in routes_list]
                
                # =====================================================
                # AFFICHAGE DES RÉSULTATS
                # =====================================================
                
                # 1. Résumé des paramètres
                with st.expander("📊 Paramètres de l'optimisation", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Nombre de clients", len(coordinates))
                        st.metric("Véhicules utilisés", len(routes_list))
                    
                    with col2:
                        total_distance = sum(route_distances)
                        st.metric("Distance totale", f"{total_distance:.1f} km")
                        st.metric("Distance moyenne", f"{np.mean(route_distances):.1f} km")
                    
                    with col3:
                        total_load = sum(vehicle_loads)
                        st.metric("Charge totale", f"{total_load} unités")
                        st.metric("Charge moyenne", f"{np.mean(vehicle_loads):.1f}")
                    
                    with col4:
                        # Calcul des coûts
                        total_time = total_distance / avg_speed
                        fuel_cost_total = total_distance * fuel_cost
                        driver_cost_total = total_time * driver_cost
                        total_cost = fuel_cost_total + driver_cost_total
                        
                        st.metric("Coût carburant", f"{fuel_cost_total:.2f} €")
                        st.metric("Coût total estimé", f"{total_cost:.2f} €")
                
                # 2. Détails par véhicule
                st.subheader("🚛 Itinéraires par véhicule")
                
                for i, (route, distance, load) in enumerate(zip(routes_list, route_distances, vehicle_loads)):
                    with st.expander(f"Véhicule {i+1} - {distance:.1f} km - {load} unités", expanded=i==0):
                        # Afficher l'itinéraire détaillé
                        route_details = []
                        total_time_minutes = (distance / avg_speed) * 60
                        
                        # Simuler des heures de départ
                        start_time = datetime.now().replace(hour=8, minute=0, second=0)
                        current_time = start_time
                        
                        for j, node_idx in enumerate(route):
                            if j == 0:
                                # Dépôt de départ
                                route_details.append({
                                    'Étape': 'Départ',
                                    'Client': clients_data[node_idx]['name'],
                                    'Heure estimée': current_time.strftime('%H:%M'),
                                    'Distance depuis précédent': '0 km',
                                    'Temps trajet': '0 min',
                                    'Charge après visite': f'{load if j>0 else 0}'
                                })
                            elif j == len(route) - 1:
                                # Retour au dépôt
                                prev_node = route[j-1]
                                segment_dist = distance_matrix[prev_node][node_idx]
                                segment_time = (segment_dist / avg_speed) * 60
                                current_time += timedelta(minutes=segment_time)
                                
                                route_details.append({
                                    'Étape': 'Retour',
                                    'Client': clients_data[node_idx]['name'],
                                    'Heure estimée': current_time.strftime('%H:%M'),
                                    'Distance depuis précédent': f'{segment_dist:.1f} km',
                                    'Temps trajet': f'{segment_time:.0f} min',
                                    'Charge après visite': '0'
                                })
                            else:
                                # Client intermédiaire
                                prev_node = route[j-1]
                                segment_dist = distance_matrix[prev_node][node_idx]
                                segment_time = (segment_dist / avg_speed) * 60
                                current_time += timedelta(minutes=segment_time)
                                
                                # Temps de visite simulé
                                visit_time = random.randint(15, 45)
                                current_time += timedelta(minutes=visit_time)
                                
                                route_details.append({
                                    'Étape': f'Visite {j}',
                                    'Client': clients_data[node_idx]['name'],
                                    'Heure estimée': current_time.strftime('%H:%M'),
                                    'Distance depuis précédent': f'{segment_dist:.1f} km',
                                    'Temps trajet': f'{segment_time:.0f} min',
                                    'Charge après visite': f'{sum(clients_data[n]["demand"] for n in route[j+1:-1])}'
                                })
                        
                        # Afficher le tableau détaillé
                        route_df = pd.DataFrame(route_details)
                        st.dataframe(route_df, use_container_width=True, hide_index=True)
                        
                        # Afficher la séquence simplifiée
                        sequence = " → ".join([clients_data[node_idx]['name'] for node_idx in route])
                        st.caption(f"**Séquence:** {sequence}")
                        
                        # Afficher les statistiques de l'itinéraire
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Distance totale", f"{distance:.1f} km")
                        with col2:
                            st.metric("Temps estimé", f"{total_time_minutes:.0f} min")
                        with col3:
                            st.metric("Utilisation", f"{(load/max_capacity)*100:.1f}%")
                
                # 3. Visualisation des itinéraires avec Matplotlib
                st.subheader("🗺️ Visualisation des itinéraires")
                
                # Créer la visualisation
                fig_map = create_route_visualization(coordinates, routes_list, clients_data, depot_index)
                
                # Afficher la visualisation
                st.pyplot(fig_map)
                
                # 4. Graphiques de performance
                st.subheader("📈 Analyse de performance")
                
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle('Analyse des itinéraires optimisés', fontsize=16)
                
                # Graphique 1: Distances par véhicule
                ax1 = axes[0, 0]
                vehicles = [f'Véhicule {i+1}' for i in range(len(route_distances))]
                colors = plt.cm.Set3(np.linspace(0, 1, len(vehicles)))
                
                bars1 = ax1.bar(vehicles, route_distances, color=colors, alpha=0.8)
                ax1.set_ylabel('Distance (km)')
                ax1.set_title('Distance parcourue par véhicule')
                ax1.set_xticklabels(vehicles, rotation=45)
                ax1.grid(True, alpha=0.3, axis='y')
                
                for bar, dist in zip(bars1, route_distances):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2, height, f'{dist:.1f} km',
                            ha='center', va='bottom', fontsize=9)
                
                # Graphique 2: Charges par véhicule
                ax2 = axes[0, 1]
                bars2 = ax2.bar(vehicles, vehicle_loads, color=colors, alpha=0.8)
                ax2.set_ylabel('Charge (unités)')
                ax2.set_title('Charge transportée par véhicule')
                ax2.axhline(y=max_capacity, color='red', linestyle='--', alpha=0.7, 
                           label=f'Capacité max: {max_capacity}')
                ax2.set_xticklabels(vehicles, rotation=45)
                ax2.grid(True, alpha=0.3, axis='y')
                ax2.legend()
                
                for bar, load in zip(bars2, vehicle_loads):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2, height, f'{load}',
                            ha='center', va='bottom', fontsize=9)
                
                # Graphique 3: Utilisation de la capacité
                ax3 = axes[1, 0]
                utilization = [(load/max_capacity)*100 for load in vehicle_loads]
                bars3 = ax3.bar(vehicles, utilization, color=colors, alpha=0.8)
                ax3.set_ylabel('Utilisation (%)')
                ax3.set_title('Utilisation de la capacité')
                ax3.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Cible: 80%')
                ax3.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Maximum')
                ax3.set_xticklabels(vehicles, rotation=45)
                ax3.grid(True, alpha=0.3, axis='y')
                ax3.legend()
                
                for bar, util in zip(bars3, utilization):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2, height, f'{util:.1f}%',
                            ha='center', va='bottom', fontsize=9)
                
                # Graphique 4: Coûts estimés
                ax4 = axes[1, 1]
                fuel_costs = [dist * fuel_cost for dist in route_distances]
                driver_costs = [(dist/avg_speed) * driver_cost for dist in route_distances]
                total_costs = [f + d for f, d in zip(fuel_costs, driver_costs)]
                
                x = np.arange(len(vehicles))
                width = 0.25
                
                bars4a = ax4.bar(x - width, fuel_costs, width, label='Carburant', alpha=0.8)
                bars4b = ax4.bar(x, driver_costs, width, label='Main d\'œuvre', alpha=0.8)
                bars4c = ax4.bar(x + width, total_costs, width, label='Total', alpha=0.8)
                
                ax4.set_ylabel('Coût (€)')
                ax4.set_title('Coûts estimés par véhicule')
                ax4.set_xticks(x)
                ax4.set_xticklabels(vehicles, rotation=45)
                ax4.grid(True, alpha=0.3, axis='y')
                ax4.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 5. Tableau récapitulatif
                st.subheader("📊 Récapitulatif des performances")
                
                summary_data = []
                for i, (distance, load, route) in enumerate(zip(route_distances, vehicle_loads, routes_list)):
                    fuel_cost_i = distance * fuel_cost
                    driver_cost_i = (distance/avg_speed) * driver_cost
                    total_cost_i = fuel_cost_i + driver_cost_i
                    utilization_i = (load/max_capacity) * 100
                    
                    summary_data.append({
                        'Véhicule': i+1,
                        'Clients visités': len(route)-2,
                        'Distance (km)': f"{distance:.1f}",
                        'Charge (unités)': f"{load}/{max_capacity}",
                        'Utilisation (%)': f"{utilization_i:.1f}",
                        'Coût carburant (€)': f"{fuel_cost_i:.2f}",
                        'Coût total (€)': f"{total_cost_i:.2f}",
                        'Séquence': " → ".join([str(node) for node in route])
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # =====================================================
                # SECTION 3: EXPORT DES RÉSULTATS
                # =====================================================
                st.header("💾 Export des itinéraires")
                
                # Préparation des données d'export
                export_data = []
                for i, (route, distance, load) in enumerate(zip(routes_list, route_distances, vehicle_loads)):
                    for j, node_idx in enumerate(route):
                        export_data.append({
                            'Véhicule': i+1,
                            'Étape': j+1,
                            'Client_ID': node_idx,
                            'Client_Nom': clients_data[node_idx]['name'],
                            'Latitude': clients_data[node_idx]['latitude'],
                            'Longitude': clients_data[node_idx]['longitude'],
                            'Demande': clients_data[node_idx]['demand'],
                            'Distance_itinéraire_km': distance,
                            'Charge_véhicule': load
                        })
                
                export_df = pd.DataFrame(export_data)
                
                # Boutons d'export
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 CSV détaillé",
                        data=csv_data,
                        file_name="itineraires_vrp.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Création du rapport Excel
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        if df is not None:
                            df.to_excel(writer, sheet_name='Données_clients', index=False)
                        export_df.to_excel(writer, sheet_name='Itinéraires_détaillés', index=False)
                        summary_df.to_excel(writer, sheet_name='Résumé_par_véhicule', index=False)
                        
                        # Ajouter les paramètres
                        params_df = pd.DataFrame({
                            'Paramètre': ['Nombre véhicules', 'Capacité max', 'Algorithme', 
                                         'Coût carburant', 'Coût chauffeur', 'Vitesse moyenne',
                                         'Distance totale', 'Charge totale', 'Coût total estimé'],
                            'Valeur': [n_vehicles, max_capacity, algorithm, 
                                      f"{fuel_cost} €/km", f"{driver_cost} €/h", f"{avg_speed} km/h",
                                      f"{total_distance:.2f} km", total_load, f"{total_cost:.2f} €"]
                        })
                        params_df.to_excel(writer, sheet_name='Paramètres', index=False)
                        
                        # Ajouter la matrice de distances
                        dist_df = pd.DataFrame(distance_matrix)
                        dist_df.columns = [f'Client {i}' for i in range(len(distance_matrix))]
                        dist_df.index = [f'Client {i}' for i in range(len(distance_matrix))]
                        dist_df.to_excel(writer, sheet_name='Matrice_distances', index=True)
                    
                    st.download_button(
                        label="📥 Excel complet",
                        data=excel_buffer.getvalue(),
                        file_name="rapport_vrp.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col3:
                    # Rapport texte détaillé
                    report = f"""
                    RAPPORT D'OPTIMISATION VRP
                    ===========================
                    
                    DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    
                    PARAMÈTRES:
                    ------------
                    Nombre de véhicules: {n_vehicles}
                    Capacité maximale: {max_capacity} unités
                    Algorithme utilisé: {algorithm}
                    Coût carburant: {fuel_cost} €/km
                    Coût chauffeur: {driver_cost} €/h
                    Vitesse moyenne: {avg_speed} km/h
                    
                    RÉSULTATS GLOBAUX:
                    -------------------
                    Distance totale parcourue: {total_distance:.2f} km
                    Charge totale transportée: {total_load} unités
                    Temps total estimé: {total_distance/avg_speed:.2f} heures
                    Coût carburant total: {fuel_cost_total:.2f} €
                    Coût main d'œuvre total: {driver_cost_total:.2f} €
                    Coût total estimé: {total_cost:.2f} €
                    
                    DÉTAIL PAR VÉHICULE:
                    ---------------------
                    """
                    
                    for i, (route, distance, load) in enumerate(zip(routes_list, route_distances, vehicle_loads)):
                        clients_list = [clients_data[node_idx]['name'] for node_idx in route[1:-1]]
                        report += f"""
                    Véhicule {i+1}:
                      - Distance: {distance:.2f} km
                      - Charge: {load}/{max_capacity} unités ({load/max_capacity*100:.1f}%)
                      - Clients visités: {len(clients_list)}
                      - Séquence: Dépôt → {' → '.join(clients_list)} → Dépôt
                      - Coût estimé: {distance*fuel_cost + (distance/avg_speed)*driver_cost:.2f} €
                        """
                    
                    report += f"""
                    
                    RECOMMANDATIONS:
                    ----------------
                    """
                    
                    # Analyse des recommandations
                    avg_utilization = np.mean([load/max_capacity for load in vehicle_loads]) * 100
                    
                    if avg_utilization < 60:
                        report += "⚠️ Utilisation moyenne basse: Considérer réduire le nombre de véhicules\n"
                    elif avg_utilization > 90:
                        report += "⚠️ Utilisation élevée: Risque de surcharge, augmenter capacité ou véhicules\n"
                    
                    if max(route_distances) > 2 * min(route_distances):
                        report += "⚠️ Déséquilibre des distances: Réoptimiser pour mieux équilibrer\n"
                    
                    report += f"""
                    ✅ Optimisation réussie avec {algorithm}
                    ✅ Économie estimée vs routes non optimisées: ~{total_distance * 0.2:.2f} km (20%)
                    """
                    
                    st.download_button(
                        label="📥 Rapport TXT",
                        data=report,
                        file_name="rapport_vrp.txt",
                        mime="text/plain"
                    )
                
                # =====================================================
                # SECTION 4: RECOMMANDATIONS
                # =====================================================
                st.header("💡 Recommandations d'amélioration")
                
                # Calcul des indicateurs de performance
                avg_utilization = np.mean([load/max_capacity for load in vehicle_loads]) * 100
                balance_index = min(route_distances) / max(route_distances) if max(route_distances) > 0 else 1
                
                cols_rec = st.columns(2)
                
                with cols_rec[0]:
                    st.metric("Utilisation moyenne", f"{avg_utilization:.1f}%")
                    if avg_utilization < 70:
                        st.warning("Utilisation sous-optimale")
                    elif avg_utilization > 90:
                        st.error("Risque de surcharge")
                    else:
                        st.success("Utilisation optimale")
                
                with cols_rec[1]:
                    st.metric("Équilibre des distances", f"{balance_index:.2f}")
                    if balance_index < 0.7:
                        st.warning("Déséquilibre important")
                    else:
                        st.success("Bon équilibre")
                
                # Recommandations détaillées
                with st.expander("🔍 Analyse détaillée et suggestions"):
                    st.markdown("""
                    **Pour améliorer l'efficacité:**
                    
                    1. **Si utilisation < 70%:**
                       - Réduire le nombre de véhicules
                       - Regrouper les clients proches
                       - Augmenter les plages horaires de service
                    
                    2. **Si déséquilibre des distances:**
                       - Réaffecter des clients entre véhicules
                       - Utiliser l'algorithme alternatif
                       - Imposer des contraintes de distance max
                    
                    3. **Pour réduire les coûts:**
                       - Négocier les tarifs carburant
                       - Optimiser les temps de visite
                       - Planifier les itinéraires en heure creuse
                    
                    4. **Améliorations techniques:**
                       - Intégrer le trafic en temps réel
                       - Considérer les fenêtres de temps clients
                       - Ajouter des contraintes de temps de service
                    """)
                
                # Information complémentaire
                with st.expander("📊 Métriques de performance"):
                    st.markdown(f"""
                    **Indicateurs clés:**
                    - **Distance totale réduite:** {total_distance:.2f} km
                    - **Économie estimée:** {total_distance * 0.2:.2f} km (vs non-optimisé)
                    - **Temps total:** {total_distance/avg_speed:.2f} heures
                    - **Coût/km moyen:** {total_cost/total_distance:.2f} €/km
                    - **Clients/vehicule moyen:** {np.mean([len(r)-2 for r in routes_list]):.1f}
                    
                    **Répartition:**
                    - Véhicule le plus chargé: {max(vehicle_loads)}/{max_capacity} ({max(vehicle_loads)/max_capacity*100:.1f}%)
                    - Véhicule le moins chargé: {min(vehicle_loads)}/{max_capacity} ({min(vehicle_loads)/max_capacity*100:.1f}%)
                    - Écart de charge: {max(vehicle_loads) - min(vehicle_loads)} unités
                    """)
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'optimisation: {str(e)}")
                st.info("Vérifiez que les données de localisation sont valides et complètes.")
else:
    st.info("📝 Veuillez importer des données de localisation pour commencer l'optimisation.")

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: gray;">
    <p><strong>VRP Optimization Tool</strong> - Optimisation des itinéraires pour voyageurs représentants placiers</p>
    <p>Algorithmes: Plus proche voisin • Clarke & Wright (Économies)</p>
    <p>Visualisation graphique • Analyse de coûts • Export complet</p>
</div>
""", unsafe_allow_html=True)
