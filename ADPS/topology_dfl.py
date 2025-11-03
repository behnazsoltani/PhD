import networkx as nx
import numpy as np

def generate_nodes_info(epoch, client_num_in_total, client_num_per_round, topology='ring', rows=None, columns=None):
    nodes_info = {}
    client_degrees = {}
    centrality = {}
    if client_num_in_total == 1:
        G = nx.empty_graph(1)  # No edges, just one isolated node

    elif topology == "random_avg_degree":
        np.random.seed(0)  # Ensure consistent randomness
        adjacency = {i: set() for i in range(client_num_in_total)}
        possible_edges = [(i, j) for i in range(client_num_in_total) for j in range(i + 1, client_num_in_total)]
        num_edges = client_num_in_total * (client_num_per_round - 1) // 2  # Average degree calculation
        selected_edges = np.random.choice(len(possible_edges), min(len(possible_edges), num_edges), replace=False)

        for idx in selected_edges:
            i, j = possible_edges[idx]
            adjacency[i].add(j)
            adjacency[j].add(i)

        G = nx.Graph(adjacency)
    
    
    elif topology == "random_regular":
        G = nx.random_regular_graph(client_num_per_round, client_num_in_total, seed=0)
    
    elif topology == "erdos_renyi":
        G = nx.erdos_renyi_graph(client_num_in_total, p=0.1, seed=0)  # 20% chance edge
    
    elif topology == "ring":
        G = nx.cycle_graph(client_num_in_total)
    
    elif topology == "full":
        G = nx.complete_graph(client_num_in_total)
    
    elif topology == "grid":
        if rows * columns != client_num_in_total:
            raise ValueError("The product of rows and columns must equal client_num_in_total")
        G = nx.grid_2d_graph(rows, columns)
        mapping = {(i, j): i * columns + j for i in range(rows) for j in range(columns)}
        G = nx.relabel_nodes(G, mapping)
    elif topology == "random":

        #np.random.seed(0)  # Ensure consistent randomness
        G = nx.random_regular_graph(5, client_num_in_total)  # Each client has exactly 5 neighbors
    elif topology == "dynamic_random":

        #np.random.seed(0)  # Ensure consistent randomness
        G = nx.random_regular_graph(5, client_num_in_total, seed = epoch)  # Each client has exactly 5 neighbors
    
    else:
        raise ValueError("Unknown topology: {}".format(topology))
    
    for cur_client in range(client_num_in_total):
        neighbor_list = list(G.neighbors(cur_client))
        nodes_info[cur_client] = neighbor_list
        client_degrees[cur_client] = len(neighbor_list)
    
    centrality = nx.degree_centrality(G)
    return nodes_info, client_degrees, centrality

def compute_weights_dict(client_id, aggregation_candidates, client_degrees):
    """
    Returns a weights dictionary including self-weight and neighbor weights.
    """
    self_degree = client_degrees[client_id]
    weights_dict = {}
    
    # Compute neighbor weights
    total_neighbor_weight = 0.0
    for neighbor_id in aggregation_candidates:
        if neighbor_id == client_id:
            continue
        neighbor_degree = client_degrees[neighbor_id]
        weight = 1.0 / (max(self_degree, neighbor_degree) + 1)
        weights_dict[neighbor_id] = weight
        total_neighbor_weight += weight
    
    # Add self-weight to the dictionary
    weights_dict[client_id] = 1.0 - total_neighbor_weight
    
    return weights_dict

