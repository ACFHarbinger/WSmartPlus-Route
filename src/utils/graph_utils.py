import torch
import numpy as np
import osmnx as ox
import networkx as nx


def generate_adj_matrix(size, num_edges, undirected=False, add_depot=True, negative=False):
    # If `num_edges` is a percentage, convert to int
    if isinstance(num_edges, float):
        if undirected:
            num_edges = int(num_edges * (size * (size - 1)) / 2)
        else:
            num_edges = int(num_edges * size * (size - 1))

    max_edges = int((size * (size - 1)) / 2) if undirected else int(size * (size - 1))
    if num_edges >= 0 and num_edges < max_edges:
        adj_matrix = np.zeros((size, size), dtype=int)
        possible_edges = [(i, j) for i in range(0, size) 
                        for j in range(0, size) if i != j]

        # For undirected graphs, avoid duplicate edges by ensuring (i, j) == (j, i)
        if undirected:
            possible_edges = [(i, j) for i, j in possible_edges if i < j]

        # Randomly select edges from all possible ones and populate adj matrix
        selected = np.random.choice(len(possible_edges), num_edges, replace=False)
        for edge_index in selected:
            i, j = possible_edges[edge_index]
            adj_matrix[i, j] = 1
            if undirected:
                adj_matrix[j, i] = 1

        # Add edges to and from the depot (without self-connection)
        if add_depot:
            adj_matrix = np.vstack((np.ones(size, dtype=int), selected))
            adj_matrix = np.hstack((np.ones(size+1, dtype=int), selected))
            adj_matrix[0, 0] = 0
    else:
        adj_matrix = np.ones((size+1, size+1), dtype=int) if add_depot else np.ones((size, size), dtype=int)
        np.fill_diagonal(adj_matrix, 0)

    # Convert the adjacency matrix to edge_index
    #edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)
    return adj_matrix if not negative else 1 - adj_matrix


def get_edge_idx_dist(dist_matrix, num_edges, add_depot=True, undirected=True):
    assert not undirected or np.allclose(dist_matrix, dist_matrix.T), \
    "Distance matrix must be symmetric for an undirected graph"
    size = len(dist_matrix)
    
    # If `num_edges` is a percentage, convert to int
    if isinstance(num_edges, float):
        if undirected:
            num_edges = int(num_edges * (size * (size - 1)) / 2)
        else:
            num_edges = int(num_edges * size * (size - 1))

    max_edges = int((size * (size - 1)) / 2) if undirected else int(size * (size - 1))
    if num_edges >= 0 and num_edges < max_edges:
        if undirected:
            # Get upper triangular part of the distance matrix (excluding the diagonal)
            upper_tri_idx = np.triu_indices_from(dist_matrix, k=1)
            upper_tri_dist = dist_matrix[upper_tri_idx]

            # Sort distances and select the edges
            sorted_indices = np.argsort(upper_tri_dist)
            selected_indices = sorted_indices[:num_edges]
            edges = np.array((upper_tri_idx[0][selected_indices], 
                            upper_tri_idx[1][selected_indices]), dtype=int)
            
            # Add edges to and from the depot
            if add_depot:
                d_edges = [[0] * size, list(range(1, size+1))]
                selected = np.hstack((edges+1, d_edges, [d_edges[1], d_edges[0]])).T
            else:
                selected = edges.T
            return selected[np.lexsort((selected[:, 1], selected[:, 0]))].T
        else:
            # Sort distances and get the threshold
            sorted_dist = np.sort(dist_matrix.flatten())
            thresh = sorted_dist[num_edges - 1]
            #thresh = np.percentile(distance_matrix, edge_threshold)

            # Select the edges and remove self-loops
            adj_matrix = (dist_matrix <= thresh).astype(int)

            # Add edges to and from the depot
            if add_depot:
                adj_matrix = np.vstack((np.ones(size, dtype=int), adj_matrix))
                adj_matrix = np.hstack((np.ones(size+1, dtype=int), adj_matrix))

            np.fill_diagonal(adj_matrix, 0)
            return adj_to_idx(np.nonzero(adj_matrix), negative=False)
    else:
        adj_matrix = np.ones((size+1, size+1), dtype=int) if add_depot else np.ones((size, size), dtype=int)
        np.fill_diagonal(adj_matrix, 0)
        return adj_to_idx(np.nonzero(adj_matrix), negative=False)


def sort_by_pairs(graph_size, edge_idx):
    assert len(edge_idx.size()) == 2
    assert edge_idx.size(dim=0) == 2 or edge_idx.size(dim=-1) == 2

    # Transpose the tensor if it has size (2, num_edges)
    is_transpose = edge_idx.size(dim=-1) != 2
    if is_transpose:
        edge_idx = torch.transpose(edge_idx)

    tmp = edge_idx.select(1, 0) * graph_size + edge_idx.select(1, 1)
    ind = tmp.sort().indices
    sorted_idx = edge_idx.index_select(0, ind)
    if is_transpose:
        sorted_idx = torch.transpose(sorted_idx)
    return sorted_idx


def get_adj_knn(dist_mat, k_neighbors, add_depot=True, negative=True):
    size = len(dist_mat)

    # If `k_neighbors` is a percentage, convert to int
    if isinstance(k_neighbors, float):
        k_neighbors = int(size * k_neighbors)
    
    if k_neighbors >= size-1 or k_neighbors == -1:
        W = np.zeros((size, size))
    else:
        W_val = np.array(dist_mat)
        W = np.ones((size, size))
        
        # Determine k-nearest neighbors for each node
        knns = np.argpartition(W_val, kth=k_neighbors, axis=-1)[:, k_neighbors::-1]

        # Make connections
        for idx in range(size):
            W[idx][knns[idx]] = 0
    
    # Add connections to and from the depot
    if add_depot:
        W = np.pad(W, ((1, 0), (1, 0)), mode='constant', constant_values=0)

    # Remove self-connections
    np.fill_diagonal(W, 1)
    return W if negative else 1 - W


def adj_to_idx(adj_matrix, negative=True):
    filter = 0 if negative else 1
    src, dst = np.where(adj_matrix == filter)
    return np.vstack((src, dst))


def idx_to_adj(edge_idx, negative=False):
    fill_values = (1, 0) if negative else (0, 1)
    num_nodes = edge_idx.max().item() + 1
    adj_matrix = np.full((num_nodes, num_nodes), fill_values[0])
    adj_matrix[edge_idx[0], edge_idx[1]] = fill_values[1]
    return adj_matrix


def tour_to_adj(tour_nodes):
    num_nodes = len(tour_nodes)
    tour_edges = np.zeros((num_nodes, num_nodes))
    for idx in range(len(tour_nodes) - 1):
        i = tour_nodes[idx]
        j = tour_nodes[idx + 1]
        tour_edges[i][j] = 1
        tour_edges[j][i] = 1

    # Add final connection
    tour_edges[j][tour_nodes[0]] = 1
    tour_edges[tour_nodes[0]][j] = 1
    return tour_edges


def get_adj_osm(coords, size, args, add_depot=True, negative=True):
    G, *args = args
    assert isinstance(G, nx.MultiDiGraph)
    df = coords.copy() if coords.shape[0] == size else coords.copy().drop(index=1)
    assert df.shape[0] == size
    
    # Find nearest locations to Open Street Maps vertices and build adjacency matrix
    df["OSM_Node"] = df.apply(lambda row: ox.distance.nearest_nodes(G, row["Lng"], row["Lat"]), axis=1)
    adj_matrix = nx.to_numpy_array(G, nodelist=df["OSM_Node"], dtype=int)
    
    # Add connections to depot and remove self-connections
    if add_depot:
        adj_matrix = np.pad(adj_matrix, ((1, 0), (1, 0)), mode='constant', constant_values=1)

    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix if not negative else 1 - adj_matrix


def find_longest_path(dist_matrix, start_vertex=0):
    """
    Find the longest path in a DAG represented by a distance matrix.
    
    Args:
        dist_matrix (torch.Tensor): n x n tensor where dist_matrix[i][j] is the weight
                                   of edge from node i to node j. Use -inf for no edge.
        start_vertex (int): Starting vertex index (default: 0).
    
    Returns:
        tuple: (max_length, path) where max_length is the longest path length
               and path is the list of node indices.
    """
    longest_path = []
    n_vertices = dist_matrix.size(0)
    longest_length = torch.tensor(float('-inf'), device=dist_matrix.device)

    def backtrack(current, visited, path, current_length):
        nonlocal longest_length, longest_path
        # If all nodes are visited, check if we can return to start_node
        if len(path) == n_vertices:
            return_weight = dist_matrix[current][start_vertex]
            if return_weight != float('-inf'):
                total_length = current_length + return_weight
                if total_length > longest_length:
                    longest_length = total_length
                    longest_path = path[:] + [start_vertex]
            return

        # Update longest path if current path is longer
        if current_length > longest_length:
            longest_length = current_length
            longest_path = path[:]

        # Explore neighbors
        for next_vertex in range(n_vertices):
            if next_vertex not in visited and dist_matrix[current][next_vertex] != float('-inf'):
                visited.add(next_vertex)
                path.append(next_vertex)
                backtrack(next_vertex, visited, path, current_length + dist_matrix[current][next_vertex])
                visited.remove(next_vertex)
                path.pop()

    # Start backtracking from start_vertex
    backtrack(start_vertex, {start_vertex}, [start_vertex], torch.tensor(0.0, device=dist_matrix.device))
    return longest_length.item(), longest_path
