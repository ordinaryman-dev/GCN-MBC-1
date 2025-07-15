import torch
from torch_geometric.data import Data

def Iterative_MBC(data: Data, tau_U, tau_V):
    # 将 torch_geometric 的 Data 对象转换为字典表示的图
    def convert_data_to_graph(data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        G = {i: set() for i in range(num_nodes)}
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            G[src].add(dst)
        return G

    G = convert_data_to_graph(data)

    def Reduce(G, tau_U, tau_V):
        def reduce(G, tau_U, tau_V):
            right_nodes_to_remove = {j for j, neighbors in G.items() if len(neighbors) < tau_U}
            new_G = {i: neighbors.copy() for i, neighbors in G.items() 
                     if len(neighbors) >= tau_V}
            for neighbors in new_G.values():
                neighbors.difference_update(right_nodes_to_remove)
            return new_G

        def reduce_2hop(G, tau_U, tau_V):
            bipartite_graph = {}
            for u_key, v_set in G.items():
                u_vertex = f'u{u_key}'
                neighbors = [f'v{v}' for v in v_set]
                bipartite_graph[u_vertex] = neighbors
                for v in v_set:
                    v_vertex = f'v{v}'
                    if v_vertex not in bipartite_graph:
                        bipartite_graph[v_vertex] = []
                    if u_vertex not in bipartite_graph[v_vertex]:
                        bipartite_graph[v_vertex].append(u_vertex)
            Gi = {u: neighbors.copy() for u, neighbors in bipartite_graph.items()}
            U = {u for u in Gi if u.startswith('u')}
            V = {v for v in Gi if v.startswith('v')}
            vertices_to_remove = []
            for u in U:
                if u not in Gi:
                    continue
                S = {}
                for v in Gi[u]:
                    if v not in Gi:
                        continue
                    for u_prime in Gi[v]:
                        if u_prime not in S:
                            S[u_prime] = 1
                        else:
                            S[u_prime] += 1
                    c = sum(1 for cnt in S.values() if cnt >= tau_V)
                    if c < tau_U:
                        vertices_to_remove.append(u)
            for u in vertices_to_remove:
                if u in Gi:
                    del Gi[u]
                for v in Gi:
                    if u in Gi[v]:
                        Gi[v].remove(u)
            vertices_to_remove = []
            for v in V:
                if v not in Gi:
                    continue
                S = {}
                for u in Gi[v]:
                    if u not in Gi:
                        continue
                    for v_prime in Gi[u]:
                        if v_prime not in S:
                            S[v_prime] = 1
                        else:
                            S[v_prime] += 1
                    c = sum(1 for cnt in S.values() if cnt >= tau_U)
                    if c < tau_V:
                        vertices_to_remove.append(v)
            for v in vertices_to_remove:
                if v in Gi:
                    del Gi[v]
                for u in Gi:
                    if v in Gi[u]:
                        Gi[u].remove(v)
            Gi = {u: neighbors for u, neighbors in Gi.items() if neighbors}
            original_dict = {}
            for vertex, neighbors in Gi.items():
                if vertex.startswith('u'):
                    u_number = vertex[1:]
                    v_numbers = {neighbor[1:] for neighbor in neighbors if neighbor.startswith('v')}
                    original_dict[u_number] = v_numbers
            return original_dict

        def convert_nested_sets_to_ints(input_dict):
            return {
                int(key): {int(element) for element in value}
                for key, value in input_dict.items()
            }
        G = reduce(G, tau_U, tau_V)
        G = reduce_2hop(G, tau_U, tau_V)
        G = convert_nested_sets_to_ints(G)
        return G

    def mbc_algorithm(G, tau_U, tau_V, initial_C):
        max_size = len(initial_C[0]) * len(initial_C[1])
        max_biclique = (set(initial_C[0]), set(initial_C[1]))

        def branch_bound(U, V, Cv, Xv):
            nonlocal max_size, max_biclique
            current_size = len(U) + len(V)
            if len(V) >= tau_V and len(V) * len(U) > max_size:
                max_size = current_size
                max_biclique = (set(U), set(V))
            while Cv:
                v = Cv.pop()
                U_prime = {u for u in U if v in G[u]}
                new_V = V | {v} | {v for v in Cv if G[v] == U_prime}
                new_Cv = {v for v in (Cv - new_V) if len(G[v] & U_prime) >= tau_U}
                new_Xv = {v for v in Xv if len(G[v] & U_prime) >= tau_U}
                if len(U_prime) >= tau_U and len(new_V) + len(new_Cv) >= tau_V and (len(U_prime) * (len(new_V) + len(new_Cv)) > max_size) and not any(U_prime.issubset(G[v]) for v in Xv):
                    branch_bound(U_prime, new_V, new_Cv, new_Xv)
                Xv.add(v)
        left_vertices = set(G.keys())
        right_vertices = set(v for u in G for v in G[u])
        branch_bound(left_vertices, set(), right_vertices, set())
        return max_biclique

    def d_max_U(G):
        max_val = 0
        for i in G.keys():
            if len(G[i]) > max_val:
                max_val = len(G[i])
        return max_val

    def init_mbc(G):
        neighbor_lengths = {node: len(neighbors) for node, neighbors in G.items()}
        for i in G:
            if neighbor_lengths[i] < 2:
                continue
            for j in G[i]:
                if i >= j:
                    continue
                if neighbor_lengths[j] < 2:
                    continue
                if neighbor_lengths[i] + neighbor_lengths[j] - len(G[i] ^ G[j]) >= 4:
                    common = G[i] & G[j]
                    if len(common) >= 2:
                        return ({i, j}, common)
        return ({}, {})

    C_current = init_mbc(G)
    tau_V_current = d_max_U(G)
    k = 0
    while tau_V_current > tau_V:
        tau_U_next = max((len(C_current[0]) + len(C_current[1])) // tau_V_current, tau_U)
        tau_V_next = max(tau_V_current // 2, tau_V)
        G_next = Reduce(G, tau_U_next, tau_V_next)
        C_next = mbc_algorithm(G_next, tau_U_next, tau_V_next, C_current)
        tau_V_current = tau_V_next
        G = G_next
        C_current = C_next
        k += 1

    # 将输出转换为 torch.Tensor
    left_nodes = torch.tensor(list(C_current[0]), dtype=torch.long)
    right_nodes = torch.tensor(list(C_current[1]), dtype=torch.long)

    return left_nodes, right_nodes