class BipartiteGraph:
    def __init__(self, U, V, adj_U, adj_V):
        self.U = U  # U节点列表
        self.V = V  # V节点列表
        self.adj_U = adj_U  # U到V的邻接表（u: list[v]）
        self.adj_V = adj_V  # V到U的邻接表（v: list[u]）
    
    def copy(self):
        return BipartiteGraph(
            self.U.copy(),
            self.V.copy(),
            {u: vs.copy() for u, vs in self.adj_U.items()},
            {v: us.copy() for v, us in self.adj_V.items()}
        )
    
    def remove_node(self, node, part):
        if part == 'U':
            self.U.remove(node)
            del self.adj_U[node]
            # 更新V节点的邻接表（删除该U节点）
            for v in self.V:
                if node in self.adj_V[v]:
                    self.adj_V[v].remove(node)
        else:
            self.V.remove(node)
            del self.adj_V[node]
            # 更新U节点的邻接表（删除该V节点）
            for u in self.U:
                if node in self.adj_U[u]:
                    self.adj_U[u].remove(node)

def Reduce1Hop(G: BipartiteGraph, tau_U, tau_V):
    Gi = G.copy()
    finish = False
    while not finish:
        finish = True
        # 检查U节点（度数到V < tau_V）
        for u in list(Gi.U):
            degree = len(Gi.adj_U[u])
            if degree < tau_V:
                Gi.remove_node(u, 'U')
                finish = False
                break  # 重新迭代检查
        if finish:
            # 检查V节点（度数到U < tau_U）
            for v in list(Gi.V):
                degree = len(Gi.adj_V[v])
                if degree < tau_U:
                    Gi.remove_node(v, 'V')
                    finish = False
                    break
    return Gi

def general_Reduce2H(Gi: BipartiteGraph, node_set, tau_node, tau_neighbor, part):
    # part: 'U'（处理U节点，邻接V，两跳U）或 'V'（处理V节点，邻接U，两跳V）
    for node in list(node_set):
        S = {}
        neighbors = Gi.adj_U[node] if part == 'U' else Gi.adj_V[node]
        for neighbor in neighbors:
            two_hop_neighbors = Gi.adj_V[neighbor] if part == 'U' else Gi.adj_U[neighbor]
            for th_node in two_hop_neighbors:
                if th_node not in S:
                    S[th_node] = 1
                else:
                    S[th_node] += 1
        c = sum(1 for cnt in S.values() if cnt >= tau_neighbor)
        if c < tau_node:
            Gi.remove_node(node, part)
    return Gi

def Reduce2Hop(G: BipartiteGraph, tau_U, tau_V):
    Gi = G.copy()
    # 处理U节点（tau_node=tau_U, tau_neighbor=tau_V，两跳U节点）
    Gi = general_Reduce2H(Gi, Gi.U, tau_U, tau_V, 'U')
    # 处理V节点（tau_node=tau_V, tau_neighbor=tau_U，两跳V节点）
    Gi = general_Reduce2H(Gi, Gi.V, tau_V, tau_U, 'V')
    return Gi

def CombinedReduce(G: BipartiteGraph, tau_U, tau_V):
    # 先执行1-hop约简，再执行2-hop约简
    Gi = Reduce1Hop(G, tau_U, tau_V)
    Gi = Reduce2Hop(Gi, tau_U, tau_V)
    return Gi

# 假设图初始化
U = ['u1', 'u2', 'u3']
V = ['v1', 'v2', 'v3']
adj_U = {'u1': ['v1'], 'u2': ['v2', 'v3'], 'u3': ['v2', 'v3']}
adj_V = {'v1': ['u1'], 'v2': ['u2', 'u3'], 'v3': ['u2', 'u3']}
G = BipartiteGraph(U, V, adj_U, adj_V)

tau_U = 2
tau_V = 2
reduced_G = CombinedReduce(G, tau_U, tau_V)
print("Reduced U:", reduced_G.U)
print("Reduced V:", reduced_G.V)