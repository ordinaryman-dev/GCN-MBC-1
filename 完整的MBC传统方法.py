def get_binary(picture,p_num):#将原始图转化为二分图（无奇数环）
    def get_side_l(picture,p_num):#把边的形式转化为每个点所连点的形式
        side_l = [[] for i0 in range(p_num + 1)]
        for [p1,p2] in picture:
            side_l[p1].append(p2)
            side_l[p2].append(p1)
        return side_l
    side_l = get_side_l(picture,p_num)
    binary_p = []
    def dfs(p,judge):#转化为二分图 judge为True时加入集合u
        if judge:
            for n_p in side_l[p]:
                if [p,n_p] not in binary_p: 
                    binary_p.append([p,n_p])
                    dfs(n_p,not judge)
        else:
            for n_p in side_l[p]:
                if [n_p,p] not in binary_p:
                    binary_p.append([n_p,p])
                    dfs(n_p,not judge)
    dfs(1,True)
    return binary_p,side_l
#二分图的形式为二位列表，每个小列表代表一条边，形式为【左侧节点，右侧节点】
#返回每个点所连的点，在后面初始化查询函数参数的时候要用


def trans_1(binary_picture):#输入二分图输出左右节点集合
    set_left = []
    set_right = []
    for [p_left,p_right] in binary_picture:
        if p_left not in set_left:
            set_left.append(p_left)
        if p_right not in set_right:
            set_right.append(p_right)
    return set_left,set_right

def mbc_algorithm(G, tau_U, tau_V, initial_C):#查找最大二分团
    max_size = len(initial_C[0]) * len(initial_C[1])
    max_biclique = (set(initial_C[0]), set(initial_C[1]))
    
    def branch_bound(U, V, Cv, Xv):
        nonlocal max_size, max_biclique
        # 更新最大 biclique（步骤4-6）
        current_size = len(U) * len(V)
        if len(V) >= tau_V and current_size > max_size:
            max_size = current_size
            max_biclique = (set(U), set(V))
        
        # 处理候选右部顶点（步骤7-15）
        while Cv:
            v = Cv.pop()  # 取出一个候选顶点
            U_prime = {u for u in U if v in G[u]}  # 左部中与v相连的顶点（步骤9）
            
            # 计算新的候选右部和排除右部（步骤10-11，简化处理，假设V'为包含v的新右部）
            new_V = V | {v}
            new_Cv = Cv  # 剩余候选顶点（v已处理，从Cv移除）
            new_Xv = Xv  # 排除右部暂时不变
            
            # 检查是否满足递归条件（步骤13）
            if len(U_prime) >= tau_U and len(new_V) + len(new_Cv) >= tau_V and (len(U_prime) * len(new_V) > max_size):
                branch_bound(U_prime, new_V, new_Cv, new_Xv)
            
            # 将v加入排除右部（步骤15，若不满足递归条件，或已处理）
            Xv.add(v)
    
    # 初始调用：左部为图的所有左顶点，右部为初始右部，候选右部为所有右顶点，排除右部为空
    left_vertices = set(G.keys())
    right_vertices = set(v for u in G for v in G[u])
    branch_bound(left_vertices, set(initial_C[1]), right_vertices - set(initial_C[1]), set())
    
    return max_biclique

def trans_2(set_left,tau_u,tau_v,side_l):#接口用于连接输出二分图形式和查找算法所需要的数据形式
    G = {}
    U_set = set_left
    for U in U_set:
        tmp = set()
        for j1 in side_l[U]:
            tmp.add(j1)
        G[U] = tmp
    initial_C = (set(), set())  # 初始空 biclique
    return mbc_algorithm(G,tau_u, tau_v, initial_C)

def optimize(b_left,b_right,side_l,p_num):#增加聚合节点优化初始图
    for p_l in b_left:
        for p_r in b_right:
            side_l[p_l].remove(p_r)
            side_l[p_r].remove(p_l)
    for p in (list(b_left) + list(b_right)):
        side_l[p_num + 1].append(p)
        side_l[p].append(p_num + 1)
    return side_l

def has_odd_cycle(graph, n):
    # 初始化邻接表
    adj = [[] for _ in range(n + 1)]
    # 将边列表转换为邻接表
    for u, v in graph:
        adj[u].append(v)
        adj[v].append(u)
    
    visited = [False] * (n + 1)
    depth = [0] * (n + 1)
    
    def dfs(node, parent):
        visited[node] = True
        for neighbor in adj[node]:
            if neighbor == parent:
                continue
            if not visited[neighbor]:
                depth[neighbor] = depth[node] + 1
                if dfs(neighbor, node):
                    return True
            else:
                cycle_length = depth[node] - depth[neighbor] + 1
                if cycle_length % 2 != 0:
                    return True
        return False
    
    for node in range(1, n + 1):
        if not visited[node]:
            if dfs(node, -1):
                return True
    return False

def main_func():
    print("第一行输入节点数量和边数量(用空格隔开)")
    print("接下来几行输入边(例如1 2)")

    p_num,side_num = map(int,input().split())
    print("--------------------------")

    graph = [list(map(int,input().split())) for _ in range(side_num)]
    if has_odd_cycle(graph, p_num): #图+点
        print('存在长度为奇数的环，因此无法生成二分图')
        return 
    print('成功生成二分图')
    binary_picture,side_l = get_binary(graph,p_num)
    print("生成的二分图为",binary_picture)
    set_left,set_right = trans_1(binary_picture)
    print("左侧点集",set_left,"右侧点集",set_right)
    result = trans_2(set_left,1,1,side_l)#先设置为1
    print("最大 biclique 左部:", result[0])
    print("最大 biclique 右部:", result[1])
    b_left = result[0]
    b_right = result[1]
    print("大小:", len(result[0]),'*', len(result[1]))
    side_l.append([])
    print("经过优化的二分图",optimize(b_left,b_right,side_l,p_num))
'-------------------------------------------------------------------------'
main_func()
#tau_u,tua_v,只为1时有效