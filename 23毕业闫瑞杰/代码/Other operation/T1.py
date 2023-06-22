from collections import defaultdict


def tree_partition(n, edges):
    graph = defaultdict(list)  # 构建邻接表表示的图，节点从1到n编号
    for edge in edges:
        u, v = edge
        graph[u].append(v)
        graph[v].append(u)

    visited = [False] * (n + 1)  # 标记已访问的节点
    subtree_sizes = [0] * (n + 1)  # 存储各子树的节点数
    min_diff = float('inf')
    count = 0

    def dfs(node):
        nonlocal min_diff, count
        visited[node] = True
        subtree_sizes[node] = 1

        for neighbor in graph[node]:
            if not visited[neighbor]:
                subtree_sizes[node] += dfs(neighbor)

        total = subtree_sizes[node]
        diff = abs(n - 2 * subtree_sizes[node])  # 计算节点数差的绝对值
        if diff < min_diff:
            min_diff = diff
            count = 1
        elif diff == min_diff:
            count += 1

        return total

    dfs(1)  # 从节点1开始遍历

    return min_diff, count


if __name__ == '__main__':
    n = 3
    edges = [(1, 2), (1, 3)]
    result = tree_partition(n, edges)
    print("最小节点数差的绝对值:", result[0])
    print("最优方案数量:", result[1])
