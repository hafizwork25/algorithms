from collections import defaultdict, deque


class Graph:
    """
    Simple Graph data structure using adjacency list representation
    Supports both directed and undirected graphs
    """

    def __init__(self, directed=False):
        """
        Initialize a graph

        Args:
            directed (bool): True for directed graph, False for undirected
        """
        self.adjacency_list = defaultdict(list)
        self.directed = directed
        self.vertices = set()

    def add_vertex(self, vertex):
        """
        Add a vertex to the graph
        Time Complexity: O(1)
        """
        self.vertices.add(vertex)
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []

    def add_edge(self, u, v):
        """
        Add an edge from vertex u to vertex v
        For undirected graphs, adds edge in both directions
        Time Complexity: O(1)
        """
        self.add_vertex(u)
        self.add_vertex(v)

        self.adjacency_list[u].append(v)

        if not self.directed:
            self.adjacency_list[v].append(u)

    def get_neighbors(self, vertex):
        """
        Get all neighbors of a vertex
        Time Complexity: O(1)
        """
        return self.adjacency_list.get(vertex, [])

    def get_vertices(self):
        """
        Get all vertices in the graph
        Time Complexity: O(1)
        """
        return list(self.vertices)

    def breadth_first_search(self, start_vertex):
        """
        Breadth First Search (BFS) traversal
        Explores vertices level by level
        Time Complexity: O(V + E) where V is vertices and E is edges
        Space Complexity: O(V)

        Returns:
            list: Vertices in BFS order
            dict: Parent mapping for path reconstruction
            dict: Distance from start vertex
        """
        if start_vertex not in self.vertices:
            return [], {}, {}

        visited = set()
        queue = deque([start_vertex])
        bfs_order = []
        parent = {start_vertex: None}
        distance = {start_vertex: 0}

        visited.add(start_vertex)

        while queue:
            vertex = queue.popleft()
            bfs_order.append(vertex)

            for neighbor in self.adjacency_list[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    parent[neighbor] = vertex
                    distance[neighbor] = distance[vertex] + 1

        return bfs_order, parent, distance

    def depth_first_search(self, start_vertex):
        """
        Depth First Search (DFS) traversal (iterative version)
        Explores as far as possible along each branch
        Time Complexity: O(V + E)
        Space Complexity: O(V)

        Returns:
            list: Vertices in DFS order
        """
        if start_vertex not in self.vertices:
            return []

        visited = set()
        stack = [start_vertex]
        dfs_order = []

        while stack:
            vertex = stack.pop()

            if vertex not in visited:
                visited.add(vertex)
                dfs_order.append(vertex)

                # Add neighbors in reverse order to maintain left-to-right traversal
                for neighbor in reversed(self.adjacency_list[vertex]):
                    if neighbor not in visited:
                        stack.append(neighbor)

        return dfs_order

    def dfs_recursive(self, start_vertex, visited=None, dfs_order=None):
        """
        Depth First Search (DFS) traversal (recursive version)
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        if visited is None:
            visited = set()
        if dfs_order is None:
            dfs_order = []

        visited.add(start_vertex)
        dfs_order.append(start_vertex)

        for neighbor in self.adjacency_list[start_vertex]:
            if neighbor not in visited:
                self.dfs_recursive(neighbor, visited, dfs_order)

        return dfs_order

    def has_path(self, start, end):
        """
        Check if there is a path from start to end vertex using BFS
        Time Complexity: O(V + E)
        """
        if start not in self.vertices or end not in self.vertices:
            return False

        if start == end:
            return True

        visited = set([start])
        queue = deque([start])

        while queue:
            vertex = queue.popleft()

            if vertex == end:
                return True

            for neighbor in self.adjacency_list[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    def get_path(self, start, end):
        """
        Get the path from start to end vertex using BFS
        Returns the shortest path in unweighted graphs
        Time Complexity: O(V + E)
        """
        if start not in self.vertices or end not in self.vertices:
            return None

        if start == end:
            return [start]

        _, parent, _ = self.breadth_first_search(start)

        if end not in parent:
            return None

        # Reconstruct path
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = parent[current]

        return list(reversed(path))

    def __str__(self):
        """
        String representation of the graph
        """
        result = []
        for vertex in sorted(self.vertices):
            neighbors = ", ".join(map(str, sorted(self.adjacency_list[vertex])))
            result.append(f"{vertex} -> [{neighbors}]")
        return "\n".join(result)


class DirectedAcyclicGraph(Graph):
    """
    Directed Acyclic Graph (DAG) implementation
    A directed graph with no cycles
    """

    def __init__(self):
        """
        Initialize a DAG
        """
        super().__init__(directed=True)

    def add_edge(self, u, v):
        """
        Add an edge from u to v
        Checks if adding this edge creates a cycle
        Time Complexity: O(V + E)

        Returns:
            bool: True if edge added successfully, False if it creates a cycle
        """
        # Temporarily add the edge
        self.add_vertex(u)
        self.add_vertex(v)
        self.adjacency_list[u].append(v)

        # Check for cycle
        if self.has_cycle():
            # Remove the edge if it creates a cycle
            self.adjacency_list[u].remove(v)
            return False

        return True

    def has_cycle(self):
        """
        Detect if the graph has a cycle using DFS
        Time Complexity: O(V + E)

        Returns:
            bool: True if cycle exists, False otherwise
        """
        visited = set()
        recursion_stack = set()

        def has_cycle_util(vertex):
            visited.add(vertex)
            recursion_stack.add(vertex)

            for neighbor in self.adjacency_list[vertex]:
                if neighbor not in visited:
                    if has_cycle_util(neighbor):
                        return True
                elif neighbor in recursion_stack:
                    return True

            recursion_stack.remove(vertex)
            return False

        for vertex in self.vertices:
            if vertex not in visited:
                if has_cycle_util(vertex):
                    return True

        return False

    def topological_sort(self):
        """
        Topological sort of the DAG
        Returns a linear ordering of vertices such that for every edge (u, v),
        u comes before v in the ordering
        Time Complexity: O(V + E)

        Returns:
            list: Topologically sorted vertices, or None if graph has cycle
        """
        if self.has_cycle():
            return None

        visited = set()
        stack = []

        def topological_sort_util(vertex):
            visited.add(vertex)

            for neighbor in self.adjacency_list[vertex]:
                if neighbor not in visited:
                    topological_sort_util(neighbor)

            stack.append(vertex)

        for vertex in self.vertices:
            if vertex not in visited:
                topological_sort_util(vertex)

        return list(reversed(stack))


class GraphReachability:
    """
    Class to handle reachability queries in directed graphs
    """

    def __init__(self, graph):
        """
        Initialize with a graph

        Args:
            graph: A Graph object (must be directed)
        """
        if not graph.directed:
            raise ValueError("Reachability analysis requires a directed graph")
        self.graph = graph

    def get_reachable_from(self, vertex):
        """
        Get all vertices reachable from the given vertex
        Time Complexity: O(V + E)

        Returns:
            set: All vertices reachable from vertex
        """
        if vertex not in self.graph.vertices:
            return set()

        reachable = set()
        stack = [vertex]

        while stack:
            current = stack.pop()
            if current not in reachable:
                reachable.add(current)
                for neighbor in self.graph.adjacency_list[current]:
                    if neighbor not in reachable:
                        stack.append(neighbor)

        return reachable

    def are_mutually_reachable(self, u, v):
        """
        Check if two vertices are mutually reachable
        (i.e., u can reach v AND v can reach u)
        Time Complexity: O(V + E)

        Returns:
            bool: True if mutually reachable, False otherwise
        """
        if u not in self.graph.vertices or v not in self.graph.vertices:
            return False

        # Check if v is reachable from u
        reachable_from_u = self.get_reachable_from(u)
        if v not in reachable_from_u:
            return False

        # Check if u is reachable from v
        reachable_from_v = self.get_reachable_from(v)
        if u not in reachable_from_v:
            return False

        return True

    def find_all_mutually_reachable_pairs(self):
        """
        Find all pairs of vertices that are mutually reachable
        Time Complexity: O(V * (V + E))

        Returns:
            list: List of tuples representing mutually reachable pairs
        """
        pairs = []
        vertices = list(self.graph.vertices)

        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if self.are_mutually_reachable(vertices[i], vertices[j]):
                    pairs.append((vertices[i], vertices[j]))

        return pairs

    def find_strongly_connected_components(self):
        """
        Find all strongly connected components (SCCs) using Kosaraju's algorithm
        An SCC is a maximal set of vertices where every vertex is reachable from every other
        Time Complexity: O(V + E)

        Returns:
            list: List of sets, each set is a strongly connected component
        """
        # Step 1: Perform DFS and store vertices by finish time
        visited = set()
        finish_stack = []

        def dfs_first_pass(vertex):
            visited.add(vertex)
            for neighbor in self.graph.adjacency_list[vertex]:
                if neighbor not in visited:
                    dfs_first_pass(neighbor)
            finish_stack.append(vertex)

        for vertex in self.graph.vertices:
            if vertex not in visited:
                dfs_first_pass(vertex)

        # Step 2: Create transpose graph
        transpose = Graph(directed=True)
        for vertex in self.graph.vertices:
            transpose.add_vertex(vertex)
        for vertex in self.graph.vertices:
            for neighbor in self.graph.adjacency_list[vertex]:
                transpose.add_edge(neighbor, vertex)

        # Step 3: Perform DFS on transpose in reverse finish order
        visited = set()
        sccs = []

        def dfs_second_pass(vertex, component):
            visited.add(vertex)
            component.add(vertex)
            for neighbor in transpose.adjacency_list[vertex]:
                if neighbor not in visited:
                    dfs_second_pass(neighbor, component)

        while finish_stack:
            vertex = finish_stack.pop()
            if vertex not in visited:
                component = set()
                dfs_second_pass(vertex, component)
                sccs.append(component)

        return sccs


if __name__ == "__main__":
    print("=" * 70)
    print("GRAPH ALGORITHMS IMPLEMENTATION")
    print("=" * 70)

    # 1. Simple Graph Data Structure
    print("\n1. SIMPLE GRAPH DATA STRUCTURE")
    print("-" * 70)

    print("\nCreating an undirected graph:")
    undirected_graph = Graph(directed=False)
    edges = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]
    for u, v in edges:
        undirected_graph.add_edge(u, v)
    print(undirected_graph)

    print("\nCreating a directed graph:")
    directed_graph = Graph(directed=True)
    edges = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (5, 3)]
    for u, v in edges:
        directed_graph.add_edge(u, v)
    print(directed_graph)

    # 2. Breadth First Search (BFS)
    print("\n\n2. BREADTH FIRST SEARCH (BFS)")
    print("-" * 70)

    bfs_order, parent, distance = undirected_graph.breadth_first_search(1)
    print(f"BFS traversal starting from vertex 1: {bfs_order}")
    print(f"Distances from vertex 1: {distance}")
    print(f"Shortest path from 1 to 5: {undirected_graph.get_path(1, 5)}")

    # 3. Depth First Search (DFS)
    print("\n\n3. DEPTH FIRST SEARCH (DFS)")
    print("-" * 70)

    dfs_order = undirected_graph.depth_first_search(1)
    print(f"DFS traversal (iterative) starting from vertex 1: {dfs_order}")

    dfs_order_recursive = undirected_graph.dfs_recursive(1)
    print(f"DFS traversal (recursive) starting from vertex 1: {dfs_order_recursive}")

    # 4. Directed Acyclic Graph (DAG)
    print("\n\n4. DIRECTED ACYCLIC GRAPH (DAG)")
    print("-" * 70)

    dag = DirectedAcyclicGraph()

    print("\nAdding edges to DAG:")
    edges_to_add = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]
    for u, v in edges_to_add:
        success = dag.add_edge(u, v)
        print(
            f"  Adding edge {u} -> {v}: {'Success' if success else 'Failed (would create cycle)'}"
        )

    print(f"\nDAG structure:")
    print(dag)

    print(f"\nHas cycle: {dag.has_cycle()}")
    print(f"Topological sort: {dag.topological_sort()}")

    print("\nTrying to add an edge that would create a cycle:")
    success = dag.add_edge(5, 1)
    print(
        f"  Adding edge 5 -> 1: {'Success' if success else 'Failed (would create cycle)'}"
    )
    print(f"  Has cycle: {dag.has_cycle()}")

    # 5. Finding Reachable Vertices
    print("\n\n5. FINDING REACHABLE VERTICES")
    print("-" * 70)

    reachability = GraphReachability(directed_graph)

    print("\nDirected graph for reachability analysis:")
    print(directed_graph)

    print("\n\nReachable vertices from each vertex:")
    for vertex in sorted(directed_graph.vertices):
        reachable = reachability.get_reachable_from(vertex)
        print(f"  From vertex {vertex}: {sorted(reachable)}")

    print("\n\nChecking mutual reachability:")
    pairs_to_check = [(1, 3), (3, 5), (1, 5), (3, 4)]
    for u, v in pairs_to_check:
        result = reachability.are_mutually_reachable(u, v)
        print(f"  Vertices {u} and {v} are mutually reachable: {result}")

    print("\n\nAll mutually reachable pairs:")
    mutually_reachable = reachability.find_all_mutually_reachable_pairs()
    if mutually_reachable:
        for pair in mutually_reachable:
            print(f"  {pair}")
    else:
        print("  No mutually reachable pairs found")

    print("\n\nStrongly Connected Components:")
    sccs = reachability.find_strongly_connected_components()
    for i, scc in enumerate(sccs, 1):
        print(f"  Component {i}: {sorted(scc)}")

    # Additional example with strongly connected components
    print("\n\nExample with strongly connected components:")
    scc_graph = Graph(directed=True)
    edges = [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 6), (6, 4), (7, 6), (7, 8)]
    for u, v in edges:
        scc_graph.add_edge(u, v)

    print(scc_graph)

    scc_reachability = GraphReachability(scc_graph)
    sccs = scc_reachability.find_strongly_connected_components()
    print("\nStrongly Connected Components:")
    for i, scc in enumerate(sccs, 1):
        print(f"  Component {i}: {sorted(scc)}")

    print("\n" + "=" * 70)
    print("All graph algorithms completed successfully!")
    print("=" * 70)
