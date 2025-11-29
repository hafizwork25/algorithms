class Node:
    """
    Node class representing each node in the BST
    """

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None


class BinarySearchTree:
    """
    Binary Search Tree implementation

    Properties:
    - Each node's key is greater than all keys in its left subtree
    - Each node's key is smaller than all keys in its right subtree
    """

    def __init__(self):
        self.root = None

    def insert(self, key):
        """
        Insert a new key into the BST
        Time Complexity: O(h) where h is the height of the tree
        """
        new_node = Node(key)

        if self.root is None:
            self.root = new_node
            return

        current = self.root
        while True:
            if key < current.key:
                if current.left is None:
                    current.left = new_node
                    new_node.parent = current
                    break
                current = current.left
            else:
                if current.right is None:
                    current.right = new_node
                    new_node.parent = current
                    break
                current = current.right

    def find_minimum(self, node=None):
        """
        Find the minimum value in the tree or subtree
        The minimum is the leftmost node
        Time Complexity: O(h)
        """
        if node is None:
            node = self.root

        if node is None:
            return None

        while node.left is not None:
            node = node.left

        return node

    def find_maximum(self, node=None):
        """
        Find the maximum value in the tree or subtree
        The maximum is the rightmost node
        Time Complexity: O(h)
        """
        if node is None:
            node = self.root

        if node is None:
            return None

        while node.right is not None:
            node = node.right

        return node

    def inorder_traversal(self, node=None):
        """
        In-order tree walk (Left -> Root -> Right)
        Returns all values in sorted order
        Time Complexity: O(n)
        """
        if node is None:
            node = self.root

        result = []

        def inorder_helper(node):
            if node is not None:
                inorder_helper(node.left)
                result.append(node.key)
                inorder_helper(node.right)

        inorder_helper(node)
        return result

    def find_successor(self, node):
        """
        Find the successor of a given node
        Successor is the node with the smallest key greater than the given node's key
        Time Complexity: O(h)

        Cases:
        1. If node has a right subtree, successor is the minimum in right subtree
        2. If node has no right subtree, go up until we find a node that is a left child
        """
        if node is None:
            return None

        # Case 1: Node has a right subtree
        if node.right is not None:
            return self.find_minimum(node.right)

        # Case 2: No right subtree, go up to find the successor
        parent = node.parent
        while parent is not None and node == parent.right:
            node = parent
            parent = parent.parent

        return parent

    def find_predecessor(self, node):
        """
        Find the predecessor of a given node
        Predecessor is the node with the largest key smaller than the given node's key
        Time Complexity: O(h)
        """
        if node is None:
            return None

        # Case 1: Node has a left subtree
        if node.left is not None:
            return self.find_maximum(node.left)

        # Case 2: No left subtree, go up to find the predecessor
        parent = node.parent
        while parent is not None and node == parent.left:
            node = parent
            parent = parent.parent

        return parent

    def search(self, key, node=None):
        """
        Search for a key in the BST
        Time Complexity: O(h)
        """
        if node is None:
            node = self.root

        while node is not None:
            if key == node.key:
                return node
            elif key < node.key:
                node = node.left
            else:
                node = node.right

        return None

    def delete(self, key):
        """
        Delete a node with the given key from the BST
        Time Complexity: O(h)
        """
        node = self.search(key)

        if node is None:
            return False

        # Case 1: Node has no left child
        if node.left is None:
            self._transplant(node, node.right)
        # Case 2: Node has no right child
        elif node.right is None:
            self._transplant(node, node.left)
        # Case 3: Node has two children
        else:
            # Find successor (minimum in right subtree)
            successor = self.find_minimum(node.right)

            if successor.parent != node:
                # Replace successor with its right child
                self._transplant(successor, successor.right)
                successor.right = node.right
                successor.right.parent = successor

            # Replace node with successor
            self._transplant(node, successor)
            successor.left = node.left
            successor.left.parent = successor

        return True

    def _transplant(self, u, v):
        """
        Helper method to replace subtree rooted at node u with subtree rooted at node v
        """
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v

        if v is not None:
            v.parent = u.parent

    def height(self, node=None):
        """
        Calculate the height of the tree
        Time Complexity: O(n)
        """
        if node is None:
            node = self.root

        if node is None:
            return -1

        left_height = self.height(node.left)
        right_height = self.height(node.right)

        return 1 + max(left_height, right_height)

    def preorder_traversal(self, node=None):
        """
        Pre-order tree walk (Root -> Left -> Right)
        Time Complexity: O(n)
        """
        if node is None:
            node = self.root

        result = []

        def preorder_helper(node):
            if node is not None:
                result.append(node.key)
                preorder_helper(node.left)
                preorder_helper(node.right)

        preorder_helper(node)
        return result

    def postorder_traversal(self, node=None):
        """
        Post-order tree walk (Left -> Right -> Root)
        Time Complexity: O(n)
        """
        if node is None:
            node = self.root

        result = []

        def postorder_helper(node):
            if node is not None:
                postorder_helper(node.left)
                postorder_helper(node.right)
                result.append(node.key)

        postorder_helper(node)
        return result

    def is_valid_bst(self, node=None, min_val=float("-inf"), max_val=float("inf")):
        """
        Validate if the tree is a valid BST
        Time Complexity: O(n)
        """
        if node is None:
            node = self.root

        if node is None:
            return True

        if node.key <= min_val or node.key >= max_val:
            return False

        return self.is_valid_bst(node.left, min_val, node.key) and self.is_valid_bst(
            node.right, node.key, max_val
        )


if __name__ == "__main__":
    # Create a BST and test all operations
    bst = BinarySearchTree()

    print("=" * 60)
    print("BINARY SEARCH TREE IMPLEMENTATION")
    print("=" * 60)

    # Insert values
    values = [50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 65]
    print("\n1. Inserting values:", values)
    for val in values:
        bst.insert(val)

    # Find minimum and maximum
    print("\n2. Finding minimum and maximum BEFORE inserting new values:")
    min_node = bst.find_minimum()
    max_node = bst.find_maximum()
    print(f"   Minimum value: {min_node.key if min_node else None}")
    print(f"   Maximum value: {max_node.key if max_node else None}")

    # Insert new values
    print("\n3. Inserting new values: [5, 90]")
    bst.insert(5)
    bst.insert(90)

    print("\n4. Finding minimum and maximum AFTER inserting new values:")
    min_node = bst.find_minimum()
    max_node = bst.find_maximum()
    print(f"   Minimum value: {min_node.key if min_node else None}")
    print(f"   Maximum value: {max_node.key if max_node else None}")

    # In-order traversal (sorted order)
    print("\n5. In-order traversal (sorted values):")
    print(f"   {bst.inorder_traversal()}")

    # Find successor
    print("\n6. Finding successors:")
    test_keys = [20, 30, 65, 80]
    for key in test_keys:
        node = bst.search(key)
        if node:
            successor = bst.find_successor(node)
            print(
                f"   Successor of {key}: {successor.key if successor else 'None (largest element)'}"
            )

    # Find predecessor
    print("\n7. Finding predecessors:")
    for key in test_keys:
        node = bst.search(key)
        if node:
            predecessor = bst.find_predecessor(node)
            print(
                f"   Predecessor of {key}: {predecessor.key if predecessor else 'None (smallest element)'}"
            )

    # Search
    print("\n8. Searching for values:")
    search_keys = [40, 100]
    for key in search_keys:
        result = bst.search(key)
        print(f"   Search for {key}: {'Found' if result else 'Not found'}")

    # Tree properties
    print("\n9. Tree properties:")
    print(f"   Height: {bst.height()}")
    print(f"   Is valid BST: {bst.is_valid_bst()}")

    # Other traversals
    print("\n10. Other traversal methods:")
    print(f"    Pre-order: {bst.preorder_traversal()}")
    print(f"    Post-order: {bst.postorder_traversal()}")

    # Delete operation
    print("\n11. Deleting node with key 30:")
    bst.delete(30)
    print(f"    In-order after deletion: {bst.inorder_traversal()}")

    print("\n" + "=" * 60)
    print("All BST operations completed successfully!")
    print("=" * 60)
