class HpoEnsmallen:
    """A class to keep track of things like information content and maps for doing sem sim
    """

    def __init__(self, g):
        self.graph = g

    # get all descendents
    def get_descendents(self, hpo_term) -> list:
        ancestors = []

        # if we're a leaf:
        if (not self.graph.get_node_neighbours_name_by_node_name(hpo_term)):
            return ancestors
        # if (root->data == target)
        #     return true;
        # /* If target is present in either left or right subtree of this node, then print this node */
        # if ( printAncestors(root->left, target) ||
        #     printAncestors(root->right, target) )
        # {
        #     cout << root->data << " ";
        #     return true;
        # }
        # /* Else return false */
        # return false;
        # }

        return ancestors

    # get all ancestors
    def get_ancestors(self, hpo_term) -> list:
        ancestors = []
        return ancestors
