from .tree import Tree
from .tree2d import Tree2D
from .tree3d import Tree3D


def _get_repeat_node_list(node_list, tree: Tree):
    for node in tree.children:
        if node is not None:
            _get_repeat_node_list(node_list, node)
        else:
            node_list.append(tree)


def get_node_list(tree: Tree):
    node_list = []
    _get_repeat_node_list(node_list, tree)
    if isinstance(tree, Tree2D):
        return node_list[::4]
    elif isinstance(tree, Tree3D):
        return node_list[::8]
    else:
        raise Exception(f"No node list for such tree.")


def update_id(tree: Tree):
    node_list = get_node_list(tree)
    for i, node in enumerate(node_list):
        node.id = i + 1


def get_max_level(tree: Tree):
    node_list = get_node_list(tree)
    level = 0
    for i, node in enumerate(node_list):
        if node.level > level:
            level = node.level
    return level