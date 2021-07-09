# ----------------------------------------------------------------------
# Created: tis apr 20 10:06:05 2021 (+0200)
# Last-Updated:
# Filename: compare.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
from copy import deepcopy
from pprint import pprint
import numpy as np
import uuid
import matplotlib.pyplot as plt

from alex.alex import dsl_parser, core, const

# FIXME: POC
delete_cost = lambda x: 1
insert_cost = lambda x: 1
update_cost = lambda x, y: 1 if x!=y else 0
match_cost = lambda x: 0

DELETE = lambda x: ["DELETE", x, ""]
INSERT = lambda x: ["INSERT", "", x]
UPDATE = lambda x, y: ["UPDATE", x, y]
MATCH = lambda x, y: ["MATCH", x, y]


def get_lmds(tree):
    return list(map(lambda x: tree[x]["lmd"], tree))


def get_keyroots(tree):
    lmds = list(set(get_lmds(tree)))
    keyroots = {x: x for x in lmds}
    for node in tree:
        keyroots[tree[node]["lmd"]] = node
    return sorted(core.name_to_index(list(keyroots.values()), tree))


def ted(t1, t2):

    Al = core.name_to_index(get_lmds(t1), t1)
    Bl = core.name_to_index(get_lmds(t2), t2)
    An = core.get_labels(t1)
    Bn = core.get_labels(t2)

    Anames = list(t1.keys())
    Bnames = list(t2.keys())

    cost_matrix = np.zeros((len(An), len(Bn)))
    operations = [[[] for _ in range(len(Bn))] for _ in range(len(An))]

    keyroots1 = get_keyroots(t1)
    keyroots2 = get_keyroots(t2)
    for i in keyroots1:
        for j in keyroots2:
            _distance(i, j, Al, Bl, An, Bn, Anames, Bnames, cost_matrix, operations)
    return cost_matrix[-1,-1], operations[-1][-1]


def annotate_ops(operations):
    annotation = {1: dict(),
                  2: dict()}
    for operation in operations:
        node_name = operation[0]
        op = operation[1][0]
        if op == "DELETE":
            annotation[1][node_name] = "red"
        elif op == "INSERT":
            annotation[2][node_name] = "green"
        elif op == "UPDATE":
            annotation[1][node_name[0]] = "orange"
            annotation[2][node_name[1]] = "orange"
    return annotation


def label_to_value(label):
    return label.split("#")[0]


def _diff_graph_list(n1, n2):
    tree1 = core.alex_graph_to_tree(n1, exclude_types=["hyperparam"], naive=False)
    tree2 = core.alex_graph_to_tree(n2, exclude_types=["hyperparam"], naive=False)
    tree_full1 = core.alex_graph_to_tree(n1, naive=False)
    tree_full2 = core.alex_graph_to_tree(n2, naive=False)
    cost, _operations = ted(tree1, tree2)
    operations = []
    for operation in _operations:
        if operation[1][1] == "root":
            break
        op = operation[1][0]
        if op == "INSERT":
            operations.append(operation)
            label2 = operation[1][2]
            if core.get_value_type(label_to_value(label2)) == "ingredient":
                affected_hyperparam_subtree = tree_full2[operation[0]]["descendants"]
                for child in affected_hyperparam_subtree:
                    cost += 1 # FIXME
                    operations.append((child, ["INSERT", "", tree_full2[child]["label"]]))
        elif op == "DELETE":
            operations.append(operation)
            label1 = operation[1][1]
            if core.get_value_type(label_to_value(label1)) == "ingredient":
                affected_hyperparam_subtree = tree_full1[operation[0]]["descendants"]
                for child in affected_hyperparam_subtree:
                    cost += 1 # FIXME
                    operations.append((child, ["DELETE", tree_full1[child]["label"], ""]))
        else:
            name1 = operation[0][0]
            name2 = operation[0][1]
            label1 = operation[1][1]
            label2 = operation[1][2]

            if label1 != label2:
                if label_to_value(label1) == label_to_value(label2):
                    subtree1 = core.get_subtree(tree_full1, name1)
                    subtree2 = core.get_subtree(tree_full2, name2)
                    _cost, __operations = ted(subtree1,
                                              subtree2)
                    cost += _cost
                    operations += __operations[:-1]
                else:
                    operations.append((name1, ("DELETE", label1, "")))
                    operations.append((name2, ("INSERT", "", label2)))
                    if core.get_value_type(label_to_value(label1)) == "ingredient":

                        affected_hyperparam_subtree = tree_full1[name1]["descendants"]
                        for child in affected_hyperparam_subtree:
                            cost += 1 # FIXME
                            _operation = (child, ["DELETE", tree_full1[child]["label"], ""])
                            operations.append(_operation)
                    if core.get_value_type(label_to_value(label2)) == "ingredient":
                        affected_hyperparam_subtree = tree_full2[name2]["descendants"]
                        for child in affected_hyperparam_subtree:
                            cost += 1 # FIXME
                            _operation = (child, ["INSERT", "", tree_full2[child]["label"]])
                            operations.append(_operation)

    return cost, operations


def _distance(i, j, Al, Bl, An, Bn, Anames, Bnames, cost_matrix, operations):
    """ Disclaimer: the _distance function is modified from the equivalent
            method implemented in the python library zss:
            https://github.com/timtadh/zhang-shasha/blob/master/zss/compare.py
    """
    m = i - Al[i] + 2
    n = j - Bl[j] + 2

    fd = np.zeros((m, n))
    partial_ops = [[[] for _ in range(n)] for _ in range(m)]

    ioff = Al[i] - 1
    joff = Bl[j] - 1

    for x in range(1, m): # δ(l(i1)..i, θ) = δ(l(1i)..1-1, θ) + γ(v → λ)
        node = An[x+ioff]
        nameA = Anames[x+ioff]
        fd[x, 0] = fd[x-1, 0] + delete_cost(node)
        op = (nameA, DELETE(node))
        partial_ops[x][0] = partial_ops[x-1][0] + [op]
    for y in range(1, n): # δ(θ, l(j1)..j) = δ(θ, l(j1)..j-1) + γ(λ → w)
        node = Bn[y+joff]
        nameB = Bnames[y+joff]
        fd[0,y] = fd[0,y-1] + insert_cost(node)
        op = (nameB, INSERT(node))
        partial_ops[0][y] = partial_ops[0][y-1] + [op]

    for x in range(1, m):  # the plus one is for the xrange impl
        for y in range(1, n):
            # x+ioff in the fd table corresponds to the same node as x in
            # the cost_matrix table (same for y and y+joff)
            node1 = An[x+ioff]
            node2 = Bn[y+joff]
            node_name1 = Anames[x+ioff]
            node_name2 = Bnames[y+joff]
            # only need to check if x is an ancestor of i
            # and y is an ancestor of j

            if Al[i] == Al[x+ioff] and Bl[j] == Bl[y+joff]:
                #                   +-
                #                   | δ(l(i1)..i-1, l(j1)..j) + γ(v → λ)
                # δ(F1 , F2 ) = min-+ δ(l(i1)..i , l(j1)..j-1) + γ(λ → w)
                #                   | δ(l(i1)..i-1, l(j1)..j-1) + γ(v → w)
                #                   +-
                costs = [fd[x-1, y] + delete_cost(node1),
                         fd[x,y-1] + insert_cost(node2),
                         fd[x-1,y-1] + update_cost(node1, node2)]
                fd[x,y] = min(costs)
                min_index = costs.index(fd[x,y])
                if min_index == 0:
                    op = (node_name1, DELETE(node1))
                    partial_ops[x][y] = partial_ops[x-1][y] + [op]
                elif min_index == 1:
                    op = (node_name2, INSERT(node2))
                    partial_ops[x][y] = partial_ops[x][y - 1] + [op]
                else:
                    op_type = MATCH if fd[x,y] == fd[x-1,y-1] else UPDATE
                    op = ((node_name1, node_name2), op_type(node1, node2))

                    partial_ops[x][y] = partial_ops[x - 1][y - 1] + [op]

                operations[x + ioff][y + joff] = partial_ops[x][y]
                cost_matrix[x+ioff,y+joff] = fd[x,y]
            else:
                #                   +-
                #                   | δ(l(i1)..i-1, l(j1)..j) + γ(v → λ)
                # δ(F1 , F2 ) = min-+ δ(l(i1)..i , l(j1)..j-1) + γ(λ → w)
                #                   | δ(l(i1)..l(i)-1, l(j1)..l(j)-1)
                #                   |                     + treedist(i1,j1)
                #                   +-
                p = Al[x+ioff]-1-ioff
                q = Bl[y+joff]-1-joff
                costs = [fd[x-1,y] + delete_cost(node1),
                         fd[x,y-1] + insert_cost(node2),
                         fd[p,q] + cost_matrix[x+ioff,y+joff]]
                fd[x,y] = min(costs)
                min_index = costs.index(fd[x,y])
                if min_index == 0:
                    op = (node_name1, DELETE(node1))
                    partial_ops[x][y] = partial_ops[x-1][y] + [op]
                elif min_index == 1:
                    op = (node_name2, INSERT(node2))
                    partial_ops[x][y] = partial_ops[x][y-1] + [op]
                else:
                    partial_ops[x][y] = partial_ops[p][q] + \
                        operations[x+ioff][y+joff]


def _dist_graph_list(tree1, tree2, exclude_types=[], render_to=None, dpi=800):
    cost, operations = ted(tree1, tree2)
    if render_to is not None:
        print("Done computing distance. Rendering image")
        annotation = annotate_ops(operations)
        img1 = core.draw(tree1, None, annotation[1], dpi=dpi)
        img2 = core.draw(tree2, None, annotation[2], dpi=dpi)
        fig, axs = plt.subplots(1, 2, dpi=dpi)
        axs[0].imshow(img1)
        axs[0].axis("off")
        axs[1].imshow(img2)
        axs[1].axis("off")
        fig.tight_layout()
        if render_to != "":
            fig.savefig(render_to, dpi="figure")
            print("Distance images written to: %s" % render_to)
        else:
            print("Distance images rendered to screen")
            plt.show()
    return cost, operations


from alex.annotators import param_count

def diff(network_config_1,
         network_config_2,
         render_to=None,
         dpi=800):
    graph_list1 = dsl_parser.parse(network_config_1)
    graph_list2 = dsl_parser.parse(network_config_2)
    # Do not include dimensionality in diff
    # graph1 = dsl_parser.list_to_graph(graph_list1)
    # graph2 = dsl_parser.list_to_graph(graph_list2)
    # tree_full1 = core.alex_graph_to_tree(graph1, naive=False)
    # tree_full2 = core.alex_graph_to_tree(graph2, naive=False)
    # Include dimensionality in diff
    tree_full1 = param_count.ParamCount(network_config_1).annotate_tree()
    tree_full2 = param_count.ParamCount(network_config_2).annotate_tree()

    operations = []
    cost = 0
    for block in ["data", "model", "loss", "optimizer"]:

        graph_list1_block = list(filter(lambda x: x["meta"]["block"]==block, graph_list1))
        graph_list2_block = list(filter(lambda x: x["meta"]["block"]==block, graph_list2))
        graph1_block = dsl_parser.list_to_graph(graph_list1_block)
        graph2_block = dsl_parser.list_to_graph(graph_list2_block)
        _cost, _operations = _diff_graph_list(graph1_block,
                                              graph2_block)
        cost += _cost
        operations += _operations

    if render_to is not None:
        print("Done computing diff. Rendering image")
        annotation = annotate_ops(operations)
        img1 = core.draw(tree_full1, None, annotation[1], dpi=dpi)
        img2 = core.draw(tree_full2, None, annotation[2], dpi=dpi)
        fig, axs = plt.subplots(1, 2, dpi=dpi)
        axs[0].imshow(img1)
        axs[0].axis("off")
        axs[1].imshow(img2)
        axs[1].axis("off")
        fig.tight_layout()
        if render_to != "":
            fig.savefig(render_to, dpi="figure")
            print("Diff images written to: %s" % render_to)
        else:
            print("Diff images rendered to screen")
            plt.show()

    return cost, operations


def dist(network_config_1,
         network_config_2,
         render_to=None,
         exclude_types=[],
         dpi=800):
    graph_list1 = dsl_parser.parse(network_config_1)
    graph_list2 = dsl_parser.parse(network_config_2)
    # Do not include dimensionality in dist
    # graph1 = dsl_parser.list_to_graph(graph_list1)
    # graph2 = dsl_parser.list_to_graph(graph_list2)
    # tree1 = core.alex_graph_to_tree(graph1, exclude_types=exclude_types)
    # tree2 = core.alex_graph_to_tree(graph2, exclude_types=exclude_types)
    # Include dimensionality in dist
    tree1 = param_count.ParamCount(graph_list1).annotate_tree()
    tree2 = param_count.ParamCount(graph_list2).annotate_tree()

    cost, operations = _dist_graph_list(tree1,
                                        tree2,
                                        exclude_types=exclude_types,
                                        render_to=render_to,
                                        dpi=dpi)
    return cost, operations


# ---------------------------------------------------------------------------- #
# Replace subtree
# ---------------------------------------------------------------------------- #
# - _, operation = dist(network1, network2, exclude_types=["hyperparam"])
# Find all matched ingredients
# Load checkpoint
def matched_ingredients(network_config_1,
                        network_config_2,
                        render_to=None,
                        dpi=800):
    if isinstance(network_config_1, list):
        graph_list1 = deepcopy(network_config_1)
    else:
        graph_list1 = dsl_parser.parse(network_config_1)
    if isinstance(network_config_1, list):
        graph_list2 = deepcopy(network_config_2)
    else:
        graph_list2 = dsl_parser.parse(network_config_2)
    # n1 = dsl_parser.list_to_graph(graph_list1)
    # n2 = dsl_parser.list_to_graph(graph_list2)
    # tree1 = core.alex_graph_to_tree(n1, exclude_types=["hyperparam"], naive=False)
    # tree2 = core.alex_graph_to_tree(n2, exclude_types=["hyperparam"], naive=False)

    tree1 = param_count.ParamCount(graph_list1,
                                   naive=False).annotate_tree()
    tree2 = param_count.ParamCount(graph_list2,
                                   naive=False).annotate_tree()
    _, operations = ted(tree1, tree2)
    # pprint(operations)
    if render_to is not None:
        print("Done computing diff. Rendering image")
        annotation = annotate_ops(operations)
        img1 = core.draw(tree1, None, annotation[1], dpi=dpi)
        img2 = core.draw(tree2, None, annotation[2], dpi=dpi)
        fig, axs = plt.subplots(1, 2, dpi=dpi)
        axs[0].imshow(img1)
        axs[0].axis("off")
        axs[1].imshow(img2)
        axs[1].axis("off")
        fig.tight_layout()
        if render_to != "":
            fig.savefig(render_to, dpi="figure")
            print("Diff images written to: %s" % render_to)
        else:
            print("Diff images rendered to screen")
            plt.show()
    matched = {}
    for operation in operations:
        if operation[1][0] == "MATCH":
            component_type = label_to_value(operation[1][1])
            if component_type in const.PARAMS:
                matched = {**matched,
                           **{operation[0][0]:
                              operation[0][1]}}
    return matched
