from enum import Enum
import json
import os
import networkx as nx
import pandas as pd

from tracktour._tracker import VirtualVertices


class OUT_FILE(Enum):
    # solutions
    TRACKED_DETECTIONS = 'tracked_detections.csv'
    TRACKED_EDGES = 'tracked_edges.csv'
    ALL_VERTICES = 'all_vertices.csv'
    ALL_EDGES = 'all_edges.csv'

    # models
    MODEL_LP = 'model.lp'
    MODEL_MPS = 'model.mps'
    MODEL_SOL = 'model.sol'

    # tracktour version
    TRACKTOUR_VERSION = 'version.txt'
    
    # timings
    TIMING = 'times.json'

    # config
    CONFIG = 'config.yaml'

    # metrics
    METRICS = 'metrics.json'
    MATCHING = 'matching.json'
    MATCHED_SOL = 'matched_solution.graphml'
    MATCHED_GT = 'matched_gt.graphml'

def load_sol_files(root_pth):
    all_vertices = pd.read_csv(join_pth(root_pth, OUT_FILE.ALL_VERTICES), index_col=0)
    if all_vertices.index.name == 't':
        all_vertices = all_vertices.reset_index()
    all_edges = pd.read_csv(join_pth(root_pth, OUT_FILE.ALL_EDGES))
    solution_graph = nx.read_graphml(join_pth(root_pth, OUT_FILE.MATCHED_SOL), node_type=int)
    gt_graph = nx.read_graphml(join_pth(root_pth, OUT_FILE.MATCHED_GT))

    with open(join_pth(root_pth, OUT_FILE.MATCHING), 'r') as f:
        node_match = json.load(f)
    gt_to_sol = {item[0]: item[1] for item in node_match}
    sol_to_gt = {item[1]: item[0] for item in node_match}
    return all_vertices, all_edges, solution_graph, gt_graph, gt_to_sol, sol_to_gt

def populate_target_label(all_edges, ground_truth, sol_to_gt):

    def is_edge_in_gt(edge):
        u, v = edge['u'], edge['v']
        # these two nodes exist in the ground truth
        # note this should be all nodes because we have no FPs
        if u in sol_to_gt and v in sol_to_gt:
            gt_u = sol_to_gt[u]
            gt_v = sol_to_gt[v]
            # if this edge is in GT, we mark it as correct
            return ground_truth.has_edge(gt_u, gt_v)
        # if real vertices aren't in GT we have a problem because we have no FPs
        if u >= 0 and v >= 0:
            raise ValueError(f'Node {u} in GT: {u in sol_to_gt}, Node {v} in GT: {v in sol_to_gt}')
        return False
    target_labels = all_edges[['u', 'v']].apply(is_edge_in_gt, axis=1)
    all_edges['oracle_is_correct'] = target_labels

def populate_solution_label_including_ws(all_edges, solution_graph,):
    def is_edge_tp(edge):
        u, v = int(edge['u']), int(edge['v'])
        if not solution_graph.has_edge(u, v):
            return 0
        return int(not solution_graph.edges[u, v]['EdgeFlag.FALSE_POS'] and not solution_graph.edges[u, v]['EdgeFlag.WRONG_SEMANTIC'])
    
    def get_error_cat(edge):
        u, v = int(edge['u']), int(edge['v'])
        if not solution_graph.has_edge(u, v):
            return 'None'
        elif solution_graph.edges[u, v]['EdgeFlag.FALSE_POS']:
            return 'FP'
        elif solution_graph.edges[u, v]['EdgeFlag.WRONG_SEMANTIC']:
            return 'WS'
        else:
            return 'Correct'
    
    all_edges['oracle_is_correct'] = all_edges.apply(is_edge_tp, axis=1)
    all_edges['error_type'] = all_edges.apply(get_error_cat, axis=1)

def populate_label_ws_enter_exit(all_edges, solution_graph, gt_graph, sol_to_gt):
    def is_edge_tp(edge):
        u, v = int(edge['u']), int(edge['v'])
        # appearance edge, correct if destination node in gt_graph
        # has incoming degree 0
        if u == VirtualVertices.APP.value:
            return int(gt_graph.in_degree(sol_to_gt[v]) == 0)
        # exit edge, correct if source node in gt_graph
        # has outgoing degree 0
        elif v == VirtualVertices.TARGET.value:
            return int(gt_graph.out_degree(sol_to_gt[u]) == 0)
        elif not solution_graph.has_edge(u, v):
            return 0
        return int(not solution_graph.edges[u, v]['EdgeFlag.FALSE_POS'] and not solution_graph.edges[u, v]['EdgeFlag.WRONG_SEMANTIC'])
    
    def get_error_cat(edge):
        u, v = int(edge['u']), int(edge['v'])
        if u == VirtualVertices.APP.value:
            if edge['flow'] <= 0:
                return 'None'
            if not edge['oracle_is_correct']:
                return 'FA'
            return 'Correct'
        if v == VirtualVertices.TARGET.value:
            if edge['flow'] <= 0:
                return 'None'
            if not edge['oracle_is_correct']:
                return 'FE'
            return 'Correct'
        if not solution_graph.has_edge(u, v):
            return 'None'
        if solution_graph.edges[u, v]['EdgeFlag.FALSE_POS']:
            return 'FP'
        if solution_graph.edges[u, v]['EdgeFlag.WRONG_SEMANTIC']:
            return 'WS'
        return 'Correct'

    all_edges['oracle_is_correct'] = all_edges.apply(is_edge_tp, axis=1)
    all_edges['error_type'] = all_edges.apply(get_error_cat, axis=1)

def join_pth(root: str, suffix: OUT_FILE) -> str:
    return os.path.join(root, suffix.value)