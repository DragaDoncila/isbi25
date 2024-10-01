import os
import numpy as np
import pandas as pd
from tracktour import load_tiff_frames
from utils import OUT_FILE, join_pth
import networkx as nx
import json
from ctc_metrics.scripts.evaluate import evaluate_sequence
from tifffile import imread
from tracktour._io_util import get_ctc_output
from tracktour.cli import _save_results


def get_remove_edges(solution_graph, u, v):
    # if it's a false positive OR a WS, we remove it
    # WS will get added back in later, because
    # we want to make sure we mark them as 
    # manual parent links if needed
    removers = [(u, v)]
    # check other successors for False Positives
    # here we either have a WS edge or a node with potentially multiple FP edges
    for dest in solution_graph.successors(u):
        edge = solution_graph.edges[u, dest]
        # if this edge was added during solving, it won't have this flag
        if edge.get('EdgeFlag.FALSE_POS', False) and (u,v) not in removers:
            removers.append((u,v))
    return removers

def get_new_node_info(dest, gt_graph):
    node_info = gt_graph.nodes[dest]
    copy_info = {}
    for k in node_copy_info:
        gt_k = node_copy_info[k]
        copy_info[k] = node_info[gt_k]
    copy_info['label'] = new_label
    return copy_info

def mask_new_node(dest, copy_info, n_digits, gt_path, gt_graph, original_seg, new_label):
    t = copy_info['t']
    frame_name = f'man_track{str(t).zfill(n_digits)}.tif'
    frame_path = os.path.join(gt_path, frame_name)
    frame = imread(frame_path)
    mask = frame == gt_graph.nodes[dest]['segmentation_id']
    original_seg[t][mask] = new_label

def add_new_vertex(dest, gt_graph, new_node_id, solution_graph, n_digits, gt_path, original_seg, new_label):
    copy_info = get_new_node_info(dest, gt_graph)
    to_add = [(new_node_id, copy_info)]
    sol_dest = new_node_id

    # add node to solution graph
    solution_graph.add_nodes_from(to_add)
    # mask segmentation
    mask_new_node(dest, copy_info, n_digits, gt_path, gt_graph, original_seg, new_label)
    return sol_dest

if __name__ == '__main__':

    # contains initial solutions generated and saved using the experiment schema
    root_pth = '/home/ddon0001/PhD/experiments/scaled/no_merges_all'
    ds_summary_path = os.path.join(root_pth, 'summary.csv')
    ds_info = pd.read_csv(ds_summary_path)[['ds_name', 'seg_path', 'tra_gt_path']]
    # path to csv generated using `ctc_evaluate` command 
    # with `--lnk --tra --det --bio --ct --tf --cca --bc` flags
    original_metrics_path = '/home/ddon0001/PhD/experiments/as_ctc/no_merges/no_merge_metrics.csv'
    original_metrics = pd.read_csv(original_metrics_path, sep=';')


    #####################################################################################
    # path to csv generated using "compute_edge_sampling_dfs.py"
    all_ds_path = os.path.join(root_pth, 'solution_edges_datasets_with_FP_WS_FA_FE.csv')
    out_root = '/home/ddon0001/PhD/experiments/error_sampling_ws_fa_fe'
    feature_of_interest = 'sensitivity_diff'
    ascending = True
    #####################################################################################

    all_ds = pd.read_csv(all_ds_path)
    ds_names = all_ds.ds_name.unique()
    for ds_name in ds_names:
        ds, seq = ds_name.split('_')
        out_ds = os.path.join(out_root, ds)
        out_res = os.path.join(out_ds, f'{seq}_RES')
        out_edge_csv_path = os.path.join(out_ds, f'{seq}_edges_{feature_of_interest}.csv')
        if os.path.exists(out_edge_csv_path):
            print(f'{out_edge_csv_path} already exists, skipping...')
            continue
        if 'H157' in ds_name:
            print(f'Skipping {ds_name}...')
            continue

        # select edges and sort by relevant feature
        ds_edges = all_ds[all_ds['ds_name'] == ds_name][[
            'ds_name',
            'u',
            'v',
            'feature_distance',
            'chosen_neighbour_rank',
            'prop_diff',
            'sensitivity_diff',
            'solution_correct',
            'solution_incorrect',
            'error_type'
        ]]
        # ds_edges = ds_edges.sort_values(by=feature_of_interest, ascending=ascending)
        total_errors = ds_edges['solution_incorrect'].sum()

        # load required dataset info
        ds_root_pth = os.path.join(root_pth, ds_name)
        solution_graph = nx.read_graphml(join_pth(ds_root_pth, OUT_FILE.MATCHED_SOL), node_type=int)
        gt_graph = nx.read_graphml(join_pth(ds_root_pth, OUT_FILE.MATCHED_GT))
        gt_path = ds_info[ds_info['ds_name'] == ds_name]['tra_gt_path'].values[0]

        with open(join_pth(ds_root_pth, OUT_FILE.MATCHING), 'r') as f:
            node_match = json.load(f)
        gt_to_sol = {item[0]: item[1] for item in node_match}
        sol_to_gt = {item[1]: item[0] for item in node_match}
        original_seg = load_tiff_frames(ds_info[ds_info['ds_name'] == ds_name]['seg_path'].values[0])

        # prepare information for copying nodes across
        node_copy_info = {}
        if len(original_seg.shape) == 4:
            node_copy_info['z'] = 'x'
            node_copy_info['y'] = 'y'
            node_copy_info['x'] = 'z'
            gt_loc = ['x', 'y', 'z']
        else:
            node_copy_info['y'] = 'x'
            node_copy_info['x'] = 'y'
            gt_loc = ['x', 'y']
        node_copy_info['t'] = 't'

        loc = gt_loc[::-1]
        new_label = original_seg.max() + 1
        new_node_id = solution_graph.number_of_nodes()
        n_digits = max(len(str(len(original_seg))), 3)

        ds_metrics = original_metrics[original_metrics['dataset'] == ds_name]
        original_lnk = ds_metrics['LNK'].values[0]
        original_bio = ds_metrics['BIO(0)'].values[0]
        original_ct = ds_metrics['CT'].values[0]
        original_tf = ds_metrics['TF'].values[0]
        original_cca = ds_metrics['CCA'].values[0]
        original_bc = ds_metrics['BC(0)'].values[0]
        original_tra = ds_metrics['TRA'].values[0]
        original_det = ds_metrics['DET'].values[0]

        ds_edges['LNK'] = -1.0
        ds_edges['BIO(0)'] = -1.0
        ds_edges['CT'] = -1.0
        ds_edges['TF'] = -1.0
        ds_edges['CCA'] = -1.0
        ds_edges['BC(0)'] = -1.0
        ds_edges['TRA'] = -1.0
        ds_edges['DET'] = -1.0

        ds_edges['presented_rank'] = -1
        resolved_since_last_eval = 0
        count_edges_presented = 0
        unpresented = set(ds_edges[ds_edges.presented_rank == -1].index)

        while len(unpresented):
            if ascending == False:
                row_idx = ds_edges[ds_edges.presented_rank == -1][feature_of_interest].idxmax()
            else:
                row_idx = ds_edges[ds_edges.presented_rank == -1][feature_of_interest].idxmin()
            row = ds_edges.loc[row_idx]
                
            is_correct = row.solution_correct
            u = row.u
            v = row.v
            if not is_correct:
                # this tracks number of errors resolved after looking at this edge
                # we might resolve more than one error per edge presented
                # but we'll never solve more errors than we originally had
                count_errors_handled = 0
                error_type = getattr(row, 'error_type', 'FP')

                if error_type == 'FA':
                    # this was an incorrect appearance, check for 'v''s
                    # predecessor
                    gt_v = sol_to_gt[v]
                    # this might be a faux FA we introduced, so we still check here
                    preds = list(gt_graph.predecessors(gt_v))
                    if len(preds):
                        gt_pred = preds[0]
                        gt_succs = list(gt_graph.successors(gt_pred))
                        if gt_pred in gt_to_sol:
                            sol_pred = gt_to_sol[gt_pred]
                        else:
                            sol_pred = add_new_vertex(gt_pred, gt_graph, new_node_id, solution_graph, n_digits, gt_path, original_seg, new_label)
                            gt_to_sol[gt_pred] = sol_pred
                            sol_to_gt[sol_pred] = gt_pred
                            new_node_id += 1
                            new_label += 1

                            # add implict FA edge into sol_pred
                            new_edge_info = {
                                'ds_name': ds_name,
                                'u': -2,
                                'v': sol_pred,
                                # max value makes sure this edge is sampled next
                                'feature_distance': np.inf,
                                'chosen_neighbour_rank': -1,
                                'prop_diff': 2,
                                # min value makes sure this edge is sampled next
                                'sensitivity_diff': 0,
                                # we pretend this edge is definitely wrong
                                # if it happens to be correct, it's just a no-op
                                'solution_correct': False,
                                'solution_incorrect': True,
                                'error_type': 'FA',
                                'LNK': -1.0,
                                'BIO(0)': -1.0,
                                'CT': -1.0,
                                'TF': -1.0,
                                'CCA': -1.0,
                                'BC(0)': -1.0,
                                'TRA': -1.0,
                                'DET': -1.0,
                                'presented_rank': -1,
                            }
                            ds_edges = pd.concat([ds_edges, pd.DataFrame(new_edge_info, index=[ds_edges.index.max() + 1])])
                        solution_graph.add_edge(sol_pred, v)
                        # this edge belongs to two different tracks
                        if len(gt_succs) == 1 and gt_graph.nodes[gt_pred]['segmentation_id'] != gt_graph.nodes[gt_v]['segmentation_id']:
                            solution_graph.edges[sol_pred, v]['manual_parent_link'] = True
                        count_errors_handled += 1
                # ws, fp or fe edges
                # may lead to removal of edges
                # will always repair sucessor(s) of u
                else:
                    # check that edge is still present
                    # could've removed in prior FP iteration
                    if solution_graph.has_edge(u, v):
                        to_remove = get_remove_edges(solution_graph, u, v)
                        solution_graph.remove_edges_from(to_remove)
                        count_errors_handled += len(to_remove)

                    # get correct destinations for u from gt
                    gt_u = sol_to_gt[u]
                    dests = list(gt_graph.successors(gt_u))
                    for dest in dests:
                        # destination exists, we just need to swap the edges
                        if dest in gt_to_sol:
                            sol_dest = gt_to_sol[dest]
                        # destination was a missing vertex
                        else:
                            sol_dest = add_new_vertex(dest, gt_graph, new_node_id, solution_graph, n_digits, gt_path, original_seg, new_label)
                            gt_to_sol[dest] = sol_dest
                            sol_to_gt[sol_dest] = dest
                            # increment information for next node to add    
                            new_label += 1
                            new_node_id += 1

                            # add implict FE edge from sol_dest
                            # because we've fixed incoming edge from u into sol_dest, but don't know where it's going
                            new_edge_info = {
                                'ds_name': ds_name,
                                'u': sol_dest,
                                'v': -4,
                                # max value makes sure this edge is sampled next
                                'feature_distance': np.inf,
                                'chosen_neighbour_rank': -1,
                                'prop_diff': 2,
                                # min value makes sure this edge is sampled next
                                'sensitivity_diff': 0,
                                # we pretend this edge is definitely wrong
                                # if it happens to be correct, it's just a no-op
                                'solution_correct': False,
                                'solution_incorrect': True,
                                'error_type': 'FE',
                                'LNK': -1.0,
                                'BIO(0)': -1.0,
                                'CT': -1.0,
                                'TF': -1.0,
                                'CCA': -1.0,
                                'BC(0)': -1.0,
                                'TRA': -1.0,
                                'DET': -1.0,
                                'presented_rank': -1,
                            }
                            ds_edges = pd.concat([ds_edges, pd.DataFrame(new_edge_info, index=[ds_edges.index.max() + 1])])


                        # add correct edge
                        if not solution_graph.has_edge(u, sol_dest):
                            solution_graph.add_edge(u, sol_dest)
                            # this edge belongs to two different tracks
                            if len(dests) == 1 and gt_graph.nodes[gt_u]['segmentation_id'] != gt_graph.nodes[dest]['segmentation_id']:
                                solution_graph.edges[u, sol_dest]['manual_parent_link'] = True
                            # this would've been a FN edge, so we just handled another error
                            count_errors_handled += 1
                    # we've repaired successors of u
                    # if gt_v's predecessor is not in gt_to_sol,
                    # we also add it here alongside its implicit FA edge
                    if error_type == 'FP' and len(preds := list(gt_graph.predecessors((gt_v := sol_to_gt[v])))) and (gt_pred := preds[0]) not in gt_to_sol:
                        # predecessor doesn't exist, needs adding
                        sol_pred = add_new_vertex(gt_pred, gt_graph, new_node_id, solution_graph, n_digits, gt_path, original_seg, new_label)
                        gt_to_sol[gt_pred] = sol_pred
                        sol_to_gt[sol_pred] = gt_pred
                        new_node_id += 1
                        new_label += 1

                        # add implict FA edge into sol_pred
                        new_edge_info = {
                            'ds_name': ds_name,
                            'u': -2,
                            'v': sol_pred,
                            # max value makes sure this edge is sampled next
                            'feature_distance': np.inf,
                            'chosen_neighbour_rank': -1,
                            'prop_diff': 2,
                            # min value makes sure this edge is sampled next
                            'sensitivity_diff': 0,
                            # we pretend this edge is definitely wrong
                            # if it happens to be correct, it's just a no-op
                            'solution_correct': False,
                            'solution_incorrect': True,
                            'error_type': 'FA',
                            'LNK': -1.0,
                            'BIO(0)': -1.0,
                            'CT': -1.0,
                            'TF': -1.0,
                            'CCA': -1.0,
                            'BC(0)': -1.0,
                            'TRA': -1.0,
                            'DET': -1.0,
                            'presented_rank': -1,
                        }
                        ds_edges = pd.concat([ds_edges, pd.DataFrame(new_edge_info, index=[ds_edges.index.max() + 1])])
                        solution_graph.add_edge(sol_pred, v)
                        # this edge belongs to two different tracks
                        if len(preds) == 1 and gt_graph.nodes[gt_pred]['segmentation_id'] != gt_graph.nodes[gt_v]['segmentation_id']:
                            solution_graph.edges[sol_pred, v]['manual_parent_link'] = True
                        count_errors_handled += 1

                # save CTC output, get new new_label based on actual segmentation
                original_seg, track_df, new_label = get_ctc_output(original_seg, solution_graph, 't', 'label', loc)
                _save_results(original_seg, track_df, out_res)
                new_label += 1
                resolved_since_last_eval += 1

                # still had wrong edges, have big number of errors, been 10 edges since we last evaluated or we are at the end
                if ((count_errors_handled > 0)\
                    and ((total_errors <= 20) or (resolved_since_last_eval >= 25)))\
                    or (count_edges_presented >= len(ds_edges) - 1):
                    # re-evaluate LNK and BIO
                    res_dict = evaluate_sequence(out_res, gt_path[:-4], ['DET', 'TRA', 'LNK', 'BIO', 'CT', 'TF', 'CCA', 'BC'])

                    # overwrite original metrics
                    original_lnk = res_dict['LNK']
                    original_bio = res_dict['BIO(0)']
                    original_ct = res_dict['CT']
                    original_tf = res_dict['TF']
                    original_cca = res_dict['CCA']
                    original_bc = res_dict['BC(0)']
                    original_tra = res_dict['TRA']
                    original_det = res_dict['DET']
                    resolved_since_last_eval = 0
            # edge was correct but we're at the last iteration
            elif count_edges_presented >= len(ds_edges) - 1:
                res_dict = evaluate_sequence(out_res, gt_path[:-4], ['DET', 'TRA', 'LNK', 'BIO', 'CT', 'TF', 'CCA', 'BC'])

                # overwrite original metrics
                original_lnk = res_dict['LNK']
                original_bio = res_dict['BIO(0)']
                original_ct = res_dict['CT']
                original_tf = res_dict['TF']
                original_cca = res_dict['CCA']
                original_bc = res_dict['BC(0)']
                original_tra = res_dict['TRA']
                original_det = res_dict['DET']


            # save into graph
            ds_edges.at[row_idx, 'LNK'] = original_lnk
            ds_edges.at[row_idx, 'BIO(0)'] = original_bio
            ds_edges.at[row_idx, 'CT'] = original_ct
            ds_edges.at[row_idx, 'TF'] = original_tf
            ds_edges.at[row_idx, 'CCA'] = original_cca
            ds_edges.at[row_idx, 'BC(0)'] = original_bc
            ds_edges.at[row_idx, 'TRA'] = original_tra
            ds_edges.at[row_idx, 'DET'] = original_det

            ds_edges.at[row_idx, 'presented_rank'] = count_edges_presented
            count_edges_presented += 1
            unpresented = set(ds_edges[ds_edges.presented_rank == -1].index)
        
        ds_edges.sort_values(by='presented_rank', ascending=True, inplace=True)
        ds_edges.to_csv(out_edge_csv_path, index=False)