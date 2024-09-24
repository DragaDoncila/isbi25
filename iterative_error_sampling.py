import os
import pandas as pd
from tracktour import load_tiff_frames
from utils import OUT_FILE, join_pth
import networkx as nx
import json
from ctc_metrics.scripts.evaluate import evaluate_sequence
from tifffile import imread
from tracktour._io_util import get_ctc_output
from tracktour.cli import _save_results


# contains initial solutions generated and saved using the experiment schema
root_pth = '/home/ddon0001/PhD/experiments/scaled/no_merges_all'
ds_summary_path = os.path.join(root_pth, 'summary.csv')
ds_info = pd.read_csv(ds_summary_path)[['ds_name', 'seg_path', 'tra_gt_path']]
# path to csv generated using `ctc_evaluate` command 
# with `--lnk --bio --ct --tf --cca --bc` flags
original_metrics_path = '/home/ddon0001/PhD/experiments/as_ctc/no_merges/no_merge_metrics.csv'
original_metrics = pd.read_csv(original_metrics_path, sep=';')


#####################################################################################
# path to csv generated using "compute_edge_sampling_dfs.py"
all_ds_path = os.path.join(root_pth, 'solution_edges_datasets_with_FP_WS.csv')
out_root = '/home/ddon0001/PhD/experiments/error_sampling_ws'
feature_of_interest = 'sensitivity_diff'
ascending = True
#####################################################################################

all_ds = pd.read_csv(all_ds_path)
all_ds = all_ds[all_ds.ds_name == 'Fluo-N2DH-SIM+_02']
ds_names = all_ds.ds_name.unique()
for ds_name in ds_names:
    edge_count = 0
    error_count = 0

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
        'distance',
        'chosen_neighbour_rank',
        'prop_diff',
        'sensitivity_diff',
        'solution_correct',
        'solution_incorrect'
    ]]
    ds_edges = ds_edges.sort_values(by=feature_of_interest, ascending=ascending)
    total_errors = ds_edges['solution_incorrect'].sum()
    total_edges = len(ds_edges)

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
    # node_copy_info['label'] = 'segmentation_id'
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

    ds_edges['LNK'] = -1.0
    ds_edges['BIO(0)'] = -1.0
    ds_edges['CT'] = -1.0
    ds_edges['TF'] = -1.0
    ds_edges['CCA'] = -1.0
    ds_edges['BC(0)'] = -1.0

    for row in ds_edges.itertuples():
        is_correct = row.solution_correct
        u = row.u
        v = row.v
        if not is_correct:
            # this tracks number of errors resolved after looking at this edge
            # we might resolve more than one error per edge presented
            # but we'll never solve more errors than we originally had
            # because the only "counted" errors are WS edges or FP edges
            count_errors_handled = 0
            # check that we haven't already resolved this error
            if solution_graph.has_edge(u, v):
                count_errors_handled += 1
                # if it's a false positive, we remove it
                if solution_graph.edges[u, v]['EdgeFlag.FALSE_POS']:
                    # remove wrong edge
                    solution_graph.remove_edge(u, v)
                # check other successors for False Positives
                # here we either have a WS edge or a node with potentially multiple FP edges
                removers = []
                for dest in solution_graph.successors(u):
                    edge = solution_graph.edges[u, dest]
                    # if this edge was added during solving, it won't have this flag
                    if edge.get('EdgeFlag.FALSE_POS', False):
                        removers.append((u,v))
                        count_errors_handled += 1
                solution_graph.remove_edges_from(removers)

            # get correct destinations for u from gt
            gt_u = sol_to_gt[u]
            dests = list(gt_graph.successors(gt_u))
            for dest in dests:
                # destination exists, we just need to swap the edges
                if dest in gt_to_sol:
                    sol_dest = gt_to_sol[dest]
                # destination was a missing vertex
                else:
                    # determine node information to copy over
                    node_info = gt_graph.nodes[dest]
                    copy_info = {}
                    for k in node_copy_info:
                        gt_k = node_copy_info[k]
                        copy_info[k] = node_info[gt_k]
                    copy_info['label'] = new_label
                    to_add = [(new_node_id, copy_info)]
                    sol_dest = new_node_id

                    # add node to solution graph
                    solution_graph.add_nodes_from(to_add)

                    # mask segmentation
                    t = copy_info['t']
                    frame_name = f'man_track{str(t).zfill(n_digits)}.tif'
                    frame_path = os.path.join(gt_path, frame_name)
                    frame = imread(frame_path)
                    mask = frame == node_info['segmentation_id']
                    original_seg[t][mask] = new_label

                    # increment information for next node to add    
                    new_label += 1
                    new_node_id += 1

                # add correct edge
                if not solution_graph.has_edge(u, sol_dest):
                    solution_graph.add_edge(u, sol_dest)

            # save CTC output, get new new_label based on actual segmentation
            original_seg, track_df, new_label = get_ctc_output(original_seg, solution_graph, 't', 'label', loc)
            _save_results(original_seg, track_df, out_res)
            new_label += 1

            # still had wrong edges, have big number of errors, been 10 edges since we last evaluated or we are at the end
            if (error_count <= total_errors)\
                and ((total_errors <= 20)\
                     or (edge_count % 10 == 0)\
                        or (edge_count == total_edges - 1)):
                # re-evaluate LNK and BIO
                res_dict = evaluate_sequence(out_res, gt_path[:-4], ['LNK', 'BIO', 'CT', 'TF', 'CCA', 'BC'])

                # overwrite original metrics
                original_lnk = res_dict['LNK']
                original_bio = res_dict['BIO(0)']
                original_ct = res_dict['CT']
                original_tf = res_dict['TF']
                original_cca = res_dict['CCA']
                original_bc = res_dict['BC(0)']

            edge_count += 1
            # might be incrementing by 0 if the current edge was already resolved
            error_count += count_errors_handled

        # save into graph
        ds_edges.at[row.Index, 'LNK'] = original_lnk
        ds_edges.at[row.Index, 'BIO(0)'] = original_bio
        ds_edges.at[row.Index, 'CT'] = original_ct
        ds_edges.at[row.Index, 'TF'] = original_tf
        ds_edges.at[row.Index, 'CCA'] = original_cca
        ds_edges.at[row.Index, 'BC(0)'] = original_bc
    ds_edges.to_csv(out_edge_csv_path, index=False)
