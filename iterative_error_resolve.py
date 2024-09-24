import os
import pandas as pd
from tracktour import Tracker, load_tiff_frames
from tracktour._io_util import read_scale
from utils import OUT_FILE, join_pth
import networkx as nx
import json
import numpy as np
from ctc_metrics.scripts.evaluate import evaluate_sequence
from tifffile import imread
from tracktour._io_util import get_ctc_output
from tracktour.cli import _save_results


root_pth = '/home/ddon0001/PhD/experiments/scaled/no_merges_all'
ds_summary_path = os.path.join(root_pth, 'summary.csv')
ds_info = pd.read_csv(ds_summary_path)[['ds_name', 'im_shape', 'seg_path', 'tra_gt_path']]
all_ds_path = os.path.join(root_pth, 'solution_edges_datasets_with_FP.csv')
all_ds = pd.read_csv(all_ds_path)

original_metrics_path = '/home/ddon0001/PhD/experiments/as_ctc/no_merges/no_merge_metrics.csv'
original_metrics = pd.read_csv(original_metrics_path, sep=';')

out_root = '/home/ddon0001/PhD/experiments/resolve_sampling'

ds_names = all_ds.ds_name.unique()

feature_of_interest = 'sensitivity_diff'
ascending = True

for ds_name in ds_names:
    edge_count = 0
    fp_edge_count = 0

    ds, seq = ds_name.split('_')
    out_ds = os.path.join(out_root, ds)
    out_res = os.path.join(out_ds, f'{seq}_RES')
    out_edge_csv_path = os.path.join(out_ds, f'{seq}_edges_{feature_of_interest}.csv')
    out_det_path = os.path.join(out_ds, f'{seq}_detections.csv')
    out_all_v_path = os.path.join(out_ds, f'{seq}_all_vertices.csv')
    out_all_e_path = os.path.join(out_ds, f'{seq}_all_edges.csv')

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
    total_fps = ds_edges['solution_incorrect'].sum()
    total_edges = len(ds_edges)

    # load required dataset info
    ds_root_pth = os.path.join(root_pth, ds_name)
    solution_graph = nx.read_graphml(join_pth(ds_root_pth, OUT_FILE.MATCHED_SOL), node_type=int)
    gt_graph = nx.read_graphml(join_pth(ds_root_pth, OUT_FILE.MATCHED_GT))
    gt_path = ds_info[ds_info['ds_name'] == ds_name]['tra_gt_path'].values[0]
    ds_scale = read_scale(ds_name)
    im_shape = eval(ds_info[ds_info['ds_name'] == ds_name]['im_shape'].values[0])

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

    added_vertices = []
    added_edges = []
    removed_edges = []

    def resolve(
        ds_root_pth,
        solution_graph,
        added_vertices,
        added_edges,
        removed_edges,
        out_all_v_path,
        out_all_e_path,
    ):

        if os.path.exists(out_all_e_path):
            first_run = False
        else:
            first_run = True
            
        tracker = Tracker(im_shape=im_shape, scale=ds_scale)
        tracker.DEBUG_MODE = True
        tracker.ALLOW_MERGES = False
        tracker.USE_DIV_CONSTRAINT = False

        # if we've added vertices, we need to recompute full_det and all_edges
        # we use solution_graph to get our det
        if len(added_vertices):
            current_det = pd.DataFrame.from_dict(solution_graph.nodes, orient="index").sort_index()
            current_det.drop(columns = ['track-id'] + [col for col in current_det.columns if 'NodeFlag' in col], inplace=True)
            for i, col in enumerate(loc):
                current_det[f'{col}_scaled'] = current_det[col] * ds_scale[i]


            # we know some things about this vertex i.e. where it comes from
            # but we still need it to be a full part of the model, including
            # candidate edges into the next frame (potentially to other new vertices)
            # so, we add all vertices to detections, and then build the model
            # "from scratch" with new vertices included, but also with
            # appropriately fixed edges
            # and then we solve "from_existing_edges"
            tdict = tracker._build_trees(current_det, 't', loc)
            edges = tracker._get_candidate_edges(current_det, 't', tdict)
            edges['cost'] = edges['distance']
            current_det['enter_exit_cost'], current_det['div_cost'] = tracker._compute_detection_costs(
                current_det,
                [f'{cd}_scaled' for cd in loc],
                edges
            )
            full_det = tracker._get_all_vertices(current_det, 't', loc)
            all_edges = tracker._get_all_edges(edges, current_det, 't')
            all_edges['flow'] = -1.0
        # we can just load existing edge df and vertices
        # but because we've re-labelled the original segmentation, we need to update the labels of the vertices
        # based on solution_graph
        else:
            if first_run:
                full_det = pd.read_csv(join_pth(ds_root_pth, OUT_FILE.ALL_VERTICES))
                all_edges = pd.read_csv(join_pth(ds_root_pth, OUT_FILE.ALL_EDGES))
            else:
                full_det = pd.read_csv(out_all_v_path, index_col=0)
                all_edges = pd.read_csv(out_all_e_path, index_col=0)
            labels = nx.get_node_attributes(solution_graph, 'label')
            full_det.loc[list(labels.keys()), 'label'] = list(labels.values())
        
        all_edges['learned_migration_cost'] = all_edges['cost']
        all_edges['oracle_is_correct'] = -1
        if not first_run:
            old_all_edges = pd.read_csv(out_all_e_path)
            all_edges = pd.merge(all_edges, old_all_edges[['u', 'v', 'oracle_is_correct']], on=['u', 'v'], how='left', suffixes=('', '_old'))
            all_edges['oracle_is_correct'] = all_edges['oracle_is_correct_old'].combine_first(all_edges['oracle_is_correct'])
            all_edges.drop(columns=['oracle_is_correct_old'], inplace=True)

        # we fix 'oracle_is_correct' to 1 for all added edges
        for edge in added_edges:
            u = edge[0]
            v = edge[1]
            if len(all_edges.loc[(all_edges['u'] == u) & (all_edges['v'] == v)]) == 0:
                new_row = {
                    'u': u,
                    'v': v,
                    # we make this edge free, but we fix its flow anyway so it makes no difference
                    'distance': 0.0,
                    'cost': 0.0,
                    'learned_migration_cost': 0.0,
                    'capacity': 1,
                    'flow': 1.0,
                    'oracle_is_correct': 1
                }
                all_edges = pd.concat([all_edges, pd.DataFrame([new_row])], ignore_index=True)
            else:
                all_edges.loc[(all_edges['u'] == u) & (all_edges['v'] == v), 'oracle_is_correct'] = 1
            # if the source of the edge is dividing, we also fix incoming division flow for it
            if solution_graph.out_degree(u) == 2:
                all_edges.loc[(all_edges['u'] == -3) & (all_edges['v'] == v), 'oracle_is_correct'] = 1

        # for each removed edge, if it's in all_edges we fix 'oracle_is_correct' to 0
        for edge in removed_edges:
            u = edge[0]
            v = edge[1]
            if len(all_edges.loc[(all_edges['u'] == u) & (all_edges['v'] == v)]) == 1:
                all_edges.loc[(all_edges['u'] == u) & (all_edges['v'] == v), 'oracle_is_correct'] = 0
        
        # we are now ready to re-solve
        tracked = tracker.solve_from_existing_edges(
            full_det,
            all_edges,
            't',
            loc,
        )

        tracked.all_vertices.to_csv(out_all_v_path, index=True)
        tracked.all_edges.to_csv(out_all_e_path, index=True)

        return tracked
    
    for row in ds_edges.itertuples():
        is_correct = row.solution_correct
        u = row.u
        v = row.v
        if not is_correct:
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
                    graph_info = {}
                    for k in node_copy_info:
                        gt_k = node_copy_info[k]
                        graph_info[k] = node_info[gt_k]
                    graph_info['label'] = new_label

                    # need scale for re-solving!
                    vertices_info = graph_info.copy()
                    for i, coord in enumerate(loc):
                        vertices_info[f'{coord}_scaled'] = vertices_info[coord] * ds_scale[i]
                    vertices_info['node_id'] = new_node_id
                    
                    to_add = [(new_node_id, graph_info)]
                    sol_dest = new_node_id

                    # add node to solution graph
                    solution_graph.add_nodes_from(to_add)
                    added_vertices.append(vertices_info)

                    # mask segmentation
                    t = graph_info['t']
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
                    added_edges.append((u, sol_dest))

            if solution_graph.has_edge(u, v):
                # remove wrong edge
                solution_graph.remove_edge(u, v)
                removed_edges.append((u, v))

            # save CTC output, get new new_label based on actual segmentation
            original_seg, track_df, new_label = get_ctc_output(original_seg, solution_graph, 't', 'label', loc)
            _save_results(original_seg, track_df, out_res)
            new_label += 1

            # still had wrong edges, have small number of errors, been 10 edges since we last evaluated or we are at the end
            if (fp_edge_count <= total_fps)\
                and ((total_fps <= 20)\
                        or (edge_count % 10 == 0)\
                        or (edge_count == total_edges - 1)):
                
                tracked = resolve(
                    ds_root_pth, 
                    solution_graph,
                    added_vertices,
                    added_edges,
                    removed_edges,
                    out_all_v_path,
                    out_all_e_path,
                )
                solution_graph = tracked.as_nx_digraph()
                original_seg, track_df, new_label = get_ctc_output(original_seg, solution_graph, 't', 'label', loc)
                _save_results(original_seg, track_df, out_res)
                new_label += 1

                added_vertices = []
                added_edges = []
                removed_edges = []

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
            fp_edge_count += 1

        # save into graph
        ds_edges.at[row.Index, 'LNK'] = original_lnk
        ds_edges.at[row.Index, 'BIO(0)'] = original_bio
        ds_edges.at[row.Index, 'CT'] = original_ct
        ds_edges.at[row.Index, 'TF'] = original_tf
        ds_edges.at[row.Index, 'CCA'] = original_cca
        ds_edges.at[row.Index, 'BC(0)'] = original_bc
    ds_edges.to_csv(out_edge_csv_path, index=False)