import os
import pandas as pd
from tqdm import tqdm
from utils import load_sol_files, populate_target_label, populate_solution_label_including_ws, populate_label_ws_enter_exit

root_pth = '/home/ddon0001/PhD/experiments/scaled/no_merges_all'
ds_summary_path = os.path.join(root_pth, 'summary.csv')
ds_info = pd.read_csv(ds_summary_path)[['ds_name', 'det_path']]



#####################################################################################
include_ws_fe_fa = True
include_ws = False
overall_df_path = os.path.join(root_pth, 'all_edges_with_target_ws_fa_fe.csv')
solution_only_path = os.path.join(root_pth, 'solution_edges_datasets_with_FP_WS_FA_FE.csv')
#####################################################################################

dfs = []
for row in tqdm(ds_info.itertuples()):
    # read all edges, solution graph, ground truth, sol_gt_mapping
    sol_pth = os.path.join(root_pth, row.ds_name)
    _, all_edges, solution_graph, gt_graph, _, sol_to_gt = load_sol_files(sol_pth)

    if include_ws_fe_fa:
        # TP and correct appearances and disappearances are correct
        populate_label_ws_enter_exit(all_edges, solution_graph, gt_graph, sol_to_gt)
    elif include_ws:
        # populate 'oracle_is_correct' column - only TP edges in solution are correct
        populate_solution_label_including_ws(all_edges, solution_graph)
    else:
        # populate 'oracle_is_correct' column - all non FP edges in solution are correct
        populate_target_label(all_edges, gt_graph, sol_to_gt)
    # correct solution edges are those with flow > 0 and in GT
    all_edges['solution_correct'] = (all_edges.flow > 0) == (all_edges.oracle_is_correct)
    # assign incorrect column as well for easy counting
    all_edges['solution_incorrect'] = all_edges.solution_correct == False
    all_edges['ds_name'] = row.ds_name
    # assign "prop diff" which is absolute difference in area proportion to 1
    all_edges['prop_diff'] = abs(all_edges['chosen_neighbour_area_prop'] - 1)
    # assign "feature distance" which is "distance" for all real edges, and otherwise, "cost"
    all_edges['feature_distance'] = all_edges.apply(lambda edg: edg['distance'] if edg['distance'] != -1 else edg['cost'], axis=1)
    # only keep real edges and appearance/exit edges
    dfs.append(all_edges[(((all_edges.u >= 0) & (all_edges.v >= 0)) | ((all_edges.u == -2) & (all_edges.cost > 0)) | ((all_edges.v == -4) & (all_edges.cost > 0)))])

overall_df = pd.concat(dfs)
overall_df.to_csv(overall_df_path, index=False)

ds_names_no_errors = []
for row in ds_info.itertuples():
    ds_name = row.ds_name
    ds_edges = overall_df[overall_df.ds_name == ds_name]
    ds_in_solution = ds_edges[ds_edges.flow > 0]
    count_edges_in_solution = len(ds_in_solution)
    count_errors_in_solution = ds_in_solution['solution_incorrect'].sum()
    if count_errors_in_solution == 0:
        print("No incorrect edges in solution for", ds_name)
        ds_names_no_errors.append(ds_name)

with_errors = overall_df[(~overall_df.ds_name.isin(ds_names_no_errors)) & (overall_df.flow > 0)]
with_errors.to_csv(solution_only_path, index=False)

