
import networkx as nx
import os
import pandas as pd
from scipy.spatial.distance import cdist
from traccuracy.loaders import load_ctc_data



data_root = '/home/ddon0001/PhD/data/cell_tracking_challenge/SUBMISSION/'
res_root = '/home/ddon0001/PhD/experiments/trackastra/'

ds_names = [name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name))]
im_names = []
err_seg_names = []
res_names = []
for name in ds_names:
    seqs = [seq for seq in os.listdir(os.path.join(data_root, name)) if '_ERR_SEG' in seq]
    im_names.extend([f'{os.path.join(data_root, name, seq[:2])}' for seq in seqs])
    err_seg_names.extend([f'{os.path.join(data_root, name, seq)}' for seq in seqs])
    res_names.extend([f'{os.path.join(res_root, name, seq[:2])}_RES' for seq in seqs])


for i in range(len(im_names)):
    im_path = im_names[i]
    seg_path = err_seg_names[i]
    res_path = res_names[i]
    det_path = os.path.join(seg_path, 'detections.csv')

    sol_g = load_ctc_data(res_path)
    # for u, v in sol_g.graph.edges:
    #     if sol_g.graph.nodes[u]['t'] +1 != sol_g.graph.nodes[v]['t']:
    #         print(f"Skip edge present in {res_path}")
    #         break

    dets = pd.read_csv(det_path)
    coord_keys = ['x', 'y', 'z'] if 'z' in dets.columns else ['x', 'y']

    # assign area to each node. node ids are '{t}_{segmentation_id}'
    area_dict = {
        f'{row.label}_{row.t}': row.area
        for row in dets.itertuples()
    }
    nx.set_node_attributes(sol_g.graph, area_dict, 'area')
    node_df = pd.DataFrame.from_dict(sol_g.graph.nodes, orient="index")[['t'] + coord_keys]
    ts = sorted(node_df.t.unique())
    for i, t in enumerate(ts[:-1]):
        t_coords = node_df[node_df.t == t][coord_keys]
        next_t_coords = node_df[node_df.t == ts[i+1]][coord_keys]
        all_dists = cdist(t_coords.values, next_t_coords.values, metric='euclidean')
        # now that we have all distances, we read from it to find
        # the distance of each edge connecting nodes in t_coords to nodes in next_t_coords
        # the rank of this distance
        # probably the easiest way to do this is to go through the nodes in frame t in the graph
        # and query their outgoing edges
        for node in t_coords.itertuples():
            for edge in sol_g.graph.out_edges(node.Index):
                u, v = edge
                # We ran a check and there shouldn't be any of these, but if there are, we raise here
                if sol_g.gragh.nodes[v]['t'] != t + 1:
                    raise ValueError(f'Skip edge present in {res_path}!')
                dist = all_dists[u, v]
                sol_g.graph.edges[edge]['distance'] = dist
                sol_g.graph.edges[edge]['chosen_neighbour_rank'] = (all_dists[u] < dist).sum()
                sol_g.graph.edges[edge]['prop_diff'] = abs((sol_g.graph.nodes[v]['area']  / sol_g.graph.nodes[u]['area']) - 1)



# get rank... will need to get all nodes in next frame, find all distances, then get distance rank
# get edge df from nx graph, save edge df to csv

# separately? or after...? 
# load gt data, evaluate using traccuracy
# write out gt graph and evaluated graph
# write out matching
# write out metrics

# then we're ready for iterative