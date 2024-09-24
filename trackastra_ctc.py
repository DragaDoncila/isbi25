import os
import pandas as pd
import torch
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc
from tracktour import load_tiff_frames

device = "cuda" if torch.cuda.is_available() else "cpu"

root_dir = '/home/ddon0001/PhD/data/cell_tracking_challenge/SUBMISSION/'
out_dir = '/home/ddon0001/PhD/experiments/trackastra/'

ds_names = [name for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))]
im_names = []
err_seg_names = []
res_names = []
for name in ds_names:
    seqs = [seq for seq in os.listdir(os.path.join(root_dir, name)) if '_ERR_SEG' in seq]
    im_names.extend([f'{os.path.join(root_dir, name, seq[:2])}' for seq in seqs])
    err_seg_names.extend([f'{os.path.join(root_dir, name, seq)}' for seq in seqs])
    res_names.extend([f'{os.path.join(out_dir, name, seq[:2])}_RES' for seq in seqs])

for i in range(len(im_names)):
    im_path = im_names[i]
    seg_path = err_seg_names[i]
    out_path = res_names[i]

    if os.path.exists(out_path):
        print(f'{out_path} already exists. Skipping...')
        continue
    print(f'Tracking {out_path}')

    # load data
    imgs = load_tiff_frames(im_path)
    masks = load_tiff_frames(seg_path)

    # Load a pretrained model
    model = Trackastra.from_pretrained("ctc", device=device)

    # or from a local folder
    # model = Trackastra.from_folder('path/my_model_folder/', device=device)

    # Track the cells
    track_graph = model.track(imgs, masks, mode="ilp")  # or mode="ilp", or "greedy_nodiv"

    # Write to cell tracking challenge format
    ctc_tracks, masks_tracked = graph_to_ctc(
        track_graph,
        masks,
        outdir=out_path,
    )