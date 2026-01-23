# %%
import cv2
import json
from src.utils.func import (crop_center, read_image_as_numpyimage, 
    pipette_color,
    group_lines)
from src.utils.const import ref_snooker_playfield
from src.utils.playfield_finder import PlayfieldFinder
from src.utils.points import Point
from src.utils.lines import Line
import numpy as np
from src.utils.plotting import display_img, plot_on_image
import src.config
from src.utils.metrics import iou, _reorder_pts
from src.utils.intersections import compute_intersections
from src.utils.annotations import transform_annotation
from pathlib import Path
import pandas as pd
from src.utils.annotations import PolygonAnnotation
import matplotlib.pyplot as plt

# %%
picname = 'pic_01_34_01'  # pic_01_125_01
pic = read_image_as_numpyimage(f'pics/{picname}.png', 'rgb') # pic_02_07_01.png   pic_05_18_01.png  pic_08_08_01.png pic_06_16_02.png
cropped_pic = crop_center(pic)

# %%
display_img(pic)

# %%
finder = PlayfieldFinder(pic)

# %%
a,b,c,d, e, f = finder._preprocess_image()

# %%
display_img(c)

# %%
locs = np.argwhere(c > 0)
top_h = locs[:, 0].min()
bottom_h = locs[:, 0].max()
diff = bottom_h - top_h
diff

# %%
prev_col_diffs = []
stability_window = 5
tolerance = 1

for col_idx in range(c.shape[1]):
    col_locs = np.argwhere(c[:, col_idx] > 0)

    if not col_locs.size:
        continue

    col_top_h = col_locs[:, 0].min()
    col_bottom_h = col_locs[:, 0].max()
    col_diff = col_bottom_h - col_top_h
    
    prev_col_diffs.append(col_diff)
    if len(prev_col_diffs) >= stability_window:
        recent_diffs = prev_col_diffs[-stability_window:]
        if max(recent_diffs) - min(recent_diffs) <= tolerance:
            top_left_col = col_idx
            break

# %%
prev_col_diffs = []
stability_window = 5
tolerance = 1

for col_idx in range(c.shape[1] - 1, -1, -1):
    col_locs = np.argwhere(c[:, col_idx] > 0)

    if not col_locs.size:
        continue

    col_top_h = col_locs[:, 0].min()
    col_bottom_h = col_locs[:, 0].max()
    col_diff = col_bottom_h - col_top_h
    
    prev_col_diffs.append(col_diff)
    if len(prev_col_diffs) >= stability_window:
        recent_diffs = prev_col_diffs[-stability_window:]
        if max(recent_diffs) - min(recent_diffs) <= tolerance:
            top_right_col = col_idx
            break


# %%
c[top_h, top_left_col:top_right_col] = 255

# %%
display_img(c)

# %%
# prev_col_bottom_diffs = []
# stability_window = 5
# tolerance = 1

# for col_idx in range(c.shape[1]):
#     col_locs = np.argwhere(c[:, col_idx] > 0)

#     if not col_locs.size:
#         continue

#     col_top_h = col_locs[:, 0].min()
#     col_bottom_h = col_locs[:, 0].max()
#     col_bottom_diff = bottom_h - col_bottom_h
    
#     prev_col_bottom_diffs.append(col_bottom_diff)
#     if len(prev_col_bottom_diffs) >= stability_window:
#         recent_diffs = prev_col_bottom_diffs[-stability_window:]
#         if max(recent_diffs) - min(recent_diffs) <= tolerance:
#             bottom_left_col = col_idx
#             break


# %%
# prev_col_bottom_diffs = []
# stability_window = 5
# tolerance = 1

# for col_idx in range(c.shape[1] - 1, -1, -1):
#     col_locs = np.argwhere(c[:, col_idx] > 0)

#     if not col_locs.size:
#         continue

#     col_top_h = col_locs[:, 0].min()
#     col_bottom_h = col_locs[:, 0].max()
#     col_bottom_diff = bottom_h - col_bottom_h
    
#     prev_col_bottom_diffs.append(col_bottom_diff)
#     if len(prev_col_bottom_diffs) >= stability_window:
#         recent_diffs = prev_col_bottom_diffs[-stability_window:]
#         if max(recent_diffs) - min(recent_diffs) <= tolerance:
#             bottom_right_col = col_idx
#             break


# %%
c[bottom_h, top_left_col:top_right_col] = 255

# %%
display_img(c)

# %%

c[top_h:bottom_h+1, top_left_col:top_right_col] = 255
display_img(c)


# %%


# %%


# %%


# %% [markdown]
# #### find playfield

# %%
finder = PlayfieldFinder(pic)

root = Path('pics')
for file in sorted(root.glob('*.png')):
    print(file.stem)
    pic = read_image_as_numpyimage(file, 'rgb')
    finder = PlayfieldFinder(pic)
    binary_mask, binary_mask_close, straighted_binary_mask, edges, copy_edges, pic_copy = finder._preprocess_image()

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0, 0].imshow(binary_mask)
    ax[0, 0].set_title(file.stem)
    ax[0, 1].imshow(binary_mask_close)
    ax[0, 2].imshow(straighted_binary_mask)
    ax[1, 0].imshow(edges)
    ax[1, 1].imshow(copy_edges)
    ax[1, 2].imshow(pic_copy)

    to_save = cv2.cvtColor(pic_copy, cv2.COLOR_RGB2BGR)
    fig.savefig(f'tests/external_edges/b/test_{file.stem}.png')
    print(file.stem)
    cv2.imwrite(f'tests/external_edges/a/test_{file.stem}.png', to_save)


