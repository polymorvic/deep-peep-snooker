# %%
import cv2
import json
from src.utils.func import (crop_center, read_image_as_numpyimage)
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
picname = 'pic_02_06_01'  # _06_09_02     pic_01_43_02
pic = read_image_as_numpyimage(f'pics/{picname}.png', 'rgb') # pic_02_07_01.png   pic_05_18_01.png  pic_08_08_01.png pic_06_16_02.png
cropped_pic = crop_center(pic)

# %%
display_img(pic)

# %%
finder = PlayfieldFinder(pic)

# %%
internal_bottom_cushion = finder.find_bottom_internal_cushion()
internal_bottom_cushion

# %%
display_img(plot_on_image(pic, lines=[internal_bottom_cushion], line_thickness=1))

# %%
polygon_annp = PolygonAnnotation(root_dir='playfield_gt')
polygon_annp.read(Path('playfield_gt/all.json'))

# %% [markdown]
# #### find playfield

# %%
root = Path('pics')
results = []
ious = []
pic_names = []
for file in sorted(root.glob('*.png')):
    # print(file.stem)
    pic = read_image_as_numpyimage(file, 'rgb')
    finder = PlayfieldFinder(pic)
    internal_bottom_cushion = None
    y_ref = None
    data = polygon_annp.filter_by_image(file.name)
    if data is None:
        print(file.name)
        
    else:
        points_gt = np.asarray(transform_annotation(pic, data.points))

        y_coords = points_gt[:, 1]
        y_ref = int(np.median(np.sort(y_coords)[-2:]))

    try:
        internal_bottom_cushion = finder.find_bottom_internal_cushion()
        pts = internal_bottom_cushion.limit_to_img(pic)
        pic_copy = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
        cv2.line(pic_copy, *pts, (255, 0, 0), 1)
        cv2.imwrite(f'bottom_line_tests/test_{file.stem}.png', pic_copy)
        
    except Exception as e:
        print(f'Error processing {file}: {e}')

    finally:
        results.append(
            {'pic_name': file.name, 
            'internal_bottom_cushion': internal_bottom_cushion, 
            'intercept_ref': y_ref,
            'intercept_pred': internal_bottom_cushion.intercept if internal_bottom_cushion is not None else None,
            }) 

# %%
df = pd.DataFrame(results)
df['diff']= df['intercept_ref'] - df['intercept_pred']
df['abs_diff']= np.abs(df['intercept_ref'] - df['intercept_pred'])

# %%
df.to_excel('bottom_cushion_results2.xlsx', index=False)

# %%
# Calculate bins based on binwidth=1
diff_clean = df['diff'].dropna()

min_val = diff_clean.min()
max_val = diff_clean.max()
bins = np.arange(min_val, max_val + 1, 1)  # binwidth = 1
plt.hist(diff_clean, bins=bins, edgecolor='black')
plt.xlabel('Difference (intercept_ref - intercept_pred)')
plt.ylabel('Frequency')
plt.title('Histogram of Intercept Differences')
plt.show()


# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
wrong_pics = pd.read_excel('/home/polymorvic/Desktop/zdjecia_do_poprawy.ods')[['col2']]

# %%
df

# %%
df[~df['pic_name'].isin(wrong_pics['col2'])].sort_values(by='abs_diff', ascending=False)['diff'].hist()

# %%
df[df['pic_name'].isin(wrong_pics['col2'])].sort_values(by='abs_diff', ascending=False)['diff'].hist()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%
df['diff'].median(), df['diff'].mean(),df['diff'].std()

# %%


# %%


# %%
df['diff'].median()

# %%
df['diff'].std()

# %%
df['diff'].mean()

# %%
picname = 'pic_07_13_01'

# %%
pic = read_image_as_numpyimage(f'pics/{picname}.png', 'rgb')
data = polygon_annp.filter_by_image('pic_01_05_01.png')
points_gt = np.asarray(transform_annotation(pic, data.points))
display_img(plot_on_image(pic, polygons=[transform_annotation(pic, data.points)]))

pic = read_image_as_numpyimage(f'pics/{picname}.png', 'rgb')

# %%
transform_annotation(pic, data.points)

# %%
with open('playfield_gt/runda_5.json', 'r') as f:
    data = json.load(f)

# %%
with open('playfield_gt/ssss.json', 'w') as f:
    json.dump(data, f, indent=4)


