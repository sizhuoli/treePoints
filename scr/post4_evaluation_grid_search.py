import os

import pandas as pd
import matplotlib.pyplot as plt
import ipdb
import numpy as np
import geopandas as gps
import glob
import rasterio
from keras.src.utils.feature_space import layers
from skimage.feature import peak_local_max
import re
from scipy import ndimage as ndi
import tqdm
from sklearn.metrics import r2_score
# ignore warnings
from sklearn.model_selection import LeaveOneOut, KFold
import warnings
warnings.filterwarnings("ignore")


heat_path = 'path to clipped heatmaps' #'../clipped_heats/'
manual_count_path = 'path to gpkg file containing manual counts in each clipped area'#'.../all_groups_count_fty.gpkg'


# load manual points and count the number of points in each polygon
manual_counts = gps.read_file(manual_count_path, layer='all_groups_count_fty')


# gridsearch
min_dis_s = [5, 10, 12, 15]
thres_abs_s = [0.0005, 0.001, 0.002]
scan_window_s = [10, 12, 15]
num_peak_chm_s = [10, 20, 30]
ndvi_tree_s = [0, 0.1]



def heat2map(heatmap, args, manual_counts):
    """returns the number of trees detected"""

    ff = os.path.basename(heatmap)
    # ipdb.set_trace()
    # handle naming differences
    if '2019' in ff:
        chm = args.chm_dir + ff.replace('2019', 'DHM').replace('_density', '')
        image = args.image_dir + ff.replace('_density', '')
    else:
        # add dhm in front of 1km
        chm = args.chm_dir + ff.replace('1km', 'DHM_1km').replace('_density', '')
        image = args.image_dir + ff.replace('_density', '')


    type = manual_counts[manual_counts['filename'] == ff.replace('_density', '')]['forest_type'].values[0]


    with rasterio.open(heatmap) as src1:
        heat = src1.read(1)
        # filter out ground/buildings using ndvi mask
        with rasterio.open(image) as src:
            # calculate ndvi from 4 band image, nir is band 4, red is band 1
            img = src.read()
            denominator = img[3].astype(float) + img[0].astype(float)
            # ipdb.set_trace()
            ndvi = np.where(denominator == 0, -1,
                            (img[3].astype(float) - img[0].astype(float)) / denominator)
            # filter out ground or buildings
            assert ndvi.shape == heat.shape
            heat[ndvi < 0] = 0 # only filter out very low ndvi
            thress = max(heat.max() * args.alpha, args.thres_abs)
            if type == 0:
                # nonforest examples can be heavily affected by the border, so exclude border
                coords = peak_local_max(heat, min_distance=args.min_dis, threshold_abs=thress, exclude_border=True)
            else:
                coords = peak_local_max(heat, min_distance=args.min_dis, threshold_abs=thress, exclude_border=False)

            with rasterio.open(chm) as src2:
                height = src2.read(1)
                height = ndi.zoom(height, (heat.shape[0]/height.shape[0], heat.shape[1]/height.shape[1]))


                scanned_area = np.zeros_like(height)
                if len(coords) != 0:
                    for c in coords:
                        x, y = c
                        # a binary mask to record scanned area on the height map, window to deal with the shift between optical and lidar and should cover the entire tree
                        scanned_area[max(0, x - args.scan_window):min(height.shape[0], x + args.scan_window), max(0, y - args.scan_window):min(height.shape[1], y + args.scan_window)] = 1

                # detect peaks from height also
                thres_h = height.copy()
                thres_h[thres_h < args.low_vege] = 0 # filter out low vegetation
                thres_h[scanned_area == 1] = 0
                thres_h[ndvi < args.ndvi_tree] = 0
                # ipdb.set_trace()
                if type == 0:
                    # nonforest examples can be heavily affected by the border, so exclude border
                    coords2 = peak_local_max(thres_h, min_distance=args.min_dis, threshold_abs=args.low_vege, num_peaks=args.num_peak_chm, exclude_border=True)
                else:
                    coords2 = peak_local_max(thres_h, min_distance=args.min_dis, threshold_abs=args.low_vege, num_peaks=args.num_peak_chm, exclude_border=False)

    #
            coords = np.concatenate([coords, coords2])

    del heat, height

    algorithm_count = len(coords)
    # get count from manual points
    # ipdb.set_trace()
    manual_c = manual_counts[manual_counts['filename'] == ff.replace('_density', '')]['NUM_POINTS'].values[0]
    forest_t = manual_counts[manual_counts['filename'] == ff.replace('_density', '')]['forest_type'].values[0]

    return algorithm_count, manual_c, forest_t


def evaluate(heatmaps, args, manual_counts):

    def re(gt, pred):
        return np.mean(np.abs(gt - pred) / gt)

    preds = []
    truths = []
    types = []

    for heat in heatmaps:
        pred, truth, forest_t = heat2map(heat, args, manual_counts)
        preds.append(pred)
        truths.append(truth)
        types.append(forest_t)

    # average relative error
    overall = re(np.array(truths), np.array(preds))
    # split forest type according to manual counts
    # ipdb.set_trace()
    B_ind = [i for i, x in enumerate(types) if x == 1]
    C_ind = [i for i, x in enumerate(types) if x == 2]
    non_ind = [i for i, x in enumerate(types) if x == 0]
    B_score = re(np.array(truths)[B_ind], np.array(preds)[B_ind])
    C_score = re(np.array(truths)[C_ind], np.array(preds)[C_ind])
    non_score = re(np.array(truths)[non_ind], np.array(preds)[non_ind])
    return overall, B_score, C_score, non_score

# other parameters
class args:
    def __init__(self):
        self.chm_dir = 'path to clipped chms' #'../clipped_chms/'
        self.image_dir = 'path to clipped images' #'../clipped_images/'
        self.low_vege = 3
        self.alpha = 0.2

args = args()

# grid search
best_para_fold = []
heat_maps_all = glob.glob(heat_path + '*.tif')
total_comb = len(min_dis_s) * len(thres_abs_s) * len(scan_window_s) * len(num_peak_chm_s) * len(ndvi_tree_s)
# outer fold
cv = KFold(n_splits=3, shuffle=True, random_state=0)
fold_count = 0
global_best_score = 100
global_best_params = None
test_scores = []
test_B_scores = []
test_C_scores = []
test_non_scores = []
fit_scores = []
for train_maps, test_maps in cv.split(heat_maps_all):
    tests = [heat_maps_all[i] for i in test_maps]
    trains = [heat_maps_all[i] for i in train_maps]
    search_count = 0
    best_score = 100
    for min_dis in min_dis_s:
        for thres_abs in thres_abs_s:

            for scan_window in scan_window_s:

                for num_peak_chm in num_peak_chm_s:
                    for ndvi_tree in ndvi_tree_s:
                        args.ndvi_tree = ndvi_tree
                        args.min_dis = min_dis
                        args.thres_abs = thres_abs
                        args.scan_window = scan_window
                        args.num_peak_chm = num_peak_chm
                        score, _, _, _ = evaluate(trains, args, manual_counts)
                        search_count += 1
                        if score < best_score:
                            best_score = score

                            best_params = (min_dis, thres_abs, scan_window, num_peak_chm, ndvi_tree)
                            if score < global_best_score:
                                global_best_score = score
                                global_best_params = best_params
                                print(f"global best score: {global_best_score}")

                        if search_count % 50 == 0:
                            print(f"searched {search_count/total_comb*100}% in fold {fold_count}")

    fit_scores.append(best_score)
    # test score using best parameters
    args.min_dis, args.thres_abs, args.scan_window, args.num_peak_chm, args.ndvi_tree = best_params
    test_score, test_B_score, test_C_score, test_non_score = evaluate(tests, args, manual_counts)
    print(f"test score: {test_score} in fold {fold_count}; B score: {test_B_score}; C score: {test_C_score}; non score: {test_non_score}")
    test_scores.append(test_score)
    test_B_scores.append(test_B_score)
    test_C_scores.append(test_C_score)
    test_non_scores.append(test_non_score)
    best_para_fold.append(best_params)
    fold_count += 1
    print('===========================')
    print(f"fold {fold_count} done.")



# the most common parameters     # note that some comb leads to overfitting
from collections import Counter
global_best_params2 = Counter(best_para_fold).most_common(1)[0][0]
# frequency of most common parameters
print(Counter(best_para_fold).most_common(1)[0][1])
# indices of most common parameters
ind = [i for i, x in enumerate(best_para_fold) if x == global_best_params2]
print('indices of most common parameters: ', ind)
# average test score if using the most common parameters
print('average test score using most common parameters: ', np.mean(np.array(test_scores)[ind]), np.std(np.array(test_scores)[ind]))
# average B score if using the most common parameters
print('average B group score using most common parameters: ', np.nanmean(np.array(test_B_scores)[ind]), np.nanstd(np.array(test_B_scores)[ind]))
# average C score if using the most common parameters
print('average C group score using most common parameters: ', np.nanmean(np.array(test_C_scores)[ind]), np.nanstd(np.array(test_C_scores)[ind]))
# average non score if using the most common parameters
print('average non group score using most common parameters: ', np.nanmean(np.array(test_non_scores)[ind]), np.nanstd(np.array(test_non_scores)[ind]))

# most common parameters: there is still a risk that the most common parameters are not the best parameters, but we rule out some overfitting combinations
print('most common parameters: ', global_best_params2)
# plot fit and test scores
xx = range(0, len(fit_scores))
plt.scatter(xx, fit_scores, label='fit', marker='o', c='red', s=40)
plt.scatter(xx, test_scores, label='test', marker='o', c='blue', s=130)
# mark the most common parameters
plt.scatter(ind, np.array(test_scores)[ind], label='most common comb', marker='o', c='green', s=100)
plt.legend()
plt.xlabel('fold')
plt.ylabel('metric score')
plt.show()

# ipdb.set_trace()










