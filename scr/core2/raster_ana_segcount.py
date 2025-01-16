#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 01:33:25 2021

@author: sizhuo
"""

import os

# import keras.saving.save

os.environ["CUDA_VISIBLE_DEVICES"]="0"
import rasterio                  # I/O raster data (netcdf, height, geotiff, ...)
import rasterio.warp             # Reproject raster samples
from rasterio import merge
from rasterio import windows
# import fiona                     # I/O vector data (shape, geojson, ...)
import geopandas as gps
from shapely.geometry import Point, Polygon
from shapely.geometry import mapping, shape
from skimage.transform import resize

import numpy as np               # numerical array manipulation
import os
from tqdm import tqdm
import PIL.Image
import time

from itertools import product
import cv2

import sys

# from rasterstats import zonal_stats
import matplotlib.patches as patches
from matplotlib.patches import Polygon

import warnings                  # ignore annoying warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# %reload_ext autoreload
# %autoreload 2
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from collections import defaultdict
from rasterio.enums import Resampling
# from osgeo import ogr, gdal

from scipy.optimize import curve_fit
from matplotlib import colors
import glob
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress
from shapely.geometry import shape
from rasterio.features import shapes
# import multiprocessing
from itertools import product
import tensorflow as tf
import ipdb
from scr.core2.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity, miou, weight_miou
from core2.UNet_attention_segcount import UNet
from core2.optimizers import adagrad, adam, nadam
from core2.frame_info import image_normalize
# from tensorflow import keras
import tensorflow.keras.backend as K

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
# ipdb.set_trace()
class anaer:
    def __init__(self, config):
        self.config = config
        self.all_files = load_files(self.config)

    def load_model(self):
        OPTIMIZER = adam

        if self.config.multires:
            from core2.UNet_multires_attention_segcount import UNet
        elif not self.config.multires:
            from core2.UNet_attention_segcount import UNet
        # load three models in order
        self.models = []
        for cc in range(3):
            self.model = UNet([self.config.BATCH_SIZE, self.config.input_size, self.config.input_size, 3+cc], inputBN=self.config.inputBN)
            self.model.load_weights(self.config.trained_model_path[cc])
            self.model.compile(optimizer=OPTIMIZER, loss=tversky, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy, miou, weight_miou])
            # prediction mode
            self.model.trainable = False
            self.models.append(self.model)


        print('Model(s) loaded')


    def segcount_RUN(self):
        _, not_work = predict_ready_run(self.config, self.all_files, self.models, self.config.output_dir, th = self.config.threshold)

        return not_work



def load_files(config):
    exclude = set(['water_new', 'md5', 'pred', 'test_kay'])
    all_files = []
    for root, dirs, files in os.walk(config.input_image_dir):
        dirs[:] = [d for d in dirs if d not in exclude]
        for file in files:
            if file.endswith(config.input_image_type) and file.startswith(config.input_image_pref):
                 all_files.append((os.path.join(root, file), file))
    print('Number of raw tif to predict:', len(all_files))
    if config.fillmiss: # only fill the missing predictions (smk) # while this include north
        doneff = gps.read_file(config.grids)
        donef2 = list(doneff['filepath'])
        done_names= set([os.path.basename(f)[:6] for f in donef2])
        all_files = [f for f in all_files if os.path.splitext(f[1])[0] not in done_names]
    # print(all_files)
    # print('Number of missing tif to predict:', len(all_files))

    return all_files




def addTOResult(res, prediction, row, col, he, wi, operator = 'MAX'):
    currValue = res[row:row+he, col:col+wi]
    newPredictions = prediction[:he, :wi]
    # # set the 4 borderlines to 0 to remove the border effect
    # newPredictions[:10, :] = 0
    # newPredictions[-10:, :] = 0
    # newPredictions[:, :10] = 0
    # newPredictions[:, -10:] = 0

# IMPORTANT: MIN can't be used as long as the mask is initialed with 0!!!!! If you want to use MIN initial the mask with -1 and handle the case of default value(-1) separately.
    if operator == 'MIN': # Takes the min of current prediction and new prediction for each pixel
        currValue [currValue == -1] = 1 #Replace -1 with 1 in case of MIN
        resultant = np.minimum(currValue, newPredictions)
    elif operator == 'MAX':
        resultant = np.maximum(currValue, newPredictions)
    else: #operator == 'REPLACE':
        resultant = newPredictions
# Alternative approach; Lets assume that quality of prediction is better in the centre of the image than on the edges
# We use numbers from 1-5 to denote the quality, where 5 is the best and 1 is the worst.In that case, the best result would be to take into quality of prediction based upon position in account
# So for merge with stride of 0.5, for eg. [12345432100000] AND [00000123454321], should be [1234543454321] instead of [1234543214321] that you will currently get.
# However, in case the values are strecthed before hand this problem will be minimized
    res[row:row+he, col:col+wi] =  resultant
    return (res)

def predict_using_model_segcount_fi(models, batch, batch_pos, maskseg, maskdens, operator):
    # batch: [[(256, 256, 5), (128, 128, 1)], [(256, 256, 5), (128, 128, 1)], ...]. len = 200
    # b1 = batch[0]
    tm1 = np.stack(batch, axis = 0) #stack a list of arrays along axis
    # rgb model
    seg0, dens0 = models[0].predict(tm1[..., :3], workers = 10, use_multiprocessing = True, verbose=0)
    # rgb+infrared model
    seg1, dens1 = models[1].predict(tm1[..., :4], workers = 10, use_multiprocessing = True, verbose=0)
    # rgb+infrared+ndvi model
    seg2, dens2 = models[2].predict(tm1, workers = 10, use_multiprocessing = True, verbose=0)
    # merge the results
    # nan sum to deal with nan values from ndvi calculation
    seg = np.nanmean([seg0, seg1, seg2], axis = 0)
    dens = np.nanmean([dens0, dens1, dens2], axis = 0)

    # ipdb.set_trace()
    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(seg[i], axis = -1)
        c = np.squeeze(dens[i], axis = -1)
        # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
        maskseg = addTOResult(maskseg, p, row, col, he, wi, operator)
        maskdens = addTOResult(maskdens, c, row, col, he, wi, operator)
    # ipdb.set_trace()
    return maskseg, maskdens






def detect_tree_segcount_fi(config, models, img, width=256, height=256, stride = 128, normalize=True):
    if 'chm' in config.channel_names1:
        raise NotImplementedError('not supporting chm as input yet')
    else:
        CHM = 0
    nols, nrows = img.meta['width'], img.meta['height']
    meta = img.meta.copy()

    if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction.
        meta['dtype'] = np.float32

    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

    masksegs = np.zeros((int(nrows), int(nols)), dtype=np.float32)
    maskdenss = np.zeros((int(nrows), int(nols)), dtype=np.float32)
    meta.update(
                {'width': int(nols),
                 'height': int(nrows)
                }
                )


    batch = []
    batch_pos = [ ]
    for col_off, row_off in tqdm(offsets):

        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        # prepare for all bands + ndvi
        patch1 = np.zeros((height, width, 5))  # Add zero padding in case of corner images
        temp_im1 = img.read(window = window)
        temp_im1 = np.transpose(temp_im1, axes=(1,2,0))

        NDVI = (temp_im1[:, :, -1].astype(float) - temp_im1[:, :, 0].astype(float)) / (temp_im1[:, :, -1].astype(float) + temp_im1[:, :, 0].astype(float))
        NDVI = NDVI[..., np.newaxis]

        temp_im1 = np.append(temp_im1, NDVI, axis = -1)

        # ndvi invalid values: when both red and nir are 0, ndvi is 0/0 = nan, this happens
        # print('ndvi', NDVI.min())
        # print('red', temp_im1[:, :, 0].min())
        # print('nir', temp_im1[:, :, -2].min())

        if normalize:
            temp_im1 = image_normalize(temp_im1, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel

        patch1[:int(window.height), :int(window.width)] = temp_im1

        batch.append(patch1)
        batch_pos.append((window.col_off, window.row_off, window.width, window.height))

        if (len(batch) == config.BATCH_SIZE):
            # print('processing one batch')
            masksegs, maskdenss = predict_using_model_segcount_fi(models, batch, batch_pos, masksegs, maskdenss, 'MAX')

            batch = []
            batch_pos = []

    if batch:
        masksegs, maskdenss = predict_using_model_segcount_fi(models, batch, batch_pos, masksegs, maskdenss, 'MAX')
        batch = []
        batch_pos = []
    return masksegs, maskdenss, meta



def predict_ready_run(config, all_files, model_segcount,output_dir, th = 0.5):
    counter = 1
    th = th
    counts = {}

    outputFiles = []
    not_work = []
    # # shuffle the files if testing only a subset
    # random.shuffle(all_files)
    for fullPath, filename in tqdm(all_files):
        outputFile = os.path.join(output_dir, filename[:-4] + config.output_suffix + config.output_image_type)
        # print(outputFile)

        if not os.path.exists(outputFile):
            try:
                # print(outputFile)
                outputFiles.append(outputFile)
                t1 = time.time()
                # outputFileChm = os.path.join(output_dir, filename.replace(config.input_image_type, config.output_image_type))

                with rasterio.open(fullPath) as img:
                    # for only south tifs
                    # print(raw.profile['transform'][5])
                    if config.segcountpred:
                        # print('creating file', outputFile)
                        detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount_fi(config, model_segcount, img, width = config.WIDTH, height = config.HEIGHT, stride = config.STRIDE, normalize=config.normalize)
                        writeMaskToDisk(detectedMaskSeg, detectedMeta, outputFile,  image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = config.output_dtype, th = th, create_countors = False)

                        # density
                        writeMaskToDisk(detectedMaskDens, detectedMeta, outputFile.replace('seg.tif', 'density.tif'), image_type = config.output_image_type, output_shapefile_type = config.output_shapefile_type, write_as_type = 'float32', th = th, create_countors = False, convert = 0)
                        counts[filename] = detectedMaskDens.sum()
                        t2 = time.time()
                        print('======== Processed image ', filename, 'in', t2-t1, 'seconds.')
                        del detectedMaskSeg, detectedMaskDens, detectedMeta
                        # clear memory
                        K.clear_session()


                    else:
                        continue
            except:
                not_work.append(fullPath)
                continue

            counter += 1


        else:
            print('Skipping: File already analysed!', fullPath)

    return counter, not_work






def writeMaskToDisk(detected_mask, detected_meta, wp, image_type, output_shapefile_type, write_as_type = 'uint8', th = 0.5, create_countors = False, convert = 1, rescale = 0):
    # Convert to correct required before writing
    meta = detected_meta.copy()
    if convert:
        if 'float' in str(detected_meta['dtype']) and 'int' in write_as_type:
            print(f'Converting prediction from {detected_meta["dtype"]} to {write_as_type}, using threshold of {th}')
            detected_mask[detected_mask<th]=0
            detected_mask[detected_mask>=th]=1

    if rescale:
        # for densty masks, multiply 10e4
        detected_mask = detected_mask*10000

    detected_mask = detected_mask.astype(write_as_type)
    if detected_mask.ndim != 2:
        detected_mask = detected_mask[0]

    meta['dtype'] =  write_as_type
    meta['count'] = 1
    if rescale:
        meta.update(
                            {'compress':'lzw',
                              'driver': 'GTiff',
                                'nodata': 32767
                            }
                        )
    else:
        meta.update(
                            {'compress':'lzw',
                              'driver': 'GTiff',
                                'nodata': 255
                            }
                        )
        ##################################################################################################
        ##################################################################################################
    with rasterio.open(wp, 'w', **meta) as outds:
        outds.write(detected_mask, 1)
    if create_countors:
        wp = wp.replace(image_type, output_shapefile_type)
        # create_contours_shapefile(detected_mask, detected_meta, wp)

