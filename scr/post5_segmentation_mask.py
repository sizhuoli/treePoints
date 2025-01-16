# morphological cleaning of the mask
import os.path

from scipy.spatial import ConvexHull
from scipy import ndimage
from skimage.morphology import isotropic_dilation, isotropic_erosion
import matplotlib.pyplot as plt
import numpy as np
from concave_hull import concave_hull, concave_hull_indexes
import rasterio
import ipdb
import glob
import multiprocessing as mp
import os
# argpas
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='', help='path to the tree segmentation files')
parser.add_argument('--out', type=str, default='...../cleaned_treemasks_radius1m/', help='output path')
parser.add_argument('--radius', type=int, default=1, help='radius for dilation and erosion, in meters')
parser.add_argument('--resolution', type=int, default=25, help='resolution of the images, in cm')
args = parser.parse_args()


def mask_one(file):
    with rasterio.open(file) as src:
        tree_cover = src.read(1)  # Read the first band (assumed binary: 1 = tree, 0 = non-tree)
        # radius in pixels, calculated based on radius in meters and resolution in cm
        out = isotropic_dilation(tree_cover, radius=args.radius * 100 / args.resolution).astype(np.ubyte)
        out2 = isotropic_erosion(out, radius=args.radius * 100 / args.resolution).astype(np.ubyte)
        # save
        out_path = args.out + '/' + file.split('/')[-1].split('.')[0] + '_treeMask.tif'
        # compress output
        with rasterio.open(out_path, 'w', driver='GTiff', width=src.width, height=src.height, count=1,
                           dtype=rasterio.ubyte, crs=src.crs, transform=src.transform, compress='ZSTD') as dst:
            dst.write(out2, 1)
    return

# all files to process
files = glob.glob(f'{args.path}/*1km*seg.tif')

if len(files) == 0:
    print('No files found')
    exit()
else:
    print(f'Found {len(files)} files to process')
    # create output directory
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    # # process files
    with mp.Pool(mp.cpu_count()-10) as p:
        p.map(mask_one, files)
