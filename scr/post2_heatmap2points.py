import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import ipdb
import geopandas as gpd
from shapely.geometry import Point
from concurrent.futures import ProcessPoolExecutor
import os
import glob
from tqdm import tqdm
# from multiprocessing import Pool
import re
import warnings
warnings.filterwarnings("ignore")


def heat2map(heatmap, args):

    locate = re.search(r'1km_(\d+_\d+)_density', heatmap).group(1)

    try:
        chm = glob.glob(args.chm_dir + f'**/*{locate}*.tif', recursive=True)[0]
    except:
        print(f'No chm found for {locate}')
        return 'error chm'
    try:
        elevation = glob.glob(args.elevation_dir + f'**/*{locate}*.tif', recursive=True)[0]
    except:
        print(f'No elevation found for {locate}')
        return 'error elevation'

    image = glob.glob(args.image_dir + f'*{locate}*.tif', recursive=True)[0]
    output = os.path.join(args.output_dir, f'1km_{locate}_treeCenters.gpkg')


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
            coords = peak_local_max(heat, min_distance=args.min_dis, threshold_abs=thress, exclude_border=False)
            with rasterio.open(chm) as src2:
                height = src2.read(1)
                height = ndi.zoom(height, (heat.shape[0]/height.shape[0], heat.shape[1]/height.shape[1]))

                with rasterio.open(elevation) as src3:
                    elev = src3.read(1)
                    elev = ndi.zoom(elev, (heat.shape[0]/elev.shape[0], heat.shape[1]/elev.shape[1]))
                    point_height = []
                    point_elev = []
                    scanned_area = np.zeros_like(height)
                    if len(coords) != 0:
                        for c in coords:
                            x, y = c
                            point_height.append(height[max(0, x-args.height_window):min(height.shape[0], x+args.height_window), max(0, y-args.height_window):min(height.shape[1], y+args.height_window)].max())
                            point_elev.append(elev[max(0, x-args.height_window):min(elev.shape[0], x+args.height_window), max(0, y-args.height_window):min(elev.shape[1], y+args.height_window)].max())
                            # a binary mask to record scanned area on the height map, window to deal with the shift between optical and lidar and should cover the entire tree
                            scanned_area[max(0, x-args.scan_window):min(height.shape[0], x+args.scan_window), max(0, y-args.scan_window):min(height.shape[1], y+args.scan_window)] = 1

                    # detect peaks from height also
                    thres_h = height.copy()
                    thres_h[thres_h < args.low_vege] = 0 # filter out low vegetation
                    thres_h[scanned_area == 1] = 0
                    thres_h[ndvi < args.ndvi_tree] = 0
                    # ipdb.set_trace()
                    coords2 = peak_local_max(thres_h, min_distance=args.min_dis, threshold_abs=args.low_vege, num_peaks=args.num_peak_chm, exclude_border=False)
                    if len(coords2) != 0:
                        for c in coords2:
                            x, y = c
                            point_height.append(height[max(0, x-args.height_window):min(height.shape[0], x+args.height_window), max(0, y-args.height_window):min(height.shape[1], y+args.height_window)].max())
                            point_elev.append(elev[max(0, x-args.height_window):min(elev.shape[0], x+args.height_window), max(0, y-args.height_window):min(elev.shape[1], y+args.height_window)].max())

                point_height = np.array(point_height)
                point_elev = np.array(point_elev)
                # save to file
                transform = src1.transform
                crs = src1.crs
                # merge coords
                coords = np.concatenate([coords, coords2])
                if len(coords) != 0:
                    geo_points = [Point(transform * (y, x)) for x, y in coords]
                    gdf = gpd.GeoDataFrame(geometry=geo_points, crs=crs)
                    gdf['TreeHeight'] = point_height
                    gdf['TreeElevation'] = point_elev
                    # remove points with height < 1 m
                    gdf = gdf[gdf['TreeHeight'] >= 1]
                    gdf.to_file(output, driver='GPKG')
                else:
                    geo_points = []
                    gdf = gpd.GeoDataFrame(geometry=[], crs=crs)
                    gdf.to_file(output, driver='GPKG')
    del heat, height, elev, coords, point_height, point_elev, geo_points, gdf

    return 'processed'


def main(args):

    heatmaps = glob.glob(args.heatmap_dir + '*1km*density.tif')
    # check if output file already exists
    locates = [re.search(r'1km_(\d+_\d+)_density', h).group(1) for h in heatmaps]
    # exclude those already processed
    heatmaps = [heatmaps[i] for i in range(len(heatmaps)) if not os.path.exists(os.path.join(args.output_dir, f'1km_{locates[i]}_treeCenters.gpkg'))]
    # ipdb.set_trace()
    print(f'Found {len(heatmaps)} heatmaps (tree density maps) to process')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with ProcessPoolExecutor(max_workers=args.maxworker) as executor:
        flag = list(tqdm(executor.map(heat2map, heatmaps, [args]*len(heatmaps)), total=len(heatmaps)))
    #summary not processed
    out_work = [heatmaps[i] for i in range(len(flag)) if flag[i] == 'processed']
    out_missing_chm = [heatmaps[i] for i in range(len(flag)) if flag[i] == 'error chm']
    out_missing_elevation = [heatmaps[i] for i in range(len(flag)) if flag[i] == 'error elevation']

    return out_work, out_missing_chm, out_missing_elevation


def merge_all(out_dir):
    from subprocess import call
    call(['ogrmerge.py', '-f', 'GPKG', '-single', '-o', out_dir + 'merged_centers.gpkg', out_dir + '1km*.gpkg', '-src_layer_field_content', '{DS_BASENAME}'])
    return

def delete_separate(out_dir):
    for f in glob.glob(out_dir + '1km*.gpkg'):
        os.remove(f)
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert heatmap (tree density map) to tree center points')
    parser.add_argument('--heatmap_dir', default='/home/sizhuo/Desktop/code_repository/TreePointsStepbyStep/test_example/preds/', type=str, help='directory to heatmaps')
    parser.add_argument('--chm_dir', default='..../elevation/DHM/', type=str, help='directory to chm files')
    parser.add_argument('--elevation_dir', default='..../elevation/DTM/', type=str, help='directory to elevation files')
    parser.add_argument('--image_dir', default='/home/sizhuo/Desktop/code_repository/TreePointsStepbyStep/test_example/', type=str, help='directory to RGBNIR images')
    parser.add_argument('--output_dir', default='/home/sizhuo/Desktop/code_repository/TreePointsStepbyStep/test_example/preds/', type=str, help='directory to save tree center points')
    parser.add_argument('--min_dis', default=10, type=int, help='for chm only, minimum distance between tree centers, in pixels, if 0.25m resolution, 8p=2m, 10p=2.5m')
    parser.add_argument('--thres_abs', default=0.0005, type=float, help='empirical threshold for kernel peak')
    parser.add_argument('--alpha', default=0.2, type=float, help='threshold for kernel peak')
    parser.add_argument('--height_window', default=6, type=int, help='window size to search for tree height and elevation, in pixels')
    parser.add_argument('--scan_window', default=15, type=int, help='window size to scan the area on height map to cover entire tree if already detected by heatmap, in pixels')
    parser.add_argument('--low_vege', default=3, type=int, help='threshold for low vegetation')
    parser.add_argument('--ndvi_tree', default=0, type=float, help='threshold for tree detection using NDVI')
    parser.add_argument('--maxworker', default=15, type=int, help='maximum number of workers')
    parser.add_argument('--num_peak_chm', default=1000, type=int, help='number of peaks to detect from chm')
    args = parser.parse_args()
    out_work, out_missing_chm, out_missing_elevation = main(args)
    merge_all(args.output_dir)
    delete_separate(args.output_dir)
    print('All processing done. :D')
    print(f'Processed {len(out_work)} heatmaps')
    print(f'Missing chm file for {len(out_missing_chm)} heatmaps')
    print(f'Missing elevation file for {len(out_missing_elevation)} heatmaps')



