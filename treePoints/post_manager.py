import numpy as np
import rasterio
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import geopandas as gpd
from shapely.geometry import Point
from concurrent.futures import ProcessPoolExecutor
import os
import glob
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings("ignore")


class PostprocessorManager:
    def __init__(self):
        ##
        print('PostprocessorManager initiated')

    def processor_run(self, config):

        heatmaps = glob.glob(config.general.output_dir + '*1km*' + config.general.output_suffix_density + '*')
        # check if output file already exists
        locates = [re.search(r'1km_(\d+_\d+)_density', h).group(1) for h in heatmaps]
        # exclude those already processed
        heatmaps = [heatmaps[i] for i in range(len(heatmaps)) if
                    not os.path.exists(os.path.join(config.general.output_dir, f'1km_{locates[i]}_treeCenters.gpkg'))]
        # ipdb.set_trace()
        print(f'Found {len(heatmaps)} heatmaps (tree density maps) to process')
        if not os.path.exists(config.general.output_dir):
            os.makedirs(config.general.output_dir)

        with ProcessPoolExecutor(max_workers=config.postprocess.maxworker) as executor:
            flag = list(tqdm(executor.map(self.heat2map, heatmaps, [config] * len(heatmaps)), total=len(heatmaps))
                       )
        # summary not processed
        out_work = [heatmaps[i] for i in range(len(flag)) if flag[i] == 'processed']
        out_missing_chm = [heatmaps[i] for i in range(len(flag)) if flag[i] == 'error chm']
        out_missing_elevation = [heatmaps[i] for i in range(len(flag)) if flag[i] == 'error elevation']

        print(f'Processed {len(out_work)} heatmaps')
        print(f'Missing chm for {len(out_missing_chm)} heatmaps')
        print(f'Missing elevation for {len(out_missing_elevation)} heatmaps')
        print('=' * 20)
        print('Merging all outputs')
        self.merge_all(config)
        print('Deleting separate outputs')
        self.delete_separate(config)
        self.ending()


    @staticmethod
    def heat2map(heatmap, config):

        locate = re.search(r'1km_(\d+_\d+)_density', heatmap).group(1)

        try:
            chm = glob.glob(config.postprocess.chm_dir + f'**/*{locate}*.tif', recursive=True)[0]
        except:
            print(f'No chm found for {locate}')
            return 'error chm'
        try:
            elevation = glob.glob(config.postprocess.elevation_dir + f'**/*{locate}*.tif', recursive=True)[0]
        except:
            print(f'No elevation found for {locate}')
            return 'error elevation'

        image = glob.glob(config.general.input_image_dir + f'*{locate}*.tif', recursive=True)[0]
        output = os.path.join(config.general.output_dir, f'1km_{locate}_treeCenters.gpkg')

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
                heat[ndvi < 0] = 0  # only filter out very low ndvi
                thress = max(heat.max() * config.postprocess.alpha, config.postprocess.thres_abs)
                coords = peak_local_max(heat, min_distance=config.postprocess.min_dis, threshold_abs=thress, exclude_border=False)
                with rasterio.open(chm) as src2:
                    height = src2.read(1)
                    height = ndi.zoom(height, (heat.shape[0] / height.shape[0], heat.shape[1] / height.shape[1]))

                    with rasterio.open(elevation) as src3:
                        elev = src3.read(1)
                        elev = ndi.zoom(elev, (heat.shape[0] / elev.shape[0], heat.shape[1] / elev.shape[1]))
                        point_height = []
                        point_elev = []
                        scanned_area = np.zeros_like(height)
                        if len(coords) != 0:
                            for c in coords:
                                x, y = c
                                point_height.append(height[max(0, x - config.postprocess.height_window):min(height.shape[0],
                                                                                              x + config.postprocess.height_window),
                                                    max(0, y - config.postprocess.height_window):min(height.shape[1],
                                                                                       y + config.postprocess.height_window)].max())
                                point_elev.append(
                                    elev[max(0, x - config.postprocess.height_window):min(elev.shape[0], x + config.postprocess.height_window),
                                    max(0, y - config.postprocess.height_window):min(elev.shape[1],
                                                                       y + config.postprocess.height_window)].max())
                                # a binary mask to record scanned area on the height map, window to deal with the shift between optical and lidar and should cover the entire tree
                                scanned_area[
                                max(0, x - config.postprocess.scan_window):min(height.shape[0], x + config.postprocess.scan_window),
                                max(0, y - config.postprocess.scan_window):min(height.shape[1], y + config.postprocess.scan_window)] = 1

                        # detect peaks from height also
                        thres_h = height.copy()
                        thres_h[thres_h < config.postprocess.low_vege] = 0  # filter out low vegetation
                        thres_h[scanned_area == 1] = 0
                        thres_h[ndvi < config.postprocess.ndvi_tree] = 0
                        # ipdb.set_trace()
                        coords2 = peak_local_max(thres_h, min_distance=config.postprocess.min_dis, threshold_abs=config.postprocess.low_vege,
                                                 num_peaks=config.postprocess.num_peak_chm, exclude_border=False)
                        if len(coords2) != 0:
                            for c in coords2:
                                x, y = c
                                point_height.append(height[max(0, x - config.postprocess.height_window):min(height.shape[0],
                                                                                              x + config.postprocess.height_window),
                                                    max(0, y - config.postprocess.height_window):min(height.shape[1],
                                                                                       y + config.postprocess.height_window)].max())
                                point_elev.append(
                                    elev[max(0, x - config.postprocess.height_window):min(elev.shape[0], x + config.postprocess.height_window),
                                    max(0, y - config.postprocess.height_window):min(elev.shape[1],
                                                                       y + config.postprocess.height_window)].max())

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

    @staticmethod
    def merge_all(config):
        from subprocess import call
        call(['ogrmerge.py', '-f', 'GPKG', '-single', '-o', config.general.output_dir + 'merged_centers.gpkg', config.general.output_dir + '1km*.gpkg',
              '-src_layer_field_content', '{DS_BASENAME}'])
        return

    @staticmethod
    def delete_separate(config):
        for f in glob.glob(config.general.output_dir + '1km*.gpkg'):
            os.remove(f)
        return

    @staticmethod
    def ending():
        victory_dance_3 = '''
             \o/
              |  
             / \
        '''
        success_banner = '''
        ****************************************
        *                                      *
        *            SUCCESS! 🏆               *
        *                                      *
        ****************************************
        '''
        print(victory_dance_3)
        print(success_banner)

