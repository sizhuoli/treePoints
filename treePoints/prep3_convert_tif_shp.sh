#!/bin/sh
DIR=$1
for i in $DIR/*.tif; do
  if [ ! -f $i.gpkg ]; then
    gdal_polygonize.py -8 $i -b mask -f "GPKG" $i.gpkg
  fi
done
ogrmerge.py -f "GPKG" -single -o $DIR/all_files.gpkg $DIR/*.gpkg -src_layer_field_content {DS_BASENAME}
#find $DIR -name "*.gpkg" ! -name "all_files.gpkg" -exec rm {} \;

# command line code: bash prep3_convert_tif_shp.sh /path/to/dir
