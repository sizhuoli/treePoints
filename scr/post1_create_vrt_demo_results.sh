#!/bin/sh
DIR=$1
data=$2
if [ $data == "prediction" ]; then
  echo "Creating VRT for predictions"
  gdalbuildvrt $DIR/result.vrt $DIR/*seg.tif
elif [ $data == "image" ]; then
  echo "Creating VRT for images"
  gdalbuildvrt $DIR/image.vrt $DIR/*.tif
else
  echo "Please specify the data type: prediction or image"
fi


# command line code to merge predictions: "bash post1_create_vrt_demo_results.sh /path/to/dir prediction"
# command line code to merge images: "bash post1_create_vrt_demo_results.sh /path/to/dir image"