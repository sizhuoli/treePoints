#!/bin/bash

input_dir=$1
output_dir=$2
mkdir -p $output_dir


for file in $input_dir/*.tif; do
  filename=$(basename $file)
  gdal_translate -of GTiff -outsize 50% 50% -r average $file $output_dir/$filename
done