#!/bin/bash
FILES_IN=./img/*.png
DIR_OUT=./img_grayscale
for f in $FILES_IN
do
    filename=$(basename -- "$f")
    ffmpeg -loglevel panic -i "$f" -vf hue=s=0 "$DIR_OUT/$filename"
done

