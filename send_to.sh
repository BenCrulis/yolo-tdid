#!/bin/bash


dst=blois

if [[ $1 == "blois" ]]; then
    dst=blois:remote/yolo_tdid
elif [[ $1 == "gm" ]]; then
    dst=gm:yolo_tdid
elif [[ $1 == "dep" ]]; then
    dst=dep:yolo_tdid
elif [[ $1 == "all" ]]; then
    sc=`basename "$0"`
    bash $sc blois
    bash $sc gm
    bash $sc dep
    exit 0
else
    echo "Usage: $0 [blois|gm|dep]"
    exit 1
fi

echo "sending to: $dst"


# Send the file to the blois server
rsync -zrtv --exclude '*.pyc' --exclude '*embeddings*.pt' --exclude '*.csv*' --exclude '*.png' --exclude '*.jpg'\
    --exclude tego_crops --exclude venv --exclude wandb . "$dst"

