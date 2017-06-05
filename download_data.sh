#! /usr/bin/env bash

DATA_FOLDER="data"
VISUAL_GENOME="../visual_genome"
DATA_URL="https://s3-sa-east-1.amazonaws.com/textobjdetection/data.tar.bz2"
DATA_FILE="data.tar.bz2"

if [ ! -d $DATA_FOLDER ]; then
    printf "\nDownloading dataset splits...\n"
    mkdir $DATA_FOLDER
    cd $DATA_FOLDER
    aria2c -x 8 $DATA_URL
    tar -xvjf $DATA_FILE
    rm $DATA_FILE
    cd ..
fi

if [ ! -d $VISUAL_GENOME ]; then
    printf "\nDownloading Visual Genome dataset (This may take a while...)"
    mkdir $VISUAL_GENOME
    cd $VISUAL_GENOME
    aria2c -x 8 "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
    aria2c -x 8 "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
    aria2c -x 8 "http://visualgenome.org/static/data/dataset/image_data.json.zip"
    aria2c -x 8 "http://visualgenome.org/static/data/dataset/region_descriptions.json.zip"
    aria2c -x 8 "http://visualgenome.org/static/data/dataset/objects.json.zip"
    aria2c -x 8 "http://visualgenome.org/static/data/dataset/attributes.json.zip"
    aria2c -x 8 "http://visualgenome.org/static/data/dataset/relationships.json.zip"
    aria2c -x 8 "http://visualgenome.org/static/data/dataset/synsets.json.zip"
    aria2c -x 8 "http://visualgenome.org/static/data/dataset/region_graphs.json.zip"
    aria2c -x 8 "http://visualgenome.org/static/data/dataset/scene_graphs.json.zip"

    printf "Uncompressing data..."
    unzip "*.zip"
    rm *.zip
fi
