# get into correct directory for multi modality data
cd data
cd T-X_pair_data/
# download cc3m
rm -rf webvid
mkdir webvid
cd webvid
cp ../../scripts/webvid.sh .
wget -nc http://www.robots.ox.ac.uk/~maxbain/webvid/results_2M_train.csv

video2dataset --url_list="results_10M_train.csv" \
        --input_format="csv" \
        --output-format="webdataset" \
	    --output_folder="dataset" \
        --url_col="contentUrl" \
        --caption_col="name" \
        --save_additional_columns='[videoid,page_idx,page_dir,duration]' \
        --enable_wandb=True \
	--config="./webvid.sh" \

cd ../

# webvid download
# wget -nc http://www.robots.ox.ac.uk/~maxbain/webvid/results_2M_train.csv

