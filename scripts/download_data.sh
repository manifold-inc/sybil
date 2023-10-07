# get into correct directory for multi modality data
cd data
cd T-X_pair_data/
# download cc3m
cd cc3m
wget -nc https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv
img2dataset --url_list Train_GCC-training.tsv --input_format "tsv" --url_col "url" --caption_col "caption" --output_format webdataset --output_folder cc3m --processes_count 16 --thread_count 64 --image_size 256 --enable_wandb True
cd ../

# webvid download
# wget -nc http://www.robots.ox.ac.uk/~maxbain/webvid/results_2M_train.csv

