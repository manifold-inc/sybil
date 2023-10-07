# get into correct directory for multi modality data
cd data
cd T-X_pair_data/
# download cc3m
rm -rf cc3m
mkdir cc3m
cd cc3m
curl -O https://storage.googleapis.com/gcc-data/Train/GCC-training.tsv

echo -e "caption\turl\n" > header.tsv
cat header.tsv GCC-training.tsv > temp.tsv && mv temp.tsv GCC-training.tsv
rm header.tsv

img2dataset --url_list GCC-training.tsv --input_format "tsv" --url_col "url" --caption_col "caption" --output_format webdataset --output_folder cc3m --processes_count 16 --thread_count 64 --image_size 256 --enable_wandb True

cd cc3m

for file in *.tar; do
    tar -xf "$file"
done

cd ../

# webvid download
# wget -nc http://www.robots.ox.ac.uk/~maxbain/webvid/results_2M_train.csv

