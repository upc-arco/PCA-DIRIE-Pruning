# Script to download and prepare the WMT Dataset.

# Variables
DATA_DIR=/datasets/wmt
RAW_DIR=/datasets/wmt/downloads

# Download training/evaluation/test datasets and measure time
echo "Dataset Preparation..."
start=`date +%s`
python3 data_download.py --data_dir=$DATA_DIR --raw_dir=$RAW_DIR
end=`date +%s`
runtime=$((end-start))
echo "Elapsed time is" $runtime "seconds."
