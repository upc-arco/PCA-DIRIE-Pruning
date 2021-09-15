# Script to prepare the ImageNet Dataset.

echo "Dataset Preparation..."
start=`date +%s`
python3 imagenet_tf_datasets.py
end=`date +%s`
runtime=$((end-start))
echo "Elapsed time is" $runtime "seconds."
