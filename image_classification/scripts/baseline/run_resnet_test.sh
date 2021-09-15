# Script to test the ResNet50 model achieving top-1 accuracy of around 76%.
# Note: Disable the Pruning in the "prune" module.

# Model Path:
MODEL_DIR="resnet_saved_model/baseline/"

# Dataset Path:
DATA_DIR="/datasets/imagenet/"

# NCCL Debug:
#export NCCL_DEBUG=WARN

# Run the evaluation, and measure the runtime:
echo "Model Testing..."
start=`date +%s`
python3 classifier_trainer.py \
  --mode=eval \
  --model_type=resnet \
  --dataset=imagenet \
  --model_dir=$MODEL_DIR \
  --data_dir=$DATA_DIR \
  --config_file=configs/examples/resnet/imagenet/gpu.yaml \
  --params_override='runtime.num_gpus=1, validation_dataset.batch_size=256'
end=`date +%s`
runtime=$((end-start))
echo "Model Testing time is" $runtime "seconds."
