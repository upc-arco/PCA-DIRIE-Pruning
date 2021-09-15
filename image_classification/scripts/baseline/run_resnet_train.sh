# Script to train the ResNet50 model achieving top-1 accuracy of around 76% in 90 epochs.
# Note: Disable the Pruning in the "prune" module.

# Model Path:
MODEL_DIR="resnet_model/"

# Dataset Path:
DATA_DIR="/datasets/imagenet/"

# NCCL Debug:
#export NCCL_DEBUG=WARN

# Run the training and evaluation loop, and measure the runtime:
echo "Model Training and Evaluation..."
start=`date +%s`
python3 classifier_trainer.py \
  --mode=train_and_eval \
  --model_type=resnet \
  --dataset=imagenet \
  --model_dir=$MODEL_DIR \
  --data_dir=$DATA_DIR \
  --config_file=configs/examples/resnet/imagenet/gpu.yaml \
  --params_override='runtime.num_gpus=3'
end=`date +%s`
runtime=$((end-start))
echo "Model Training time is" $runtime "seconds."
