# Script to retrain the ResNet50 model with pruning.
# Note: Enable the Pruning in the "prune" module.

# Model Path:
MODEL_DIR="resnet_pruned_model/"

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
  --params_override='runtime.num_gpus=3, train.epochs=90'
end=`date +%s`
runtime=$((end-start))
echo "Model Training time is" $runtime "seconds."
