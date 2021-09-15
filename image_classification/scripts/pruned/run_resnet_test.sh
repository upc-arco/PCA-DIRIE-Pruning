# Script to test the ResNet50 model with pruning.
# Note: Disable (i.e. For Inference) the Pruning in the "prune" module.

# Model Path:
#MODEL_DIR="resnet_saved_model/pruned/uc_0p75/"
MODEL_DIR="resnet_saved_model/pruned/uc_1p00/"

# Dataset Path:
DATA_DIR="/datasets/imagenet/"

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
