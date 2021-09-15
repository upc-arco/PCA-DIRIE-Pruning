# Script to train the Transformer-big model with UC pruning.
# Note: Enable the UC Pruning in the "prune" module.

# Export variables
TF_CPP_MIN_LOG_LEVEL=2 # 0:All, 1:Info, 2:Warning , 3:None
PARAM_SET=big
DATA_DIR=/datasets/wmt
MODEL_DIR=model_uc_pruned_$PARAM_SET
VOCAB_FILE=$DATA_DIR/vocab.ende.32768
NUM_GPUS=4
BATCH_SIZE=$((4096 * NUM_GPUS)) # Default Test = 4096
EPOCHS=30 # Default Test = 20
STEPS_EPOCH=20000 # Default Test = 5000
TRAIN_STEPS=$((STEPS_EPOCH * EPOCHS))
MAX_LENGTH=64 # Default Test = 64

# Train the model for 600000 steps and evaluate every 20000 steps on multiple GPUs.
# Each train step, takes 4096 tokens per GPU as a batch budget with 64 as sequence maximal length.
echo "Model Training and Evaluation..."
start=`date +%s`
python3 transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
  --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET \
  --train_steps=$TRAIN_STEPS --steps_between_evals=$STEPS_EPOCH \
  --batch_size=$BATCH_SIZE --max_length=$MAX_LENGTH \
  --bleu_source=$DATA_DIR/newstest2014.en \
  --bleu_ref=$DATA_DIR/newstest2014.de \
  --num_gpus=$NUM_GPUS \
  --enable_time_history=true
end=`date +%s`
runtime=$((end-start))
echo "Model Training time is" $runtime "seconds."
