# Script to test the Transformer-big model with PCA pruning achieving a BLEU Score of 28+.
# Note: Disable (i.e. For Inference) the Pruning in the "prune" module.

# Export variables
TF_CPP_MIN_LOG_LEVEL=3 # 0:All, 1:Info, 2:Warning , 3:None
PARAM_SET=big
DATA_DIR=/datasets/wmt
MODEL_DIR=model_saved/pca_pruned_$PARAM_SET
VOCAB_FILE=$DATA_DIR/vocab.ende.32768
NUM_GPUS=1
BATCH_SIZE=$((4096 * NUM_GPUS))
DECODE_BATCH_SIZE=32
EPOCHS=20
STEPS_EPOCH=5000
TRAIN_STEPS=$((STEPS_EPOCH * EPOCHS))
MAX_LENGTH=64

# Test the model.
echo "Model Testing..."
start=`date +%s`
python3 transformer_main.py --mode=eval --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
    --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET \
    --train_steps=$TRAIN_STEPS --steps_between_evals=$STEPS_EPOCH \
    --batch_size=$BATCH_SIZE --decode_batch_size=$DECODE_BATCH_SIZE --max_length=$MAX_LENGTH \
    --bleu_source=$DATA_DIR/newstest2014.en \
    --bleu_ref=$DATA_DIR/newstest2014.de \
    --num_gpus=$NUM_GPUS \
    --enable_time_history=true
end=`date +%s`
runtime=$((end-start))
echo "Model Testing time is" $runtime "seconds."
