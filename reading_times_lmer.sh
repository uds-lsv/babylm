
#!/bin/bash

PROJECT_PATH=/nethome/jsteuer/git/lsv/babylm
cd $PROJECT_PATH

PYTHON_BIN="/nethome/jsteuer/miniconda3/envs/babylm-rts/bin"
$PYTHON_BIN/python --version

# SAVED_MODELS_DIR="/local/models/babylm/opt_with_alibi"
# SAVED_MODELS_DIR="/local/models/babylm/opt_with_alibi_save_steps"
SAVED_MODELS_DIR="/local/models/babylm/opt_with_alibi_save_steps_wikitext"
OUTPUT_DIR="/nethome/jsteuer/git/lsv/babylm/lmer/wikitext"
NATURALSOTRIES_DIR="/nethome/jsteuer/git/lsv/babylm/naturalstories/naturalstories_RTS"
# BABYLM_TRAIN_FILE="/data/corpora/babylm/babylm_data/babylm_100M/babylm_full.train.txt"
BABYLM_TRAIN_FILE="/data/users/jsteuer/datasets/wikitext-103-v1/wikitext_full.txt"

# for checkpoint in {100,250,500,1000,10000,20000,1}
# for checkpoint in {500,1000,2000,3000,1}
# for checkpoint in {500,}
# for checkpoint in {2000,3000}
# for checkpoint in {1,}
for checkpoint in {5,}
do
    $PYTHON_BIN/python reading_times_lmer.py \
        --checkpoint $checkpoint \
        --babylm_train_file $BABYLM_TRAIN_FILE \
        --models_dir $SAVED_MODELS_DIR \
        --output_dir $OUTPUT_DIR \
        --naturalstories_dir $NATURALSOTRIES_DIR
done

