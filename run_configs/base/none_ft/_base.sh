DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
source $DIR/../_base.sh

BERT_MODEL_INIT_PATH=$BERT_DIR/bert_model.ckpt
BERT_PYTORCH_MODEL_INIT_PATH=$BERT_DIR/pytorch_model.bin
BERT_EXP_NAME=none_ft
