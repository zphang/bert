DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
source $DIR/../_base.sh

BERT_MODEL_INIT_PATH=$BERT_DIR/bert_model.ckpt
BERT_EXP_NAME=large_none_ft
BERT_EXP_RAND=10000
