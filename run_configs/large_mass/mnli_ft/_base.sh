DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
source $DIR/../_base.sh

BERT_MODEL_INIT_PATH=/home/zp489/scratch/working/bowman/bert/large/none_ft/mnli/model.ckpt-49087
BERT_EXP_NAME=large_mnli_ft
BERT_EXP_RAND=20000
