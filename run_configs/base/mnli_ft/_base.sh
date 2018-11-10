DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
source $DIR/../_base.sh

BERT_MODEL_INIT_PATH=/scratch/zp489/working/bowman/bert/initial/bert_test_mnli/model.ckpt-36815
BERT_EXP_NAME=mnli_ft
