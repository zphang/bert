DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
source $DIR/_base.sh

BERT_TASK_NAME=mnli
BERT_FOL_NAME=MNLI

BERT_OUTPUT_DIR=$OUTPUT_BASE_DIR/$BERT_EXP_NAME/$BERT_TASK_NAME