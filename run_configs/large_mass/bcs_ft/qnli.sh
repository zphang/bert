DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
source $DIR/_base.sh

BERT_TASK_NAME=qnli
BERT_FOL_NAME=QNLI
BERT_TASK_RAND=1006

BERT_RAND_SEED=`expr $BERT_EXP_RAND + $BERT_TASK_RAND + $BERT_RAND_INPUT`
BERT_OUTPUT_DIR=$OUTPUT_BASE_DIR/$BERT_EXP_NAME/run_$BERT_RAND_INPUT/$BERT_TASK_NAME