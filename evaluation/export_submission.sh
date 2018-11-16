mkdir $2
PYTHONPATH=$PYTHONPATH:. python evaluation/format_for_glue.py \
	--input-base-path $1 \
	--output-base-path $2
cp /home/zp489/scratch/working/bowman/bert_submissions/template/*.tsv $2
zip -j -D $2/submission.zip $2/*.tsv
