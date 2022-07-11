export BUGGER_OFF=True

python ../prepare_mini_dataset.py \
--splits-dir "$SCRATCH/bigearthnet/splits/" \
--output-dir "$SCRATCH/bigearthnet/bigearthnet-medium/" \
--dataset-root-dir $SCRATCH/bigearthnet/BigEarthNet-v1.0 \
--seed 42 \
--split-samples 4000 500 500 \

python ../data_parser.py \
--root-path "$SCRATCH/bigearthnet/bigearthnet-medium/data/" \
--splits-path "$SCRATCH/bigearthnet/bigearthnet-medium/splits/" \
--output-hub-path "$SCRATCH/bigearthnet/bigearthnet-medium/" \

cd $SCRATCH/bigearthnet/
tar  -zcf bigearthnet-medium.tar bigearthnet-medium/
