python ../prepare_mini_dataset.py \
--splits-dir "$SCRATCH/bigearthnet/splits/" \
--output-dir "$SCRATCH/bigearthnet/bigearthnet-mini/" \
--dataset-root-dir $SCRATCH/bigearthnet/BigEarthNet-v1.0 \
--split-samples 80 10 10 \

python ../data_parser.py \
--root-path "$SCRATCH/bigearthnet/bigearthnet-mini/data/" \
--splits-path "$SCRATCH/bigearthnet/bigearthnet-mini/splits/" \
--output-hub-path "$SCRATCH/bigearthnet/bigearthnet-mini/" \

cd $SCRATCH/bigearthnet/
tar  -zcf bigearthnet-mini.tar bigearthnet-mini/
