python ../prepare_mini_dataset.py \
--splits-dir "$SCRATCH/bigearth/splits/" \
--output-dir "$SCRATCH/bigearth/bigearthnet-mini/" \
--dataset-root-dir $SCRATCH/bigearth/BigEarthNet-v1.0 \
--split-samples 80 10 10 \

python ../data_parser.py \
--root-path "$SCRATCH/bigearth/bigearthnet-mini/data/" \
--splits-path "$SCRATCH/bigearth/bigearthnet-mini/splits/" \
--output-hub-path "$SCRATCH/bigearth/bigearthnet-mini/" \

cd $SCRATCH/bigearth/
tar  -zcf bigearthnet-mini.tar bigearthnet-mini/
