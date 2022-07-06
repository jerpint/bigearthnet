python ../prepare_mini_dataset.py \
--splits-dir "$SCRATCH/bigearth/splits/" \
--output-dir "$SCRATCH/bigearth/bigearthnet-mini/" \
--dataset-root-dir $SCRATCH/bigearth/BigEarthNet-v1.0 \
--split-samples 160 20 20 \

python ../data_parser.py \
--root-path "$SCRATCH/bigearth/bigearthnet-mini/" \
--splits-path "$SCRATCH/bigearth/bigearthnet-mini/splits/" \
--output-hub-path "$SCRATCH/bigearth/bigearthnet-mini/hub/" \
