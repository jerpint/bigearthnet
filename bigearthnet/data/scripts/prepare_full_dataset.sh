export BUGGER_OFF=True

python ../data_parser.py \
--root-path "$SCRATCH/bigearthnet/BigEarthNet-v1.0/" \
--splits-path "$SCRATCH/bigearthnet/splits/" \
--output-hub-path "$SCRATCH/bigearthnet/bigearthnet-hub/" \
