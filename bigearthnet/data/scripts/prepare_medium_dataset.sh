export BUGGER_OFF=True

set -e

python prepare_mini_dataset.py \
--splits-dir "$SCRATCH/bigearthnet/splits/" \
--output-dir "$SCRATCH/bigearthnet/bigearthnet-medium/" \
--dataset-root-dir $SCRATCH/bigearthnet/BigEarthNet-v1.0 \
--seed 42 \
--split-samples 8000 1000 1000 \

python data_parser.py \
--root-path "$SCRATCH/bigearthnet/bigearthnet-medium/data/" \
--splits-path "$SCRATCH/bigearthnet/bigearthnet-medium/splits/" \
--output-hub-path "$SCRATCH/bigearthnet/bigearthnet-medium/" \

echo "compressing folder to tar archive..."
cd $SCRATCH/bigearthnet/
tar  -zcf bigearthnet-medium.tar bigearthnet-medium/
echo "done"
