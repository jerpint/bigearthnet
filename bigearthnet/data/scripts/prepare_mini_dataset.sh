export BUGGER_OFF=True

set -e

python prepare_dataset_subset.py \
--splits-dir "$SCRATCH/bigearthnet/splits/" \
--output-dir "$SCRATCH/bigearthnet/bigearthnet-mini/" \
--dataset-root-dir $SCRATCH/bigearthnet/BigEarthNet-v1.0 \
--split-samples 80 40 40 \
--seed 42 \

python data_parser.py \
--root-path "$SCRATCH/bigearthnet/bigearthnet-mini/raw_data/" \
--splits-path "$SCRATCH/bigearthnet/bigearthnet-mini/splits/" \
--output-hub-path "$SCRATCH/bigearthnet/bigearthnet-mini/" \

echo "compressing folder to tar archive..."
cd $SCRATCH/bigearthnet/
rm -r bigearthnet-mini/raw_data/  # Remove the raw data from the archive
tar -zcf bigearthnet-mini.tar bigearthnet-mini/
echo "done"
