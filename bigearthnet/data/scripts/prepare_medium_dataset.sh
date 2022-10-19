export BUGGER_OFF=True

set -e

python prepare_dataset_subset.py \
--splits-dir "$SCRATCH/bigearthnet/splits/" \
--output-dir "$SCRATCH/bigearthnet/bigearthnet-medium/" \
--dataset-root-dir $SCRATCH/bigearthnet/BigEarthNet-v1.0 \
--seed 42 \
--split-samples 25000 10000 10000 \

python data_parser.py \
--root-path "$SCRATCH/bigearthnet/bigearthnet-medium/raw_data/" \
--splits-path "$SCRATCH/bigearthnet/bigearthnet-medium/splits/" \
--output-hub-path "$SCRATCH/bigearthnet/bigearthnet-medium/" \

echo "compressing folder to tar archive..."
cd $SCRATCH/bigearthnet/
rm -r bigearthnet-medium/raw_data/  # Remove the raw data from the archive
tar  -zcf bigearthnet-medium.tar bigearthnet-medium/
echo "done"
