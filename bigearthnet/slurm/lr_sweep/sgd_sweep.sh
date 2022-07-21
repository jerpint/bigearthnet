#!/bin/bash
#SBATCH --partition=main                           # Ask for unkillable job
#SBATCH -o /network/scratch/p/pintojer/slurm/slurm-%j.out  # Write the log on scratch
#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=32G                                        # Ask for 10 GB of RAM
#SBATCH --time=24:00:00                                  # The job will run this long

# 1. Load the required modules
set -e

module --quiet load miniconda/3

# 2. Load your environment
conda activate bigearth
DATASET=bigearthnet-full

# 3. Copy your dataset on the compute node
echo "moving dataset..."
cp $SCRATCH/bigearthnet/$DATASET.tar $SLURM_TMPDIR
tar -xzf $SLURM_TMPDIR/$DATASET.tar -C $SLURM_TMPDIR/
echo "Done moving dataset..."
ls $SLURM_TMPDIR/$DATASET

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
cd $HOME/bigearthnet/
pip install -e .
cd bigearthnet
echo "beginning training..."

# Try different learning rates + optimizer on baseline model
python train.py -m \
++datamodule.dataset_dir=$SLURM_TMPDIR ++datamodule.dataset_name=$DATASET ++datamodule.num_workers=2 ++datamodule.batch_size=256 \
++trainer.max_epochs=20 +trainer.accelerator='gpu' +trainer.devices=1 \
++optimizer.name="sgd" ++optimizer.lr=0.01,0.001,0.0001,0.00001 \
++experiment.group="lr_sweep" \

echo "All done."
