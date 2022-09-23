#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH -o /network/scratch/p/pintojer/slurm/slurm-%j.out  # Write the log on scratch
#SBATCH --cpus-per-task=4                                # Ask for 4 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=32G                                        # Ask for 32 GB of RAM
#SBATCH --time=24:00:00                                  # The job will run this long

set -e

# Load the required modules and environment
module --quiet load miniconda/3
conda activate bigearth

# Specify the dataset
# DATASET=bigearthnet-full
DATASET=bigearthnet-medium
# DATASET=bigearthnet-mini

# Copy the dataset on the compute node
echo "copying $DATASET to SLURM_TMPDIR..."
cp $SCRATCH/bigearthnet/$DATASET.tar $SLURM_TMPDIR
tar -xzf $SLURM_TMPDIR/$DATASET.tar -C $SLURM_TMPDIR/
echo "Done moving dataset..."
ls $SLURM_TMPDIR/$DATASET

# Launch the job, tell it to save the model in $SLURM_TMPDIR
# and look for the dataset into $SLURM_TMPDIR
cd $HOME/bigearthnet/
pip install -e .
cd bigearthnet
echo "beginning training..."

# Try different learning rates + optimizers on baseline model
python train.py -m \
++datamodule.dataset_dir=$SLURM_TMPDIR ++datamodule.dataset_name=$DATASET ++datamodule.num_workers=4 ++datamodule.batch_size=256 \
++trainer.max_epochs=100 +trainer.accelerator='gpu' +trainer.devices=1 \
++optimizer.name="adam","sgd" ++optimizer.lr=0.01,0.001,0.0001,0.00001 \
++model.hidden_dim=256,512 \
++experiment.group="lr_sweep" \

echo "All done."
