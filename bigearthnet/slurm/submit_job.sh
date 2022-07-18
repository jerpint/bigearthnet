#!/bin/bash
#SBATCH --partition=unkillable                           # Ask for unkillable job
#SBATCH -o /network/scratch/p/pintojer/slurm/slurm-%j.out  # Write the log on scratch
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=20G                                        # Ask for 10 GB of RAM
#SBATCH --time=3:00:00                                   # The job will run for 3 hours

# 1. Load the required modules
set -e

module --quiet load miniconda/3

# 2. Load your environment
conda activate bigearth
DATASET=bigearthnet-mini

# 3. Copy your dataset on the compute node
cp $SCRATCH/bigearthnet/$DATASET.tar $SLURM_TMPDIR
ls $SLURM_TMPDIR
tar -xzf $SLURM_TMPDIR/$DATASET.tar -C $SLURM_TMPDIR/

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
cd $HOME/bigearthnet/
pip install -e .
cd bigearthnet
ls $SLURM_TMPDIR/$DATASET
python train.py ++datamodule.dataset_path=$SLURM_TMPDIR/$DATASET ++datamodule.num_workers=8 +trainer.accelerator='gpu' +trainer.devices=1
