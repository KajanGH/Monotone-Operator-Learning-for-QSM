#!/bin/bash -l



# 1. Force bash as the executing shell.
#$ -S /bin/bash

# 2. Request wallclock time (format hours:minutes:seconds).
#$ -l h_rt=20:00:00

# 3. Request 16 gigabyte of RAM.
#$ -l mem=32G
#$ -l gpu=2

# 7. Set the name of the job.
#$ -N mol_training

# 8. Set the working directory to somewhere in your scratch space.  This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID
#$ -wd /lustre/scratch/scratch/zcemksu/mol

#Request 5 gigabyte of TMPDIR space (default is 10GB)
#$ -l tmpfs=5G

# Please send me e-mails of progress :D
#$ -m bae

# 9. Your work should be done in $TMPDIR
cd $TMPDIR
cp -r /lustre/scratch/scratch/zcemksu/mol/data_sim_analytic $TMPDIR/
cp /lustre/scratch/scratch/zcemksu/mol/MonotoneOperatorQSM.py $TMPDIR/

# 11. load modules
module -f unload compilers mpi gcc-libs
module load beta-modules
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0/recommends
module load cuda/11.3.1/gnu-10.2.0
module load cudnn/8.2.1.32/cuda-11.3
module load pytorch/1.11.0/gpu

nvidia-smi



# Set CUDA_LAUNCH_BLOCKING environment variable
export CUDA_LAUNCH_BLOCKING=1

# These echoes output what you are about to run
echo ""
echo "/shared/ucl/apps/python/bundles/gnu-10.2.0/1634821926-39/venv/bin/python MonotoneOperatorQSM.py"
echo "" 
/shared/ucl/apps/python/bundles/gnu-10.2.0/1634821926-39/venv/bin/python MonotoneOperatorQSM.py

# Automatically transfer output to Scratch from $TMPDIR
#Local2Scratch

# Make sure you have given enough time for the copy to complete!
