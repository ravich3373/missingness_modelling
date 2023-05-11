#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=32GB
#SBATCH --job-name=TransformImpute
#SBATCH --output=imputation_output_%j.txt
#SBATCH --gres=gpu:mi50:1

module purge

singularity exec --nv \
	    --overlay /scratch/sp6686/pytorch-example/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python ./TransformImpute.py"
