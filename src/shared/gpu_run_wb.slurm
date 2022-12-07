#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=myTest
#SBATCH --mail-type=END
#SBATCH --mail-user=yp2201@nyu.edu
#SBATCH --output=slurm_%j.out

module purge    
singularity exec --nv --bind $SCRATCH/comp_gen --overlay $SCRATCH/overlay-25GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
pip install rouge
CUDA_LAUNCH_BLOCKING=1 python $SCRATCH/capstone_wb/wb_code_v2.py conclusion --finetune
"

## choices=["Justia_summary", "Justia_holding", "facts_of_the_case", "question", "conclusion"]
## python $SCRATCH/capstone_wb/wb_code.py Justia_holding --finetune
## python $SCRATCH/capstone_wb/wb_code.py
## CUDA_LAUNCH_BLOCKING=1 python $SCRATCH/capstone_wb/wb_code_v2.py Justia_summary --finetune