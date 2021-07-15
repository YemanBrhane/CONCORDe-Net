#!/bin/bash
#!
#! SLURM job script for clust1 - gpunodes
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#SBATCH --job-name=IMC
#SBATCH --partition=gpunodes
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=30-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yeman.hagos@cruk.cam.ac.uk
#SBATCH --no-requeue
#SBATCH --output=/%x-slurm.%A.%a.out
#SBATCH --error=/%x-slurm.%A.%a.err

###############################################################
### Execute script                                         ####
###############################################################
data_dir='/path/to/data'
counter_dir='/path/to/cell countter model'
output='/path/to/output'
code='/path/to/code'

echo -e "JobID: $SLURM_JOB_ID\n======"
echo -e "ArrayID: $SLURM_ARRAY_TASK_ID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

HOME=/home/$USER \
    singularity exec \
    --bind "$data_dir:/data" \
    --bind "$code:/code" \
    --bind "$counter_dir:/counter_dir" \
    --bind "$output:/output" \
    docker://yhagos/concorde:imcgpu /bin/bash --norc --noprofile -c "
    which ls;
    python /code/main.py  --data_dir /data --counter_dir /counter_dir --output /output
    "

exit 0
