#!/bin/bash
#SBATCH --job-name=tds_vanilla
#SBATCH --output=/scratch/users/yaowei/tds/experiments_new/outputs/logs/tds_aug%A_%a.out
#SBATCH --error=/scratch/users/yaowei/tds/experiments_new/outputs/logs/tds_aug%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu,btrippe,owners,roxanad,stat
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM:80GB
#SBATCH --array=1-30
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yaowei@berkeley.edu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source /home/groups/btrippe/yaowei/miniconda3/etc/profile.d/conda.sh
conda activate tds_new
module load python/3.9 cuda/11.7
module load system ruse

# absolute path to motif_pdbs directory:
PDB_DIR=/home/groups/btrippe/yaowei/esm_motifbench/MotifBench/motif_pdbs

# build a sorted bash array of all .pdb stems:
mapfile -t PDB_LIST < <(ls $PDB_DIR/*.pdb | sort | xargs -n1 basename -s .pdb)

# pick the test‐name for this task index:
TEST_NAME=${PDB_LIST[$SLURM_ARRAY_TASK_ID-1]}
SAMPLE_IDX=$(( SLURM_ARRAY_TASK_ID - 1 )) # this isn't actually right -> fix later

echo "Running test case: $TEST_NAME (task $SLURM_ARRAY_TASK_ID of ${#PDB_LIST[@]})"

BASE_DIR=/scratch/users/yaowei/tds/twisted_diffusion_sampler/protein_exp

####### Hyperparameters #######
WEIGHTS_PATH=${BASE_DIR}/weights/paper_weights.pth
NUM_SAMPLES=100
K=8
NUM_STEPS_GEOM=200
STRUCT_TWIST_SCALE=2
TEST_CASES_CSV=${BASE_DIR}/motif_scaffolding/mb_test_cases.csv
OUT_DIR=/scratch/users/yaowei/tds/experiments_new/outputs/run_vanilla_${NUM_STEPS_GEOM}_${STRUCT_TWIST_SCALE}_${K}

python experiments/inference_particle_filter.py \
    inference.weights_path=$WEIGHTS_PATH \
    inference.output_dir=$OUT_DIR \
    inference.motif_scaffolding.inpaint_cases_csv=$TEST_CASES_CSV \
    inference.motif_scaffolding.number_of_samples=$NUM_SAMPLES \
    inference.particle_filtering.number_of_particles=$K \
    inference.diffusion.num_t=$NUM_STEPS_GEOM \
    inference.motif_scaffolding.test_name=$TEST_NAME \
    inference.pmpnn_dir=${BASE_DIR}/ProteinMPNN \
    inference.pt_hub_dir=${BASE_DIR}/.cache/torch \

# python collect_tds_results.py