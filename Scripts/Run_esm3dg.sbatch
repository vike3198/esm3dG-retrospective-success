#!/bin/bash
# ESM3dG stability scoring. Edit the 2 marked lines, then: sbatch run_esm3dg.sbatch
# Needs read access to /projects/ryde3462 (ryde3462grp). Nothing to install.
# Output: scores_shard0_of1.csv — use the dG_ensemble column. Higher = more stable.
 
#SBATCH --account=ucb715_asc1                        # <-- EDIT
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=esm3dg_%j.out
 
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"                             # Alpine starts jobs in $HOME
module purge
ml anaconda
conda activate stability

 
ls -1 /scratch/alpine/vike3198/esm3dg/overath/input_pdbs/*.pdb > overath_manifest.txt      # <-- EDIT
 
python /projects/vike3198/absolute-stability-predictor/scripts/score_structures.py \
    --manifest overath_manifest.txt --outdir . --mode monomer --chain A 