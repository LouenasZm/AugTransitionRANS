#!/bin/bash
#SBATCH --account=ACCOUNT_NAME
#SBATCH --job-name=bilevel-rans-t3b
#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=92G
#SBATCH --cpus-per-task=40
#SBATCH --time=23:30:00
#SBATCH --output=slurm-%j.out
 
set -euo pipefail
 
# Cap BLAS thread pools to 1 per worker process. LeaderGA already does this
# defensively via threadpoolctl at each follower solve, but exporting it here
# too is a cheap belt-and-braces guard against any library that reads these
# at import time (before threadpoolctl gets a chance to patch it).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1                                                                                                                                                       
 
# --- cluster-specific setup: edit these two lines for your site ---
# module load python/3.11
source /path/to/framework/.venv/bin/activate
 
cd $SCRATCHDIR/t3b
 
cp $HOMEDIR/loo_bilevel/t3b/*py .
cp $HOMEDIR/loo_bilevel/t3b/config.json .
cp -r $HOMEDIR/loo_bilevel/t3b/AugCfd/ .
 
python learnRANStranition.py config.json
