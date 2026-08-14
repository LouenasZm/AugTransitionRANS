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
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

# --- Restarting after the time limit is hit ---
# learnRANStranition.py checkpoints the leader GA to <output_dir>/checkpoint.pkl
# after every generation and resumes from it automatically if that file
# exists (see src/bilevel/checkpoint.py). --requeue asks SLURM to resubmit
# this same job if it's killed for exceeding --time; on the next run it just
# picks the checkpoint back up, no flags needed. If your site doesn't permit
# --requeue, the same recovery happens if you just resubmit by hand:
# `sbatch submit_slurm.sh` again. Note the checkpoint lives under
# $SCRATCHDIR/t3b/results/ (per output_dir in config.json) -- make sure the
# cp step below doesn't overwrite it and that resubmissions reuse the same
# scratch directory.

set -euo pipefail

# SLURM sends SIGUSR1 120s before the hard kill (--signal above). Progress is
# already checkpointed every generation, so there's nothing to save here --
# just let the shell exit promptly instead of waiting out SIGTERM/SIGKILL.
trap 'echo "[$(date)] Caught SIGUSR1 - near time limit, exiting for requeue."; exit 0' USR1

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

# Run in the background and `wait` on it rather than blocking in the
# foreground -- bash only runs trap handlers between commands, so if
# `python ...` were run directly in the foreground the USR1 trap above
# would never get a chance to fire while it's running.
python learnRANStranition.py config.json &
wait $!
