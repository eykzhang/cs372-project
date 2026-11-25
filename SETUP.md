# Setup

Tested on macOS with Python 3.13 and Node.js 26.

## 1. Python environment

```bash
git clone <this repo> && cd <repo>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Local Pokemon Showdown server

All battles run on a local Showdown server. Clone it anywhere outside this repo:

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
git checkout c7fa6f1   # the version all results were produced with
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```

Leave it running. It listens on `localhost:8000`, which is where poke-env connects by default. `--no-security` turns off login and rate limits, which a local training server needs.

## 3. Teams

The teams in `data/teams/` are committed. To rebuild them from the source dataset (needs internet):

```bash
python -m src.build_teams
```

## 4. Run

From the repo root, with the venv active and the server running:

```bash
# experiment 1: scripted players against each other on validation teams
python -m src.run_baselines --split val --battles 500

# train each agent through the opponent curriculum (random, then max_power, then heuristic)
python -m src.train_dqn --run dqn_realistic --obs realistic --opponents random,max_power,heuristic --steps 200000
python -m src.train_ppo --run ppo_realistic --obs realistic --opponents random,max_power,heuristic --steps 200000
```

Training writes `config.json`, `episodes.csv`, `evals.csv` and checkpoints (`best.pt`, `final.pt`) to `models/<run>/`. Use `--obs oracle` to train with the opponent's full team visible. The agent moves to the next opponent once its validation win rate reaches `--advance-at` (default 0.6).

To use a server on another port, set `SHOWDOWN_PORT` (for example `SHOWDOWN_PORT=8001 python -m src.train_dqn ...`). Agent hyperparameters can be overridden without editing code: `--config lr=3e-4 gamma=0.995` sets fields of `DQNConfig` or `PPOConfig`, and the values used are saved in the run's `config.json`.

## 5. Running sweeps on a SLURM cluster (optional)

`slurm/train_array.sbatch` runs one line of a sweep file per array task. Each task copies the Showdown server to node-local disk, starts it on its own port bound to localhost, trains, and shuts it down. `slurm/sweep.txt` lists the hyperparameter sweep.

One-time setup on the cluster. These steps assume Node.js is installed but npm is not, as on the cluster this was developed for, so the built server is copied from a machine that already ran step 2:

```bash
# from your own machine: copy the installed, built Showdown server
rsync -a --exclude .git pokemon-showdown/ <user>@<cluster>:/usr/xtmp/<user>/pokemon-showdown/

# on the cluster, in the repo checkout
uv venv --python 3.13 .venv          # or python3 -m venv .venv with Python 3.13
uv pip install -r requirements.txt   # or pip install -r requirements.txt
mkdir -p slurm/logs
```

Submit the sweep from the repo root, then check on it:

```bash
sbatch --array=1-$(grep -c . slurm/sweep.txt) slurm/train_array.sbatch slurm/sweep.txt
squeue -u $USER
tail -f slurm/logs/cs372-train_<jobid>_<task>.out
```

Set `SHOWDOWN_DIR` if the server is somewhere other than `/usr/xtmp/$USER/pokemon-showdown`. Results land in `models/<run>/` in the checkout.
