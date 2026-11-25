# Iteration Log

Each entry records what was tried, what was measured, what changed because of it, and the result. Win rates are on validation teams. Intervals are 95% Wilson.

## Iteration 0: train directly against the strongest opponent (2026-11-25)

**Tried.** Double DQN (`dqn_realistic_heuristic_v1`) and PPO (`ppo_realistic_heuristic_v1`), realistic observations, 200K environment steps each, trained only against `SimpleHeuristicsPlayer`. Default configs in `src/dqn.py` and `src/ppo.py`.

**Measured.**

| Agent | Val win rate vs heuristic, best checkpoint | vs heuristic, last eval | Training win rate, first to last fifth of episodes | Mean battle length, first to last fifth |
|---|---|---|---|---|
| DQN | 0.10 (step 10K) | 0.01 | 0.03 to 0.11 | 41 to 109 turns |
| PPO | 0.05 (step 30K) | 0.03 | 0.04 to 0.05 | 49 to 88 turns |

For reference, `RandomPlayer` beats the heuristic 3.6% of the time (experiment 1). Neither agent improved on that against the heuristic. The best DQN checkpoint came from step 10K, before learning had properly started, so it reflects evaluation noise.

200 extra validation battles per opponent with the saved `best.pt` checkpoints, plus an untrained network as a reference:

| Policy | vs random | vs max_power | vs heuristic |
|---|---|---|---|
| Untrained Q-network, greedy | 0.655 | 0.295 | |
| DQN v1 | 0.92 | 0.395 | |
| PPO v1 | 0.35 | 0.10 | 0.04 |

**Reading.** DQN learned something general (0.655 to 0.92 against random), but against the heuristic almost every training battle is a loss. With nearly every episode ending at -1, the clearest signal left is that a loss further in the future is discounted more, and in both runs battles got longer over training. PPO also switched on 40% of its turns against the heuristic (393 of 968 actions over 30 battles), and its policy is worse than an untrained network against random.

**Changed.** Added the opponent curriculum planned in `docs/design.md`: train against `RandomPlayer`, move to `MaxBasePowerPlayer` once the validation win rate reaches 0.6, then to `SimpleHeuristicsPlayer` (`Curriculum` in `src/env.py`). The best checkpoint is now picked only from evaluations against the last opponent. Everything else is unchanged.

**Result.** Runs `dqn_realistic_v2` and `ppo_realistic_v2`, 200K steps each.

| Agent | Stage reached | Val win rate per stage at advance or end | Best val win rate vs heuristic |
|---|---|---|---|
| DQN | max_power (never advanced) | random 0.86 at 20K; max_power 0.20 at 200K, peak 0.34 | not reached |
| PPO | heuristic at 90K | random 0.65 at 20K; max_power 0.68 at 90K; heuristic 0.06 at 200K | 0.16 at 140K (v1: 0.05) |

Mean battle length no longer grows: training battles averaged 66 turns in the first fifth and 57 in the last for PPO, 75 and 59 for DQN (v1: 49 to 88 and 41 to 109). For PPO the curriculum helped: its best win rate against the heuristic went from 0.05 to 0.16. It then declined to 0.06 over the last 60K steps, while policy entropy fell from 1.76 to about 0.8.

DQN's training win rate against max_power climbed to 0.69 in the last fifth of episodes, while its validation win rate stayed between 0.10 and 0.34. To separate the training teams from the validation teams, both agents were evaluated on each split for 300 battles (both sides always draw from the same split, so team strength is symmetric):

| Policy | max_power, train teams | max_power, val teams | heuristic, train teams | heuristic, val teams |
|---|---|---|---|---|
| DQN v2, final | 0.70 | 0.16 | 0.337 | 0.013 |
| PPO v2, best | 0.50 | 0.64 | 0.183 | 0.137 |

DQN overfits the 11 training teams: it wins 0.70 on them and 0.16 on unseen ones against the same opponent, and it beats the heuristic, which it never trained against, 34% of the time on training teams but 1% on validation teams. PPO shows no such gap. The curriculum gate is what exposed this, since it advances on validation win rate and DQN never got there.

## Iteration 2: more training teams (2026-11-25)

**Tried.** Iteration 1's train/val split showed DQN overfitting the 11 training teams, so the training pool was enlarged. 2,475 teams from Metamon's `hl_05_26` ladder set were added to the training split, after removing any team with the same six Pokemon as a validation or test team (`data/README.md`). Validation and test teams are unchanged, so results stay comparable with iteration 1. Configs, curriculum and step budget are also unchanged: runs `dqn_realistic_v3` and `ppo_realistic_v3`, 200K steps each.

On the new pool the scripted baselines match the competitive teams: over 300 battles each, max_power beats the heuristic 0.19 (val teams: 0.178) and random beats max_power 0.09 (val: 0.116).

**Result.**

| Agent | Stage reached | Val win rate per stage at advance or end | Best val win rate vs heuristic |
|---|---|---|---|
| DQN | max_power (never advanced) | random 1.00 at 10K; max_power 0.30 at 200K, range 0.18 to 0.51 | not reached |
| PPO | heuristic at 110K | random 0.62 at 20K; max_power 0.61 at 100K; heuristic 0.08 at 200K | 0.20 at 190K (v2: 0.16) |

The same 300-battle evaluation per split as iteration 1. "Train teams" now means the expanded pool, which is almost all ladder teams. Validation teams are the same as before, so the validation columns compare directly with v2.

| Policy | max_power, train teams | max_power, val teams | heuristic, train teams | heuristic, val teams |
|---|---|---|---|---|
| DQN v2, final | 0.70 | 0.16 [0.12, 0.21] | 0.337 | 0.013 [0.005, 0.034] |
| DQN v3, final | 0.303 | 0.327 [0.28, 0.38] | 0.087 | 0.11 [0.08, 0.15] |
| PPO v2, best | 0.50 | 0.64 | 0.183 | 0.137 [0.10, 0.18] |
| PPO v3, best | 0.387 | 0.50 | 0.127 | 0.157 [0.12, 0.20] |

DQN's train/val gap is gone, and its validation win rates roughly doubled against max_power and rose from 1% to 11% against the heuristic. Both changes are outside the confidence intervals. The agent is no longer memorizing teams, but it is also not learning much: its training win rate against max_power reached only 0.23 in the last fifth of episodes (v2: 0.69 on 11 teams), and it never reached the 0.6 gate. PPO is unchanged within noise against the heuristic. Its max_power numbers are lower, but its best checkpoint is picked on heuristic win rate, so they are not tuned for max_power.

**Next.** DQN's limit has moved from overfitting to slow learning on a varied pool, so regularization is no longer the indicated fix. The planned hyperparameter search (learning rate, discount, exploration schedule) and a longer step budget are the next candidates.
