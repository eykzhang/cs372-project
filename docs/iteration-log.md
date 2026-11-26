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

## Iteration 3: hyperparameter search and a longer step budget (2026-11-25)

**Tried.** Iteration 2 left DQN learning slowly on the varied team pool, so both agents were trained for 500K steps (up from 200K) with one change at a time from the default config: learning rate, discount and exploration schedule for DQN, and learning rate, discount and entropy coefficient for PPO. Same curriculum and teams as iteration 2, one seed per config, run as a SLURM array (`slurm/sweep.txt`).

**Measured.** In-training evaluations are not comparable across these runs: each one met its opponents at different steps, and three DQN runs never reached the heuristic stage, so they have no `best.pt`. Every run's `final.pt` was therefore evaluated under one protocol, 300 battles per opponent on validation teams (`src/evaluate_runs.py`, `data/results/sweep_val.csv`).

| Run | Change from default | vs max_power | vs heuristic | Mean turns vs heuristic |
|---|---|---|---|---|
| `dqn_sweep_base` | none | 0.173 [0.13, 0.22] | 0.053 [0.03, 0.08] | 37.5 |
| `dqn_sweep_lr3e-4` | lr 1e-4 to 3e-4 | 0.283 [0.24, 0.34] | 0.043 [0.03, 0.07] | 24.4 |
| `dqn_sweep_gamma995` | gamma 0.99 to 0.995 | 0.377 [0.32, 0.43] | 0.070 [0.05, 0.10] | 31.3 |
| `dqn_sweep_eps150k` | epsilon decay 60K to 150K steps | 0.213 [0.17, 0.26] | 0.040 [0.02, 0.07] | 57.0 |
| `ppo_sweep_base` | none | 0.430 [0.38, 0.49] | 0.083 [0.06, 0.12] | 23.3 |
| `ppo_sweep_lr1e-4` | lr 3e-4 to 1e-4 | 0.573 [0.52, 0.63] | 0.150 [0.11, 0.19] | 16.3 |
| `ppo_sweep_gamma995` | gamma 0.99 to 0.995 | 0.687 [0.63, 0.74] | 0.337 [0.29, 0.39] | 16.8 |
| `ppo_sweep_ent03` | entropy coefficient 0.01 to 0.03 | 0.207 [0.16, 0.26] | 0.037 [0.02, 0.06] | 25.3 |

For reference, max_power beats the heuristic 0.178 (experiment 1) and the best earlier agent, PPO v3, reached 0.157 [0.12, 0.20].

**Reading.** PPO with gamma 0.995 is the first agent to beat the heuristic more often than max_power does, and its interval clears both PPO base and PPO v3. A lower learning rate also helped PPO. A larger entropy bonus hurt: the run took 380K steps to pass the max_power gate and finished weakest. No DQN config separates from the others against the heuristic. Gamma 0.995 is the best DQN config against max_power, and the only DQN change whose interval clears the default.

To see what DQN is doing, 40 battles per opponent were played with `dqn_sweep_base` and `ppo_sweep_gamma995` and their protocol logs tallied. DQN switched on 63% of its actions against max_power and 66% against the heuristic (PPO: 38% and 40%, both counts including forced switches after a faint), with 752 and 492 immediate switch-backs (A to B to A) over the 40 battles, against 40 and 32 for PPO. In one battle two DQN Pokemon swapped places for over ten turns without attacking while the opponent's Zapdos chipped both, and in another Golem used Earthquake into Zapdos ten times. The move effectiveness feature for that matchup is the immune value (-1.5), so the information was in the observation. Against max_power the stalling ran the opponent out of PP: it used Struggle 278 times over the 40 battles. Neither agent used Protect or Detect.

**Selected.** Gamma 0.995 for both agents in the experiments that follow. It is the best config for PPO by a wide margin and the best DQN config against max_power, and keeping the discount the same for both algorithms keeps the comparison controlled. All results are single-seed.

## Iteration 4: mixed-opponent curriculum and a longer step budget (in progress)

**Why.** The information-gap runs (experiment 3: both agents, realistic and oracle observations, three seeds each, gamma 0.995, 500K steps) were evaluated with 500 battles per opponent on validation teams (`data/results/info_gap_val.csv`). For DQN the result depends on where each run ended in the curriculum more than on what it observed:

| DQN runs | Runs | vs max_power | vs heuristic |
|---|---|---|---|
| Stayed at the max_power stage | 3 (`dqn_realistic_s0`, `dqn_oracle_s1`, `dqn_oracle_s2`) | 0.27 to 0.40 | 0.05 to 0.12 |
| Advanced to the heuristic stage | 3 (`dqn_realistic_s1`, `dqn_realistic_s2`, `dqn_oracle_s0`) | 0.03 to 0.14 | 0.01 to 0.03 |

`dqn_sweep_base` from iteration 3 also advanced and also collapsed (0.173 vs max_power). Every DQN run that reached the heuristic declined from then on. Validation win rate against the heuristic, averaged over each 100K-step window:

| Run | Heuristic stage from | 200K | 300K | 400K | 500K |
|---|---|---|---|---|---|
| `dqn_realistic_s1` | 140K | 0.08 | 0.04 | 0.02 | 0.01 |
| `dqn_oracle_s0` | 80K | 0.07 | 0.03 | 0.02 | 0.02 |
| `ppo_realistic_s1` | 180K | 0.22 | 0.24 | 0.24 | 0.26 |
| `ppo_realistic_s2` | 90K | 0.12 | 0.07 | 0.17 | 0.21 |

DQN's training win rate in that stage fell to 1-2%, so almost every episode ends at -1, and the collapsed runs also lost what they had learned against max_power (0.03 to 0.06, below `RandomPlayer`'s 0.116). PPO in the same stage was still improving slowly at 500K steps. While DQN wins under 5%, its information gap cannot be measured.

A second reading: in a lost battle the shaping can tell a close loss from a rout by at most 0.2 (knocking out three opponents instead of none is worth 0.05), so every loss returns between about -1.2 and -0.8. A value-based agent has to rank actions by small differences inside that band.

**Plan.** Three arms, each one change from a shared baseline, run with both agents, two seeds each, realistic observations and gamma 0.995 (`slurm/iteration4.txt`, 12 runs):

| Arm | Curriculum | Shaping weights (HP + knockouts) | Steps |
|---|---|---|---|
| `it4_base` | as before | 0.1 + 0.1 | 1M |
| `it4_mixed` | mixed | 0.1 + 0.1 | 1M |
| `it4_shaped` | as before | 0.5 + 0.5 | 1M |

- Mixed curriculum (`--mixed`): after an advance, half of the training battles are against the current opponent and the rest are split evenly among earlier ones, so winnable battles stay in the training data. The advance gate still evaluates against the current opponent only.
- Stronger shaping (`--reward hp_weight=0.5 faint_weight=0.5`): widens the spread between losses fivefold. Shaping stays potential-based, so it cannot be farmed, and a win still always outscores a loss (the potential is positive whenever the opponent's team has all fainted and negative whenever ours has).
- The step budget doubles from 500K for all arms, so `it4_base` against iteration 3 shows what longer training alone does.

All arms also use the exact potential-based form of the shaping, gamma times the new potential minus the old one, where earlier runs used the plain difference. At the old weights the two differ by at most 0.001 per turn.

Episode logs now record each battle's opponent, so training win rates can be split by opponent in the mixed runs. The better curriculum and shaping are then used to rerun the information-gap pairs, with both agents trained the same way.
