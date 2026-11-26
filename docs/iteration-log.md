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

## Iteration 4: mixed-opponent curriculum and a longer step budget (2026-11-26)

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

**Measured.** Each run's `final.pt` was evaluated with 500 battles per opponent on validation teams (`src/evaluate_runs.py`, `data/results/iteration4_val.csv`). The heuristic stage column comes from `evals.csv`. The arm means average the two seeds.

| Run | Heuristic stage from | vs max_power | vs heuristic |
|---|---|---|---|
| `ppo_it4_base_s0` | 180K | 0.528 [0.48, 0.57] | 0.136 [0.11, 0.17] |
| `ppo_it4_base_s1` | 190K | 0.722 [0.68, 0.76] | 0.284 [0.25, 0.33] |
| `ppo_it4_mixed_s0` | 220K | 0.712 [0.67, 0.75] | 0.220 [0.19, 0.26] |
| `ppo_it4_mixed_s1` | 190K | 0.676 [0.63, 0.72] | 0.186 [0.15, 0.22] |
| `ppo_it4_shaped_s0` | 160K | 0.708 [0.67, 0.75] | 0.254 [0.22, 0.29] |
| `ppo_it4_shaped_s1` | 130K | 0.702 [0.66, 0.74] | 0.238 [0.20, 0.28] |
| `dqn_it4_base_s0` | 140K | 0.092 [0.07, 0.12] | 0.052 [0.04, 0.08] |
| `dqn_it4_base_s1` | never | 0.244 [0.21, 0.28] | 0.098 [0.08, 0.13] |
| `dqn_it4_mixed_s0` | never | 0.272 [0.24, 0.31] | 0.046 [0.03, 0.07] |
| `dqn_it4_mixed_s1` | never | 0.204 [0.17, 0.24] | 0.038 [0.02, 0.06] |
| `dqn_it4_shaped_s0` | 180K | 0.184 [0.15, 0.22] | 0.008 [0.00, 0.02] |
| `dqn_it4_shaped_s1` | never | 0.424 [0.38, 0.47] | 0.140 [0.11, 0.17] |

| Arm | PPO vs max_power | PPO vs heuristic | DQN vs max_power | DQN vs heuristic |
|---|---|---|---|---|
| `it4_base` | 0.625 | 0.210 | 0.168 | 0.075 |
| `it4_mixed` | 0.694 | 0.203 | 0.238 | 0.042 |
| `it4_shaped` | 0.705 | 0.246 | 0.304 | 0.074 |

For the step budget, the three 500K PPO realistic runs from experiment 3 are the comparison: the same curriculum, shaping weights and gamma. They averaged 0.707 against max_power and 0.271 against the heuristic (`data/results/info_gap_val.csv`).

Training win rates in the last 200K steps, split by the `opponent` column of `episodes.csv`: the mixed PPO runs won 0.90 to 0.91 against random, 0.55 to 0.58 against max_power and 0.19 to 0.20 against the heuristic. The mixed DQN runs, which never left the max_power stage, won 0.60 to 0.61 against random and 0.15 to 0.16 against max_power.

**Result.** Doubling the step budget did not help PPO. `it4_base` at 1M steps averaged 0.210 against the heuristic, below the 0.271 of the 500K runs. Most of that gap is `ppo_it4_base_s0`, whose in-training heuristic evaluation peaked at 0.28 and ended at 0.12.

Stronger shaping gave PPO the highest mean on both opponents and the smallest spread between seeds (0.238 and 0.254 against the heuristic, against 0.136 and 0.284 for base). Both shaped PPO runs also reached the heuristic stage earliest, at 130K and 160K steps. With two seeds the intervals still overlap with base, so this is a lead and not a separation. The mixed curriculum kept PPO strong against the earlier opponents in training, but on validation it was no better than base against the heuristic.

No arm fixed the DQN collapse. The two DQN runs that reached the heuristic ended worse than the four that did not: `dqn_it4_shaped_s0` peaked at 0.26 in training and finished at 0.008, and `dqn_it4_base_s0` fell to 0.092 against max_power. Neither mixed DQN run passed the max_power gate in 1M steps, so the mixed curriculum never got to test the heuristic stage it was meant for. The best DQN result, `dqn_it4_shaped_s1` (0.424 and 0.140), never reached the heuristic either. That is the same split by curriculum stage as in the information-gap runs.

**Selected.** Stronger shaping (0.5 + 0.5), the standard curriculum and the 500K step budget, with gamma 0.995 for both agents. The information-gap pairs are rerun on this setup: both agents, realistic and oracle observations, three seeds each (`slurm/info_gap_shaped.txt`). Evaluation stays greedy for DQN (epsilon 0), the same protocol as every earlier DQN result. If DQN collapses again at the heuristic stage, its information gap is reported as not measurable, and the collapse itself is reported as a finding.

**Information-gap rerun on the selected setup.** The 12 runs in `slurm/info_gap_shaped.txt` (both agents, realistic and oracle observations, three seeds, 500K steps, gamma 0.995, shaping 0.5 + 0.5) were each evaluated from `final.pt` with 500 battles per opponent on validation teams (`data/results/info_gap_shaped_val.csv`).

| Run | Heuristic stage from | vs max_power | vs heuristic |
|---|---|---|---|
| `ppo_shaped_realistic_s0` | 130K | 0.708 [0.67, 0.75] | 0.256 [0.22, 0.30] |
| `ppo_shaped_realistic_s1` | 130K | 0.654 [0.61, 0.69] | 0.266 [0.23, 0.31] |
| `ppo_shaped_realistic_s2` | 150K | 0.686 [0.64, 0.73] | 0.260 [0.22, 0.30] |
| `ppo_shaped_oracle_s0` | 200K | 0.522 [0.48, 0.57] | 0.196 [0.16, 0.23] |
| `ppo_shaped_oracle_s1` | 100K | 0.550 [0.51, 0.59] | 0.138 [0.11, 0.17] |
| `ppo_shaped_oracle_s2` | 160K | 0.710 [0.67, 0.75] | 0.244 [0.21, 0.28] |
| `dqn_shaped_realistic_s0` | 380K | 0.316 [0.28, 0.36] | 0.068 [0.05, 0.09] |
| `dqn_shaped_realistic_s1` | never | 0.284 [0.25, 0.33] | 0.062 [0.04, 0.09] |
| `dqn_shaped_realistic_s2` | 350K | 0.222 [0.19, 0.26] | 0.066 [0.05, 0.09] |
| `dqn_shaped_oracle_s0` | never | 0.390 [0.35, 0.43] | 0.156 [0.13, 0.19] |
| `dqn_shaped_oracle_s1` | 50K | 0.164 [0.13, 0.20] | 0.068 [0.05, 0.09] |
| `dqn_shaped_oracle_s2` | 380K | 0.168 [0.14, 0.20] | 0.088 [0.07, 0.12] |

Information gap (oracle minus realistic win rate, three seeds pooled, with a binomial 95% interval that leaves out seed-to-seed variation), next to the experiment 3 runs at the old shaping weights (`data/results/info_gap_val.csv`):

| Agent | Opponent | Gap, shaping 0.1 + 0.1 | Gap, shaping 0.5 + 0.5 |
|---|---|---|---|
| PPO | max_power | -0.126 ± 0.034 | -0.089 ± 0.034 |
| PPO | heuristic | -0.105 ± 0.029 | -0.068 ± 0.030 |
| DQN | max_power | +0.166 ± 0.028 | -0.033 ± 0.031 |
| DQN | heuristic | +0.044 ± 0.015 | +0.039 ± 0.020 |

**Result of the rerun.** PPO's gap stayed negative and shrank by about a third. Realistic PPO beat oracle PPO against the heuristic on all three seeds, as in experiment 3, so six of six seed pairs now point the same way. Realistic PPO was also far more consistent across seeds (0.256 to 0.266) than oracle PPO (0.138 to 0.244).

DQN is still confounded by curriculum stage, though this time evenly: each observation mode has one run that stayed at max_power and two that reached the heuristic and declined. The collapsed runs finished at 0.06 to 0.09 against the heuristic, higher than the 0.01 to 0.03 of experiment 3, most likely because three of the four reached the heuristic only at 350K to 380K steps and had less time to decay. The one matched pair at the max_power stage favors oracle (`dqn_shaped_oracle_s0` 0.390 and 0.156 against `dqn_shaped_realistic_s1` 0.284 and 0.062). That is the opposite direction from PPO, but one pair is not a reportable gap.

**Train-split check.** In experiment 3, oracle PPO did worse on validation teams than on training teams while realistic PPO did the reverse. The six shaped PPO runs were evaluated again with 500 battles per opponent on the training split (`data/results/info_gap_shaped_train.csv`), three seeds pooled:

| Mode | vs max_power, train | vs max_power, val | vs heuristic, train | vs heuristic, val |
|---|---|---|---|---|
| Realistic | 0.585 | 0.683 | 0.234 | 0.261 |
| Oracle | 0.595 | 0.594 | 0.243 | 0.193 |

On training teams the two modes are level (oracle ahead by 0.010 and 0.009), so PPO's negative information gap appears only on validation teams. Realistic PPO does better on validation teams than on training teams against both opponents; oracle PPO does not improve against max_power and drops 0.051 against the heuristic, with the largest drop on `ppo_shaped_oracle_s1` (0.236 to 0.138).

Memorizing individual teams is an unlikely cause. Each run played about 12,000 battles drawn from 2,486 training teams, so a given team appeared about ten times. The training split is almost all ladder teams and the validation split is four expert-built teams, so the more likely reading is that oracle PPO learned something from full movesets and items on ladder teams that does not hold on expert teams, while the coarser realistic view transfers. This is not yet tested.

**Correction, made while planning iteration 5.** The Result above says doubling the step budget did not help PPO. That compared one arm's `final.pt` at 1M steps with different runs at 500K. The in-training evaluations of the same 1M runs say otherwise: all six PPO runs set their best evaluation against the heuristic after 500K steps (540K to 990K), and the shaped arm averaged 0.17 before 600K and 0.23 to 0.29 after. Averaged over 600K to 800K and 800K to 1M, three runs rose slightly, one was flat and two fell, so PPO may be close to levelling off by 1M, but these are 100-battle evaluations on four teams. The 500K budget chosen in **Selected** was based on the wrong comparison.

## Iteration 5: a proper validation set, best checkpoints and the final experiments (planned 2026-11-26)

This is the last training iteration. Everything after it is analysis of the models it produces.

**Why.** Reviewing iterations 2 to 4 turned up four problems.

1. Validation was four expert teams and test three. Ladder teams went into training in iteration 2, but validation and test were never enlarged. Since both players draw from the same split, every validation number came from 16 team pairings, and the curriculum gate and checkpoint selection were tuned to those 16 too. Comparisons within a split (DQN vs. PPO, realistic vs. oracle, arm vs. arm) were fair, but no result can be claimed for Gen 2 OU teams in general. The PPO information-gap result is affected directly: oracle PPO matched realistic PPO on training teams and fell behind only on the four validation teams.
2. Runs were compared on `final.pt`. The best checkpoint of a run can come well before its end (DQN peaked within 50K steps of reaching the heuristic) or well after a fixed budget (PPO above), so `final.pt` measures when training happened to stop.
3. DQN's results depend on whether a run reached the heuristic stage, which differs by seed. Its information gap cannot be read until the stage is held fixed.
4. In realistic observations, an opponent's revealed Hidden Power was encoded as a Normal-type attack (poke-env does not know its type, and Gen 2 Hidden Power is never Normal), adding a false attack-type flag and false matchup values for most opponents. The agent's own Hidden Power and the oracle observation were correct.

**Changes.**

- Team split (`data/README.md`). Validation and test each gain a ladder pool of 100 held-out team compositions, at most three teams each (183 and 181 teams), drawn so that no set of six Pokemon appears in two splits. Training keeps 1,859 ladder teams and the 11 expert teams. The four and three expert teams stay as a separate expert pool in validation and test. The ladder teams turned out to repeat compositions heavily (724 compositions over 2,475 teams), which is why the held-out pools are counted in compositions.
- Training evaluates on the validation ladder pool only.
- Checkpoint selection. Every 25K steps, 200 validation battles against the heuristic, whatever the current curriculum stage, and `best.pt` keeps the best. The curriculum gate is unchanged.
- Step budget 2M, with early stopping after 600K steps on the final curriculum stage without a new best. In the six 1M-step PPO runs of iteration 4 the longest wait between new bests was 270K to 530K steps, and PPO set a new best as late as 990K. Every DQN run that reached the heuristic set its best within 50K steps of arriving and never beat it, so a collapsed DQN run stops around 600K steps after its peak. The countdown starts only on the final stage, because DQN runs still on max_power have advanced as late as 380K.
- An opponent's Hidden Power is encoded as a move of unknown type (`known_type` in `src/encoding.py`).

Unchanged from iteration 4: gamma 0.995, shaping 0.5 + 0.5, the standard curriculum, network sizes and every other hyperparameter.

**Runs.** `slurm/final.txt`, 30 runs, three seeds per arm:

| Group | Runs | Arms | Question |
|---|---|---|---|
| `i5_full` | 12 | DQN and PPO, realistic and oracle, curriculum random, max_power, heuristic | Information gap, the headline result |
| `i5_capped` | 12 | Same, curriculum stopped at max_power | Information gap with the curriculum stage held fixed, so DQN's is measurable |
| `i5_dueling` | 6 | Dueling DQN, realistic and oracle, full curriculum | Whether a dueling head prevents the collapse. Run in both modes so that, if it does, it also gives a DQN information gap. |

**Evaluation protocol.** Each run's `best.pt`, 500 battles per opponent against max_power and the heuristic on the test split, with the ladder and expert pools reported separately. Validation picked the checkpoint, so it is not used for reported numbers. Results are given per seed with the spread across seeds alongside the pooled Wilson interval. `final.pt` is kept to show how runs ended, which for DQN measures the collapse.

**Not run.** Sparse reward, a vanilla DQN target, the masking-vs-penalty ablation, the memory extensions and self-play (`docs/design.md`).
