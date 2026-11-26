# Data

## Teams

`python -m src.build_teams` downloads both team sets at the pinned revision and rebuilds everything under `teams/`, including the split and the filters below. It reproduces the committed files exactly.

`teams/` holds Gen 2 OU teams in Showdown teambuilder format from two sets of Metamon's team dataset ([jakegrigsby/metamon-teams](https://huggingface.co/datasets/jakegrigsby/metamon-teams), dataset revision `bff785c`). Each split has two pools, told apart by file name:

| Split | Ladder pool (`ladder_*`) | Expert pool (`team_*`) | Used for |
|---|---|---|---|
| `train` | 1,859 teams, 524 compositions | 11 teams | training both agents |
| `val` | 183 teams, 100 compositions | 4 teams | curriculum advances and checkpoint selection (ladder pool) |
| `test` | 181 teams, 100 compositions | 3 teams | final numbers only |

A composition is a team's set of six Pokemon, ignoring moves and items. In every battle both players draw a team at random from the same split, or from one pool of it when a script is given `--pool ladder` or `--pool expert`, so a battle can be a mirror match.

### Expert pool

18 expert-built sample teams scraped from Smogon forum threads (file `competitive/gen2ou.tar.gz`). The archive numbers them 0 to 18 with no team 10. They were shuffled once with a fixed seed (`random.Random(0)`) and split 11 / 4 / 3:

| Split | Teams |
|---|---|
| `train` | 0, 2, 3, 5, 9, 11, 12, 15, 16, 17, 18 |
| `val` | 4, 6, 7, 8 |
| `test` | 1, 13, 14 |

The expert pool has little variety. Snorlax is on all 18 teams, Cloyster on 13 and Zapdos on 12, and 102 of 108 Pokemon hold Leftovers. That reflects the Gen 2 OU metagame, but it means held-out teams share most of their Pokemon with training teams. The generalization test measures new team compositions, not unseen species.

### Ladder pool

2,616 teams from Metamon's `hl_05_26` set (file `hl_05_26/gen2ou.tar.gz`). Metamon built them from high-rated ladder and tournament replays of 2026, filling in unrevealed Pokemon and moves from usage statistics, so they are predicted teams, not hand-built ones. Every one passes Showdown's `gen2ou` validator. The original file numbers are kept in the names (`team_002661` became `ladder_002661`). In order:

1. 3 exact duplicates removed.
2. 138 teams removed because their six Pokemon match a validation or test expert team exactly. Ladder players often copy published sample teams, and keeping these would leak the held-out teams into training.
3. The remaining 2,475 teams form 724 compositions. Ladder teams repeat compositions heavily: 376 compositions appear once, while the 50 most common cover 1,067 teams and the most common alone covers 75. Validation and test are therefore drawn by composition, not by team. Compositions that match an expert training team stay in training. From the rest, 100 compositions were drawn at random (same seed) for validation and 100 for test, keeping at most 3 teams of each so that no composition dominates a held-out pool. The 252 extra copies of held-out compositions were dropped, since training on them would leak the composition. Everything else is training data.

No composition appears in more than one split, in either pool. Teams sharing five of six Pokemon were kept apart from that, since overlap at that level is normal in this metagame (Snorlax is on 97% of the ladder teams).

The ladder teams list Hidden Power without a type and with every IV at 31, which makes it Dark type in Gen 2. 2,291 of the 2,475 ladder teams carry it. The expert teams set Hidden Power types deliberately (Ice, Fire, Water and others). This is one concrete difference between the two pools.

History: iterations 0 and 1 trained on the 11 expert training teams. Iteration 2 added the ladder teams to training only, leaving validation and test at 4 and 3 expert teams. Iteration 5 moved held-out ladder compositions into validation and test (`docs/iteration-log.md`).

## Results

`results/` holds CSV outputs of the experiment scripts in `src/`. `baselines_<split>[_<pool>].csv` comes from `python -m src.run_baselines`. Results from iteration 4 and earlier were measured on the four expert validation teams, before the ladder validation pool existed.
