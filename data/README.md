# Data

## Teams

`python -m src.build_teams` downloads both team sets at the pinned revision and rebuilds everything under `teams/`, including the split and the filters below. It reproduces the committed files exactly.

`teams/` holds 18 Gen 2 OU teams in Showdown teambuilder format, from the `competitive` set of Metamon's team dataset ([jakegrigsby/metamon-teams](https://huggingface.co/datasets/jakegrigsby/metamon-teams), file `competitive/gen2ou.tar.gz`, dataset revision `bff785c`). They are expert-built sample teams scraped from Smogon forum threads. The archive numbers them 0 to 18 with no team 10.

The teams were shuffled once with a fixed seed (`random.Random(0)`) and split 11 / 4 / 3:

| Split | Teams | Share | Used for |
|---|---|---|---|
| `train` | 0, 2, 3, 5, 9, 11, 12, 15, 16, 17, 18 | 61% | training both agents |
| `val` | 4, 6, 7, 8 | 22% | checkpoint and hyperparameter selection |
| `test` | 1, 13, 14 | 17% | final numbers only |

In every battle both players draw a team at random from the same split, so a battle can be a mirror match.

### Ladder teams in the training split

`teams/train/` also holds 2,475 teams named `ladder_*.gen2ou_team`, from Metamon's `hl_05_26` set (file `hl_05_26/gen2ou.tar.gz`, same dataset revision). Metamon built them from high-rated ladder and tournament replays of 2026, filling in unrevealed Pokemon and moves from usage statistics, so they are predicted teams, not hand-built ones. They were added in iteration 2 (`docs/iteration-log.md`) because 11 training teams were few enough for DQN to memorize.

The archive has 2,616 teams. Every one passes Showdown's `gen2ou` validator. Two filters were applied:

- 3 exact duplicates removed.
- 138 teams removed because their six Pokemon match a validation or test team exactly. Ladder players often copy published sample teams, and keeping these would leak the held-out teams into training. Teams sharing five of six Pokemon were kept, since overlap at that level is normal in this metagame (Snorlax is on 97% of the ladder teams).

The ladder pool covers 92 species. The original file numbers are kept in the names (`team_002661` became `ladder_002661`). Training now draws from 2,486 teams, so a competitive team comes up in about 0.4% of training battles.

The competitive pool has little variety. Snorlax is on all 18 teams, Cloyster on 13 and Zapdos on 12, and 102 of 108 Pokemon hold Leftovers. That reflects the Gen 2 OU metagame, but it means the held-out test teams share most of their Pokemon with training teams. The generalization test measures new team compositions, not unseen species.

## Results

`results/` holds CSV outputs of the experiment scripts in `src/`. `baselines_<split>.csv` comes from `python -m src.run_baselines`.
