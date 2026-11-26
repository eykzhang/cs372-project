# Design: Double DQN vs. PPO Under Hidden Information in Gen 2 OU Pokemon

Status: planning document, written 2026-11-23 before implementation. Sections describe intended design; results go in the README's Evaluation section once they exist.

## Research question

Pokemon battles are an incomplete-information game: each player sees only what the opponent has revealed, and both players choose their moves at the same time. This project asks which of two classic deep RL algorithms copes better with that, a value-based agent (Double DQN) or a policy-gradient agent (PPO), when both are implemented from scratch, see the same state encoding, and train against the same opponents in Gen 2 OU singles on Pokemon Showdown.

"Copes better" is measured three ways:

1. **Information gap.** Each agent is trained and evaluated twice, once observing only what a real player could see and once with an oracle observation that also exposes the opponent's full team, moves and held items. The drop in win rate from oracle to realistic observation measures how much each algorithm is hurt by hidden information.
2. **Randomization.** Under simultaneous moves, a predictable player can be exploited, so good play is often mixed, the way rock-paper-scissors is. A greedy DQN policy picks one action per state; PPO's policy is a probability distribution and can randomize. We measure each agent's policy entropy and how accurately a simple classifier can predict its next action from the state.
3. **Hidden-information situations.** Performance on decision points where the correct play depends on something unrevealed: an opponent Pokemon not yet seen, an item that has not activated, a move not yet used.

Overall strength is also reported: win rate against scripted opponents, sample efficiency, generalization to unseen teams, and head-to-head play.

The two algorithms have different strengths here. DQN reuses every past transition from a replay buffer, which matters when each sample comes from a slow simulator. PPO represents a stochastic policy directly and learns only from recent experience. Both learn to play; the difference is that DQN derives its play from estimated action values, while PPO optimizes the playing distribution itself.

## Why Gen 2 OU

- The game is small enough to encode by hand. There are no abilities, natures or Terastallization, and the state fits in under a thousand features, which keeps training on a laptop.
- It has real hidden information: unrevealed team members, unused moves, and held items that stay hidden until they activate (Leftovers recovery, a berry being eaten).
- Gen 2 removed Gen 1's most irregular mechanics. Hyper Beam always requires a recharge turn, critical-hit rate no longer depends on Speed, frozen Pokemon can thaw, and partial-trapping moves no longer stop the target from attacking.
- It is an active research setting. Metamon (Grigsby et al., RLC 2025) trains offline RL agents on Gen 1-4 OU and reports human-level play on the Showdown ladder, and the NeurIPS 2025 PokeAgent Challenge ran a competitive Showdown battling track. Our agents are far smaller than Metamon's transformers; the question here is how two online RL algorithms compare on the same problem.
- OU uses fixed, player-built teams. Training on one set of teams and testing on another gives a clean generalization test.

Gen 2 adds mechanics the encoding has to cover: held items, weather from Rain Dance, Sunny Day and Sandstorm, Spikes, the Dark and Steel types, and separate Special Attack and Special Defense stats.

## System components

### Environment

A Gymnasium environment built on poke-env's `SinglesEnv`, playing against a local Pokemon Showdown server in the `gen2ou` format. One episode is one battle.

- **Action space:** 10 discrete actions, poke-env's Gen 2 layout: 6 switch slots (one per team member, in team order) and 4 moves of the active Pokemon. The active Pokemon's own switch slot is always masked.
- **Action masking:** a legality mask from poke-env each turn (fainted or active Pokemon cannot be switched to, a trapped Pokemon cannot switch, forced switches after a faint allow only switches).
- **Turn cap:** battles are cut off at 200 turns, because Rest and Leftovers stalling between weak policies can otherwise run to Showdown's 1000-turn tie. A cut-off battle counts as a non-win, and training bootstraps through the cutoff since it is not a real end of the game.

### Teams

Validation and test teams are expert-built Gen 2 OU teams from Metamon's `competitive` set: 18 teams, split 11 / 4 / 3 into training, validation and held-out test teams. The training split also holds 2,475 teams from Metamon's `hl_05_26` set, predicted from high-rated ladder replays, with any team that has the same six Pokemon as a validation or test team removed. These were added in iteration 2 after DQN overfit the 11 competitive training teams (`docs/iteration-log.md`). Validation teams pick checkpoints and hyperparameters; test teams are touched only for final numbers. Each battle, both players draw a team at random from the same split. Details in `data/README.md`.

### State encoding

Hand-designed features. In the realistic observation, opponent information appears only once revealed, with an explicit "unknown" indicator otherwise:

- For each of the 12 Pokemon (6 own, 6 opponent): types, HP fraction, status condition, stat boosts, held item (or unknown), fainted flag, active flag.
- For each of the active Pokemon's 4 moves: base power, type, accuracy, remaining PP fraction, and derived features computed against the opponent's active Pokemon (type effectiveness multiplier, same-type attack bonus).
- Battle-level features: turn number, weather and turns remaining, Spikes on each side, Pokemon remaining on each side.

Each Pokemon slot also carries derived features: the types and roles (recovery, setup, Explosion, sleep, Spikes, phazing, Rapid Spin, Sleep Talk) of its known moves, its best known attack into the opposing active Pokemon, and the worst type matchup it faces from that Pokemon. An unrevealed opponent slot is encoded as full HP with an unknown item, which is known for certain because it has never been in battle. The vector has 956 features.

The oracle observation uses the same layout with every opponent field filled in from the opponent's true team, so the two variants differ only in what is hidden. Opponent slots are ordered by reveal in both modes; the oracle fills the remaining slots with the unrevealed Pokemon. The true team comes from the opponent's own battle object, which poke-env keeps in the same process.

### Reward

A terminal reward of +1 for a win and -1 for a loss, plus shaping for HP and knockouts. The shaping is potential-based: each turn's shaping reward is gamma times the new potential minus the old one, where the potential is a weighted HP difference between the teams plus a weighted knockout difference, each scaled to [-1, 1]. The per-turn terms telescope, so total shaping over a battle comes down to the final potential, which acts as a margin-of-victory bonus. Because shaping depends only on the state, damage dealt and then healed back earns nothing, so the agent cannot farm damage without winning. That happened in earlier work of mine with shaping that was not potential-based. A win also always outscores a loss at any weights: the final potential is positive when the opponent's whole team has fainted and negative when ours has. The default weights are 0.1 and 0.1. Iteration 4 (`docs/iteration-log.md`) tests 0.5 and 0.5, which spreads returns between close losses and routs fivefold.

### Agents

**Double DQN.** Written from scratch in PyTorch.
- Online and target Q-networks, with periodic target updates.
- Experience replay buffer that stores the next state's legality mask alongside each transition.
- Huber loss and epsilon-greedy exploration with a decaying epsilon.
- Masking applies in two places: illegal actions get a Q-value of negative infinity when choosing an action, and again when taking the max over next-state actions in the bootstrap target. Without the second one, the target can bootstrap from a move the agent could never have made.
- A dueling network head is the planned architecture variant.

**PPO.** Written from scratch in PyTorch.
- Actor-critic network with a shared trunk.
- Generalized advantage estimation.
- Clipped surrogate objective with an entropy bonus, updated over several epochs of minibatches per rollout.
- Masked categorical policy: illegal actions get negative-infinity logits before the softmax.
- Evaluated by sampling from its policy, since the stochastic policy is what it learned and what the randomization question is about. DQN is evaluated greedily.

### Memory (stretch)

Both core agents are memoryless MLPs: anything from earlier turns reaches them only through hand-designed features such as revealed moves and items. Two optional extensions test whether learned memory helps under hidden information:

1. **Frame stacking.** Concatenate the last few turns' observations into the MLP input, for both agents. This works with DQN's replay buffer unchanged and gives a first, inexpensive measure of whether history adds anything beyond the hand-designed features.
2. **Transformer PPO.** A small causal transformer (about 2 layers, 4 heads, width 64-128) over the sequence of per-turn observation embeddings, with policy and value heads at every step. Attention is causal so turn t sees only turns 1 through t, matching what the agent knows at play time. PPO fits naturally because its rollouts are already whole battles. A transformer DQN would need sequence replay with burn-in, a substantially larger piece of work, and is out of scope.

The transformer agent is trained with realistic observations only. The oracle observation already contains everything, so memory should not help there. The prediction is that learned memory closes part of the realistic-to-oracle gap. If it does not, that suggests the hand-designed features already capture the history that matters.

### Opponents and baselines

Scripted players that ship with poke-env: `RandomPlayer`, `MaxBasePowerPlayer` (always picks the highest-power move) and `SimpleHeuristicsPlayer` (type matchups, switching and stat boosts). These serve as both training opponents, in a curriculum from weakest to strongest, and as the baselines every agent is measured against. A mixed variant of the curriculum keeps earlier opponents in the training battles after each advance. Self-play is a possible extension once both agents beat the scripted opponents. It would also let each agent face an opponent that adapts to its predictability, which bears directly on the randomization question.

## Experiments

1. **Baselines.** Win rates among the scripted players, to calibrate how hard each opponent is.
2. **DQN and PPO against scripted opponents.** Learning curves (win rate and episode return over environment steps) and final win rates on validation and test teams.
3. **Information gap.** Each agent trained with realistic and oracle observations, with all other settings fixed. The headline result is the realistic-vs-oracle win-rate drop for each algorithm.
4. **Randomization analysis.** Policy entropy over the course of battles, and the accuracy of a classifier trained to predict each agent's actions from the state.
5. **Head-to-head.** DQN against PPO, each matchup of teams played from both sides so neither agent benefits from a stronger team.
6. **Ablations**, each varying one design choice with everything else fixed:
   - shaped vs. sparse reward
   - action masking vs. a penalty for illegal actions
   - Double DQN vs. vanilla DQN target
   - dueling vs. plain Q-network head
7. **Hyperparameter search** over at least three configurations per agent (learning rate, discount, exploration schedule or entropy coefficient), selected on validation teams.
8. **Generalization.** Performance on held-out test teams compared with training teams.
9. **Error analysis.** Losses broken down by cause, with replay inspection of representative failures. Examples: attacking into a type immunity, failing to switch out of a losing matchup, misplaying against an item that had not yet been revealed.
10. **Behavioral probes.** Edit one feature of a real game state, such as revealing or hiding an opponent's Pokemon, item or move, or changing our HP. Then measure how each agent's action choice and value estimate respond.
11. **Memory (stretch).** Information gap for frame-stacked agents, and for transformer PPO against MLP PPO, both under realistic observations.

## Metrics

- Win rate with 95% Wilson confidence intervals, from at least 500 battles for any headline number.
- Information gap: oracle win rate minus realistic win rate, per agent, with intervals.
- Policy entropy and action predictability (classifier accuracy against a majority-class baseline).
- Sample efficiency: environment steps needed to reach a fixed win rate against each opponent.
- Battle length in turns, and Pokemon remaining at the end of a battle as a margin-of-victory measure.
- Inference latency per decision.

## Prior work this builds on

- Mnih et al., "Human-level control through deep reinforcement learning," Nature 2015 (DQN).
- van Hasselt, Guez and Silver, "Deep Reinforcement Learning with Double Q-learning," AAAI 2016.
- Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning," ICML 2016.
- Schulman et al., "Proximal Policy Optimization Algorithms," arXiv 2017.
- Grigsby et al., "Human-Level Competitive Pokemon via Scalable Offline Reinforcement Learning with Transformers," RLC 2025 (Metamon).
- The PokeAgent Challenge, NeurIPS 2025 competition track.
- My own earlier project, battle-engine, which trained a library PPO agent on Gen 9 OU. No code is reused; two lessons carry over. First, self-play destabilized when team variety grew too large. Second, uncapped reward shaping could outweigh winning.
