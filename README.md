# Tichu
Implementation of the Tichu game and agents able to play it.

-------------------------

## Run core regression tests with Docker

Docker Desktop must be running. Build the test image and run the tests with:

```bash
docker compose build
docker compose run --rm tests
```

The core test image contains only the dependencies required for the card and game-state model. It does not include the legacy neural-network training stack.

## Run reproducible agent benchmarks

The benchmark runner plays every seed twice and swaps the teams between the
even and odd seats. Per-game results are written to CSV.

```bash
docker compose run --rm tests python benchmark.py \
  --team-a mcts \
  --team-b balanced-random \
  --games 10 \
  --seed 42 \
  --target 100 \
  --iterations 10 \
  --max-time 0.2
```

`--games 10` means 10 seeded deals and 20 games after the seat-swapped
rematches. See [milestone.md](./milestone.md) for the AI development roadmap.
Available benchmark agents are `random`, `balanced-random`, `fuegi`, `mcts`,
and `fuegi-mcts`.

## Collect teacher decisions for learning

The dataset collector records only information visible to the acting player,
the legal actions, the teacher's choice, MCTS root statistics, and the final
team result. Paired games from the same seed are always assigned to the same
train or validation split.

```bash
docker compose run --rm tests python collect_dataset.py \
  --team-a fuegi-mcts \
  --team-b fuegi \
  --record-agent fuegi-mcts \
  --games 100 \
  --seed 40000 \
  --output-dir datasets/tichu-decisions-v1
```

See [docs/training-dataset-v1.md](./docs/training-dataset-v1.md) for the schema
and privacy rules. Generated datasets are intentionally ignored by Git.

The verified Brettspielwelt decisions can also train a supervised legal-action
ranking baseline. See [docs/behavior-cloning.md](./docs/behavior-cloning.md) for
the Docker smoke test, full training command, and validation metrics.

## Download recent Brettspielwelt logs

The downloader can discover the newest numeric log id, resume from cached
files, and work backwards until the directory contains a requested number of
complete games. This PowerShell command collects 10,000 complete logs with at
most eight concurrent requests:

```powershell
docker compose run --rm --entrypoint python tests `
  scraper/download_brettspielwelt.py `
  --discover-latest `
  --latest-id 2419834 `
  --target-total 10000 `
  --workers 8 `
  --delay 0.1 `
  --progress-every 250
```

`--latest-id` is a known-valid hint, not a fixed upper bound. Re-running the
same command safely skips cached files. Raw logs and the append-only download
manifest are stored under `datasets/raw/brettspielwelt` and are ignored by Git.

## Dependencies
**Python 3.6+**

And following packages (all should be installable with pip or anaconda).

- **profilehooks**: To profile/measure function execution time (https://pypi.python.org/pypi/profilehooks)
- **keras-rl**: Neural Network Reinforcement learning (https://github.com/matthiasplappert/keras-rl). 
- **keras**: https://keras.io/.
- either **Tensorflow** or **Theano** (used by keras)
- **h5py**: For the h5f files.
- **gymnasium**: Maintained successor to OpenAI Gym (https://gymnasium.farama.org/)
- **networkx**: For the Game-graph
- **numpy**
- **argparse**: To parse command line inputs
- **requests**: For the [tichumania](http://log.tichumania.de) scraper
- **BeautifulSoup**: For the [tichumania](http://log.tichumania.de) scraper

Then do (to install and register the Gymnasium environment):
```bash
cd gym-tichu
pip install -e .
```

## Play a game
Gamelogs are written to the folder _Tichu/logs_

To play a game against three agents:
```bash
python play.py
```

To play a game against three agents and see all cards.
```bash
python play.py --cheat
```

To watch a game amongst four agents:
```bash
python play.py --lazy
```

The default agents use the core random-rollout MCTS implementation and do not
require the legacy neural-network dependencies. You can limit the search work
for each decision, for example:

```bash
python play.py --lazy --target 100 --iterations 10 --max-time 0.2
```

More games can be found in the [game_starter.py](./game_starter.py)

## Train a Deep-Q-learning agent
Training results are written to the folder _nn_training/logs_

Example (train against random agents for 10000 steps (10000 decisions taken by the agent)): 
```bash
python nn_training/train_dqn.py dqn_2l17x5_2_sep random 10000
```

Following command shows all options
```bash
python nn_training/train_dqn.py -h
```

To visualize the training afterwards:
```bash
python nn_training/visualize_logs.py nn_training/logs/**/*.json
```

To save the plots:
```bash
python nn_training/visualize_logs.py nn_training/logs/**/*.json --save
```


# Run Experiments / Tournaments
Experiment results are written to the folder _experiments/logs_.

Launch the _**experiments/run_experiments.py**_ script.

For example, to launch a Tournament between the 4 DQN-agents, each game lasts until one team reached 1000 points, do:
```bash
python experiments/run_experiments.py nn_tournament --target 1000
```

To list all options:
```bash
python experiments/run_experiments.py -h
```
