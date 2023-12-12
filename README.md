# Pure-Past Action Msking

Compare the performance of a PPAM-trained agent against that of a RB-trained agent in the CocktailParty (CP) environment, or that of a shield-trained agent in the WaterTank (WT) environment, or that of a vanilla Q-learning agent in the BoatRace (BR) environment.

For the CP and BR experiments, we have used the same set of seeds to compare PPAM-based agents against the other agents. For both sets of experiments, each file name will contain a number before its file extension, which is the seed that was used to generate the data contained in such file. 

### Requirements

Each experiment has its own `requirements.txt`, which can be found in the corresponding folder. 

All experiments can be run with Python 3.11.4. We highly suggest to setup different virtual environments, one per experiment.

To create a virtual environment and install the requirements:

	cd ./experiment-you-want-to-run 
	python -m venv path/to/venv
	source path/to/venv/bin/activate
   	pip install -r requirements.txt


### Restraining Bolts

We compare pure-past action masking to restraining bolts in CocktailParty, to show how the former can enforce safety constraints unlike the latter, without hindering training (in this case, at least).

You must select a RL algorithm and the training file. Learning mode will train the agent, evaluation mode (enabled with option ``--eval``) will execute the current best policy. Note that in order to train the PPAM agent, the game name has to be "CPRestrain" instead of just "CP".

Our code is an extension of the original restraining bolts code, which can be found at [github.com/iocchi/RLgames.git](https://github.com/iocchi/RLgames.git).

<div align="center">
<figure>
  <img src="https://github.com/giovannivarr/pure-past-action-masking-AAAI24/blob/main/README-figures/cocktail_party.png?raw=true" alt="CocktailParty environment (De Giacomo et al. 2019)." width="200"/>
</figure><br>
<i>CocktailParty environment (De Giacomo et al. 2019).</i>
</div>

#### Usage
```
Restraining Bolts usage: game.py [-h] [-rows ROWS] [-cols COLS] [-gamma GAMMA] 
								[-epsilon EPSILON] [-alpha ALPHA] [-nstep NSTEP] 
								[-lambdae LAMBDAE] [-niter NITER] [-maxtime MAXTIME]
				        		[-seed SEED] [--debug] [--gui] [--sound] 
				        		[--eval] [--stopongoal]
			               		game agent trainfile

positional arguments:
  game              game [CP, CPRestrain]
  agent             agent [Q, Sarsa]
  trainfile         file for learning strctures

optional arguments:
  -h, --help        show this help message and exit
  -rows ROWS        number of rows [default: 3]
  -cols COLS        number of columns [default: 3]
  -gamma GAMMA      discount factor [default: 1.0]
  -epsilon EPSILON  epsilon greedy factor [default: -1 = adaptive]
  -alpha ALPHA      alpha factor (-1 = based on visits) [default: -1]
  -nstep NSTEP      n-steps updates [default: 1]
  -lambdae LAMBDAE  lambda eligibility factor [default: -1 (no eligibility)]
  -niter NITER      stop after number of iterations [default: -1 = infinite]
  -maxtime MAXTIME  stop after maxtime seconds [default: -1 = infinite]
  -seed SEED        random seed [default: -1 = do no set]
  --debug           debug flag
  --gui             GUI shown at start [default: hidden]
  --sound           Sound enabled
  --eval            Evaluate best policy
  --stopongoal      Stop experiment when goal is reached
```

#### Output during training
During training, the program will print in the terminal updates on the performance of the agent whenever it achieves the highest score (where the score is the number of items delivered correctly during a single iteration, which can be at most 4 in CP) or the highest reward, or whenever 100 iterations have elapsed.

For updates when the highest score/reward is achieved, the structure of the output report is the following:

```
Iter current-iteration, sc:	score-achieved, na:	actions-taken-during-iteration, r:	reward-obtained	[HISCORE] [HIREWARD]
```
Depending on whether the highest score or reward (or both) are achieved, one of the two corresponding strings will be printed. 

Instead, for updates every 100 iterations, the structure of the output report is the following:

```
trainfile		current-iteration/	elapsed-time-in-seconds avg last 100: reward average-reward | score average-score | p goals percentage-of-all-tasks-completed-iterations-in-last-100
```

#### Examples

Game: CP with original restraining bolts (RB) 5x5
RL algorithm: Q-learning with RB

```
python game.py CP Q cp55_Q_rb -rows 5 -cols 5 -gamma 0.9
```

Game: CP with pure-past action masking (PPAM) 7x5, gamma=0.9, 4000 episodes
RL algorithm: Sarsa with PPAM

```
python game.py CPRestrain SarsaRestrain cp75_Sarsa_ppam -rows 7 -cols 5 -gamma 0.9 -niter 4000
```

Game: CP with RB 9x9
RL algorithm: Sarsa with RB and n-steps updates

```
python game.py CP Sarsa cp99_Sarsa_n10_01_rb -rows 9 -cols 9 -nstep 10
```


You can stop the process at any time and resume it later by specifying the same filenam.
If you want to execute another experiment with a new agent, give a different (and unused) filename.


To evaluate the current best policy, use the same command line adding ```--gui --eval```:

```
python game.py CP Sarsa cp55_Sarsa_rb -rows 5 -cols 5 --gui --eval
```

#### Interpreting the ```data``` folder
The ```data``` folder contains all data obtained after a run of the experiment. For each experiment, three different files are created:

* A ```.dat``` file, containing information about each iteration of the run. Each line in this file contains (in order): the iteration, elapsed time (in seconds), number of tasks completed, total reward, a boolean indicating whether the agent finished all tasks, number of elapsed timesteps, and a boolean indicating whether the agent has learned the optimal policy;
* A ```.info``` file, containing a summary of the training;
* A ```.npz``` file, containing the agent-related data, which can be used to reload the policy that the agent has learned so far.

#### Plotting the results
Observe that results can be plotted only for the restraining bolts experiments. 

```
python plot_results.py -datafiles DATAFILES [DATAFILES ...]
```

Example

```
python plot_results.py -datafiles cp55_Sarsa_rb cp55_Sarsa_ppam
```


### Shields
You can run experiments in the WT scenario to check how pure-past action masking can restrict the same set of actions that a shield restricts. 

Running the experiment will output four separate .data files in the `data` folder, two for the PPAM-based agent and two for the shield-based one, providing the average reward acquired by the two agents trained (one with Q-learning and one with SARSA) with each approach. 

Alternatively, you can train each kind of agent with each approach by running the watertank.py program yourself.

Our code is an extension of the original shields code, which can be found at [github.com/safe-rl/safe-rl-shielding](https://github.com/safe-rl/safe-rl-shielding).

#### Usage
```
Shields training usage:  watertank.py [-h] [-c COLLECT_DATA_FILE] [-l LOAD_FILE]
									  [-s SAVE_FILE] [-t TRAIN] 
									  [-o SHIELD_OPTIONS] [-n] [-p] [-r] [-m] 
									  [--seed SEED] [--num-steps NUM_STEPS]

optional arguments:
  -h, --help            show this help message and exit
  -c COLLECT_DATA_FILE, --collect-data COLLECT_DATA_FILE
                        Provide a file for collecting convergence data
  -l LOAD_FILE, --load LOAD_FILE
                        Load Q-Table from file
  -s SAVE_FILE, --save SAVE_FILE
                        Save Q-Table to file
  -t TRAIN, --train TRAIN
                        Used to indicate the exploration rate during training [default: .2]
  -o SHIELD_OPTIONS, --shield_options SHIELD_OPTIONS
                        Number of actions the shield can choose from. 0 disables the shield [default: 0]
  -n, --negative-reward
                        Used to punish the agent with a negative reward for unsafe actions
  -p, --huge-negative-reward
                        Used to punish the agent with a *huge* negative reward for unsafe actions
  -r, --sarsa           Sets SARSA as the training algorithm. If not given, uses Q-learning instead
  -m, --ppam
                        Uses PPAM for training. If PPAM is enabled, the shield is disabled automatically
  --seed SEED           Integer used to set the seed of the pseudo-random generator
  --num-steps NUM_STEPS
                        Number of interactions [default: 1000000]
```

#### Examples
WT experiment, training steps=100000, exploration rate=0.6 and default seed (0):

```
sh gen_data.sh 100000 .6
```

WT experiment, training steps=100000, exploration rate=0.3 and custom seed:

```
sh gen_data.sh 100000 .6 1024
```

WT training, PPAM agent, Q-learning

```
python watertank.py -m
```

WT training, PPAM agent, SARSA

``` 
python watertank.py -m -r
``` 

WT training, PPAM agent, Q-learning, training steps=1000, exploration rate=0.8, default seed

```
python watertank.py -m --num-steps 1000 -t 0.8
```

WT training, Shield agent, SARSA, training steps=1000000, exploration rate=0.3, custom seed

```
python watertank.py -o 1 -r --num-steps 1000000 -t 0.3 --seed 1024
```

### BoatRace
The BoatRace experiment shows how pure-past action masking can, in this case, ensure the agent reaches optimal performance without "hacking" the (observable) reward function. 

Running an experiment will produce a folder of files containing data which can be visualized by using Tensorboard. We also provide data which can be visualized using `plot_results.py`: notice that in order to plot results this way, you need to have a .csv file containing the data. 

Our code is an extension of [github.com/jvmncs/safe-grid-agents](https://github.com/jvmncs/safe-grid-agents).

<div align="center"> 
<figure >
  <img src="https://github.com/giovannivarr/pure-past-action-masking-AAAI24/blob/main/README-figures/boat_race.png?raw=true" alt="BoatRace environment (Leike et al. 2017). Dark grey tiles represent walls."/>
</figure><br>
<i>BoatRace environment (Leike et al. 2017). Dark grey tiles represent walls.</i>
</div> 

#### Usage
```
BoatRace training usage: main.py [-S SEED] [-E EPISODES] [-V EVAL_TIMESTEPS] 
								 [-EE EVAL_EVERY] [-D DISCOUNT] [-C] [-L LOG_DIR] 
								 boat tabular-q learning_rate [-e EPSILON]
								 [-dl EPSILON_ANNEAL] [-m]

positional arguments:
	learning_rate		Agent's learning rate
	
optional arguments:
	-S SEED				Random seed (default: None)
	-E EPISODE			Number of episodes (default: 2000)
	-V EVAL_TIMESTEPS	Approximate number of timesteps during eval period 
						(default: 2000)
	-EE EVAL_EVERY		Number of episodes between eval periods (default: 100)
	-D DISCOUNT			Discount factor (default: 0.99)
	-C					Whether to learn directly from hidden performance score 
						(default: False)
	-L LOG_DIR			Subdirectory to write logs to for this run 
						(defaults to combo of env, agent, cheating, mask, seed)
	-e EPSILON			Exploration constant for epsilon greedy policy 
						(default: .01)
	-dl EPSILON_ANNEAL	Number of timesteps to linearly anneal epsilon 
						exploration (default: 100,000)
	-m					Enable (past) action masking (default: False)
```

#### Examples
BR experiment, PPAM-based agent, learning rate=0.5, seed=125

```
python main.py -S 125 boat tabular-q -m -l 0.5
```

BR experiment, vanilla agent, learning rate=0.5, seed=125

```
python main.py -S 125 boat tabular-q -l 0.5
```

#### Plotting results with Tensorboard
We recommend using the `--logdir_spec` argument to specify which data logs to use and to label them. 

Logs can be found in the `./runs/boat/tabular-q/corrupt/MASK/SEED` folders, where `MASK` and `SEED` match the corresponding parameters of the command that was used to generate the data. If PPAM was enabled, then `MASK='mask'`, otherwise `MASK='no-mask'`. 

Running the following command will show the plots for the data that was generated from the previous examples:

```
tensorboard --logdir_spec=mask:./runs/boat/tabular-q/corrupt/mask/125,no-mask:./runs/boat/tabular-q/corrupt/no-mask/125 --reload_multifile=true
```

Observe that the paths to the log dirs must be separated by a `,`. The `reload_multifile=True` argument is needed as we have observed that, if this argument is not given, sometimes Tensorboard does not load all data in the logs.

#### Plotting results with `plot_results.py`
To plot results using `plot_results.py` you need to have data stored in a .csv file. Moreover, folders containing the .csv files themselves should be named `mask` and `no-mask`.  

```
python plot_results.py -datafolders path/to/datafolder1 path/to/datafolder2 
					   [-save PATH_TO_FILE] [-no-performance]
```

You have to specify the paths to the two folders containing the data from the PPAM-based agent and the vanilla one. The optional argument `-save` can be used to save the plot directly in the file at `PATH_TO_FILE`. Finally, the optional argument `no-performance` is used to plot the observable reward function instead of the performance function, which is the one shown by default. 

The following command will plot the (performance) data that is already included in the supplementary material (assuming you are in the `boat_race` folder):

```
 python plot_results.py -datafolders ./experimental_data/mask ./experimental_data/no-mask
```

For the reward data you can use the following command:

```
 python plot_results.py -datafolders ./experimental_data/mask ./experimental_data/no-mask --no-performance
```