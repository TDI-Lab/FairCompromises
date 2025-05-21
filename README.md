# pb_env
participatory budgeting game environment utilising gym and pabutools.

**custom_environment** contains the code for the PB voting environment built ontop of gymnasium.

**voting_agent** is where the agent model is found.

**training** contains the learning environment in which the baseline/ RL training is carried out.

**utils** is where helper functions are stored. This includes for data manipulation and also evaluation. 

Files with "main" at the start are main functions that produce experiment data, unless followed by "eval" in which case they produce the evaluation data for that experiment.

Evaluation.py, evaluation_comparison.py and plots.py all contain the functions for producing evaluations.



