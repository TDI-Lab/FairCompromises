from utils import *
from voting_agent.baseline_agent import BaselineAgent
from training.trainer import learning_environment, baseline_learning_environment
from voting_agent.BQ import BranchingQN
from voting_agent.MultiLabelQ import MultiLabelDQ
import ipdb
import random


def experiment_part(params, experiment_name, voter_preferences, instance, profile, voter_ballot, device):
    voters = [voter for voter in voter_preferences.keys()]
    params["env_config"]['first_instance'] = instance
    params["env_config"]['voter_preferences'] = voter_preferences
    params["env_config"]["voter_ballot"] = voter_ballot
    env = env_creator(params["env_config"], profile)

    # ------------ Agent initialization ----------------
    ACTION_SPACE = env.action_space(voters[0])
    observations, infos = env.reset()
    
    if instance.meta['vote_type'] == 'approval':
        ACTION_DIMENSTION = len(instance)
    if instance.meta['vote_type'] == 'cumulative':
        if instance.meta['unit']=='Toulouse':
            ACTION_DIMENSTION = 7
        if instance.meta['unit']=='Aaurau':
            ACTION_DIMENSTION = 10

    state, info = env.reset()
    params["voter_config"] = {
        "number_of_tokens" : ACTION_DIMENSTION,
        "input_dims" : len(observations[voters[0]].flatten()),
        "action_space" : ACTION_SPACE.n
    }
    voters = {}
    for agent in env.agents:
        voters[agent] = params['training_config']['algorithm'](
            params['voter_config']['input_dims'],
            params['voter_config']['number_of_tokens'], 
            params['voter_config']['action_space'], 
            params,
            device=device
            )
    
    data = learning_environment(params, env, voters, experiment_name)
    save_object(data, experiment_name, params['exp_part'])
    