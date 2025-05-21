import pickle
import matplotlib.pyplot as plt
import ipdb
from pabutools.election import parse_pabulib
import pandas as pd
import numpy as np
import os
import torch
from voting_agent.BDQ import BranchingDQN 
from utils import add_categories_v2, population_scaler

def load_models(data, epoch):
    config = data['infos']['config']

    path_start = os.path.join("exp_data", config['experiment_name'])
    path_middle = os.path.join(path_start, config['experiment_part'])
    path_epoch = os.path.join(path_middle, "epoch"+str(epoch*5))
    voters = {}
    for agent in data['infos']['agents']:
        voters[agent] = BranchingDQN(
                config['voter_config']['input_dims'],
                config['voter_config']['action_space'], 
                config['voter_config']['number_of_tokens'], 
                config
                )
        PATH = os.path.join(path_epoch, agent)
        voters[agent].load_state_dict(torch.load(PATH))
    return voters

def category_dict_to_name(category_dict):
    present_categories = []
    for category, presence in category_dict.items():
        if presence == 1:
            present_categories.append(category)
    return present_categories

def open_data(experiment, part):
    path = os.path.join('exp_data', str(experiment),str(part))
    file = open(path+'.pickle', 'rb')
    data = pickle.load(file)
    file.close()
    return data

def save_figure(experiment, part, graph):
    PATH = os.path.join("graphs", experiment)
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    PATH = os.path.join(PATH, part)
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    PATH = os.path.join(PATH, graph)
    try:
        plt.savefig(PATH+".png", bbox_inches="tight")
        print("Plot saved as {}".format(PATH))
    except:
        print("Could not save plot")

def cat_rep(data):
    """
    This function takes a set of winners and returns a value of how often 
    a category is represented in the winning set.
    """
    winners = data['validation']['winners']['voter_0']
    category_representation = {category:[[0 for j in i] for i in winners] for category in winners[0][0][0].categories.keys()}
    for i, epoch in enumerate(winners):
        for j, stage in enumerate(epoch):
            for winner in stage:
                for category, presence in winner.categories.items():
                    category_representation[category][i][j] += presence
    cr = {}
    for category, value in category_representation.items():
        cr[category] = []
        for epoch in value:
            cr[category].append(epoch[0])
    return cr

def satisfaction(observations, votes, winners):
    """
    For each agent, this is the cost of the winning projects that the agent 
    voted for /approves of. This is a welfare measure calculated for each agent 
    which is then fed into gini function.

    This definition is lifted and ammended from "fairness in Long Term PB"
    """
    # TODO: The original definition for this is for approval voting, 
    # there is a possiblity of doing this for
    # characteristic approval, as well as voted for. 
    winners_names = [int(winner.name) for winner in winners]
    satisfaction = {voter:0 for voter in votes}
    for voter, vote in votes.items():
        intersection = list(set(vote) & set(winners_names))
        for project_index in intersection:
            satisfaction[voter] += observations[project_index][0] # cost is first entry

    return satisfaction

def satisfaction_v2(votes, winners, instance, voter_prefs):
    """
    For each agent, this is the cost of the winning projects that the agent 
    voted for /approves of. This is a welfare measure calculated for each agent 
    which is then fed into gini function.

    This definition is lifted and ammended from "fairness in Long Term PB"
    """
    # TODO: The original definition for this is for approval voting, 
    # there is a possiblity of doing this for
    # characteristic approval, as well as voted for. 
    winners_names = [int(winner.name) for winner in winners]
    vote_corrected = {}
    for voter, vote in votes.items():
        vote_corrected[voter] = [token+1 for token in vote]
    satisfaction = {voter:0 for voter in votes}
    for voter, vote in vote_corrected.items():
        intersection = list(set(vote) & set(winners_names))
        for project_index in intersection:
            satisfaction[voter] += int(instance.get_project(str(project_index)).cost)


    return satisfaction

def share(observations, votes, winners):
    """
    share is welfare measure. For each agent, this is the cost of the winning projects that the agent 
    voted for /approves of divided by the amount of other agents that voted for that project.
    This is a welfare measure calculated for each agent which is then fed into gini function.

    This is lifted and ammended from "fairness in Long Term PB"
    """
    winners_names = [int(winner.name) for winner in winners]
    share = {voter:0 for voter in votes}
    for voter, vote in votes.items():
        intersection = list(set(vote) & set(winners_names))
        for project_index in intersection:
            number_of_other_voters_that_favor = 0
            for inner_voter in votes:
                if project_index in votes[inner_voter]:
                    number_of_other_voters_that_favor += 1
            cost = observations[voter][project_index][0]
            share[voter] += cost / number_of_other_voters_that_favor
    
    return share

def gini_coefficient(values):
    """
    The gini coefficient is a measure of inequality. I am using this with satisfaction and share for 
    the evaluation section.
    """
    if len(values)==0:
        print(values)
        ipdb.set_trace()
    if sum(values)==0:
        print(values)
        ipdb.set_trace()
    sorted_values = sorted(values)
    total_cum_sum = 0
    for i, v in enumerate(sorted_values):
        total_cum_sum += v * (len(values) - i)
    return (len(values) +1 - (2*total_cum_sum)/sum(values)) /len(values)

def borda_score(project, actions):
    num_projects = 33
    num_voters = len(actions.keys())
    """
    This gives the normalised sum of number of people that prefer the chosen project to other projects
    for every other project.
    """
    borda_count = 0
    for iter_proj in range(num_projects):
        Nab = 0
        if iter_proj == project:
            pass
        else:
            for voter, action in actions.items():
                #how many times does project feature in the vote
                chosen_projects = action.tolist()
                if chosen_projects.count(project) > chosen_projects.count(iter_proj):
                    Nab += 1
        # normalise it
        borda_count += Nab/((num_projects - 1) * num_voters)
    
    return borda_count

def divisiveness(project, sub_population, actions):
    # sub_population is a list of names of voters who favor a to b

    other_sub_population = list(set(actions.keys()) - set(sub_population))
    
    people_who_prefered_a_to_b = {person: actions[person] for person in sub_population}
    everyone_else =  {person: actions[person] for person in other_sub_population}

    
    div_score = abs(borda_score(project, people_who_prefered_a_to_b) - borda_score(project, everyone_else))
    return div_score

def alpha_divisiveness(project, actions, alpha=1, l=4):
    num_projects = 33
    num_voters = len(actions.keys())
    alpha_div = 0
    
    for iter_proj in range(num_projects):
        sub_population = []
        if iter_proj == project:
            pass
        else:
            #who prefers project to iterproj
            for voter, action in actions.items():
                #how many times does project feature in the vote
                chosen_projects = action.tolist()
                if chosen_projects.count(project) > chosen_projects.count(iter_proj):
                    sub_population.append(voter)
        
        if len(sub_population) > 0 and len(sub_population) < num_voters:
            competing_pop_sizes = (4* len(sub_population) * (num_voters-len(sub_population)))/(num_voters*num_voters)
            alpha_div += pow(competing_pop_sizes, alpha) * divisiveness(project, sub_population, actions)
    
    alpha_div = alpha_div/(num_projects - 1)

    return alpha_div

def save_object(obj, exp, part):
    parent_dir = "exp_data"
    try:
        path = os.path.join(parent_dir, exp)
        os.mkdir(path)
        print("Directory '%s' created" %exp)
    except:
        pass
    try:
        with open("exp_data/"+str(exp)+"/"+str(part)+".pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)








