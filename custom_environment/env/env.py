from pettingzoo.utils.env import ParallelEnv
from pabutools.election import Project, Instance
from pabutools.election import Project,CardinalBallot, CardinalProfile
from gymnasium.spaces import MultiDiscrete, Discrete
from pabutoolsOld.model import Voter, Candidate, Election
from pabutoolsOld.rules import utilitarian_greedy, equal_shares
from collections import Counter
import ipdb
import numpy as np
import functools
from random import randint, sample
from gmpy2 import mpq
from utils import gini_coefficient
import math
import torch

class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "PB_environment"
    }

    def __init__(self, env_config, aarau_profile, render_mode=None):
        self.first_instance = env_config["first_instance"] #This is the first stage projects, costs and categories
        self.categories = [x for x in env_config["first_instance"].categories]
        self.voting_rule = env_config["voting_rule"]
        self.num_rounds = env_config["num_rounds"]
        self.ballot_format = env_config["ballot"]
        self.render_mode = render_mode
        self.voting_stage = None
        self.instances = [env_config["first_instance"]]
        self.projects = sorted([project for project in self.first_instance])
        self.possible_agents = [voter for voter in env_config["voter_preferences"].keys()]
        self._agent_ids = set(self.possible_agents) # required by rllib error message
        self.voter_preferences = env_config["voter_preferences"]
        # Instantiate the action and observation spaces
        number_of_projects = len(self.first_instance)
        number_of_categories = len(self.first_instance.categories)
        observation_space = MultiDiscrete(np.array([[30000]+[2]*number_of_categories]*number_of_projects, dtype=np.int64))
        self.observation_spaces = {agent: observation_space for agent in self.possible_agents}
        #action_space = MultiDiscrete([number_of_projects]*10)
        if env_config["ballot"] == "cumulative":
            action_space = Discrete(number_of_projects)
        elif env_config["ballot"] == "approval":
            action_space = Discrete(number_of_projects)
        
        self.action_spaces = {agent: action_space for agent in self.possible_agents}
        self.strategy = None
        self.aarau_profile = aarau_profile

        # reward function variants
        self.utilitarian_altruism = env_config["utilitarian_altruism"]
        self.egalitarian_altruism = env_config["egalitarian_altruism"]
        self.project_voting_cap = env_config["project_voting_cap"]
        self.voted_for = env_config["voted_for"]

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialise the agents variable, and set up the environment so that
        step and render can be called without issue. 

        This returns the first observation and will be called to start an episode.
        """
        # voting_stage is the number of the round.
        self.voting_stage = 0

        # get_obs is a function that takes a pabutools instance and turns it into the
        # Discrete format so that it can be read by AI.
        obs = self.get_obs(self.instances[self.voting_stage])

        # for rllib
        self.agents = self.possible_agents[:]

        # Generate observation
        observations = {agent: obs for agent in self.agents}

        # infos is something required by gym. A useful way to save and visualise training data.
        infos = {agent: {} for agent in self.agents}
        return observations, infos


    def step(self, actions):
        """
        Takes on board actions and changes the environment, moves to the next state and 
        distributes rewards accordingly.
        """
        # This puts voting data in a format suitable for pabutools to compute the winners.
        election = self.discrete_to_election(actions, self.instances[self.voting_stage])

        # Compute relevant voting rule
        if self.voting_rule == "method_of_equal_shares":
            oldPabWinners = equal_shares(election, completion="add1_utilitarian")
        if self.voting_rule == "greedy":
            oldPabWinners = utilitarian_greedy(election)
        winners = []
        for winner in oldPabWinners:
            winners.append(self.first_instance.get_project(winner.id))

        rewards = self.reward_signal(winners, actions)
        
        if self.egalitarian_altruism:
            minimum_reward = min(rewards.values())
            rewards = {voter:reward + minimum_reward for voter, reward in rewards.items()}
        elif self.utilitarian_altruism:
            average_reward = sum(rewards.values())/len(rewards)
            rewards = {voter:reward + average_reward for voter, reward in rewards.items()}

        self.voting_stage += 1

        # terminations produce a DONE signal for reasons *internal* to game logic
        terminations = {agent: False for agent in self.agents}
        if self.voting_stage > self.num_rounds -1 :
            terminations = {agent: True for agent in self.agents}
            obs = self.get_obs(self.first_instance, end=True)
            observations = {agent: obs for agent in self.agents}
        else:
            #produce new observation
            new_instance = self.project_proposal(winners)#, self.instances[self.voting_stage])
            self.instances.append(new_instance)
            obs = self.get_obs(new_instance)
            observations = {agent: obs for agent in self.agents}
        
        # truncations produce a DONE signal for reasons *external* to game logic
        # eg. its running too long. I wont need this.
        env_truncation = False
        truncations = {agent: env_truncation for agent in self.agents}

        #infos are typically empty
        infos = {agent: {
            "winners": winners,
            "instance": self.instances[self.voting_stage-1]
            } for agent in self.agents}
        return  observations, rewards, terminations, truncations, infos


    def reward_signal(self, winners, actions):
        if self.ballot_format=="cumulative":
            return self.reward_cumulative(winners, actions)
        if self.ballot_format=="approval":
            return self.reward_approval(winners, actions)


    def reward_cumulative(self, winners, actions):
        """
        Reward to each agent is the sum over the projects in the winning set that the agent voted for of the log 
        of the cost of the project multiplied by the proportion of project categories favoured
        multiplied by proportion of voter preferences that are satisfied by the project.
        """
        rewards = {voter:0 for voter in self.voter_preferences}
        current_instance = self.instances[self.voting_stage]
  
        # for 0 indexing
        vote_corrected = {}
        for voter, vote in actions.items():
            vote_corrected[voter] = [token+1 for token in vote.tolist()]
        num_tokens = len(vote)

        category_overlap = {voter:{project:0 for project in current_instance} for voter in self.voter_preferences.keys()}
        for voter, preferences in self.voter_preferences.items():
            for project in current_instance:
                for category, there in project.categories.items():
                    if there == 1:
                        if preferences[category]==1:
                            category_overlap[voter][project] +=1

        total_features = {project:0 for project in current_instance}                    
        for project in current_instance:
                for category, there in project.categories.items():
                    if there == 1:
                        total_features[project] +=1
        
        total_preferences = {voter:0 for voter in self.voter_preferences.keys()}
        for voter, preferences in self.voter_preferences.items():
            for _, there in preferences.items():
                if there == 1:
                    total_preferences[voter] +=1

        for voter, preferences in self.voter_preferences.items():
            for project in winners:
                tokens = vote_corrected[voter].count(int(project.name))
                if self.voted_for:
                    if tokens != 0:
                        if total_features[project] != 0:
                            if total_preferences[voter] !=0:
                                reward =math.log(project.cost/2)*category_overlap[voter][project]*category_overlap[voter][project]/(total_features[project]*total_preferences[voter]*num_tokens)*tokens
                                rewards[voter] += reward
                else:
                    if total_features[project] != 0:
                            if total_preferences[voter] !=0:
                                reward =math.log(project.cost/2)*category_overlap[voter][project]*category_overlap[voter][project]/(total_features[project]*total_preferences[voter]*num_tokens)
                                rewards[voter] += reward


        return rewards
    
    def reward_approval(self, winners, actions):
        """
        Reward to each agent is the sum over the projects in the winning set that the agent voted for of the log 
        of the cost of the project multiplied by the proportion of project categories favoured
        multiplied by proportion of voter preferences that are satisfied by the project.
        """
        rewards = {voter:0 for voter in self.voter_preferences}
        current_instance = self.instances[self.voting_stage]
        
        vote_corrected = {}
        for voter, vote in actions.items():
            vote_corrected[voter] = list(set(self.projects[action] for action in vote))
        
        category_overlap = {voter:{project:0 for project in current_instance} for voter in self.voter_preferences.keys()}
        for voter, preferences in self.voter_preferences.items():
            for project in current_instance:
                for category in project.categories:
                    if preferences[category]==1:
                        category_overlap[voter][project] +=1

        total_features = {project:0 for project in current_instance}                    
        for project in current_instance:
                for category in project.categories:
                    total_features[project] +=1

        total_preferences = {voter:0 for voter in self.voter_preferences.keys()}
        for voter, preferences in self.voter_preferences.items():
            for _ in preferences:
                total_preferences[voter] +=1
        for voter, preferences in self.voter_preferences.items():
            for project in winners:
                
                if self.voted_for:
                    tokens = vote_corrected[voter].count(project.name)
                    if tokens != 0:
                        if total_features[project] != 0:
                            if total_preferences[voter] !=0:
                                reward =math.log(project.cost/2)*category_overlap[voter][project]*category_overlap[voter][project]/(total_features[project]*total_preferences[voter]*10)
                                rewards[voter] += reward
                else:
                    if total_features[project] != 0:
                        if total_preferences[voter] !=0:
                            reward =math.log(project.cost/2)*category_overlap[voter][project]*category_overlap[voter][project]/(total_features[project]*total_preferences[voter]*10)
                            rewards[voter] += reward
        if self.project_voting_cap:
            for voter, vote in vote_corrected.items():
                if len(vote)>len(self.first_instance)/4:
                    rewards[voter]=0  
        return rewards

    def discrete_to_election(self, actions, instance):
        """
        Converts the actions of a population of agents from a 
        format suitable for RL, to a format suitable for pabutoolsv1.
        """
        new_election = Election()
        new_election.budget = int(instance.budget_limit)
        new_election.method = instance.meta['vote_type']
        
        for project in instance:
            proj = {
                "cost":str(int(project.cost)),
                "votes":'',
                "score":'', 
                "name":"Project"+project.name
                }
            c = Candidate(
                project.name, 
                int(project.cost), 
                "Project"+project.name,
                proj = proj
                )
            new_election.profile[c] = {}

        vote_corrected = {}
        if new_election.method=='cumulative':
            for voter, vote in actions.items():
                #ipdb.set_trace()
                vc = [token+1 for token in vote.tolist()]
                vote_corrected[voter] = dict(Counter(vc))
        if new_election.method=='approval':
            for voter, vote in actions.items():
                vc = set(token+1 for token in vote.tolist())
                vote_corrected[voter] = dict(Counter(vc))

        if new_election.method=='approvalOLD':
            for voter, vote in actions.items():
                vote_corrected[voter] = [self.projects[i] for i, action in enumerate(vote) if action==1]
        
        for voter in vote_corrected:
            v = Voter(
                voter,
                None,
                None,
            )
            new_election.voters.add(v)
            if new_election.method=='cumulative':
                ballot = vote_corrected[voter]
                
                for project, score in ballot.items():
                    for candidate in new_election.profile.keys():
                        #ipdb.set_trace()
                        if candidate.id == str(project):
                            new_election.profile[candidate][v] = int(score)

            if new_election.method=='approval':
                ballot = vote_corrected[voter]
                for project, score in ballot.items():
                    for candidate in new_election.profile.keys():
                        if candidate.id == self.projects[project-1]:
                            new_election.profile[candidate][v] = 1

            if new_election.method=='approvalOLD':
                ballot = vote_corrected[voter]
                for project in ballot:
                    for candidate in new_election.profile.keys():
                        if candidate.id == str(project):
                            new_election.profile[candidate][v] = 1

        for candidate, votes in new_election.profile.items():
            candidate.votes = str(len(votes))
            candidate.score = str(sum(votes.values()))

        for c in set(c for c in new_election.profile):
            if sum(new_election.profile[c].values()) == 0: #nobody voted for the project; usually means the project was withdrawn
                del new_election.profile[c]
                
        return new_election

    def get_obs(self, first_instance, end=False):
        """
        Converts the instance from pabutools format to a format 
        suitable for RL.

        Also extrapolates to produce data for future rounds. This just repeats 
        first round. Rudimentary, but will be imporved upon as soon as working end-to-end.
        """
        num_projects = int(first_instance.meta['num_projects'])
        num_categories = len(self.categories)
        cost = 1
        # build the first instance where each position (besides cost) is a binary indicator
        # of the presence of a category
        instance_one = torch.zeros([num_projects, cost + num_categories], dtype=torch.float32)
        if end==True:
            return instance_one
        for i, project in enumerate(first_instance.project_meta):
            instance_one[i, 0] = int(project.cost/10000)
            if self.ballot_format=="approval":
                for category in project.categories:
                    j = self.categories.index(category)
                    instance_one[i, 1 + j] = 1 # 1 if that category is present
            if self.ballot_format=="cumulative":
                for category, presence in project.categories.items():
                    if presence == 1:
                        j = self.categories.index(category)
                        instance_one[i, 1 + j] = 1 # 1 if that category is present
        return instance_one   


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # The observation space is the same for every agent
        return self.observation_spaces[agent]


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # The action space is the same for every agent
        return self.action_spaces[agent]
    

    def render(self):
        # so that it can be seen by a human user, last thing to do.
        pass
    

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass



