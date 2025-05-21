from utils import *
from pabutools.election import parse_pabulib
import pandas as pd
import seaborn as sns
from matplotlib.cm import register_cmap
from pabutoolsOld.model import Voter, Candidate, Election
from pabutoolsOld.rules import utilitarian_greedy, equal_shares
from collections import Counter

def calculate_real_winners(instance, actions ,vam):
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

    for voter in actions:
        v = Voter(
            voter,
            None,
            None,
        )
        new_election.voters.add(v)

        ballot = actions[voter]
        if isinstance(ballot, ApprovalBallot):
            for project in ballot:
                for candidate in new_election.profile.keys():
                    if candidate.id == str(project):
                        new_election.profile[candidate][v] = 1
        if isinstance(ballot, CardinalBallot):
            for project, score in ballot.items():
                for candidate in new_election.profile.keys():
                    if candidate.id == str(project):
                        new_election.profile[candidate][v] = int(score)

    for candidate, votes in new_election.profile.items():
        candidate.votes = str(len(votes))
        candidate.score = str(sum(votes.values()))
    
    for c in set(c for c in new_election.profile):
        if sum(new_election.profile[c].values()) == 0: #nobody voted for the project; usually means the project was withdrawn
            del new_election.profile[c]

    if vam == "mes":
        oldPabWinners = equal_shares(new_election, completion="add1_utilitarian")
    if vam == "greedy":
        oldPabWinners = utilitarian_greedy(new_election)
    
    winners = []
    for winner in oldPabWinners:
        winners.append(instance.get_project(winner.id))
    return winners

def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)