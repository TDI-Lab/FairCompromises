import  params_greedy as pms
from run import experiment_part
import ipdb
from utils import *
import random
import cProfile
from voting_agent.BQ import BranchingQN
import copy

if __name__ == "__main__":
    path = "toulouse_data/france_toulouse_2019_.pb"
    instance, profile, voter_preferences, voter_ballot = get_voter_data(path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = copy.deepcopy(pms.PARAMS)
    params['data path']=  path
    params['samples'] = 10
    params['robustness']=1
    params['experiment_name']= "toulouse_greedy"
    
    params['experiments']['greedy']['training_config']['num_iters']=300
    params['experiments']['greedy']['training_config']['algorithm']=BranchingQN
    params['experiments']['greedy']['env_config']['ballot']='cumulative'
    params['experiments']['greedy']['env_config']['project_voting_cap']=False
    params['experiments']['greedy']['env_config']['voted_for']=True
    params['experiments']['greedy']['env_config']['utilitarian_altruism'] = False
    params['experiments']['greedy']['env_config']['egalitarian_altruism'] = False

    
    TESTING=True
    if TESTING:
        params['samples'] = 5
        params['experiments']['greedy']['training_config']['num_iters']=100
        voter_preferencesF, voter_ballotF = usable_preferences(profile, instance)
        params['num_agents'] = 30
        agent_sample = random.sample(sorted(voter_preferencesF), params['num_agents'])
        voter_preferences = {voter:voter_preferencesF[voter] for voter in agent_sample}
        voter_ballot = {voter:voter_ballotF[voter] for voter in agent_sample}

    for exp_name, exp_params in params['experiments'].items():
        params['experiments'][exp_name]['env_config']['num_agents']=len(voter_preferences)
        for sample in range(params['samples']):
            for time in range(params['robustness']):
                exp_params['exp_part']=exp_name+str(len(voter_preferences))+"Agents"+"Sample"+str(sample)+"Time"+str(time)
                experiment_part(
                    exp_params, params['experiment_name'],
                    voter_preferences=voter_preferences,
                    instance=instance,
                    profile=profile, 
                    voter_ballot=voter_ballot,
                    device=device)
                #cProfile.run("experiment_part(exp_params, params.PARAMS['experiment_name'],voter_preferences=voter_preferences[time],aarau_instance=aarau_instance,aarau_profile=aarau_profile, voter_ballot)")
