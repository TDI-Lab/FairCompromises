from utils.rl_utils import e_greedy, ExperienceReplayMemory
import torch.optim as optim
import torch
import random
import ipdb
import os

def learning_environment(config, environment, voters, experiment_name):
    """
    This is ammending the learning environment so that It works with RL.
    learning environments simplify data production for evaluation and test/train splits. 
    """
    print("-------- Running {} -------- ".format(experiment_name+" "+config["exp_part"]))
    # For saving data
    measures = ["rewards", "actions", "winners", "obs", "instance", "loss"]
    data = {
        #"training":{measure:{voter:[] for voter in voters} for measure in measures},
        "validation":{measure:{voter:[] for voter in voters} for measure in measures},
        "test":{measure:{voter:[] for voter in voters} for measure in measures},
        "infos":{
            "voter_preferences": environment.voter_preferences,
            "config" : config,
            "agents": environment.possible_agents
        }
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_interval = config['training_config']["val_interval"]  

    training_steps = 0

    #Initialize dictionaries of buffers and optimizers.
    memory = {voter: ExperienceReplayMemory(config["training_config"]["buffer_capacity"]) for voter in voters}
    optimizers = {voter: optim.AdamW(
        voters[voter].q.parameters(), 
        lr=config["training_config"]["learning_rate"], 
        amsgrad=True) for voter in voters}

    path_start = os.path.join("exp_data", experiment_name)
    if not os.path.exists(path_start):
        os.mkdir(path_start)
    path_middle = os.path.join(path_start, config['exp_part'])
    if not os.path.exists(path_middle):
        os.mkdir(path_middle)

    # Training & Validation
    for iter in range(config['training_config']["num_iters"]):
                
        if iter % val_interval == 0:
            with torch.no_grad():
                for voter in voters:
                    voters[voter].eval()
                # validation epoch
                observations, _ = environment.reset()
                observations = {observer:observations[observer].to(device)
                                for observer in observations
                                }
                val_stage = 0
                # create transition
                terminations = {'a':False} #placeholder
                while all(terminations.values())==False:
                    actions = {
                        voter: e_greedy(voters[voter], observations[voter], training_steps, environment, voter, test=True, cap=False, ballot=config['env_config']['ballot']) 
                        for voter in voters
                        } 
                    
                    if config['env_config']['ballot'] == 'approval':
                        saving_actions = {}
                        for voter, action in actions.items():
                            new_action = torch.zeros_like(action)
                            for x in action:
                                new_action[x]=1
                            saving_actions[voter] = new_action
                    else:
                        saving_actions = actions
                    next_observation, rewards, terminations, truncations, infos = environment.step(actions)
                    next_observation = {
                        observer: observations[observer].unsqueeze(0).type(torch.float32)#.device(device)
                        for observer in next_observation
                        }

                    loss = {voter:[] for voter in voters}
                    if iter > config["training_config"]["batch_size"]:
                        if config["training_config"]["batch_size"]==1:
                            batch = [0]
                        else:
                            max_value = min(iter, 1000)
                            batch = random.sample([x for x in range(max_value)], int(config["training_config"]["batch_size"]/2)-1)
                            batch_b = random.sample([x for x in range(max_value-config["training_config"]["batch_size"], max_value)], int(config["training_config"]["batch_size"]/2)-1)
                            batch.extend(batch_b)
                        for voter in voters:
                            #ipdb.set_trace()
                            loss[voter]=voters[voter].get_loss(
                                optimizers[voter],
                                memory[voter],
                                config,
                                batch
                                ).item()
                    else:
                        for voter in voters:
                            loss[voter]=0
                    if val_stage == 0:
                        for voter in voters:
                            data["validation"]["loss"][voter].append([loss[voter]])
                            data["validation"]["rewards"][voter].append([rewards[voter]])
                            data["validation"]["actions"][voter].append([saving_actions[voter]])
                            data["validation"]["obs"][voter].append([observations[voter]])
                            data["validation"]["winners"][voter].append([infos[voter]['winners']])
                            data['validation']['instance'][voter].append([infos[voter]['instance']])
                    else:
                        for voter in voters:
                            data["validation"]["loss"][voter][-1].append(loss[voter])
                            data["validation"]["rewards"][voter][-1].append(rewards[voter])
                            data["validation"]["actions"][voter][-1].append(saving_actions[voter])
                            data["validation"]["obs"][voter][-1].append(observations[voter])
                            data["validation"]["winners"][voter][-1].append(infos[voter]['winners'])
                            data['validation']['instance'][voter][-1].append(infos[voter]['instance'])
                    val_stage +=1

        for voter in voters:
            voters[voter].train()  

        # reset
        observations, _ = environment.reset()
        observations = {observer:observations[observer].to(device)
                                for observer in observations
                                }

        # create transition
        terminations = {'a':False} #placeholder
        training_stage = 0

        while all(terminations.values())==False:

            actions = {
                voter: e_greedy(voters[voter], observations[voter], training_steps, environment, voter) 
                for voter in voters
                } 
            if config['env_config']['ballot'] == 'approval':
                saving_actions = {}
                for voter, action in actions.items():
                    new_action = torch.zeros_like(action)
                    for x in action:
                        new_action[x]=1
                    saving_actions[voter] = new_action
            else:
                saving_actions = actions
            
            next_observation, rewards, terminations, _, infos = environment.step(actions)
            next_observation = {
                observer: observations[observer].unsqueeze(0).type(torch.float32)#.device(device)
                for observer in next_observation
                }
            training_steps += 1

            #store transition
            for voter in voters:
                memory[voter].push((
                    observations[voter].flatten().reshape(-1).numpy().tolist(),
                    actions[voter].numpy(),
                    rewards[voter],
                    next_observation[voter].flatten().reshape(-1).numpy().tolist(),
                    0. if terminations[voter] == False else 1.
                        ))   

            #call optimizer
            if iter > config["training_config"]["batch_size"]:
                if config["training_config"]["batch_size"]==1:
                    batch = [0]
                else:
                    max_value = min(iter, 1000)
                    batch = random.sample([x for x in range(max_value)], int(config["training_config"]["batch_size"]/2)-1)
                    batch_b = random.sample([x for x in range(max_value-config["training_config"]["batch_size"], max_value)], int(config["training_config"]["batch_size"]/2)-1)
                    batch.extend(batch_b)
                for voter in voters:
                    voters[voter].update_policy(
                        optimizers[voter],
                        memory[voter],
                        config,
                        batch)
            
            observations = next_observation
            training_stage +=1

    with torch.no_grad():
        for voter in voters:
                voters[voter].eval()
        for _ in range(config['training_config']["test_iters"]):
            
            # reset
            observations, _ = environment.reset()
            observations = {observer:observations[observer].to(device)
                                for observer in observations
                                }

            # create transition
            terminations = {'a':False} #placeholder
            test_stage = 0
            while all(terminations.values())==False:

                actions = {
                    voter: e_greedy(voters[voter], observations[voter], training_steps, environment, voter, test=True) 
                    for voter in voters
                    } 
                if config['env_config']['ballot'] == 'approval':
                    saving_actions = {}
                    for voter, action in actions.items():
                        new_action = torch.zeros_like(action)
                        for x in action:
                            new_action[x]=1
                        saving_actions[voter] = new_action

                else:
                    saving_actions = actions
                next_observation, rewards, terminations, truncations, infos = environment.step(actions)
                next_observation = {
                    observer: observations[observer].unsqueeze(0).type(torch.float32)#.device(device)
                    for observer in next_observation
                    }
                
            if test_stage == 0:
                for voter in voters:
                    data["test"]["obs"][voter].append([observations[voter]])
                    data["test"]["actions"][voter].append([saving_actions[voter]])
                    data["test"]["rewards"][voter].append([rewards[voter]])
                    data["test"]["winners"][voter].append([infos[voter]['winners']])
            else:
                for voter in voters:
                    data["test"]["obs"][voter][-1].append(observations[voter])
                    data["test"]["actions"][voter][-1].append(saving_actions[voter])
                    data["test"]["rewards"][voter][-1].append(rewards[voter])
                    data["test"]["winners"][voter][-1].append(infos[voter]['winners']) 
            test_stage +=1  
    return data
