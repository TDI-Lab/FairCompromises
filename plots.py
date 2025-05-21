from evaluation import *
from pabutools.rules import method_of_equal_shares, greedy_utilitarian_welfare, max_additive_utilitarian_welfare
import pandas as pd
import numpy as np
from pabutools.election import Cost_Sat, CumulativeProfile, CumulativeBallot
from pabutoolsOld.model import Voter, Candidate, Election
from pabutoolsOld.rules import utilitarian_greedy, equal_shares
from collections import Counter
import ipdb


def rewardsloss(experiments, data, stages, font):
    def aluq(data, stages):
        keys = list(data.keys())
        akey = keys[0]
        agents = data[akey]['infos']['agents']
        num_epochs = len(data[akey]['validation']['rewards'][agents[0]])
        rewards =  {stage:[[] for epoch in range(num_epochs)] for stage in stages}
        for stage in stages:
            agents = [voter for voter in data[stage]['test']['winners'].keys()]
            for epoch in range(num_epochs):
                for agent in agents:
                    rewards[stage][epoch].append(data[stage]['validation']['rewards'][agent][epoch][0])
        #calculate mean
        average = [sum(list(flatten(column)))/(len(stages)*len(agents)) for column in zip(*rewards.values())]
        #calculate min
        #minimum = [min(list(flatten(column))) for column in zip(*rewards.values())]
        lq = [np.percentile(np.array(list(flatten(column))), 25) for column in zip(*rewards.values())]
        #calculate max
        #maximum = [max(list(flatten(column))) for column in zip(*rewards.values())]
        uq = [np.percentile(np.array(list(flatten(column))), 75) for column in zip(*rewards.values())]
        return {"lq":lq, "ave":average, "uq":uq}
    
    def loss(data, stages):
        keys = list(data.keys())
        akey = keys[0]
        agents = data[akey]['infos']['agents']
        num_epochs = len(data[akey]['validation']['rewards'][agents[0]])
        loss =  {stage:[[] for epoch in range(num_epochs)] for stage in stages}
        df = pd.DataFrame(columns=['loss', 'epoch'])
        for stage in stages:
            agents = [voter for voter in data[stage]['test']['winners'].keys()]
            for epoch in range(7, num_epochs):
                for agent in agents:
                    los = data[stage]['validation']['loss'][agent][epoch][0].item()
                    loss[stage][epoch].append(los)
                    df.loc[len(df.index)] = [los, epoch] 
        
        #calculate mean
        average = [sum(list(flatten(column)))/(len(stages)*len(agents)) if i>7 else 0 for i, column in enumerate(zip(*loss.values()))][8:]
        #calculate min
        #minimum = [min(list(flatten(column))) for column in zip(*rewards.values())]
        lq = [np.percentile(np.array(list(flatten(column))), 25)  if i>7 else 0 for i, column in enumerate(zip(*loss.values()))][8:]
        #calculate max
        #maximum = [max(list(flatten(column))) for column in zip(*rewards.values())]
        uq = [np.percentile(np.array(list(flatten(column))), 75)  if i>7 else 0 for i, column in enumerate(zip(*loss.values()))][8:]
        return {"lq":lq, "ave":average, "uq":uq}

    keys = list(data[experiments[0]].keys())
    akey = keys[0]
    agents = data[experiments[0]][akey]['infos']['agents']
    num_epochs = len(data[experiments[0]][akey]['validation']['rewards'][agents[0]])

    lines = []
    loss_lines = []
    for i, experiment in enumerate(experiments):
        lines.append(aluq(data[experiment], stages[i]))
        loss_lines.append(loss(data[experiment], stages[i]))

    nrow = 2
    ncols = 4
    
    plt.rc('font', **font)
    fig, axs = plt.subplots(nrow,ncols, sharex=True,sharey=False, figsize=(14,6))
    fig.text(0.5, 0.04, 'Episodes', ha='center')
    #fig.text(0.04, 0.5, 'Rewards', va='center', rotation='vertical') 
    labels = ["Equal shares, Aarau", "Greedy, Aarau", "Equal Shares, Toulouse", "Greedy, Toulouse"]  
    for i in range(ncols):
        ax = axs[0][i]
        ax.plot(lines[i]["ave"])
        ax.plot(lines[i]["lq"], 'g--', linewidth=0.5)
        ax.plot(lines[i]["uq"], 'g--', linewidth=0.5)
        ax.fill_between(list(range(num_epochs)), lines[i]["lq"], lines[i]["uq"], alpha=0.2)
        ax1=axs[1][i]
        ax1.plot(list(range(8, num_epochs)),loss_lines[i]["ave"], label="average")
        ax1.plot(list(range(8, num_epochs)),loss_lines[i]["lq"], 'g--', linewidth=0.5, label="upper quartile")
        ax1.plot(list(range(8, num_epochs)),loss_lines[i]["uq"], 'g--', linewidth=0.5, label= "lower quartile")
        ax1.fill_between(list(range(8, num_epochs)), loss_lines[i]["lq"], loss_lines[i]["uq"], alpha=0.2)
        ax.set_title(labels[i])
        if i == 0:
            ax.set_ylabel("Rewards")
            ax1.set_ylabel("Loss")
        if i ==3:
            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncols=3 )
    save_figure("final", "final", "RewardLoss")
    plt.close()

def relative_cost(data, experiments, stages, paths, font, ballot="cumulative"):
    def vote_share_to_SML_costed_projects(data, experiment, stages, path, ballot="cumulative"):
        """
        This function produces a line graph that shows the share of voters' selections that go to projects that have small, medium, large,
        and extra large costs. 
        """
        def vote_share_df_cost(data, aarau_instance, experiment, stages, agents, place, sizes, costs, projects):
            df = pd.DataFrame(columns=['stage', 'percentage', 'at least'])
            average_cost_rep = {key:0 for key in sizes}
            for stage in stages:
                if place == "end":
                    voters = [voter for voter in data[experiment][stage]['test']['winners']]
                else:
                    voters = [voter for voter in data[experiment][stage]['validation']['winners']]
                cost_rep = {key:0 for key in sizes}
                
                for voter in voters:
                    if ballot=="approval":
                        if place=="end":
                            actions = [projects[i] for i, x in enumerate(data[experiment][stage]['test']['actions'][voter][0][0].tolist()) if x ==1]
                        else:
                            actions = [projects[i] for i, x in enumerate(data[experiment][stage]['validation']['actions'][voter][place][0].tolist()) if x ==1]
                    if ballot=="cumulative":
                        if place=="end":
                            actions = [x+1 for x in data[experiment][stage]['test']['actions'][voter][0][0].tolist()]
                        else:
                            actions = [x+1 for x in data[experiment][stage]['validation']['actions'][voter][place][0].tolist()]
                    for action in actions:
                        action_cost = int(aarau_instance.get_project(str(action)).cost)
                        if action_cost <= costs[int(len(costs)/4)]:
                            cost_rep['small']+=float(100)/(len(voters)*len(actions))
                        elif action_cost <= costs[int(2*len(costs)/4)]:
                            cost_rep['medium']+=float(100)/(len(voters)*len(actions))
                        elif action_cost <= costs[int(3*len(costs)/4)]:
                            cost_rep['large']+=float(100)/(len(voters)*len(actions))
                        else:
                            cost_rep['extra large']+=float(100)/(len(voters)*len(actions))
                    
                for k, v in cost_rep.items():
                    average_cost_rep[k] += v/len(stages)    
                for key, value in cost_rep.items():
                    df.loc[len(df.index)] = [stage, value, key] 
            return df, average_cost_rep

        sizes = ["small", "medium", "large", "extra large"]
        stages = list(data[experiment].keys())
        agents = data[experiment][stages[0]]['infos']['agents']

        instance, _,_ , voter_ballot = get_voter_data(path)
        costs = [int(project.cost) for project in instance]
        costs = sorted(costs)
        actions = {agent:voter_ballot[agent] for agent in agents}

        #if ballot=="approval":
        projects = [project for project in instance]
        projects = sorted(projects)

        df, average_cr = vote_share_df_cost(data, instance, experiment, stages, agents, "end", sizes, costs, projects)
        df2, average_cr2 = vote_share_df_cost(data, instance, experiment, stages, agents, 0, sizes, costs, projects)

        real_cost_rep = {key:0 for key in sizes}
        for vote in actions.values():
            if ballot=="cumulative":
                num_tokens = float(sum(vote.values()))
                for action, quant in vote.items():
                    action_cost = int(action.cost)
                    if action_cost <= costs[int(len(costs)/4)]:
                        real_cost_rep['small']+=100.*float(quant)/(len(agents)*num_tokens)
                    elif action_cost <= costs[int(2*len(costs)/4)]:
                        real_cost_rep['medium']+=100.*float(quant)/(len(agents)*num_tokens)
                    elif action_cost <= costs[int(3*len(costs)/4)]:
                        real_cost_rep['large']+=100.*float(quant)/(len(agents)*num_tokens)
                    else:
                        real_cost_rep['extra large']+=100.*float(quant)/(len(agents)*num_tokens)
            
            if ballot=="approval":
                num_tokens = len(vote)
                for action in vote:
                    action_cost = int(action.cost)
                    #
                    if action_cost <= costs[int(len(costs)/4)]:
                        real_cost_rep['small']+=100.*1/(len(agents)*num_tokens)
                    elif action_cost <= costs[int(2*len(costs)/4)]:
                        real_cost_rep['medium']+=100.*1/(len(agents)*num_tokens)
                    elif action_cost <= costs[int(3*len(costs)/4)]:
                        real_cost_rep['large']+=100.*1/(len(agents)*num_tokens)
                    else:
                        real_cost_rep['extra large']+=100.*1/(len(agents)*num_tokens)
        return {"df":df, "df2":df2, "acr":average_cr, "acr2":average_cr2, "rcp":real_cost_rep}
    
    sizes = ["small", "medium", "large", "extra large"]
    lines = []
    for i, experiment in enumerate(experiments):
        lines.append(vote_share_to_SML_costed_projects(data, experiment, stages[i], paths[i]))

    nrows = 1
    ncols = 4
    fig, axs = plt.subplots(nrows, ncols , sharex=True,sharey=True, figsize=(14,3))
    fig.text(0.5, -0.1, 'Cost category of winning projects', ha='center')
    plt.rc('font', **font)
    plt.tight_layout()
    labels = ["Equal shares, Aarau", "Greedy, Aarau", "Equal Shares, Toulouse", "Greedy, Toulouse"]  
    for i in range(ncols):
        ax = axs[i]
        sns.scatterplot(data = lines[i]["df"], x='at least', y='percentage', alpha = 0.3, ax=ax, color = "#ff3319", legend=False)
        sns.lineplot(x = range(len(sizes)), y = lines[i]["acr"].values(), label="Marl model after training", ax=ax, color = "#ff3319", legend=False)
        sns.scatterplot(data = lines[i]["df2"], x='at least', y='percentage', alpha = 0.3, ax=ax, color="#ffe600", legend=False)
        sns.lineplot(x = range(len(sizes)), y = lines[i]["acr2"].values(), label="Marl model before training", ax=ax, color="#ffe600", legend=False)
        sns.lineplot(x = range(len(sizes)), y = lines[i]["rcp"].values(), label="Actual vote", ax=ax, color='#96bfe6', legend=False)
        ax.set_xticks(range(len(sizes)), sizes)
        ax.set_title(labels[i])
        ax.set_xlabel("")
        if i == 0:
            ax.set_ylabel('Token Share, [%]')
        if i ==3:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, bbox_to_anchor =(0.5,1.2), loc='upper center', ncols=3 )
    save_figure("final", "final", "cost")
    plt.close()

def how_well_are_they_doin(data, experiments, stages, paths, font, vam="mes", ballot="cumulative"):
    def proportion_of_vote_to_winners(data, experiment, stages, path, vam="mes", ballot="cumulative"):
        """
        This function produces a line plot showing the proportion of voters' selected projects that get selected in the winning set. 
        """
        def vote_share_df(data, aarau_instance, experiment, stages, agents, place, projects, voter_preferences):
            df = pd.DataFrame(columns=['stage', 'project satisfaction', 'cost satisfaction', 'at least'])
            average_al =  {key:0 for key in range(10,110,10)}
            budget_average_al = copy.deepcopy(average_al)
            for stage in stages:
                if place == "end":
                    voters = [voter for voter in data[experiment][stage]['test']['winners']]
                    winners = data[experiment][stage]['test']['winners'][voters[0]][0][0]
                else:
                    voters = [voter for voter in data[experiment][stage]['validation']['winners']]
                    winners = data[experiment][stage]['validation']['winners'][voters[0]][0][0]
                al = {key:0 for key in range(10,110,10)}
                budget_al = copy.deepcopy(al)
                
                for voter in voters:
                    if ballot=="approval":
                        if place=="end":
                            actions = [projects[i] for i, x in enumerate(data[experiment][stage]['test']['actions'][voter][0][0].tolist()) if x ==1]
                        else:
                            actions = [projects[i] for i, x in enumerate(data[experiment][stage]['validation']['actions'][voter][place][0].tolist()) if x ==1]
                    if ballot=="cumulative":
                        if place=="end":
                            actions = [x+1 for x in data[experiment][stage]['test']['actions'][voter][0][0].tolist()]
                        else:
                            actions = [x+1 for x in data[experiment][stage]['validation']['actions'][voter][place][0].tolist()]
                    projects_voted_for = set(action for action in actions)

                    num_voted_for_project_winning = 0
                    voted_for_budget_that_wins = 0
                    for action in projects_voted_for:
                        action_project = aarau_instance.get_project(str(action))
                        if action_project in winners:
                            project_cats = category_dict_to_name(action_project.categories)
                            voter_prefs = category_dict_to_name(voter_preferences[voter])
                            overlap = list(set(voter_preferences[voter]) & set(project_cats))
                            if len(overlap) > 0:
                                num_voted_for_project_winning +=1
                                voted_for_budget_that_wins += int(action_project.cost)

                    percent_voted_project_winning = 100 * num_voted_for_project_winning / len(winners)#num_project_voted_for
                    percent_voted_project_winning = 10*int(percent_voted_project_winning/10)
                    for num in range(10,percent_voted_project_winning+10, 10):
                        al[num] += 100/len(agents)
                    percent_voted_for_budget_acquired = 100 * voted_for_budget_that_wins / int(aarau_instance.budget_limit)
                    percent_voted_for_budget_acquired = 10*int(percent_voted_for_budget_acquired/10)
                    for num in range(10,percent_voted_for_budget_acquired+10, 10):
                        budget_al[num] += 100/len(agents)
                    
                for k, v in al.items():
                    average_al[k] += v/len(stages)    
                for k, v in budget_al.items():
                    budget_average_al[k] += v/len(stages)  
                for key, value in al.items():
                    df.loc[len(df.index)] = [stage, value, budget_al[key], key] 
                #ipdb.set_trace()
            return df, average_al, budget_average_al
        
        stages = list(data[experiment].keys())
        agents = data[experiment][stages[0]]['infos']['agents']

        instance, _, voter_preferences, voter_ballot = get_voter_data(path)
        projects = [project for project in instance]
        projects = sorted(projects)
        df, average_al, budget_average_al = vote_share_df(data, instance, experiment, stages, agents, "end", projects, voter_preferences)
        df2, average_al2, budget_average_al2 = vote_share_df(data, instance, experiment, stages, agents, 1, projects, voter_preferences)
        actions = {agent:voter_ballot[agent] for agent in agents}
        real_winners = calculate_real_winners(instance, actions, vam)

        real_proportion = {key:0 for key in range(10,110,10)} 
        budget_real_proportion = {key:0 for key in range(10,110,10)}
        for voter in actions:
            projects_voted_for = set(action for action in actions[voter])
            num_voted_for_project_winning = 0
            voted_for_budget_that_wins = 0
            for action in projects_voted_for:
                action_project = instance.get_project(str(action))
                if action_project in real_winners:
                    num_voted_for_project_winning +=1
                    voted_for_budget_that_wins += int(action_project.cost)
            percent_voted_project_winning = 100 * num_voted_for_project_winning / len(real_winners)
            percent_voted_project_winning = 10*int(percent_voted_project_winning/10)
            percent_voted_for_budget_acquired = 100 * voted_for_budget_that_wins / int(instance.budget_limit)
            percent_voted_for_budget_acquired = 10*int(percent_voted_for_budget_acquired/10)
            for num in range(10,percent_voted_project_winning+10, 10):
                real_proportion[num] += 100/len(agents)
                budget_real_proportion[num] += 100/len(agents)

        return {"df":df, "df2":df2, "rw":real_winners, "al":average_al, "al2": average_al2, "bal": budget_average_al, "bal2": budget_average_al2, "instance":instance, "rp":real_proportion, "brp": budget_real_proportion}

    lines = []
    for i, experiment in enumerate(experiments):
        lines.append(proportion_of_vote_to_winners(data, experiment, stages[i], paths[i], vam="mes", ballot="cumulative"))

    nrow = 2
    ncols = 4
    plt.rc('font', **font)
    
    fig, axs = plt.subplots(nrow,ncols, sharex=False,sharey=True, figsize=(14,6))
    plt.subplots_adjust(hspace=0.3)
    #fig.tight_layout()
    fig.text(0.5, 0.47, 'Project Satisfaction, [%]', ha='center')
    fig.text(0.5, 0.02, 'Cost Satisfaction, [%]', ha='center')
    #fig.text(0.04, 0.5, 'Rewards', va='center', rotation='vertical') 
    labels = ["Equal shares, Aarau", "Greedy, Aarau", "Equal Shares, Toulouse", "Greedy, Toulouse"]  
    for i in range(ncols):

        ax = axs[0][i]

        sns.scatterplot(data = lines[i]['df'], x='at least', y='project satisfaction', alpha = 0.3, ax=ax, color = "#ff3319", legend=False)
        sns.lineplot(x = range(10,110,10), y = lines[i]['al'].values(), label="Marl model after training", ax=ax, color = "#ff3319", legend=False)
        sns.scatterplot(data = lines[i]['df2'], x='at least', y='project satisfaction', alpha = 0.3, ax=ax, color="#ffe600", legend=False)
        sns.lineplot(x = range(10,110,10), y = lines[i]['al2'].values(), label="Marl model before training", ax=ax, color="#ffe600", legend=False)
        sns.lineplot(x = range(10,110,10), y = lines[i]['rp'].values(), label="Actual vote", ax=ax, color='#96bfe6', legend=False)
        ax.set_xticks( range(10,110,10), range(10,110,10))
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax1=axs[1][i]
        sns.scatterplot(data = lines[i]['df'], x='at least', y='cost satisfaction', alpha = 0.3, ax=ax1, color = "#ff3319", legend=False)
        sns.lineplot(x = range(10,110,10), y = lines[i]['bal'].values(), label="Marl model after training", ax=ax1, color = "#ff3319", legend=False)
        sns.scatterplot(data = lines[i]['df2'], x='at least', y='cost satisfaction', alpha = 0.3, ax=ax1, color="#ffe600", legend=False)
        sns.lineplot(x = range(10,110,10), y = lines[i]['bal2'].values(), label="Marl model before training", ax=ax1, color="#ffe600", legend=False)
        sns.lineplot(x = range(10,110,10), y = lines[i]['brp'].values(), label="Actual vote", ax=ax1, color='#96bfe6', legend=False)
        ax1.set_xticks( range(10,110,10), range(10,110,10))
        ax1.set_ylabel("")
        ax1.set_xlabel("")

        ax.set_title(labels[i])
        if i == 0:
            ax.set_ylabel("Token Share, [%]")
            ax1.set_ylabel("Token Share, [%]")
        if i ==3:
            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncols=3 )
    save_figure("final","final", "satisfaction")
    plt.close()
