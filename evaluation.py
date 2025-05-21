import matplotlib.pyplot as plt
import ipdb
from utils import *
from pabutools.election import parse_pabulib
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import register_cmap
from evaluation_comparison import *
from pabutools.election import  CumulativeProfile, CumulativeBallot, ApprovalBallot, ApprovalProfile
import copy


def project_votes_and_winners_over_time(data, experiment, vam, stages, pal, ballot):    
    """
    This function produces a heat map of how many votes each project recieves and how this changes over validation
    epochs. This includes total and a breakdown of agent votes. project heatmaps are sorted by volume of votes recieved, and agents
    are also sorted by number of votes. 
    """
    
    mes_data = data[vam]
    
    voters = {}
    winners = {}
    actions = {}
    for stage in mes_data:
        actions[stage] = mes_data[stage]['validation']['actions']
        voters[stage] = [voter for voter in actions[stage].keys()]
        winners[stage] = mes_data[stage]['validation']['winners'][voters[stage][0]]

    instance = mes_data[stages[0]]['validation']['instance'][voters[stage][0]][0][0]
    if ballot=="cumulative":
        projects = [int(project.name) for project in instance]
        projects = sorted(projects, reverse=True)
    if ballot=="approval":
        projects = [project for project in instance]
        projects = sorted(projects)

    #define the dataframe
    num_epochs = len(winners[stages[0]])
    num_voters = len(voters[stages[0]])
    num_projects = len(projects)
    data_frame = {project: np.array([[0. for epoch in range(num_epochs)] for voter in range(num_voters)]) for project in projects}
    #fill data frame from with voter actions over validation epochs
    for stage in stages:
        for i, voter in enumerate(actions[stage]):
            for epoch, _ in enumerate(actions[stage][voter]):
                for j, action in enumerate(actions[stage][voter][epoch][0]):
                    if ballot=="cumulative":
                        data_frame[action.item()+1][i][epoch] += 1/len(stages)
                    if ballot=="approval":
                        data_frame[projects[j]][i][epoch] += action.item()/len(stages)

    # set max value for colour bar
    vmax = 0
    for project in projects:
        for epoch in range(num_epochs):
            for voter in range(num_voters):
                score = data_frame[project][voter][epoch]
                if score > vmax:
                    vmax=score

    #sort projects by volume of votes over all validation epochs
    projects_sorted = {k:v for k, v in sorted(data_frame.items(), key=lambda item: sum(sum(item[1])), reverse=True)}
    projects_sorted = list(projects_sorted.keys()) 
    
    #sort voters by volume of votes
    data_frame_sorted = {}
    voters_sorted = {}
    for project in projects_sorted:
        df = data_frame[project]
        order = {}
        for i, voter in enumerate(df):
            order[voters[stages[0]][i]]=sum(voter)
        vs = list({k:v for k, v in sorted(order.items(), key=lambda item: item[1])}.keys())
        df_sorted = sorted(df, key=lambda item: sum(item))
        data_frame_sorted[project] = df_sorted
        voters_sorted[project] = vs



    # add labels which are the number of times this is featured in winning set
    pro_pres = {project: [0. for epoch in data_frame_sorted[projects[0]][0]]
                for project in projects
                }
    for i, epoch in enumerate(range(0, num_epochs)):
        for stage in stages:
            for winning_set in winners[stage][epoch]:
                for winner in winning_set:
                    if ballot=="cumulative":
                        pro_pres[int(winner.name)][i] +=1*num_voters/len(stages)
                    if ballot=="approval":
                        pro_pres[winner.name][i] +=1*num_voters/len(stages)

    fig, axs = plt.subplots(1,num_projects,figsize=(24, 6))
    #fig.set_figheight(2)
    #fig.set_figwidth(36)
    cbar_ax = fig.add_axes([.91, .1, .01, .8])
    for i, ax in enumerate(axs):
        ax.set_title(projects_sorted[i])
        sns.heatmap(
                data_frame_sorted[projects_sorted[i]],
                cmap=pal, 
                ax=ax, 
                cbar=i == 0, 
                vmin=0,
                vmax=vmax,
                cbar_ax= cbar_ax,
                cbar_kws={'label': 'project presence\n in voter ballot'},
                #annot = pro_pres[projects_sorted[i]],
                fmt=''
                )
        #ax.set_yticks([i+0.5 for i in range(num_voters)],voters_sorted[projects_sorted[i]], rotation='horizontal')
        sns.lineplot(x=[i for i in range(num_epochs)],y=pro_pres[projects_sorted[i]],ax=ax, color="red")
        if i ==0:
            ax.tick_params(axis='y', colors='red')  
            ax.spines['left'].set_color('red') 
            ax.set_yticks([0,num_voters/2,num_voters],[0,50,100], rotation='horizontal')
            ax.set_ylabel("average representation in\nwinning set, %", rotation='vertical')
        else:
            ax.set_yticks([],[], rotation='horizontal')
        if i == num_projects-1:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel("voters", rotation='vertical')
        ax.set_xticks([num_epochs],[num_epochs], rotation='horizontal')
        ax.invert_yaxis()
    fig.text(0.5, 0.04, 'Validation Episodes', ha='center')
    save_figure(experiment, vam, "AverageProjectChangeOT")  
    plt.close()

def category_votes_and_winners_over_time(data, experiment, vam, stages, pal, ballot):
    """
    This function produces a heat map of how many votes each category recieves and how this changes over validation
    epochs. This includes total and a breakdown of agent votes. category heatmaps are sorted by volume of votes recieved, and agents
    are also sorted by number of votes.
    """
    data = data[vam]

    voters = {}
    winners = {}
    actions = {}
    
    for stage in data:
        actions[stage] = data[stage]['validation']['actions']
        voters[stage] = [voter for voter in actions[stage].keys()]
        winners[stage] = data[stage]['validation']['winners'][voters[stage][0]]
    if ballot=="cumulative":
        categories = [category for category in winners[stages[0]][0][0][0].categories.keys()]
    if ballot=="approval":
        categories = set()
        for _, meta in data[stages[0]]['infos']['config']['env_config']['first_instance'].project_meta.items():
            for thing in meta['categories']:
                categories.add(thing)
        categories = sorted(list(categories))
        
    instance = data[stages[0]]['validation']['instance'][voters[stage][0]][0][0]
    if ballot=="cumulative":
        projects = [int(project.name) for project in instance]
        projects = sorted(projects, reverse=True)
    if ballot=="approval":
        projects = [project for project in instance]
        projects = sorted(projects)

    num_epochs = len(winners[stages[0]])
    num_voters = len(voters[stages[0]])
    num_projects = len(projects)

    #define the dataframe
    data_frame = {category: np.array([[0. for epoch in range(num_epochs)] for voter in range(num_voters)]) for category in categories}
    #fill dataframe with category voting data
    for stage in stages:
        for i, voter in enumerate(actions[stage]):
            for epoch, _ in enumerate(actions[stage][voter]):
                for j, action in enumerate(actions[stage][voter][epoch][0]):
                    if ballot=="cumulative":
                        for category, presence in instance.get_project(str(action.tolist()+1)).categories.items():
                            if int(presence)==1:
                                data_frame[category][i][epoch] += 1/len(stages)
                    if ballot=="approval":
                        for category in instance.get_project(str(projects[j])).categories:
                            data_frame[category][i][epoch] += action.item()/len(stages)

    # set maximum value for colourbar
    vmax = 0
    for category in categories:
        for epoch in range(num_epochs):
            for voter in range(num_voters):
                score = data_frame[category][voter][epoch]
                if score > vmax:
                    vmax=score

    #sort category graphs by volume of votes
    categories_sorted = {k:v for k, v in sorted(data_frame.items(), key=lambda item: sum(sum(item[1])), reverse=True)}
    categories_sorted = list(categories_sorted.keys()) 

    #sort dataframe and voters by volume of votes
    data_frame_sorted = {}
    voters_sorted = {}
    for category in categories_sorted:
        df = data_frame[category]
        order = {}
        for i, voter in enumerate(df):
            order[voters[stages[0]][i]]=sum(voter)
        vs = list({k:v for k, v in sorted(order.items(), key=lambda item: item[1])}.keys())
        df_sorted = sorted(df, key=lambda item: sum(item))
        data_frame_sorted[category] = df_sorted
        voters_sorted[category] = vs


    cat_pres = {category: [0 for epoch in data_frame_sorted[categories[0]][0]]
                for category in categories
                }
    for i, epoch in enumerate(range(0, num_epochs)):
        for stage in stages:
            for winning_set in winners[stage][epoch]:
                num_categories_in_winning_set = 0
                for winner in winning_set:
                    if ballot=="approval":
                        num_categories_in_winning_set += len(winner.categories)
                    if ballot=="cumulative":
                        num_categories_in_winning_set += len(category_dict_to_name(winner.categories))

                for winner in winning_set:
                    if ballot=="approval":
                        cat = winner.categories
                    if ballot=="cumulative":
                        cat = category_dict_to_name(winner.categories)
                    for category_name in cat:
                        cat_pres[category_name][i] +=100/(num_categories_in_winning_set*len(stages))
    max_percent = 0.
    for category in categories:
        for epoch in range(num_epochs):
            if cat_pres[category][epoch]>max_percent:
                max_percent=float(cat_pres[category][epoch])
    
    for category in categories:
        for epoch in range(num_epochs):
            cat_pres[category][epoch] = cat_pres[category][epoch]*num_voters/max_percent

    fig, axs = plt.subplots(1,len(categories),figsize=(24, 6))
    #fig.set_figheight(2.5)
    #fig.set_figwidth(25)
    cbar_ax = fig.add_axes([.91, .1, .01, .8])

    for i, ax in enumerate(axs):
        ax.set_title(categories_sorted[i])
        sns.heatmap(
                data_frame_sorted[categories_sorted[i]],
                cmap=pal, 
                ax=ax, 
                #cbar=i == 8, 
                cbar_kws={'label': 'category presence\n in voter ballot'},
                vmin=0,
                vmax=vmax,
                cbar_ax= cbar_ax,
                #annot = cat_pres[categories_sorted[i]],
                fmt='.2f'
                )
        sns.lineplot(x=[i for i in range(num_epochs)],y=cat_pres[categories_sorted[i]],ax=ax, color="red")
        #ax.set_yticks([i+0.5 for i in range(num_voters)],voters_sorted[categories_sorted[i]], rotation='horizontal')
        #ax.set_xticks([0, len(shrunken_data_frame[categories[0]][0])],[0, num_epochs], rotation='vertical')
        #ax.set_ylim(0,10)
        if i ==0:
            ax.tick_params(axis='y', colors='red')  
            ax.spines['left'].set_color('red') 
            ax.set_yticks([0,num_voters/2,num_voters],[0,int(max_percent/2),int(max_percent)], rotation='horizontal')
            ax.set_ylabel("percentage category\nrepresentation in\nwinning set, %", rotation='vertical')
        else:
            ax.set_yticks([],[], rotation='horizontal')
        if i == 8:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel("voters", rotation='vertical')
        ax.set_xticks([0,int(num_epochs/2),num_epochs],[0,int(num_epochs/2),num_epochs], rotation='horizontal')
        

        ax.invert_yaxis()
    #fig.tight_layout(rect=[0, 0, .9, 1])
    fig.text(0.5, 0.04, 'Validation Episodes', ha='center')
    save_figure(experiment, vam,"AverageCategoryChangeOT")  
    plt.close()

def colour_palette(colour1="#e5f5f9", colour2="#006d2c"):
    """ 
    This produces a seaborn colour palette for colourbars to range between two values
    """
    pos = [0.0, 1.0]
    colors=[colour1,colour2]
    cmap = LinearSegmentedColormap.from_list("", list(zip(pos, colors)))
    name = colour1+colour2
    register_cmap(name, cmap)
    pal= sns.color_palette(colour1+colour2, as_cmap=True)#,n_colors=50)
    return pal, name

def fairness_metrics(data, experiment, vam="mes", real_winners = False, path="amsterdam_data/netherlands_amsterdam_304_.pb", ballot="approval"):
    """
    Calculates fainess metrics for a participatory budgeting election. Can be set to real data, or experimental data.
    """ 
    stages = list(data[vam].keys())
    winning_voted = []
    winning_budget = []
    winning_personal_budget = []
    
    instance, aarau_profile, voter_preferences_full, voter_ballot = get_voter_data(path)
    real_winners_yej = calculate_real_winners(instance, voter_ballot, vam)
    agents = data[vam][stages[0]]['infos']['agents']
    if ballot=="approval":
        projects = [project for project in instance]
        projects = sorted(projects)

    for stage in stages:
        
        winners = data[vam][stage]['test']['winners'][agents[0]][0][0]
        if real_winners == True:
            winners = real_winners_yej
            
        num_voters_who_voted_for_a_winner = {winner:0 for winner in winners}

        for voter in agents:
            if isinstance(voter_ballot[voter], ApprovalBallot):
                actions = [projects[i] for i, x in enumerate(data[vam][stage]['test']['actions'][voter][0][0].tolist())  if x ==1]
            if isinstance(voter_ballot[voter], CardinalBallot):
                actions = [x+1 for x in data[vam][stage]['test']['actions'][voter][0][0].tolist()]
            if real_winners == True:

                if isinstance(voter_ballot[voter], ApprovalBallot):
                    actions = set(voter_ballot[voter])
                if isinstance(voter_ballot[voter], CardinalBallot):
                    actions = set(voter_ballot[voter].keys())
            
            for action in actions:
                if str(action) in winners:
                    num_voters_who_voted_for_a_winner[str(action)]+=1

        for voter in agents:
            if isinstance(voter_ballot[voter], ApprovalBallot):
                actions = [projects[i] for i, x in enumerate(data[vam][stage]['test']['actions'][voter][0][0].tolist())  if x ==1]
            if isinstance(voter_ballot[voter], CardinalBallot):
                actions = [x+1 for x in data[vam][stage]['test']['actions'][voter][0][0].tolist()]
            
            if real_winners == True:
                if isinstance(voter_ballot[voter], ApprovalBallot):
                    actions = set(voter_ballot[voter])
                if isinstance(voter_ballot[voter], CardinalBallot):
                    actions = set(voter_ballot[voter].keys())
            projects_voted_for = set(instance.get_project(str(action)) for action in actions)

            winning_voted_for = 0
            winning_voted_for_budget = 0
            winning_voted_for_budget_per_person = 0

            for winner in winners:
                win_proper = instance.get_project(winner.name)
                if winner in projects_voted_for:
                    winning_voted_for +=1
                    winning_voted_for_budget += int(win_proper.cost)
                    winning_voted_for_budget_per_person += int(win_proper.cost)/num_voters_who_voted_for_a_winner[winner.name]

            winning_voted.append(100*winning_voted_for/len(winners) if len(winners) > 0 else 0)
            winning_budget.append(100*winning_voted_for_budget/int(instance.budget_limit))
            winning_personal_budget.append(winning_voted_for_budget_per_person)

    data = {"gini":{}, "min":{}, "average":{}}

    data["average"]["num"]=sum(winning_voted)/(len(stages)*len(agents))
    data["min"]["num"]=min(winning_voted)
    data["gini"]["num"] = (gini_coefficient(winning_voted) if sum(winning_voted)>1 else 1)

    data["average"]["budget"]=sum(winning_budget)/(len(stages)*len(agents))
    data["min"]["budget"]=min(winning_budget)
    data["gini"]["budget"] = gini_coefficient(winning_budget)

    data["average"]["share"]=sum(winning_personal_budget)/(len(stages)*len(agents))
    data["min"]["share"]=min(winning_personal_budget)
    data["gini"]["share"] = gini_coefficient(winning_personal_budget)
    print(data)
    return data

def vote_share_to_SML_costed_projects(data, experiment, vam="mes", ballot="approval", path="amsterdam_data/netherlands_amsterdam_304_.pb"):
    """
    This function produces a line graph that shows the share of voters' selections that go to projects that have small, medium, large,
    and extra large costs. 
    """
    def vote_share_df_cost(data, aarau_instance, vam, stages, agents, place, sizes, costs, projects):
        df = pd.DataFrame(columns=['stage', 'percentage', 'at least'])
        average_cost_rep = {key:0 for key in sizes}
        for stage in stages:
            if place == "end":
                voters = [voter for voter in data[vam][stage]['test']['winners']]
            else:
                voters = [voter for voter in data[vam][stage]['validation']['winners']]
            cost_rep = {key:0 for key in sizes}
            
            for voter in voters:
                if ballot=="approval":
                    if place=="end":
                        actions = [projects[i] for i, x in enumerate(data[vam][stage]['test']['actions'][voter][0][0].tolist()) if x ==1]
                    else:
                        actions = [projects[i] for i, x in enumerate(data[vam][stage]['validation']['actions'][voter][place][0].tolist()) if x ==1]
                if ballot=="cumulative":
                    if place=="end":
                        actions = [x+1 for x in data[vam][stage]['test']['actions'][voter][0][0].tolist()]
                    else:
                        actions = [x+1 for x in data[vam][stage]['validation']['actions'][voter][place][0].tolist()]
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
    stages = list(data[vam].keys())
    agents = data[vam][stages[0]]['infos']['agents']


    instance, _,_ , voter_ballot = get_voter_data(path)
    costs = [int(project.cost) for project in instance]
    costs = sorted(costs)
    actions = {agent:voter_ballot[agent] for agent in agents}

    #if ballot=="approval":
    projects = [project for project in instance]
    projects = sorted(projects)

    df, average_cr = vote_share_df_cost(data, instance, vam, stages, agents, "end", sizes, costs, projects)
    df2, average_cr2 = vote_share_df_cost(data, instance, vam, stages, agents, 1, sizes, costs, projects)

    real_cost_rep = {key:0 for key in sizes}
    for vote in actions.values():
        if ballot=="cumulative":
            num_tokens = float(sum(vote.values()))
            for action, quant in vote.items():
                action_cost = int(action.cost)
                #ipdb.set_trace()
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

    fig, ax = plt.subplots(figsize=(6,4))#1,2, sharex=True, sharey=True)
    sns.scatterplot(data = df, x='at least', y='percentage', alpha = 0.3, ax=ax, color = "#ff3319")
    sns.lineplot(x = range(len(sizes)), y = average_cr.values(), label="Marl model after training", ax=ax, color = "#ff3319")
    sns.scatterplot(data = df2, x='at least', y='percentage', alpha = 0.3, ax=ax, color="#ffe600")
    sns.lineplot(x = range(len(sizes)), y = average_cr2.values(), label="Marl model before training", ax=ax, color="#ffe600")
    sns.lineplot(x = range(len(sizes)), y = real_cost_rep.values(), label="Actual vote", ax=ax, color='#96bfe6')
    ax.set_xticks(range(len(sizes)), sizes)
    ax.set_xlabel('Cost category of winning projects')
    ax.set_ylabel('Token Share, [%]')

    #fig.suptitle("Proportion of vote going to each cost category: {}".format(vam))

    save_figure(experiment, vam, "vscost")
    plt.close()

def proportion_of_vote_to_winners(data, experiment, vam="mes", ballot="approval", path="amsterdam_data/netherlands_amsterdam_304_.pb"):
    """
    This function produces a line plot showing the proportion of voters' selected projects that get selected in the winning set. 
    """
    def vote_share_df(data, aarau_instance, vam, stages, agents, place, projects):
        df = pd.DataFrame(columns=['stage', 'project satisfaction', 'cost satisfaction', 'at least'])
        average_al =  {key:0 for key in range(10,110,10)}
        budget_average_al = copy.deepcopy(average_al)
        for stage in stages:
            if place == "end":
                voters = [voter for voter in data[vam][stage]['test']['winners']]
                winners = data[vam][stage]['test']['winners'][voters[0]][0][0]
            else:
                voters = [voter for voter in data[vam][stage]['validation']['winners']]
                winners = data[vam][stage]['validation']['winners'][voters[0]][0][0]
            al = {key:0 for key in range(10,110,10)}
            budget_al = copy.deepcopy(al)
            
            for voter in voters:
                if ballot=="approval":
                    if place=="end":
                        actions = [projects[i] for i, x in enumerate(data[vam][stage]['test']['actions'][voter][0][0].tolist()) if x ==1]
                    else:
                        actions = [projects[i] for i, x in enumerate(data[vam][stage]['validation']['actions'][voter][place][0].tolist()) if x ==1]
                if ballot=="cumulative":
                    if place=="end":
                        actions = [x+1 for x in data[vam][stage]['test']['actions'][voter][0][0].tolist()]
                    else:
                        actions = [x+1 for x in data[vam][stage]['validation']['actions'][voter][place][0].tolist()]
                projects_voted_for = set(action for action in actions)

                num_voted_for_project_winning = 0
                voted_for_budget_that_wins = 0
                for action in projects_voted_for:
                    action_project = aarau_instance.get_project(str(action))
                    if action_project in winners:
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
    
    stages = list(data[vam].keys())
    agents = data[vam][stages[0]]['infos']['agents']

    instance, aarau_profile, voter_preferences_full, voter_ballot = get_voter_data(path)
    #if ballot=="approval":
    projects = [project for project in instance]
    projects = sorted(projects)
    df, average_al, budget_average_al = vote_share_df(data, instance, vam, stages, agents, "end", projects)
    df2, average_al2, budget_average_al2 = vote_share_df(data, instance, vam, stages, agents, 1, projects)
    actions = {agent:voter_ballot[agent] for agent in agents}
    real_winners = calculate_real_winners(instance, actions, vam)

    def representation_graph(df, average_al, df2, average_al2, vam, actions, real_winners, agents, type):
        real_proportion = {key:0 for key in range(10,110,10)}
        for voter in actions:
            projects_voted_for = set(action for action in actions[voter])
            num_voted_for_project_winning = 0
            for action in projects_voted_for:
                action_project = instance.get_project(str(action))
                if action_project in real_winners:
                    num_voted_for_project_winning +=1
            percent_voted_project_winning = 100 * num_voted_for_project_winning / len(real_winners)
            percent_voted_project_winning = 10*int(percent_voted_project_winning/10)
            for num in range(10,percent_voted_project_winning+10, 10):
                real_proportion[num] += 100/len(agents)

        fig,ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(data = df, x='at least', y=type, alpha = 0.3, ax=ax, color = "#ff3319")
        sns.lineplot(x = range(10,110,10), y = average_al.values(), label="Marl model after training", ax=ax, color = "#ff3319")
        sns.scatterplot(data = df2, x='at least', y=type, alpha = 0.3, ax=ax, color="#ffe600")
        sns.lineplot(x = range(10,110,10), y = average_al2.values(), label="Marl model before training", ax=ax, color="#ffe600")
        sns.lineplot(x = range(10,110,10), y = real_proportion.values(), label="Actual vote", ax=ax, color='#96bfe6')
        ax.set_xticks( range(10,110,10), range(10,110,10))
        ax.set_xlabel(str(type)+', [%]')
        ax.set_ylabel('Voter Share, [%]')
        #fig.suptitle("percentage of selected projects that win: {}".format(vam))
        save_figure(experiment, vam, "vs"+str(type))
        plt.close() 
    #ipdb.set_trace()
    representation_graph(df, average_al, df2, average_al2, vam, actions, real_winners, agents, 'project satisfaction')
    representation_graph(df, budget_average_al, df2, budget_average_al2, vam, actions, real_winners, agents, 'cost satisfaction')

def proportion_of_voted_budget_to_winners(data, experiment, vam="mes", ballot="approval", path="amsterdam_data/netherlands_amsterdam_304_.pb"):
    """
    This function produces a line plot showing the proportion of voters' selected budget that get selected in the winning set. 
    """
    def vote_share_df(data, aarau_instance, vam, stages, agents, place, projects):
        df = pd.DataFrame(columns=['stage', 'percentage', 'at least'])
        average_al =  {key:0 for key in range(10,110,10)}
        for stage in stages:
            if place == "end":
                voters = [voter for voter in data[vam][stage]['test']['winners']]
                winners = data[vam][stage]['test']['winners'][voters[0]][0][0]
            else:
                voters = [voter for voter in data[vam][stage]['validation']['winners']]
                winners = data[vam][stage]['validation']['winners'][voters[0]][0][0]
            al = {key:0 for key in range(10,110,10)}
            
            for voter in voters:
                if ballot=="approval":
                    if place=="end":
                        actions = [projects[i] for i, x in enumerate(data[vam][stage]['test']['actions'][voter][0][0].tolist()) if x ==1]
                    else:
                        actions = [projects[i] for i, x in enumerate(data[vam][stage]['validation']['actions'][voter][place][0].tolist()) if x ==1]
                if ballot=="cumulative":
                    if place=="end":
                        actions = [x+1 for x in data[vam][stage]['test']['actions'][voter][0][0].tolist()]
                    else:
                        actions = [x+1 for x in data[vam][stage]['validation']['actions'][voter][place][0].tolist()]
                projects_voted_for = set(action for action in actions)

                voted_for_budget_that_wins = 0
                for action in projects_voted_for:
                    action_project = aarau_instance.get_project(str(action))
                    if action_project in winners:
                        voted_for_budget_that_wins += int(action_project.cost)
                percent_voted_for_budget_acquired = 100 * voted_for_budget_that_wins / int(aarau_instance.budget_limit)
                percent_voted_for_budget_acquired = 10*int(percent_voted_for_budget_acquired/10)
                for num in range(10,percent_voted_for_budget_acquired+10, 10):
                    al[num] += 1/len(agents)
                
            #df.loc[len(df.index)] = [stage, *[x for x in al.values()]]
            for k, v in al.items():
                average_al[k] += v/len(stages)    
            for key, value in al.items():
                df.loc[len(df.index)] = [stage, value, key] 
        return df, average_al
    
    stages = list(data[vam].keys())
    agents = data[vam][stages[0]]['infos']['agents']

    instance, aarau_profile, voter_preferences_full, voter_ballot = get_voter_data(path)
    
    #if ballot=="approval":
    projects = [project for project in instance]
    projects = sorted(projects)

    df, average_al = vote_share_df(data, instance, vam, stages, agents, "end", projects)
    df2, average_al2 = vote_share_df(data, instance, vam, stages, agents, 5, projects)
            
    actions = {agent:voter_ballot[agent] for agent in agents}
    real_winners = calculate_real_winners(instance, actions, vam)

    real_proportion_of_budget = {key:0 for key in range(10,110,10)}
    for voter in actions:
        projects_voted_for = set(action for action in actions[voter])
        projects_voted_for_costs = [int(instance.get_project(str(action)).cost) for action in projects_voted_for]
        budget_voted_for = sum(projects_voted_for_costs)
        voted_for_budget_that_wins = 0
        for action in projects_voted_for:
            action_project = instance.get_project(str(action))
            if action_project in real_winners:
                voted_for_budget_that_wins += int(action_project.cost)
        percent_voted_for_budget_acquired = 100 * voted_for_budget_that_wins / int(instance.budget_limit)
        percent_voted_for_budget_acquired = 10*int(percent_voted_for_budget_acquired/10)
        for num in range(10,percent_voted_for_budget_acquired+10, 10):
            real_proportion_of_budget[num] += 1/len(agents)

    fig, ax = plt.subplots(figsize = (6,4))
    sns.scatterplot(data = df, x='at least', y='percentage', alpha = 0.3, ax=ax, color = "#ff3319")
    sns.lineplot(x = range(10,110,10), y = average_al.values(), label="Marl model after training", ax=ax, color = "#ff3319")
    sns.scatterplot(data = df2, x='at least', y='percentage', alpha = 0.3, ax=ax, color="#ffe600")
    sns.lineplot(x = range(10,110,10), y = average_al2.values(), label="Marl model before training", ax=ax, color="#ffe600")
    sns.lineplot(x = range(10,110,10), y = real_proportion_of_budget.values(), label="Actual vote", ax=ax, color='#96bfe6')
    ax.set_xticks( range(10,110,10), range(10,110,10))
    ax.set_xlabel('selected budget representation, [%]')
    ax.set_ylabel('Voter Share, [%]')
    #fig.suptitle("Proportion of chosen budget that gets selected: {}".format(vam))
    save_figure(experiment, vam, "vs3")
    plt.close()
    return

def vote_share_to_multi_valent_projects(data, experiment, vam="mes", ballot="approval", path="amsterdam_data/netherlands_amsterdam_304_.pb"):
    """
    This function produces a line plot showing the amount of votes going to projects with 1, 2, 3 ... categories. This plot compares the votes from the model experiments to the
    the amount of votes received in the real instance.
    """
    def vote_share_df(data, aarau_instance, vam, stages, agents, place, projects):  
        df = pd.DataFrame(columns=['stage', 'percentage', 'num_categories'])
        average_al =  {key:0 for key in range(6)}
        for stage in stages:
            if place == "end":
                voters = [voter for voter in data[vam][stage]['test']['winners']]
            else:
                voters = [voter for voter in data[vam][stage]['validation']['winners']]
            cat_rep = {key:0 for key in range(6)}
            
            for voter in voters:
                if ballot=="approval":
                    if place=="end":
                        actions = [projects[i] for i, x in enumerate(data[vam][stage]['test']['actions'][voter][0][0].tolist()) if x ==1]
                    else:
                        actions = [projects[i] for i, x in enumerate(data[vam][stage]['validation']['actions'][voter][place][0].tolist()) if x ==1]
                if ballot=="cumulative":
                    if place=="end":
                        actions = [x+1 for x in data[vam][stage]['test']['actions'][voter][0][0].tolist()]
                    else:
                        actions = [x+1 for x in data[vam][stage]['validation']['actions'][voter][place][0].tolist()]
                for action in actions:
                    action_project = aarau_instance.get_project(str(action))
                    if ballot=="cumulative":
                        num_categories = sum([int(x) for x in action_project.categories.values()])
                    if ballot=="approval":
                        num_categories = len(action_project.categories)
                    cat_rep[num_categories] += 100/(len(agents)*len(actions))
                
            for k, v in cat_rep.items():
                average_al[k] += v/len(stages)    
            for key, value in cat_rep.items():
                df.loc[len(df.index)] = [stage, value, key] 
        return df, average_al
        
    stages = list(data[vam].keys())
    agents = data[vam][stages[0]]['infos']['agents']

    instance, aarau_profile, voter_preferences_full, voter_ballot = get_voter_data(path)

    #if ballot=="approval":
    projects = [project for project in instance]
    projects = sorted(projects)

    df, average_al = vote_share_df(data, instance, vam, stages, agents, "end", projects)
    df2, average_al2 = vote_share_df(data, instance, vam, stages, agents, 0, projects)
            
    actions = {agent:voter_ballot[agent] for agent in agents}

    real_cat_rep= {key:0 for key in range(6)}
    for vote in actions.values():
        num_tokens = float(sum(vote.values()))
        if ballot =="cumulative":
            for action, quant in vote.items():
                aarau_proj = instance.get_project(action.name).categories
                num_categories = sum([int(x) for x in aarau_proj.values()])
                real_cat_rep[num_categories] += 100.*float(quant)/(len(agents)*num_tokens)
        if ballot =="approval":
            for action in vote:
                aarau_proj = instance.get_project(action.name).categories
                num_categories = len(aarau_proj)
                real_cat_rep[num_categories] += 100.*float(quant)/(len(agents)*num_tokens)

    #.set_trace()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data = df, x='num_categories', y='percentage', alpha = 0.3, ax=ax, color = "#ff3319")
    sns.lineplot(x = range(6), y = average_al.values(), label="Marl model after training", ax=ax, color = "#ff3319")
    sns.scatterplot(data = df2, x='num_categories', y='percentage', alpha = 0.3, ax=ax, color="#ffe600")
    sns.lineplot(x = range(6), y = average_al2.values(), label="Marl model before training", ax=ax, color="#ffe600")
    sns.lineplot(x = range(6), y = real_cat_rep.values(), label="Actual vote", ax=ax, color='#96bfe6')
    ax.set_xticks(range(6), range(6))
    ax.set_xlabel('Number of project impact area contributions')
    ax.set_ylabel('Token share, [%]')
    #fig.suptitle("Share of votes that go to projects with x categories : {}".format(vam))
    save_figure(experiment, vam, "categoryShare")
    plt.close()

def vote_share_prefered(data, experiment, vam="mes", ballot="approval", path="amsterdam_data/netherlands_amsterdam_304_.pb"):
    def vote_share_df(data, aarau_instance, vam, stages, agents, place, projects, voter_preferences_full):  
        df = pd.DataFrame(columns=['stage', 'percentage', 'num_categories'])
        average_al =  {key:0 for key in range(6)}
        for stage in stages:
            if place == "end":
                voters = [voter for voter in data[vam][stage]['test']['winners']]
            else:
                voters = [voter for voter in data[vam][stage]['validation']['winners']]
            cat_rep = {key:0 for key in range(6)}
            
            for voter in voters:
                if ballot=="approval":
                    if place=="end":
                        actions = [projects[i] for i, x in enumerate(data[vam][stage]['test']['actions'][voter][0][0].tolist()) if x ==1]
                    else:
                        actions = [projects[i] for i, x in enumerate(data[vam][stage]['validation']['actions'][voter][place][0].tolist()) if x ==1]
                if ballot=="cumulative":
                    if place=="end":
                        actions = [x+1 for x in data[vam][stage]['test']['actions'][voter][0][0].tolist()]
                    else:
                        actions = [x+1 for x in data[vam][stage]['validation']['actions'][voter][place][0].tolist()]
                for action in actions:
                    action_project = aarau_instance.get_project(str(action))
                    num_prefered_categories = 0
                    if ballot=="cumulative":
                        for category in category_dict_to_name(action_project.categories):
                            if category in category_dict_to_name(voter_preferences_full[voter]):
                                num_prefered_categories +=1
                    if ballot=="approval":
                        for category in action_project.categories:
                            if category in voter_preferences_full[voter]:
                                num_prefered_categories +=1
                    
                    cat_rep[num_prefered_categories] += 100/(len(agents)*len(actions))
                
            for k, v in cat_rep.items():
                average_al[k] += v/len(stages)    
            for key, value in cat_rep.items():
                df.loc[len(df.index)] = [stage, value, key]
        return df, average_al
        
    stages = list(data[vam].keys())
    agents = data[vam][stages[0]]['infos']['agents']

    instance, aarau_profile, voter_preferences_full, voter_ballot = get_voter_data(path)

    projects = [project for project in instance]
    projects = sorted(projects)

    df, average_al = vote_share_df(data, instance, vam, stages, agents, "end", projects, voter_preferences_full)
    df2, average_al2 = vote_share_df(data, instance, vam, stages, agents, 0, projects, voter_preferences_full)
            
    actions = {agent:voter_ballot[agent] for agent in agents}

    real_cat_rep= {key:0 for key in range(6)}
    for voter, vote in actions.items():
        for action, quant in vote.items():
            num_prefered_categories = 0
            if ballot=="cumulative":
                aarau_proj = instance.get_project(action.name).categories
                for category in category_dict_to_name(aarau_proj):
                    if category in category_dict_to_name(voter_preferences_full[voter]):
                        num_prefered_categories +=1
            if ballot=="approval":
                aarau_proj = instance.get_project(action.name).categories
                for category in aarau_proj:
                    if category in voter_preferences_full[voter]:
                        num_prefered_categories +=1
            real_cat_rep[num_prefered_categories] += 100./(len(agents)*len(vote))

    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data = df, x='num_categories', y='percentage', alpha = 0.3, ax=ax)
    sns.lineplot(x = range(6), y = average_al.values(), label="Marl model after training", ax=ax)
    sns.scatterplot(data = df2, x='num_categories', y='percentage', alpha = 0.3, ax=ax)
    sns.lineplot(x = range(6), y = average_al2.values(), label="Marl model before training", ax=ax)
    sns.lineplot(x = range(6), y = real_cat_rep.values(), label="Actual vote", ax=ax)
    ax.set_xticks(range(6), range(6))
    ax.set_xlabel('Number of prefered categories')
    ax.set_ylabel('Token share, [%]')
    #fig.suptitle("share of votes that that go to projects with x categories that the voter favours : {}".format(vam))
    save_figure(experiment, vam, "preferedCategoryShare")
    plt.close()

def vote_share_not_prefered(data, experiment, vam="mes", ballot="approval", path="amsterdam_data/netherlands_amsterdam_304_.pb"):
    def vote_share_df(data, aarau_instance, vam, stages, agents, place, projects, voter_preferences_full):  
        df = pd.DataFrame(columns=['stage', 'percentage', 'num_categories'])
        average_al =  {key:0 for key in range(6)}
        for stage in stages:
            if place == "end":
                voters = [voter for voter in data[vam][stage]['test']['winners']]
            else:
                voters = [voter for voter in data[vam][stage]['validation']['winners']]
            cat_rep = {key:0 for key in range(6)}
            
            for voter in voters:
                if ballot=="approval":
                    if place=="end":
                        actions = [projects[i] for i, x in enumerate(data[vam][stage]['test']['actions'][voter][0][0].tolist()) if x ==1]
                    else:
                        actions = [projects[i] for i, x in enumerate(data[vam][stage]['validation']['actions'][voter][place][0].tolist()) if x ==1]
                if ballot=="cumulative":
                    if place=="end":
                        actions = [x+1 for x in data[vam][stage]['test']['actions'][voter][0][0].tolist()]
                    else:
                        actions = [x+1 for x in data[vam][stage]['validation']['actions'][voter][place][0].tolist()]
                for action in actions:
                    action_project = aarau_instance.get_project(str(action))
                    num_prefered_categories = 0
                    if ballot=="cumulative":
                        for category in category_dict_to_name(action_project.categories):
                            if category not in category_dict_to_name(voter_preferences_full[voter]):
                                num_prefered_categories +=1
                    if ballot=="approval":
                        for category in action_project.categories:
                            if category not in voter_preferences_full[voter]:
                                num_prefered_categories +=1
                    
                    cat_rep[num_prefered_categories] += 100/(len(agents)*len(actions))
                
            for k, v in cat_rep.items():
                average_al[k] += v/len(stages)    
            for key, value in cat_rep.items():
                df.loc[len(df.index)] = [stage, value, key]
        return df, average_al
        
    stages = list(data[vam].keys())
    agents = data[vam][stages[0]]['infos']['agents']

    instance, aarau_profile, voter_preferences_full, voter_ballot = get_voter_data(path)

    projects = [project for project in instance]
    projects = sorted(projects)

    df, average_al = vote_share_df(data, instance, vam, stages, agents, "end", projects, voter_preferences_full)
    df2, average_al2 = vote_share_df(data, instance, vam, stages, agents, 0, projects, voter_preferences_full)
            
    actions = {agent:voter_ballot[agent] for agent in agents}

    real_cat_rep= {key:0 for key in range(6)}
    for voter, vote in actions.items():
        for action, quant in vote.items():
            num_prefered_categories = 0
            if ballot=="cumulative":
                aarau_proj = instance.get_project(action.name).categories
                for category in category_dict_to_name(aarau_proj):
                    if category not in category_dict_to_name(voter_preferences_full[voter]):
                        num_prefered_categories +=1
            if ballot=="approval":
                aarau_proj = instance.get_project(action.name).categories
                for category in aarau_proj:
                    if category not in voter_preferences_full[voter]:
                        num_prefered_categories +=1
            real_cat_rep[num_prefered_categories] += 100./(len(agents)*len(vote))

    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data = df, x='num_categories', y='percentage', alpha = 0.3, ax=ax)
    sns.lineplot(x = range(6), y = average_al.values(), label="Marl model after training", ax=ax)
    sns.scatterplot(data = df2, x='num_categories', y='percentage', alpha = 0.3, ax=ax)
    sns.lineplot(x = range(6), y = average_al2.values(), label="Marl model before training", ax=ax)
    sns.lineplot(x = range(6), y = real_cat_rep.values(), label="Actual vote", ax=ax)
    ax.set_xticks(range(6), range(6))
    ax.set_xlabel('Number of not prefered categories')
    ax.set_ylabel('Token share, [%]')
    #fig.suptitle("share of votes that that go to projects with x categories that the voter favours : {}".format(vam))
    save_figure(experiment, vam, "NotPreferedCategoryShare")
    plt.close()
