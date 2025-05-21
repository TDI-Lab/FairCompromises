import pandas as pd
import numpy as np
from pabutools.election import parse_pabulib, ApprovalBallot, CardinalBallot
import ipdb

def get_category_csv():
    acd = pd.read_csv("aarau_data/new_categories.csv", encoding='latin-1')
    ocd = pd.read_csv("aarau_data/aarau_categories.csv", encoding='latin-1')
    acd = acd.fillna(0)
    acd = acd.replace(to_replace='?',value=int(1))
    acd['Project ID']=ocd['project_id']
    acd['Education'] = acd['Education'].astype(int)
    acd['Welfare'] = acd['Welfare'].astype(int)
    acd['Sport'] = acd['Sport'].astype(int)
    acd['Urban Greenery'] = acd['Urban Greenery'].astype(int)
    acd['Public Transit and Roads'] = acd['Public Transit and Roads'].astype(int)
    acd['Culture'] = acd['Culture'].astype(int)
    acd['Health'] = acd['Health'].astype(int)
    acd['Environmental Protection'] = acd['Environmental Protection'].astype(int)
    acd['Public Space'] = acd['Public Space'].astype(int)
    #ipdb.set_trace()
    return acd

def get_categories_toulouse(data_path):
    category_data = pd.read_csv("toulouse_data/toulouse_categories.csv", encoding='latin-1')
    category_data = category_data.fillna(0)
    category_data = category_data.rename(columns={"ï»¿Project ID": "Project ID"})
    return category_data

def add_categories_v3(category_data, instance):
    pre_categories = category_data.columns.values.tolist()[-9:]
    #ipdb.set_trace()
    project_meta = instance.project_meta
    for project in instance:
        project_id = project.name
        categories = {category: 0 for category in pre_categories}
        project_row = category_data.loc[category_data['Project ID'] == int(project_id)]
        for category in category_data.columns:
            if category != 'Project ID':
                if project_row[category].values[0] == 1:
                    categories[category] = 1
        project.categories = categories
        project_meta[project]['categories'] = categories
        project_meta[project]['score'] = ''
        project_meta[project]['votes'] = ''
        project_meta[project]['name'] = category_data[category_data['Project ID']==int(project_id)]['Project Name (English)'].item()


    instance.project_meta = project_meta
    instance.categories = [category for category in category_data.columns 
        if category not in ["project ID", "Project Name", "Project Name (English)", "Project Description"]]

    meta = instance.meta
    meta['num_votes'] = ''
    instance.meta = meta

    return instance

def env_creator(env_config, aarau_profile):
    from custom_environment.env.env import CustomEnvironment
    return CustomEnvironment(env_config, aarau_profile)

def get_voter_data(path):
    """
    reproduces the voter preferences for analysis
    """
    instance, profile = parse_pabulib(path)
    if instance.meta['unit'] == 'Toulouse':
        category_data = get_categories_toulouse(path)
        instance = add_categories_v3(category_data, instance)

    if instance.meta['unit']=='Aaurau':
        category_data = get_category_csv()
        instance = add_categories_v3(category_data, instance)
    voter_preferences, voter_ballot = usable_preferences(profile, instance)

    return instance, profile, voter_preferences, voter_ballot

def usable_preferences(profile, instance):
    voter_preferences_2 = {}
    voter_ballot = {}
    for i, ballot in enumerate(profile):
        voter_pref = {category:0 for category in instance.categories}
        if isinstance(ballot, ApprovalBallot):
            for project in ballot:
                for category in instance.get_project(project.name).categories:
                    if voter_pref[category] ==1:
                        pass
                    else:
                        voter_pref[category] = 1
        elif isinstance(ballot, CardinalBallot):
            for project in ballot.keys():
                for category, presence in instance.get_project(project.name).categories.items():
                    if voter_pref[category] ==1:
                        pass
                    else:
                        voter_pref[category] = presence

        voter_preferences_2["voter_{}".format(i)] = voter_pref
        voter_ballot["voter_{}".format(i)] = ballot
    return voter_preferences_2, voter_ballot