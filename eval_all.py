from evaluation import *
from plots import *


PATHS = ['aarau_data/switzerland_aarau_2023.pb', 'aarau_data/switzerland_aarau_2023.pb', "toulouse_data/france_toulouse_2019_.pb", "toulouse_data/france_toulouse_2019_.pb"]
experiments = ["aarau_mes", "aarau_good_categories", "toulouse_mes", "toulouse_shourt_new_algo"]

testing=False
if testing:
    mes_stages = ["mes30AgentsSample"+str(x)+"Time0" for x in range(5)]
    greedy_stages = ["greedy30AgentsSample"+str(x)+"Time0" for x in range(5)]
    stages = [mes_stages, greedy_stages, mes_stages, greedy_stages]
else:
    aarau_mes_stages = ["mes1703AgentsSample"+str(x)+"Time0" for x in range(10)]
    aarau_greedy_stages = ["greedy1703AgentsSample"+str(x)+"Time0" for x in range(10)]
    toulouse_mes_stages = ["mes1494AgentsSample"+str(x)+"Time0" for x in range(10)]
    toulouse_greedy_stages = ["greedy1494AgentsSample"+str(x)+"Time0" for x in range(10)]
    stages = [aarau_mes_stages, aarau_greedy_stages, toulouse_mes_stages, toulouse_greedy_stages]

font = {'size': 12}

pal, _ = colour_palette()
pal2, pal2_name = colour_palette("#ffe600" ,"#ff3319")

VAMs = ["mes", "greedy", "mes", "greedy"]
BALLOT = "cumulative"
data = {experiment:{} for experiment in experiments}
for i, experiment in enumerate(experiments):
    for stage in stages[i]:
        data[experiment][stage] = open_data(experiment, stage)

rewardsloss(experiments, data, stages, font)
relative_cost(data, experiments, stages, PATHS, font, ballot="cumulative")
how_well_are_they_doin(data, experiments, stages, PATHS, font)