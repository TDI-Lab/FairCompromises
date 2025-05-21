
import matplotlib.pyplot as plt
from utils import *
import pandas as pd
import numpy as np
import os 
from pandas.core.common import flatten

font = {'size'   : 10}

def rewards(experiment, data, stages, vam):
    long=True
    keys = list(data.keys())
    akey = keys[0]
    agents = data[akey]['infos']['agents']
    num_epochs = len(data[akey]['validation']['rewards'][agents[0]])
    rewards =  {stage:[[] for epoch in range(num_epochs)] for stage in stages}
    for stage in stages:
        agents = [voter for voter in data[stage]['test']['winners'].keys()]
        for epoch in range(num_epochs):
            for agent in agents:
                if long:
                    rewards[stage][epoch].append(sum(data[stage]['validation']['rewards'][agent][epoch]))
                else:
                    rewards[stage][epoch].append(data[stage]['validation']['rewards'][agent][epoch][0])
    #calculate mean
    average = [sum(list(flatten(column)))/(len(stages)*len(agents)) for column in zip(*rewards.values())]
    #calculate min
    lq = [np.percentile(np.array(list(flatten(column))), 25) for column in zip(*rewards.values())]
    #calculate max
    uq = [np.percentile(np.array(list(flatten(column))), 75) for column in zip(*rewards.values())]
    plt.rc('font', **font)
    fig, axs = plt.subplots()
    axs.plot(average)
    axs.plot(lq, 'g--', linewidth=0.5)
    axs.plot(uq, 'g--', linewidth=0.5)
    axs.fill_between(list(range(num_epochs)), lq, uq, alpha=0.2)
    #axs.set_title('Validation rewards for population over training')
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Rewards')

    PATH = os.path.join("graphs", experiment)
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    PATH = os.path.join(PATH, "rewards_envelope_{}".format(vam))

    try:
        plt.savefig(PATH+".png", bbox_inches="tight")
        print("Plot saved as {}".format(PATH))
    except:
        print("Could not save plot")
    plt.close()

def loss(experiment, data, stages, vam):
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
    lq = [np.percentile(np.array(list(flatten(column))), 25)  if i>7 else 0 for i, column in enumerate(zip(*loss.values()))][8:]
    #calculate max
    uq = [np.percentile(np.array(list(flatten(column))), 75)  if i>7 else 0 for i, column in enumerate(zip(*loss.values()))][8:]

    fig, axs = plt.subplots()
    plt.tight_layout()
    axs.plot(list(range(8, num_epochs)),average)
    axs.plot(list(range(8, num_epochs)),lq, 'g--', linewidth=0.5)
    axs.plot(list(range(8, num_epochs)),uq, 'g--', linewidth=0.5)
    axs.fill_between(list(range(8, num_epochs)), lq, uq, alpha=0.2)
    #axs.set_title('Training loss for population over training')
    axs.set_xlabel('Epochs')
    axs.set_ylabel('loss')
    PATH = os.path.join("graphs", experiment)
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    PATH = os.path.join(PATH, "loss_envelope_{}".format(vam))

    try:
        plt.savefig(PATH+".png", bbox_inches="tight")
        print("Plot saved as {}".format(PATH))
    except:
        print("Could not save plot")
    plt.close()
