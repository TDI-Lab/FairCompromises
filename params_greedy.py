PARAMS = {
    "experiment_name": "nd2",
    "robustness": 2,
    "samples": 1,
    "num_agents":None,

    "experiments":{

        "greedy": { 
            #"part_description":PART_DISC,
            "exp_part":None,
            "model": "BDQ",
            "training_config": {
                "num_iters": 100,
                "val_interval": 5,
                "test_iters": 1,
                "buffer_capacity": 1000,
                "learning_rate": 1e-4,
                "target_net_update_freq": 50,
                "batch_size": 32
                    },
            "env_config": {"num_rounds":1, 
                    "voting_rule":"greedy", 
                    "first_instance":None,
                    "num_agents":None,
                    "voter_preferences": None,
                    "ballot":"approval"
                    }
        },
}
}