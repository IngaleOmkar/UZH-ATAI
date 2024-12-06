from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
import pandas as pd
import json

def evaluate_crowd_data(input_path, output_path):
    data = pd.read_csv(input_path, sep="\t")

    data = data.sort_values(by=['HITTypeId', 'HITId', 'WorkerId'])
    grouped = data.groupby("HITTypeId")

    batches = {}
    majorty_data = {}
    for batch, group in grouped:
        batches[batch] = group.reset_index(drop=True)
        majorty_data[batch] = {}
        task_groups = batches[batch].groupby('HITId')
        batch_votes = []
        majorty_data[batch]['tasks'] = {} 
        for task, task_row in task_groups:
            answer_list = list(task_row['AnswerLabel'])
            vote_count = [0, 0]

            for entry in answer_list:
                if (entry == "INCORRECT"):
                    vote_count[1] += 1
                else:
                    vote_count[0] += 1
                
            batch_votes.append(vote_count)
            task_obj = {}
            task_obj['Majority Element'] = float(vote_count.index(max(vote_count))) + 1.0
            task_obj['Votes'] = vote_count
            task_obj['Entity'] = task_row['Input1ID'].iloc[0]
            task_obj['Relation'] = task_row['Input2ID'].iloc[0]
            task_obj['Answer'] = task_row['Input3ID'].iloc[0]
            majorty_data[batch]['tasks'][task] = task_obj

        batch_votes_mat = np.array(batch_votes)
        kappa = fleiss_kappa(batch_votes_mat)
        print(f"Fleiss' Kappa: {kappa}")
        majorty_data[batch]['Kappa'] = kappa


    with open(output_path, 'w') as json_file:
        json.dump(majorty_data, json_file, indent=4)