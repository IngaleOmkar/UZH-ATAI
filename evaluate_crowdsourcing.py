from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
import pandas as pd
import json


def get_mode(values):
    occurrences = {}
    for item in values:
        if (isinstance(item, float) and np.isnan(item)):
            continue
        occurrences[item] = occurrences.get(item, 0) + 1
    max_value = max(occurrences.values())
    print(occurrences)
    candidates = [key for key, value in occurrences.items() if value == max_value]
    for candidate in candidates:
        if (candidate.startswith("Q") or candidate.startswith("P")):
            return candidate
        
    return candidates[0]


def evaluate_crowd_data(input_path, output_path, triplet_path):
    data = pd.read_csv(input_path, sep="\t")

    data = data.sort_values(by=['HITTypeId', 'HITId', 'WorkerId'])
    grouped = data.groupby("HITTypeId")

    batches = {}
    majorty_data = {}
    triplets = { "update" : [], "delete" : []}
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
            og_triplet = (task_obj['Entity'], task_obj['Relation'], task_obj['Answer'])
            print(f"task: {task}, maj: {task_obj['Majority Element']}")
            if (task_obj['Majority Element'] != 1.0):
                if (len(task_row['FixPosition'].mode()) > 0 and len(task_row['FixValue'].mode()) > 0):
                    position = task_row['FixPosition'].mode()[0]
                    value = get_mode(task_row['FixValue'])
                    if (position == "Subject"):
                        task_obj['Entity'] = value if value.startswith("wd:") else "wd:" + value
                    elif (position == "Predicate"):
                        task_obj['Relation'] =  value if value.startswith("wdt:") else "wdt:" + value
                    elif (position == "Object"):
                        task_obj['Answer'] = "wd:" + value if value.startswith("Q") else value

                    task_obj['Majority Element'] = 1.0
                    print(og_triplet)
                    new_triplet = (task_obj['Entity'], task_obj['Relation'], task_obj['Answer'])
                    print(new_triplet)
                    triplets["update"].append((og_triplet, new_triplet))
                else:
                    triplets["delete"].append(og_triplet)
            else: 
                triplets["update"].append((og_triplet, og_triplet))

            majorty_data[batch]['tasks'][task] = task_obj

        batch_votes_mat = np.array(batch_votes)
        kappa = fleiss_kappa(batch_votes_mat)
        print(f"Fleiss' Kappa: {kappa}")
        majorty_data[batch]['Kappa'] = kappa


    with open(output_path, 'w') as json_file:
        json.dump(majorty_data, json_file, indent=4)

    with open(triplet_path, 'w') as json_file:
        json.dump(triplets, json_file, indent=4)