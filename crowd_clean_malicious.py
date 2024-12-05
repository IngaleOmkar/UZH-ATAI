from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
import pandas as pd
import json

def response_time_variance(worker, batch):
    worker_data = batch[batch['WorkerId'] == worker]
    min_time = batch['WorkTimeInSeconds'].min()
    max_time = batch['WorkTimeInSeconds'].max()
    rtv = worker_data['WorkTimeInSeconds'].var()
    normalized_rtv = (rtv - min_time) / (max_time - min_time)
    return rtv, normalized_rtv

def mean_response_time(worker, batch):
    worker_data = batch[batch['WorkerId'] == worker]
    rtv = worker_data['WorkTimeInSeconds'].mean()
    return rtv

def overall_disagreement(worker, batch, majority_data):
    tasks = batch.groupby('HITId')
    nr_disagreements = 0
    batch_name = batch['HITTypeId'].iloc[0]
    for task_name, task in tasks:
        majority_elem = majority_data[batch_name]['tasks'][str(task_name)]['Majority Element']
        worker_answer = task.loc[task['WorkerId'] == worker]
        if (worker_answer.iloc[0]['AnswerID'] != majority_elem):
            nr_disagreements += 1

    disagreement_rate = float(nr_disagreements)/float(len(tasks))
    return disagreement_rate

def answer_consistency(worker, batch):
    tasks = batch.groupby('HITId')
    is_consistent = True
    prev_answer = 0.0
    for i, (task_name, task) in enumerate(tasks):
        answer = task.loc[task['WorkerId'] == worker, 'AnswerID'].values[0]
        if (i > 0 and prev_answer != answer):
            is_consistent = False
        prev_answer = answer
    
    return is_consistent

def min_response_time(worker, batch):
    return batch.loc[batch['WorkerId'] == worker, 'WorkTimeInSeconds'].min()

def score_malicious(worker, batch, majority_data):
    reputation = int(batch.loc[batch['WorkerId'] == worker, 'LifetimeApprovalRate'].iloc[0].replace('%', ''))
    rtv, normalized_rtv = response_time_variance(worker, batch)
    mrt = mean_response_time(worker, batch)
    disagreement = overall_disagreement(worker, batch, majority_data)
    consistent = answer_consistency(worker, batch)
    min_time = min_response_time(worker, batch)

    if (reputation < 50 or min_time < 10 or consistent):
        return True
    else: 
        return False

def clean_malicious(data_path, majority_path, output_path):
    data = pd.read_csv(data_path, sep="\t")

    with open(majority_path, 'r') as json_file:
        majority_data = json.load(json_file)

    cleaned_data = pd.DataFrame()

    batches = data.groupby('HITTypeId')
    for batch_name, batch in batches:
        filtered_batch = batch[batch.apply(
            lambda row: not score_malicious(row['WorkerId'], batch, majority_data), axis=1
        )]
    
        cleaned_data = pd.concat([cleaned_data, filtered_batch], ignore_index=True)

    print(f"original: {len(data)}, cleaned: {len(cleaned_data)}")

    cleaned_data.to_csv(output_path, sep='\t', index=False)
