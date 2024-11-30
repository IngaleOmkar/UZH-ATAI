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

def score_malicious(worker, batch, majority_data):
    reputation = int(batch.loc[batch['WorkerId'] == worker, 'LifetimeApprovalRate'].iloc[0].replace('%', ''))
    rtv, normalized_rtv = response_time_variance(worker, batch)
    mrt = mean_response_time(worker, batch)
    disagreement = overall_disagreement(worker, batch, majority_data)

    print(f"-----------\nworker: {worker}\nreputation: {reputation}\nrtv: {rtv}\nmrt: {mrt}\ndisagreement: {disagreement}")

    malicious_score = 0

    if (normalized_rtv < 0.2 or mrt < 10):
        if (reputation < 85):
            malicious_score += 1
        else:
            malicious_score += 0.5

    if (disagreement > 0.3):
        malicious_score += 0.5

    return malicious_score

def clean_malicious(data_path, majority_path, output_path):
    data = pd.read_csv(data_path, sep="\t")

    with open(majority_path, 'r') as json_file:
        majority_data = json.load(json_file)

    batches = data.groupby('HITTypeId')
    for batch_name, batch in batches:
        workers = batch['WorkerId'].unique()
        malicious_scores = {}

        for worker in workers:
            malicious_scores[worker] = score_malicious(worker, batch, majority_data)

    malicious_condition = data['WorkerId'].apply(lambda x: malicious_scores.get(x, 0) > 0.5)
    cleaned_data = data[malicious_condition]
    print(f"original: {len(data)}, cleaned: {len(cleaned_data)}")

    data.to_csv(output_path, sep='\t', index=False)
