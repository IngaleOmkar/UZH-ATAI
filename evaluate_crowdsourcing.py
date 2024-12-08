from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
import pandas as pd
import json
import rdflib

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

def replace_subject(graph, subject, predicate, obj, new_subject):
    query = f"""
    DELETE {{ <{subject}> <{predicate}> <{obj}> }}
    INSERT {{ <{new_subject}> <{predicate}> <{obj}> }}
    WHERE {{ <{subject}> <{predicate}> <{obj}> }}
    """
    graph.update(query)

def replace_predicate(graph, subject, predicate, obj, new_predicate):
    query = f"""
    DELETE {{ <{subject}> <{predicate}> <{obj}> }}
    INSERT {{ <{subject}> <{new_predicate}> <{obj}> }}
    WHERE {{ <{subject}> <{predicate}> <{obj}> }}
    """
    graph.update(query)

def replace_object(graph, subject, predicate, obj, new_object):
    query = f"""
    DELETE {{ <{subject}> <{predicate}> <{obj}> }}
    INSERT {{ <{subject}> <{predicate}> <{new_object}> }}
    WHERE {{ <{subject}> <{predicate}> <{obj}> }}
    """
    graph.update(query)

def delete_triplet(graph, subject, predicate, obj):
    query = f"""
    DELETE WHERE {{ <{subject}> <{predicate}> <{obj}> }}
    """
    graph.update(query)

def add_triplet(graph, subject, predicate, obj):
    query = f"""
    INSERT DATA {{ <{subject}> <{predicate}> <{obj}> }}
    """
    graph.update(query)

class CrowdEvaluation():
    def __init__(self):
        self.graph = rdflib.Graph().parse('data/14_graph.nt', format='turtle')
        print(len(self.graph))

    def evaluate_crowd_data(self, input_path, output_path, triplet_path, save_graph=False):
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
                            ent = value if value.startswith("wd:") else "wd:" + value
                            replace_subject(self.graph, task_obj['Entity'], task_obj['Relation'], task_obj['Answer'], ent)
                            task_obj['Entity'] = ent
                        elif (position == "Predicate"):
                            pred = value if value.startswith("wdt:") else "wdt:" + value
                            replace_predicate(self.graph, task_obj['Entity'], task_obj['Relation'], task_obj['Answer'], pred)
                            task_obj['Relation'] = pred
                        elif (position == "Object"):
                            obj = "wd:" + value if value.startswith("Q") else value
                            replace_object(self.graph, task_obj['Entity'], task_obj['Relation'], task_obj['Answer'], obj)
                            task_obj['Answer'] = obj

                        task_obj['Majority Element'] = 1.0
                        print(og_triplet)
                        new_triplet = (task_obj['Entity'], task_obj['Relation'], task_obj['Answer'])
                        print(new_triplet)
                        triplets["update"].append((og_triplet, new_triplet))
                    else:
                        triplets["delete"].append(og_triplet)
                        delete_triplet(self.graph, task_obj['Entity'], task_obj['Relation'], task_obj['Answer'])
                else: 
                    triplets["update"].append((og_triplet, og_triplet))
                    add_triplet(self.graph, task_obj['Entity'], task_obj['Relation'], task_obj['Answer'])

                majorty_data[batch]['tasks'][task] = task_obj

            batch_votes_mat = np.array(batch_votes)
            kappa = fleiss_kappa(batch_votes_mat)
            print(f"Fleiss' Kappa: {kappa}")
            majorty_data[batch]['Kappa'] = kappa

            if (save_graph):
                self.graph.serialize('data/14_graph_updated.nt', format='turtle')
                print(len(self.graph))


        with open(output_path, 'w') as json_file:
            json.dump(majorty_data, json_file, indent=4)

        with open(triplet_path, 'w') as json_file:
            json.dump(triplets, json_file, indent=4)