from evaluate_crowdsourcing import evaluate_crowd_data
from crowd_clean_malicious import clean_malicious
import rdflib
from data_repository import DataRepository
import pandas as pd
import json

evaluate_crowd_data('data/crowd_data_olat_P344FullstopCorrected.tsv', 'data/crowd_majorty_kappa.json')
clean_malicious('data/crowd_data.tsv', 'data/crowd_majorty_kappa.json', 'data/crowd_data_cleaned.tsv')
evaluate_crowd_data('data/crowd_data_cleaned.tsv', 'data/crowd_majorty_kappa_cleaned.json')

# UPDATE GRAPH

data_repository = DataRepository()
g = data_repository.graph

with open('data/crowd_majorty_kappa.json', 'r') as file:
    answers = json.load(file)

all_tasks = {}

def insert(g, ent, rel, obj):
    if isinstance(obj, rdflib.Literal):
        obj_str = f'"{obj}"'
    else:
        obj_str = f'<{obj}>'

    query = f"""
        INSERT DATA {{
            <{ent}> <{rel}> {obj_str} .
        }}
    """
    print(query)
    g.update(query)

def delete(g, ent, rel, obj):
    if isinstance(obj, rdflib.Literal):
        obj_str = f'"{obj}"'
    else:
        obj_str = f'<{obj}>'


    query = f"""
        DELETE WHERE {{
            <{ent}> <{rel}> {obj_str} .
        }}
    """
    print(query)
    g.update(query)

def entity_to_uri(x):
    x = x.split(":")[1]
    if (x in data_repository.WD):
        return data_repository.WD[x]
    else:
        raise KeyError(f"entity {x} does not exist")
    
def relation_to_uri(x):
    x = x.split(":")[1]
    if (x in data_repository.WD):
        return data_repository.WD[x]
    elif (x in data_repository.DDIS):
        return data_repository.DDIS[x]
    else:
        raise KeyError(f"relation {x} does not exist")

def object_to_uri_or_literal(x):
    y = x.split(":")[1]
    if (y in data_repository.WD):
        return data_repository.WD[x]
    else:
        return x
    
def create_entity_uri(x):
    x = x.split(":")[1]
    return f"http://www.wikidata.org/entity/{x}"

def create_relation_uri(x):
    parts = x.split(":")
    rel = parts[1]
    type = parts[0]
    if (type == "wdt"):
        return f"http://www.wikidata.org/prop/direct/{rel}"
    else:
        return f"http://ddis.ch/atai/{rel}"
    
def create_object_uri(x):
    parts = x.split(":")
    if (parts[0] == "wd" and len(parts) == 2):
        return f"http://www.wikidata.org/entity/{parts[1]}"
    else:
        return x

for batch in answers:
    for task, val in answers[batch]["tasks"].items():
        triple = (val["Entity"], val["Relation"], val["Answer"])
        all_tasks[triple] = val["Majority Element"]

initial_len = len(g)
for task, result in all_tasks.items():
    if (result == 1.0):
        ent = create_entity_uri(task[0])
        rel = create_relation_uri(task[1])
        obj = create_object_uri(task[2])
    
        insert(g, ent, rel, obj)
    else:
        try:
            ent = entity_to_uri(task[0])
            rel = relation_to_uri(task[1])
            obj = object_to_uri_or_literal(task[2])
        except Exception as e:
            print(e)
            print("skipping")
            continue
        
        delete(g, ent, rel, obj)

print(f"initial length: {initial_len}, final length: {len(g)}")
g.serialize(destination="data/14_graph_updated.nt", format="turtle")


df = pd.read_csv('data/crowd_data_cleaned.tsv', sep="\t")

relations = df["Input2ID"].unique()
objs = pd.concat([df["Input1ID"], df["Input3ID"]]).unique()

entities = list(filter(lambda x: x.startswith("wd:"), objs))

pd.DataFrame(entities, columns=["Entities"]).to_csv("data/crowdsourcing_entities.csv", index=False)
pd.DataFrame(relations, columns=["Relations"]).to_csv("data/crowdsourcing_relations.csv", index=False)
