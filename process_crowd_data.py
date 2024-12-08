from evaluate_crowdsourcing import CrowdEvaluation
from crowd_clean_malicious import clean_malicious
import pandas as pd

crowd_eval = CrowdEvaluation()

crowd_eval.evaluate_crowd_data('data/crowd_data_olat_P344FullstopCorrected.tsv', 'data/crowd_majorty_kappa.json', 'data/new_triplets.json')
clean_malicious('data/crowd_data.tsv', 'data/crowd_majorty_kappa.json', 'data/crowd_data_cleaned.tsv')
crowd_eval.evaluate_crowd_data('data/crowd_data_cleaned.tsv', 'data/crowd_majorty_kappa_cleaned.json', 'data/new_triplets.json', save_graph=True)

df = pd.read_csv('data/crowd_data_cleaned.tsv', sep="\t")

relations = df["Input2ID"].unique()
objs = pd.concat([df["Input1ID"], df["Input3ID"]]).unique()

entities = list(filter(lambda x: x.startswith("wd:"), objs))

pd.DataFrame(entities, columns=["Entities"]).to_csv("data/crowdsourcing_entities.csv", index=False)
pd.DataFrame(relations, columns=["Relations"]).to_csv("data/crowdsourcing_relations.csv", index=False)
