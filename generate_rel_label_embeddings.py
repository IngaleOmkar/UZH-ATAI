import json
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import csv
import rdflib

 # define some prefixes
WD = rdflib.Namespace('http://www.wikidata.org/entity/')
WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
DDIS = rdflib.Namespace('http://ddis.ch/atai/')
RDFS = rdflib.namespace.RDFS
SCHEMA = rdflib.Namespace('http://schema.org/')

print("========== Initializing Data Repository ==========")

# Loading graph
print("========== Loading graph ==========")
graph = rdflib.Graph().parse('data/14_graph.nt', format='turtle')

rel2lbl = {rel: str(lbl) for rel, lbl in graph.subject_objects(RDFS.label) if str(rel).startswith(WDT)}
lbl2rel = {lbl: rel for rel, lbl in rel2lbl.items()}

# Initialize the model and tokenizer
model = SentenceTransformer('data/lbl_nlp_embeddings_model')

# Function to get the embedding for a given text
def get_embedding(text):
    return model.encode(text)

# Step 1: Save embeddings
def save_embeddings(dict_keys, filename="data/rel_nlp_embeddings.json"):
    embeddings = {key: get_embedding(key).tolist() for key in dict_keys}
    with open(filename, "w") as f:
        json.dump(embeddings, f)

# Run this only once to save the embeddings
save_embeddings(lbl2rel.keys())