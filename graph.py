from rdflib.namespace import Namespace, RDF, RDFS, XSD
from rdflib.term import URIRef, Literal
import csv
import json
import networkx as nx
import pandas as pd
import rdflib
from collections import defaultdict, Counter
import locale
_ = locale.setlocale(locale.LC_ALL, '')
from _plotly_future_ import v4_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = 'jupyterlab+svg'
from pandas import DataFrame


class Graph:

    def __init__(self, file_name):
        self.graph = rdflib.Graph()
        self.graph.parse(file_name, format='turtle')
    
    def query(self, q):
        q_result = self.graph.query(q)
        q_result = DataFrame(q_result,columns=q_result.vars)
        print(f"q_result: {q_result}")
        lens = q_result.shape[0]
        items = q_result.iloc[:,0].tolist()
        if lens > 1:
            last_item = items.pop()
            result = ', '.join(items) + ' and ' + last_item
        elif items:
                result = items[0]
        else:
            result = None

        return str(result.encode('utf-8'))

    
