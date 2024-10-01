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
        result = self.graph.query(q)

        ans = [str(s) for s, in result]

        
        return ans

    
