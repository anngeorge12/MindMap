# pipeline/concept_graph.py

import re
import networkx as nx
from pyvis.network import Network

def parse_triplets(rel_text):
    tokens = rel_text.split()
    triplets = []
    for i in range(0, len(tokens) - 2, 3):
        triplets.append((tokens[i], tokens[i+1], tokens[i+2]))
    return triplets

def build_graph(triplets):
    G = nx.DiGraph()
    for subj, rel, obj in triplets:
        G.add_node(subj)
        G.add_node(obj)
        G.add_edge(subj, obj, label=rel)
    return G

def visualize_graph(G, out_file="outputs/concept_map.html"):
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")

    for node in G.nodes:
        net.add_node(node, label=node, title=node)
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, title=data.get("label", ""), label=data.get("label", ""))

    net.write_html(out_file, notebook=False)
