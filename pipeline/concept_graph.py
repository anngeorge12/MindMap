# pipeline/concept_graph.py

import re
import networkx as nx
from pyvis.network import Network

def parse_triplets(rel_text):
    """
    Parse triplets from relation extraction output.
    Handles both Groq format and REBEL format.
    """
    if not rel_text or "error" in rel_text.lower():
        return []
    
    triplets = []
    lines = rel_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Handle Groq format: (subject, relation, object)
        if line.startswith('(') and line.endswith(')'):
            try:
                content = line[1:-1]  # Remove parentheses
                parts = [part.strip() for part in content.split(',')]
                if len(parts) >= 3:
                    subject = parts[0]
                    relation = parts[1]
                    object_part = ','.join(parts[2:])  # Handle objects with commas
                    triplets.append((subject, relation, object_part))
            except Exception as e:
                print(f"Error parsing triplet line '{line}': {e}")
                continue
        
        # Handle REBEL format: subject relation object
        else:
            tokens = line.split()
            if len(tokens) >= 3:
                # Try to find the relation (usually the middle token)
                for i in range(1, len(tokens) - 1):
                    subject = ' '.join(tokens[:i])
                    relation = tokens[i]
                    obj = ' '.join(tokens[i+1:])
                    if len(subject) > 0 and len(obj) > 0:
                        triplets.append((subject, relation, obj))
                        break
    
    return triplets

def build_graph(triplets):
    """
    Build a directed graph from triplets.
    """
    G = nx.DiGraph()
    
    if not triplets:
        print("Warning: No triplets provided to build graph")
        return G
    
    for subj, rel, obj in triplets:
        # Clean up node names
        subj = subj.strip()
        obj = obj.strip()
        rel = rel.strip()
        
        if subj and obj and rel:
            G.add_node(subj)
            G.add_node(obj)
            G.add_edge(subj, obj, label=rel)
    
    return G

def visualize_graph(G, out_file="outputs/concept_map.html"):
    """
    Create an interactive HTML visualization of the concept graph.
    """
    if not G.nodes():
        print("Warning: No nodes in graph to visualize")
        return
    
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    
    # Add nodes
    for node in G.nodes:
        net.add_node(node, label=node, title=node, size=20)
    
    # Add edges
    for u, v, data in G.edges(data=True):
        label = data.get("label", "")
        net.add_edge(u, v, title=label, label=label, arrows='to')
    
    # Configure physics for better layout
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 200,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "solver": "forceAtlas2Based",
        "timestep": 0.35
      }
    }
    """)
    
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    net.write_html(out_file, notebook=False)
    print(f"Graph visualization saved to: {out_file}")
