# pipeline/concept_graph.py

import re
import networkx as nx
from pyvis.network import Network
from networkx.algorithms.community import greedy_modularity_communities
import random

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

def get_layout_options():
    """
    Return available layout options for the graph.
    """
    return {
        "force": "Force-directed (Default)",
        "hierarchical": "Hierarchical (Tree)",
        "circular": "Circular",
        "spring": "Spring Physics"
    }

def get_physics_config(layout_type="force"):
    """
    Get physics configuration for different layout types.
    """
    if layout_type == "hierarchical":
        return """
        var options = {
          "physics": {
            "hierarchicalRepulsion": {
              "centralGravity": 0.0,
              "springLength": 100,
              "springConstant": 0.01,
              "nodeDistance": 120,
              "damping": 0.09
            },
            "maxVelocity": 50,
            "minVelocity": 0.1,
            "solver": "hierarchicalRepulsion",
            "timestep": 0.35
          }
        }
        """
    elif layout_type == "circular":
        return """
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -1000,
              "centralGravity": 0.01,
              "springLength": 200,
              "springConstant": 0.08,
              "damping": 0.4,
              "avoidOverlap": 0.5
            },
            "maxVelocity": 50,
            "minVelocity": 0.1,
            "solver": "forceAtlas2Based",
            "timestep": 0.35
          }
        }
        """
    elif layout_type == "spring":
        return """
        var options = {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -2000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09,
              "avoidOverlap": 0.1
            },
            "maxVelocity": 50,
            "minVelocity": 0.1,
            "solver": "barnesHut",
            "timestep": 0.35
          }
        }
        """
    else:  # force (default)
        return """
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
        """

def get_relation_color(relation):
    """
    Assign colors to different types of relations.
    """
    relation_lower = relation.lower()
    
    # Define color schemes for different relation types
    if any(word in relation_lower for word in ['contains', 'has', 'includes', 'consists']):
        return "#74c476"  # Green for containment
    elif any(word in relation_lower for word in ['causes', 'leads', 'results', 'creates']):
        return "#fd8d3c"  # Orange for causation
    elif any(word in relation_lower for word in ['transmits', 'sends', 'carries', 'flows']):
        return "#6baed6"  # Blue for transmission
    elif any(word in relation_lower for word in ['controls', 'regulates', 'manages']):
        return "#e377c2"  # Pink for control
    elif any(word in relation_lower for word in ['connects', 'links', 'joins']):
        return "#ff9896"  # Light red for connection
    else:
        return "#c7c7c7"  # Gray for other relations

def visualize_graph(G, out_file="outputs/concept_map.html", layout_type="force", search_term=None):
    """
    Create an interactive HTML visualization of the concept graph.
    Node size = degree centrality, color = community, enhanced tooltips.
    """
    if not G.nodes():
        print("Warning: No nodes in graph to visualize")
        return
    
    # --- Filter graph based on search term ---
    if search_term:
        search_words = search_term.lower().split()
        
        # Find matching nodes
        matching_nodes = set()
        for node in G.nodes():
            node_words = node.lower().split()
            for search_word in search_words:
                if search_word in node_words:
                    matching_nodes.add(node)
                    break
        
        # Include immediate neighbors of matching nodes
        neighbor_nodes = set()
        for node in matching_nodes:
            neighbor_nodes.update(G.predecessors(node))  # Incoming connections
            neighbor_nodes.update(G.successors(node))    # Outgoing connections
        
        # Create filtered graph with only matching nodes and their neighbors
        filtered_nodes = matching_nodes.union(neighbor_nodes)
        G_filtered = G.subgraph(filtered_nodes).copy()
        
        print(f"Filtered graph: showing {len(G_filtered.nodes())} nodes (searched: {len(matching_nodes)}, neighbors: {len(neighbor_nodes)})")
        G = G_filtered
    else:
        matching_nodes = set()
    
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    
    # --- Node size by degree centrality ---
    centrality = nx.degree_centrality(G)
    min_size, max_size = 15, 40
    
    # --- Community detection for coloring ---
    try:
        communities = list(greedy_modularity_communities(G.to_undirected()))
        node_community = {}
        palette = [
            "#6baed6", "#fd8d3c", "#74c476", "#9e9ac8", "#e377c2", "#ff9896", "#c7c7c7", "#bcbd22", "#17becf"
        ]
        random.shuffle(palette)
        for idx, comm in enumerate(communities):
            for node in comm:
                node_community[node] = palette[idx % len(palette)]
    except Exception as e:
        print(f"Community detection failed: {e}")
        node_community = {n: "#6baed6" for n in G.nodes}
    
    # --- Enhanced tooltips with connections ---
    for node in G.nodes:
        size = min_size + (max_size - min_size) * centrality.get(node, 0)
        color = node_community.get(node, "#6baed6")
        degree = G.degree[node]
        
        # Check if node matches search term
        is_highlighted = False
        if search_term:
            search_words = search_term.lower().split()  # Split into individual words
            node_words = node.lower().split()
            
            # Check if any search word exactly matches any word in the node
            for search_word in search_words:
                if search_word in node_words:
                    is_highlighted = True
                    break
            
            if is_highlighted:
                # Make highlighted nodes larger and add border
                size += 10
                border_color = "#ffff00"  # Yellow border for highlighted nodes
            else:
                border_color = "#ffffff"
        else:
            border_color = "#ffffff"
        
        # Get sample connections for tooltip
        neighbors = list(G.neighbors(node))[:3]  # First 3 neighbors
        neighbor_text = ", ".join(neighbors) if neighbors else "No direct connections"
        if len(list(G.neighbors(node))) > 3:
            neighbor_text += f" (+{len(list(G.neighbors(node))) - 3} more)"
        
        # Get community info
        community_idx = None
        for idx, comm in enumerate(communities):
            if node in comm:
                community_idx = idx
                break
        
        community_name = ["Core concepts", "Supporting concepts", "Related concepts", "Secondary concepts", "Additional concepts"][community_idx] if community_idx is not None and community_idx < 5 else f"Cluster {community_idx + 1}" if community_idx is not None else "Unknown"
        
        # Enhanced tooltip
        title = (
            f'<div style="text-align:left;max-width:250px;">'
            f'<b style="color:#fff;font-size:16px;">{node}</b><br>'
            f'<span style="color:#aaa;">Degree: {degree} connections</span><br>'
            f'<span style="color:#aaa;">Community: {community_name}</span><br>'
            f'<span style="color:#aaa;">Connections: {neighbor_text}</span>'
            f'{"<br><span style=\"color:#ffff00;font-weight:bold;\">🔍 SEARCH MATCH</span>" if is_highlighted else ""}'
            f'</div>'
        )
        
        net.add_node(node, label=node, title=title, color=color, size=size, border=border_color)
    
    # --- Edge styling by relation type ---
    edge_counts = {}
    for u, v in G.edges():
        edge_counts[(u, v)] = edge_counts.get((u, v), 0) + 1
    
    for u, v, data in G.edges(data=True):
        label = data.get("label", "")
        width = 2 + edge_counts.get((u, v), 1)  # Thicker if more frequent
        color = get_relation_color(label)
        
        # Highlight edges connected to searched nodes
        edge_highlighted = False
        if search_term:
            search_words = search_term.lower().split()
            u_words = u.lower().split()
            v_words = v.lower().split()
            
            # Check if any search word exactly matches any word in either node
            for search_word in search_words:
                if (search_word in u_words or search_word in v_words):
                    edge_highlighted = True
                    width += 2  # Make highlighted edges thicker
                    break
        
        net.add_edge(u, v, title=label, label=label, arrows='to', width=width, color=color)
    
    # Configure physics based on layout type
    physics_config = get_physics_config(layout_type)
    net.set_options(physics_config)
    
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    net.write_html(out_file, notebook=False)

    # --- Add legend to HTML output ---
    # Read the generated HTML
    with open(out_file, "r", encoding="utf-8") as f:
        html = f.read()

    # Build legend HTML
    legend_items = []
    if 'communities' in locals():
        # Generic cluster names
        cluster_names = ["Core concepts", "Supporting concepts", "Related concepts", "Secondary concepts", "Additional concepts"]
        
        for idx, comm in enumerate(communities):
            color = palette[idx % len(palette)]
            cluster_name = cluster_names[idx] if idx < len(cluster_names) else f"Cluster {idx+1}"
            node_count = len(comm)
            legend_items.append(f'<span style="display:inline-block;width:16px;height:16px;background:{color};border-radius:8px;margin-right:6px;"></span><b>{cluster_name}</b> ({node_count} nodes)')
    
    # Add search filter legend if search is active
    search_legend = ""
    if search_term:
        search_legend = f'''
        <div style="margin-top:10px;padding-top:10px;border-top:1px solid #444;">
          <b>Search Filter:</b><br>
          <span style="color:#ffff00;">■</span> Searched: "{search_term}" matches<br>
          <span style="color:#fff;">■</span> Connected concepts (neighbors)<br>
          <span style="color:#aaa;font-size:12px;">(Showing {len(G.nodes())} of {len(G.nodes())} filtered nodes)</span>
        </div>
        '''
    
    # Add edge color legend
    edge_legend = '''
    <div style="margin-top:10px;padding-top:10px;border-top:1px solid #444;">
      <b>Edge Colors:</b><br>
      <span style="color:#74c476;">■</span> Contains/Has <span style="color:#fd8d3c;">■</span> Causes/Leads <span style="color:#6baed6;">■</span> Transmits/Carries<br>
      <span style="color:#e377c2;">■</span> Controls/Regulates <span style="color:#ff9896;">■</span> Connects/Links <span style="color:#c7c7c7;">■</span> Other
    </div>
    '''
    
    legend_html = f'''
    <div style="position:fixed;top:20px;right:20px;z-index:1000;background:#222;color:#fff;padding:12px 18px;border-radius:8px;max-width:300px;font-size:13px;box-shadow:0 2px 8px #0007;max-height:80vh;overflow-y:auto;">
      <b>Legend</b><br>
      {'<br>'.join(legend_items) if legend_items else '<span style="color:#aaa">(Single cluster)</span>'}<br>
      <span style="font-size:11px;color:#aaa;">Node size = importance (number of connections)<br>
      Edge thickness = frequency of connection</span>
      {search_legend}
      {edge_legend}
    </div>
    '''
    # Insert legend just after <body>
    html = html.replace('<body>', '<body>' + legend_html, 1)

    # Write back the HTML with legend
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Graph visualization saved to: {out_file} (with enhanced features)")

def get_learning_path_mermaid(G):
    """
    Generate a Mermaid flowchart for the learning path (topological sort) of the concept map graph G.
    Escapes node names for Mermaid compatibility.
    """
    import networkx as nx
    import re
    def sanitize(node):
        # Remove newlines, tabs, and escape quotes/backslashes
        node = str(node).replace('"', '\"').replace('\\', '\\')
        node = re.sub(r'[\n\r\t]', ' ', node)
        return node.strip()
    if not G or not G.nodes:
        return "flowchart TD\n  %% No concepts found"
    try:
        order = list(nx.topological_sort(G))
    except Exception:
        order = list(G.nodes)
    edges = list(G.edges())
    if not edges:
        return "flowchart TD\n  " + " --> ".join([f'"{sanitize(n)}"' for n in order])
    mermaid = ["flowchart TD"]
    for u, v in edges:
        mermaid.append(f'  "{sanitize(u)}" --> "{sanitize(v)}"')
    return "\n".join(mermaid)
