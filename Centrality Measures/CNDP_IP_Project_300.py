from gurobipy import GRB
import gurobipy as gp
import networkx as nx
import pandas as pd
from collections import defaultdict
import warnings
import time
import os
warnings.filterwarnings("ignore", category=DeprecationWarning)

threads = 8
def build_graph(filename):
    df = pd.read_csv(filename)
    G = nx.Graph()
    df = df.values.tolist()
    G.add_edges_from(df)
    return G

def agile_HDA_org(G, B):
    start_graph = G.copy()
    start = time.time()
    degrees = dict(G.degree())
    degree_buckets = defaultdict(set)
    max_degree = 0

    for vertex, degree in degrees.items():
        degree_buckets[degree].add(vertex)
        max_degree = max(max_degree, degree)

    # Initialize the ordering list
    ordering = []
    for q in range(B):
        for degree in range(max_degree, -1, -1):
            if degree_buckets[degree]:
                break
            else:
                degree_buckets[degree] = None
        if (degree == 0):
            break

        max_degree_vertex = degree_buckets[degree].pop()

        ordering.append(max_degree_vertex)

        degrees[max_degree_vertex] = -1
        # Update the degree of neighboring vertices
        for neighbor in list(G.neighbors(max_degree_vertex)):
            # if neighbour still exists in the network
            if degrees[neighbor] > 0:
                old_degree = degrees[neighbor]
                new_degree = old_degree - 1
                degrees[neighbor] = new_degree

                degree_buckets[old_degree].remove(neighbor)
                degree_buckets[new_degree].add(neighbor)

        # Remove the vertex from the graph
        G.remove_node(max_degree_vertex)
    if (len(ordering) < B):
        ordering.extend(list(G.nodes())[:B - len(ordering)])

    return time.time() - start, len(start_graph.edges()) - len(G.edges()), ordering

def one_DCNDP(G, budget):
    start = time.time()
    # initial
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.Params.Threads = threads

    # node binary vars
    node_vars = m.addVars(sorted(G.nodes), vtype=GRB.BINARY)

    X = m.addVars(G.edges, vtype=GRB.BINARY)
    m.addConstr(gp.quicksum(node_vars) == budget)

    # edges
    m.addConstrs(1 - node_vars[u] - node_vars[v] <= X[u, v] for u, v in G.edges)

    m.setObjective(gp.quicksum(X), GRB.MINIMIZE)

    mip_build = time.time() - start

    start = time.time()
    m.optimize()

    return mip_build, time.time() - start, m.ObjVal

def one_DCNDP_branch(G, budget, measure):
    start = time.time()
    # initial
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.Params.Threads = threads

    # node binary vars
    node_vars = m.addVars(sorted(G.nodes), vtype=GRB.BINARY)

    for i in node_vars:
        node_vars[i].setAttr("BranchPriority", int(100000*measure[i]))

    X = m.addVars(G.edges, vtype=GRB.BINARY)
    m.addConstr(gp.quicksum(node_vars) == budget)

    # edges
    m.addConstrs(1 - node_vars[u] - node_vars[v] <= X[u, v] for u, v in G.edges)

    m.setObjective(gp.quicksum(X), GRB.MINIMIZE)

    mip_build = time.time() - start

    start = time.time()
    m.optimize()

    return mip_build, time.time() - start, m.ObjVal

def one_DCNDP_warmstart(G, budget, measure):
    start_1 = time.time()
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.Params.Threads = threads

    node_vars = m.addVars(sorted(G.nodes), vtype=GRB.BINARY)
    edge_vars = m.addVars(G.edges, vtype=GRB.BINARY)

    warm_start = measure.sort_values(1, ascending=False)[0].tolist()[:int(budget)]
    for v in warm_start:
        node_vars[v].Start = 1

    m.addConstr(gp.quicksum(node_vars) == budget)
    for u, v in G.edges:
        m.addConstr(1 - node_vars[u] - node_vars[v] <= edge_vars[u, v])

    m.setObjective(gp.quicksum(edge_vars), GRB.MINIMIZE)
    mip_build = time.time() - start_1

    start = time.time()
    m.optimize()

    return mip_build, time.time() - start, m.ObjVal

def one_DCNDP_preprocess(G, budget, measure, alpha):
    start = time.time()
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.Params.Threads = threads

    node_vars = m.addVars(sorted(G.nodes), vtype=GRB.BINARY)
    edge_vars = m.addVars(G.edges, vtype=GRB.BINARY)

    m.addConstr(gp.quicksum(node_vars) == budget)

    best_nodes = measure.sort_values(1, ascending=False)[0].tolist()[:int(budget * (1 + alpha))]
    for u, v in G.edges:
        m.addConstr(1 - node_vars[u] - node_vars[v] <= edge_vars[u, v])
        if u not in best_nodes:
            m.addConstr(node_vars[u] <= 0)
        if v not in best_nodes:
            m.addConstr(node_vars[v] <= 0)

    m.setObjective(gp.quicksum(edge_vars), GRB.MINIMIZE)
    mip_build = time.time() - start

    start = time.time()
    m.optimize()

    return mip_build, time.time() - start, m.ObjVal

def experiment():
    one_DCNDP_data = []
    hda_data = []
    branch_data = []
    warmstart_data = []
    preprocess_data = []

    instance_size = "300"
    path = r"Instances"+"\\\\"+instance_size
    graphs = list(os.listdir(path))
    for i in graphs:
        print(i)
        G = build_graph(path + "\\\\" + i)

        if nx.is_connected(G):
            degree_centrality = nx.degree_centrality(G)
            eigenvector_centrality = nx.eigenvector_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)
            current_flow_closeness_centrality = nx.current_flow_closeness_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            current_flow_betweenness_centrality = nx.current_flow_betweenness_centrality(G)
            communicability_betweenness_centrality = nx.communicability_betweenness_centrality(G)
            load_centrality = nx.load_centrality(G)
            subgraph_centrality = nx.subgraph_centrality(G)
            harmonic_centrality = nx.harmonic_centrality(G)
            second_order_centrality = nx.second_order_centrality(G)
            laplacian_centrality = nx.laplacian_centrality(G)

            measures = {"degree_centrality" : {k: (v - min(degree_centrality.values())) / (max(degree_centrality.values()) - min(degree_centrality.values())) for k, v in degree_centrality.items()},
                    "eigenvector_centrality": {k: (v - min(eigenvector_centrality.values())) / (max(eigenvector_centrality.values()) - min(eigenvector_centrality.values())) for k, v in eigenvector_centrality.items()},
                    "closeness_centrality": {k: (v - min(closeness_centrality.values())) / (max(closeness_centrality.values()) - min(closeness_centrality.values())) for k, v in closeness_centrality.items()},
                    "current_flow_closeness_centrality": {k: (v - min(current_flow_closeness_centrality.values())) / (max(current_flow_closeness_centrality.values()) - min(current_flow_closeness_centrality.values())) for k, v in current_flow_closeness_centrality.items()},
                    "betweenness_centrality": {k: (v - min(betweenness_centrality.values())) / (max(betweenness_centrality.values()) - min(betweenness_centrality.values())) for k, v in betweenness_centrality.items()},
                    "current_flow_betweenness_centrality": {k: (v - min(current_flow_betweenness_centrality.values())) / (max(current_flow_betweenness_centrality.values()) - min(current_flow_betweenness_centrality.values())) for k, v in current_flow_betweenness_centrality.items()},
                    "communicability_betweenness_centrality": {k: (v - min(communicability_betweenness_centrality.values())) / (max(communicability_betweenness_centrality.values()) - min(communicability_betweenness_centrality.values())) for k, v in communicability_betweenness_centrality.items()},
                    "load_centrality": {k: (v - min(load_centrality.values())) / (max(load_centrality.values()) - min(load_centrality.values())) for k, v in load_centrality.items()},
                    "subgraph_centrality": {k: (v - min(subgraph_centrality.values())) / (max(subgraph_centrality.values()) - min(subgraph_centrality.values())) for k, v in subgraph_centrality.items()},
                    "harmonic_centrality": {k: (v - min(harmonic_centrality.values())) / (max(harmonic_centrality.values()) - min(harmonic_centrality.values())) for k, v in harmonic_centrality.items()},
                    "second_order_centrality": {k: (1 - (v - min(second_order_centrality.values())) / (max(second_order_centrality.values()) - min(second_order_centrality.values()))) for k, v in second_order_centrality.items()},
                    "laplacian_centrality": {k: (v - min(laplacian_centrality.values())) / (max(laplacian_centrality.values()) - min(laplacian_centrality.values())) for k, v in laplacian_centrality.items()},
                    }

            for k in [0.05, 0.1, 0.2]:
                print(k)
                budget = int(round(len(G.nodes) * k))

                # HDA
                hda_solve_time, hda_obj, hda_solution = agile_HDA_org(G.copy(), budget)
                hda_data.append([i, k, hda_solve_time, hda_obj])

                # 1-DCNDP
                one_DCNDP_build_time, one_DCNDP_solve_time, one_DCNDP_obj = one_DCNDP(G.copy(), budget)
                one_DCNDP_data.append([i, k, one_DCNDP_build_time, one_DCNDP_solve_time, one_DCNDP_obj])

                # Speedups

                measure_names = list(measures.keys())
                for j in measure_names:
                    # Branch
                    branch_build_time, branch_solve_time, branch_obj = one_DCNDP_branch(G.copy(), budget, measures[j])
                    branch_data.append([i, k, j, branch_build_time, branch_solve_time, branch_obj])

                    # Warm start
                    warmstart_build_time, warmstart_solve_time, warmstart_obj = one_DCNDP_warmstart(G.copy(), budget, pd.DataFrame(measures[j].items()))
                    warmstart_data.append([i, k, j, warmstart_build_time, warmstart_solve_time, warmstart_obj])

                    # Preprocess
                    alpha = 2
                    preprocess_build_time, preprocess_solve_time, preprocess_obj = one_DCNDP_preprocess(G.copy(), budget, pd.DataFrame(measures[j].items()), alpha)
                    preprocess_data.append([i, k, j, preprocess_build_time, preprocess_solve_time, preprocess_obj])
        else:
            print(i + " NOT CONNECTED")
    pd.DataFrame(one_DCNDP_data).to_csv("Results" +"\\\\" + instance_size+r"\one_DCNDP_data"+"_"+instance_size+ ".csv")
    pd.DataFrame(hda_data).to_csv("Results" +"\\\\" + instance_size+r"\hda_data"+"_"+instance_size+ ".csv")
    pd.DataFrame(branch_data).to_csv("Results" +"\\\\" + instance_size+r"\branch_data"+"_"+instance_size+ ".csv")
    pd.DataFrame(warmstart_data).to_csv("Results" +"\\\\" + instance_size+r"\warmstart_data"+"_"+instance_size+ ".csv")
    pd.DataFrame(preprocess_data).to_csv("Results" +"\\\\" + instance_size+r"\preprocess_data"+"_"+instance_size+ ".csv")
experiment()