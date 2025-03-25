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

def one_DCNDP_branch_warmstart(G, budget, measure_branch, measure_warmstart):
    start = time.time()
    # initial
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.Params.Threads = threads

    # node binary vars
    node_vars = m.addVars(sorted(G.nodes), vtype=GRB.BINARY)

    warm_start = measure_warmstart.sort_values(1, ascending=False)[0].tolist()[:int(budget)]
    for v in warm_start:
        node_vars[v].Start = 1

    for i in node_vars:
        node_vars[i].setAttr("BranchPriority", int(100000*measure_branch[i]))

    X = m.addVars(G.edges, vtype=GRB.BINARY)
    m.addConstr(gp.quicksum(node_vars) == budget)

    # edges
    m.addConstrs(1 - node_vars[u] - node_vars[v] <= X[u, v] for u, v in G.edges)

    m.setObjective(gp.quicksum(X), GRB.MINIMIZE)

    mip_build = time.time() - start

    start = time.time()
    m.optimize()

    return mip_build, time.time() - start, m.ObjVal


def one_DCNDP_preprocess_branch(G, budget, measure_preprocess, measure_branch, alpha):
    start = time.time()
    # initial
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.Params.Threads = threads

    best_nodes = measure_preprocess.sort_values(1, ascending=False)[0].tolist()[:int(budget * (1 + alpha))]
    # node binary vars
    node_vars = m.addVars(sorted(G.nodes), vtype=GRB.BINARY)

    for i in node_vars:
        node_vars[i].setAttr("BranchPriority", int(100000 * measure_branch[i]))

    X = m.addVars(G.edges, vtype=GRB.BINARY)
    m.addConstr(gp.quicksum(node_vars) == budget)

    # edges
    for u, v in G.edges:
        m.addConstr(1 - node_vars[u] - node_vars[v] <= X[u, v])
        if u not in best_nodes:
            m.addConstr(node_vars[u] <= 0)
        if v not in best_nodes:
            m.addConstr(node_vars[v] <= 0)


    m.setObjective(gp.quicksum(X), GRB.MINIMIZE)

    mip_build = time.time() - start

    start = time.time()
    m.optimize()

    return mip_build, time.time() - start, m.ObjVal


def one_DCNDP_preprocess_warmstart(G, budget, measure_preprocess, measure_warmstart, alpha):
    start = time.time()
    # initial
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.Params.Threads = threads

    best_nodes = measure_preprocess.sort_values(1, ascending=False)[0].tolist()[:int(budget * (1 + alpha))]

    # node binary vars
    node_vars = m.addVars(sorted(G.nodes), vtype=GRB.BINARY)

    warm_start = measure_warmstart.sort_values(1, ascending=False)[0].tolist()[:int(budget)]
    for v in warm_start:
        node_vars[v].Start = 1


    X = m.addVars(G.edges, vtype=GRB.BINARY)
    m.addConstr(gp.quicksum(node_vars) == budget)

    # edges
    for u, v in G.edges:
        m.addConstr(1 - node_vars[u] - node_vars[v] <= X[u, v])
        if u not in best_nodes:
            m.addConstr(node_vars[u] <= 0)
        if v not in best_nodes:
            m.addConstr(node_vars[v] <= 0)

    m.setObjective(gp.quicksum(X), GRB.MINIMIZE)

    mip_build = time.time() - start

    start = time.time()
    m.optimize()

    return mip_build, time.time() - start, m.ObjVal

def one_DCNDP_branch_warmstart_preprocess(G, budget, measure_branch, measure_warmstart, measure_preprocess, alpha):
    start = time.time()
    # initial
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.Params.Threads = threads

    best_nodes = measure_preprocess.sort_values(1, ascending=False)[0].tolist()[:int(budget * (1 + alpha))]

    # node binary vars
    node_vars = m.addVars(sorted(G.nodes), vtype=GRB.BINARY)

    warm_start = measure_warmstart.sort_values(1, ascending=False)[0].tolist()[:int(budget)]
    for v in warm_start:
        node_vars[v].Start = 1

    for i in node_vars:
        node_vars[i].setAttr("BranchPriority", int(100000 * measure_branch[i]))

    X = m.addVars(G.edges, vtype=GRB.BINARY)
    m.addConstr(gp.quicksum(node_vars) == budget)

    # edges
    for u, v in G.edges:
        m.addConstr(1 - node_vars[u] - node_vars[v] <= X[u, v])
        if u not in best_nodes:
            m.addConstr(node_vars[u] <= 0)
        if v not in best_nodes:
            m.addConstr(node_vars[v] <= 0)

    m.setObjective(gp.quicksum(X), GRB.MINIMIZE)

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
    branch_warmstart_data = []
    preprocess_branch_data = []
    preprocess_warmstart_data = []
    preprocess_branch_warmstart_data = []

    instance_size = "500"
    path = r"Instances"+"\\\\"+instance_size
    graphs = list(os.listdir(path))
    for l in ["hda", "1dcndp", "branch", "warmstart", "preprocess", "branch_warmstart", "preprocess_branch", "preprocess_warmstart", "preprocess_branch_warmstart"]:
        for i in graphs:
            print(i)
            G = build_graph(path + "\\\\" + i)

            if nx.is_connected(G):
                if l == "branch":
                    start = time.time()
                    eigenvector_centrality = nx.eigenvector_centrality(G)
                    measures = {"eigenvector_centrality": {k: (v - min(eigenvector_centrality.values())) / (max(eigenvector_centrality.values()) - min(eigenvector_centrality.values())) for k, v in eigenvector_centrality.items()}}
                    measure_time = time.time() - start

                elif l == "warmstart":
                    start = time.time()
                    betweenness_centrality = nx.betweenness_centrality(G)
                    measures = {"betweenness_centrality": {k: (v - min(betweenness_centrality.values())) / (max(betweenness_centrality.values()) - min(betweenness_centrality.values())) for k, v in betweenness_centrality.items()}}
                    measure_time = time.time() - start

                elif l == "preprocess":
                    start = time.time()
                    load_centrality = nx.load_centrality(G)
                    measures = {"load_centrality": {k: (v - min(load_centrality.values())) / (max(load_centrality.values()) - min(load_centrality.values())) for k, v in load_centrality.items()}}
                    measure_time = time.time() - start

                elif l == "branch_warmstart":
                    start = time.time()
                    betweenness_centrality = nx.betweenness_centrality(G)
                    measures_warmstart = {"betweenness_centrality": {k: (v - min(betweenness_centrality.values())) / (max(betweenness_centrality.values()) - min(betweenness_centrality.values())) for k, v in betweenness_centrality.items()}}

                    eigenvector_centrality = nx.eigenvector_centrality(G)
                    measures_branch = {"eigenvector_centrality": {k: (v - min(eigenvector_centrality.values())) / (max(eigenvector_centrality.values()) - min(eigenvector_centrality.values())) for k, v in eigenvector_centrality.items()}}
                    measure_time = time.time() - start

                elif l == "preprocess_branch":
                    start = time.time()
                    load_centrality = nx.load_centrality(G)
                    measures_preprocess = {"load_centrality": {k: (v - min(load_centrality.values())) / (max(load_centrality.values()) - min(load_centrality.values())) for k, v in load_centrality.items()}}

                    eigenvector_centrality = nx.eigenvector_centrality(G)
                    measures_branch = {"eigenvector_centrality": {k: (v - min(eigenvector_centrality.values())) / (max(eigenvector_centrality.values()) - min(eigenvector_centrality.values())) for k, v in eigenvector_centrality.items()}}
                    measure_time = time.time() - start

                elif l == "preprocess_warmstart":
                    start = time.time()
                    betweenness_centrality = nx.betweenness_centrality(G)
                    measures_warmstart = {"betweenness_centrality": {k: (v - min(betweenness_centrality.values())) / (max(betweenness_centrality.values()) - min(betweenness_centrality.values())) for k, v in betweenness_centrality.items()}}

                    load_centrality = nx.load_centrality(G)
                    measures_preprocess = {"load_centrality": {k: (v - min(load_centrality.values())) / (max(load_centrality.values()) - min(load_centrality.values())) for k, v in load_centrality.items()}}
                    measure_time = time.time() - start

                elif l == "preprocess_branch_warmstart":
                    start = time.time()
                    betweenness_centrality = nx.betweenness_centrality(G)
                    measures_warmstart = {"betweenness_centrality": {k: (v - min(betweenness_centrality.values())) / (max(betweenness_centrality.values()) - min(betweenness_centrality.values())) for k, v in betweenness_centrality.items()}}

                    load_centrality = nx.load_centrality(G)
                    measures_preprocess = {"load_centrality": {k: (v - min(load_centrality.values())) / (max(load_centrality.values()) - min(load_centrality.values())) for k, v in load_centrality.items()}}

                    eigenvector_centrality = nx.eigenvector_centrality(G)
                    measures_branch = {"eigenvector_centrality": {k: (v - min(eigenvector_centrality.values())) / (max(eigenvector_centrality.values()) - min(eigenvector_centrality.values())) for k, v in eigenvector_centrality.items()}}
                    measure_time = time.time() - start

                for k in [0.05, 0.1, 0.2]:
                    budget = int(round(len(G.nodes) * k))

                    if l == "hda":
                        hda_solve_time, hda_obj, hda_solution = agile_HDA_org(G.copy(), budget)
                        hda_data.append([i, k, hda_solve_time, hda_obj])

                    elif l == "1dcndp":
                        one_DCNDP_build_time, one_DCNDP_solve_time, one_DCNDP_obj = one_DCNDP(G.copy(), budget)
                        one_DCNDP_data.append([i, k, one_DCNDP_build_time, one_DCNDP_solve_time, one_DCNDP_obj])

                    elif l == "branch":
                        branch_build_time, branch_solve_time, branch_obj = one_DCNDP_branch(G.copy(), budget, measures["eigenvector_centrality"])
                        branch_data.append([i, k, "eigenvector_centrality", measure_time, branch_build_time, branch_solve_time, branch_obj])

                    elif l == "warmstart":
                        warmstart_build_time, warmstart_solve_time, warmstart_obj = one_DCNDP_warmstart(G.copy(), budget, pd.DataFrame(measures["betweenness_centrality"].items()))
                        warmstart_data.append([i, k, "betweenness_centrality", measure_time, warmstart_build_time, warmstart_solve_time, warmstart_obj])

                    elif l == "preprocess":
                        alpha = 2
                        preprocess_build_time, preprocess_solve_time, preprocess_obj = one_DCNDP_preprocess(G.copy(), budget, pd.DataFrame(measures["load_centrality"].items()), alpha)
                        preprocess_data.append([i, k, "load_centrality", measure_time, preprocess_build_time, preprocess_solve_time, preprocess_obj])

                    elif l == "branch_warmstart":
                        build_time, solve_time, obj = one_DCNDP_branch_warmstart(G.copy(), budget, measures_branch["eigenvector_centrality"], pd.DataFrame(measures_warmstart["betweenness_centrality"].items()))
                        branch_warmstart_data.append([i, k, "eigenvector_centrality_betweenness_centrality", measure_time, build_time, solve_time, obj])

                    elif l == "preprocess_branch":
                        alpha = 2
                        build_time, solve_time, obj = one_DCNDP_preprocess_branch(G.copy(), budget, pd.DataFrame(measures_preprocess["load_centrality"].items()), measures_branch["eigenvector_centrality"], alpha)
                        preprocess_branch_data.append([i, k, "load_centrality+eigenvector_centrality", measure_time, build_time, solve_time, obj])

                    elif l == "preprocess_warmstart":
                        alpha = 2
                        build_time, solve_time, obj = one_DCNDP_preprocess_warmstart(G.copy(), budget, pd.DataFrame(measures_preprocess["load_centrality"].items()), pd.DataFrame(measures_warmstart["betweenness_centrality"].items()), alpha)
                        preprocess_warmstart_data.append([i, k, "load_centrality_betweenness_centrality", measure_time, build_time, solve_time, obj])

                    elif l == "preprocess_branch_warmstart":
                        alpha = 2
                        build_time, solve_time, obj = one_DCNDP_branch_warmstart_preprocess(G.copy(), budget, measures_branch["eigenvector_centrality"], pd.DataFrame(measures_warmstart["betweenness_centrality"].items()), pd.DataFrame(measures_preprocess["load_centrality"].items()), alpha)
                        preprocess_branch_warmstart_data.append([i, k, "load_centrality_eigenvector_centrality_betweenness_centrality", measure_time, build_time, solve_time, obj])
            else:
                print(i + " NOT CONNECTED")
    pd.DataFrame(one_DCNDP_data).to_csv("Results" +"\\\\" + instance_size+r"\one_DCNDP_data"+"_"+instance_size+ ".csv")
    pd.DataFrame(hda_data).to_csv("Results" +"\\\\" + instance_size+r"\hda_data"+"_"+instance_size+ ".csv")
    pd.DataFrame(branch_data).to_csv("Results" +"\\\\" + instance_size+r"\branch_data"+"_"+instance_size+ ".csv")
    pd.DataFrame(warmstart_data).to_csv("Results" +"\\\\" + instance_size+r"\warmstart_data"+"_"+instance_size+ ".csv")
    pd.DataFrame(preprocess_data).to_csv("Results" +"\\\\" + instance_size+r"\preprocess_data"+"_"+instance_size+ ".csv")
    pd.DataFrame(branch_warmstart_data).to_csv("Results" +"\\\\" + instance_size+r"\branch_warmstart_data"+"_"+instance_size+ ".csv")
    pd.DataFrame(preprocess_branch_data).to_csv("Results" +"\\\\" + instance_size+r"\preprocess_branch_data"+"_"+instance_size+ ".csv")
    pd.DataFrame(preprocess_warmstart_data).to_csv("Results" +"\\\\" + instance_size+r"\preprocess_warmstart_data"+"_"+instance_size+ ".csv")
    pd.DataFrame(preprocess_branch_warmstart_data).to_csv("Results" +"\\\\" + instance_size+r"\preprocess_branch_warmstart_data"+"_"+instance_size+ ".csv")
experiment()