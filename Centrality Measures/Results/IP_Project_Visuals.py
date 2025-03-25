import pandas as pd
pd.options.display.max_columns = None
import matplotlib.pyplot as plt
import numpy as np

def centrality_measures(method, method_name, budget):
    plt.rc('font', size=40)
    plt.rc('legend', fontsize=30)

    data = method[method["1"] == budget]
    data_100 = data[data["0"].str.contains("Graph_N100")]
    data_300 = data[data["0"].str.contains("Graph_N300")]
    data_500 = data[data["0"].str.contains("Graph_N500")]
    data_100 = data_100[["2", "4"]].sort_values("2")
    data_300 = data_300[["2", "4"]].sort_values("2")
    data_500 = data_500[["2", "4"]].sort_values("2")

    dict_100 = {}
    for i in data_100["2"].unique():
        dict_100[i] = data_100[data_100["2"] == i]["4"].tolist()

    dict_300 = {}
    for i in data_300["2"].unique():
        dict_300[i] = data_300[data_300["2"] == i]["4"].tolist()

    dict_500 = {}
    for i in data_500["2"].unique():
        dict_500[i] = data_500[data_500["2"] == i]["4"].tolist()

    if budget != 0.2:
        combined = {"100": dict_100, "300": dict_300, "500": dict_500}
        positions = [2,5,8]
    else:
        combined = {"100": dict_100, "300": dict_300}
        positions = [2,5]

    sizes = combined.keys()

    fig, ax = plt.subplots()

    offsets = np.linspace(-1.2, 1.2, 13)

    colors = plt.cm.viridis(np.linspace(0, 1, 13))

    boxes = []
    for i, j in enumerate(dict_100.keys()):
        box = ax.boxplot(
            [combined[s][j] for s in sizes],
            positions=[p + offsets[i] for p in positions],
            widths=0.175, patch_artist=True
        )

        for patch in box['boxes']:
            patch.set_facecolor(colors[i])

        boxes.append(box)

    ax.set_xticks(positions)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("# of nodes", fontsize= 40)
    ax.set_ylabel("time (s)", fontsize=40)
    #time by centrality measure and network size -
    ax.set_title("budget: " + str(budget))
    ax.set_yscale('log')
    legend_labels = list(dict_100.keys())
    handles = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(13)]
    ax.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=6, frameon=False)

    fig.set_figwidth(40)
    fig.set_figheight(38)
    plt.yticks(fontsize=40)
    plt.xticks(fontsize=40)
    plt.savefig("centralitymeasures_"+method_name+"_"+str(budget)+".png")


for i in ["100", "300", "500"]:
    if i == "100":
        branch = pd.read_csv(i +r"\branch_data_"+i+".csv")
        preprocess = pd.read_csv(i +r"\preprocess_data_"+i+".csv")
        warmstart = pd.read_csv(i +r"\warmstart_data_"+i+".csv")
    else:
        branch = pd.concat([branch, pd.read_csv(i +r"\branch_data_"+i+".csv")])
        preprocess = pd.concat([preprocess, pd.read_csv(i +r"\preprocess_data_"+i+".csv")])
        warmstart = pd.concat([warmstart, pd.read_csv(i +r"\warmstart_data_"+i+".csv")])

#centrality_measures(branch, "branch", 0.05)
#centrality_measures(branch, "branch", 0.1)
#centrality_measures(branch, "branch", 0.2)
#centrality_measures(preprocess, "preprocess", 0.05)
#centrality_measures(preprocess, "preprocess", 0.1)
#centrality_measures(preprocess, "preprocess", 0.2)
#centrality_measures(warmstart, "warmstart", 0.05)
#centrality_measures(warmstart, "warmstart", 0.1)
#centrality_measures(warmstart, "warmstart", 0.2)