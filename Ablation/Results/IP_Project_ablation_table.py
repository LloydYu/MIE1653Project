import pandas as pd
pd.options.display.max_columns = None
import numpy as np
import sys

temp1 = []
for i in ["100", "300", "500"]:
    data_cndp = pd.read_csv(i+r"\one_DCNDP_data_"+i+".csv", header=0, index_col=0)
    temp_cndp = data_cndp
    t = ["", "default"]
    average_solve_agg_CNDP = []
    average_obj_agg_CNDP = []
    for k in [0.05, 0.1, 0.2]:
        average_solve_cndp = temp_cndp[temp_cndp["1"]==k]["3"].mean() * 1000
        t.append(average_solve_cndp)
        average_solve_agg_CNDP.append(average_solve_cndp)
        average_obj_agg_CNDP.append(temp_cndp[temp_cndp["1"]==k]["4"])
    temp1.append(t)

    for j in ["branch", "preprocess", "warmstart", "branch_warmstart", "preprocess_branch", "preprocess_warmstart", "preprocess_branch_warmstart"]:

        data_method = pd.read_csv(i+r"\\\\"+j+"_data_"+i+".csv", header=0, index_col=0)
        temp_method = data_method
        t = [i, j]
        average_solve_agg = []
        average_obj_agg = []
        for k in [0.05, 0.1, 0.2]:

            average_solve_method = temp_method[temp_method["1"]==k]["5"].mean()*1000
            t.append(average_solve_method)
            average_solve_agg.append(average_solve_method)
            average_obj_agg.append(temp_method[temp_method["1"]==k]["6"])

        average_solve_agg_agg = []
        average_obj_agg_agg = []
        for l in range(len(average_solve_agg)):
            average_solve_agg_agg.append((average_solve_agg_CNDP[l] - average_solve_agg[l])/average_solve_agg_CNDP[l])
            average_obj_agg_agg.append(((average_obj_agg_CNDP[l] - average_obj_agg[l])/average_obj_agg_CNDP[l]).mean())

        t.append(np.mean(average_solve_agg_agg))
        t.append(np.mean(average_obj_agg_agg))
        t.append(temp_method["3"].mean()*1000)
        temp1.append(t)

pd.DataFrame(temp1).to_csv(r"ablation_agg.csv")
