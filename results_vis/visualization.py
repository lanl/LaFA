import argparse
import pickle
import os
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

import file_name_gens
def arg_parse():
    parser = argparse.ArgumentParser(description="Visualization for results of experiments on NMF implicit function.")

    parser.add_argument(
            "--dataset", dest="dataset", help="Name of the dataset"
        )
    parser.add_argument("--norm", dest="norm", type = str, help = "Attack norm: L2 or Linf.")
    
    parser.set_defaults(
        dataset = "Synthetic",
        norm = "L2"
    )
    
    return parser.parse_args()

OUTPUT_PATH = 'result/log'
prog_args = arg_parse()
DATASET = prog_args.dataset
METHOD = 'PGD'
NORM = prog_args.norm
GRADs = ['BackProp', 'Implicit']
log_name_bp = f'{OUTPUT_PATH}/{DATASET}_{METHOD}_{NORM}_{GRADs[0]}.pkl'
log_name_im = f'{OUTPUT_PATH}/{DATASET}_{METHOD}_{NORM}_{GRADs[1]}.pkl'

SAVE_PLOT = f'{OUTPUT_PATH}/{DATASET}_{METHOD}_{NORM}.png'

df_bp = pd.read_pickle(log_name_bp)
mem_bp = df_bp['memory'].mean()/1000000
df_im = pd.read_pickle(log_name_im)
mem_im = df_im['memory'].mean()/1000000

df = pd.concat([df_bp, df_im])
print(df)

df['Epsilon'] = df['in'].astype(float)
df['Feature error'] = df['out fe'].astype(float)

def label_implicit(row):
    if row['Implicit gradient'] == True:
        return "Implicit gradient"
    else:
        return "Backpropagate"
    
df['Method'] = df.apply(label_implicit, axis=1)

memory_sum_text = f'Backpropagate: {mem_bp:.1f}Mbs/Implicit: {mem_im:.1f}Mbs'

fig = plt.figure(figsize = (8,8))
sns.scatterplot(data=df, x="Epsilon", y="Feature error", hue="Method", palette = 'tab10')
plt.grid()
# plt.ylim([0,0.3])
plt.title("Feature Errors of Adversarial Attacks.")
fig.text(.5, .02, memory_sum_text, ha='center')
plt.savefig(SAVE_PLOT)

print("Save to: ", SAVE_PLOT)
    


