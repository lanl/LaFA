import argparse
import pickle
import os
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

import os.path as osp


def combine_pickle_files(directory_path, output_file):
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame to store the merged data

    for file_name in os.listdir(directory_path):
        if file_name.endswith('.pkl'):
            file_path = osp.join(directory_path, file_name)
            with open(file_path, 'rb') as f:
                content = pickle.load(f)
                if isinstance(content, pd.DataFrame):
                    combined_df = pd.concat([combined_df, content], ignore_index=True)
    
    with open(output_file, 'wb') as out:
        pickle.dump(combined_df, out, protocol=pickle.HIGHEST_PROTOCOL)

# Example usage:

directory_path = '/usr/projects/unsupgan/mvu_experiments/NMF_attacks_old/log/'
output_file = directory_path + 'WTSI_PGD_L2_Implicit.pkl'
# output_file = 'pca_results/pca_wtsi_combined.pkl'
# combine_pickle_files(directory_path, output_file)


with open(output_file, 'rb') as f:
    df = pickle.load(f)
# output_file
print(df)


# import file_name_gens
# def arg_parse():
#     parser = argparse.ArgumentParser(description="Visualization for results of experiments on SVD attack.")

#     parser.add_argument(
#             "--dataset", dest="dataset", help="Name of the dataset"
#         )
#     parser.add_argument("--norm", dest="norm", type = str, help = "Attack norm: L2 or Linf.")
    
#     parser.set_defaults(
#         dataset = "Synthetic",
#         norm = "L2"
#     )
    
#     return parser.parse_args()

# OUTPUT_PATH = 'pca_results/'
# prog_args = arg_parse()
# # DATASET = prog_args.dataset
# # METHOD = 'PGD'
# # NORM = prog_args.norm
# # GRADs = ['BackProp', 'Implicit']
# log_name_bp = f'{OUTPUT_PATH}/{DATASET}_{METHOD}_{NORM}_{GRADs[0]}.pkl'
# log_name_im = f'{OUTPUT_PATH}/{DATASET}_{METHOD}_{NORM}_{GRADs[1]}.pkl'

# log_name_fe = f'{OUTPUT_PATH}/{DATASET}_{METHOD}_{NORM}_{GRADs[0]}.pkl'

SAVE_PLOT = 'result/unknown_rank.png'
# SAVE_PLOT = 'pca_results/pca_wtsi_feloss.png'

# df_bp = pd.read_pickle(log_name_bp)
# mem_bp = df_bp['memory'].mean()/1000000
# df_im = pd.read_pickle(log_name_im)
# mem_im = df_im['memory'].mean()/1000000

# df = pd.concat([df_bp, df_im])
# print(df)

df['Epsilon'] = df['eps'].astype(float)
df['Feature error'] = df['out fe'].astype(float)
df['W error'] = df['out W'].astype(float)
df['H error'] = df['out H'].astype(float)
df['Recon error'] = df['out rec'].astype(float)

df = df[df['rank'].isin([2, 4, 6, 9])]

df = df[df['Epsilon'].isin([0.000, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018])]
# df = df[df['norm'].isin(['L2'])]
# df = df[df['Implicit gradient'] == False]
# def label_implicit(row):
#     if row['Implicit gradient'] == True:
#         return "Implicit gradient"
#     else:
#         return "Backpropagate"
    
# df['Method'] = df.apply(label_implicit, axis=1)

# memory_sum_text = f'Backpropagate: {mem_bp:.1f}Mbs/Implicit: {mem_im:.1f}Mbs'

fig = plt.figure(figsize = (14,8))
sns.set_style('darkgrid')
sns.set_context('poster', font_scale = 1.5)
# sns.scatterplot(data=df, x="Epsilon", y="Feature error", palette = 'tab10')
# sns.lineplot(data=df, x="Epsilon", y="Feature error", palette = 'tab10', hue = 'rank', linestyle='--', markers=True, style="rank")
sns.lineplot(data=df, x="Epsilon", y="W error", palette = 'tab10', hue = 'rank', linestyle='--', markers=True, style="rank")
# sns.scatterplot(data=df, x="Epsilon", y="V error", palette = 'tab10')
# sns.lineplot(data=df, x="Epsilon", y="V error", palette = 'tab10')÷
# sns.lineplot(data=df, x="Epsilon", y="Recon error", palette = 'tab10')
plt.grid()
# plt.ylim([0,0.3])
# plt.title("Feature Errors of Adversarial Attacks.")
# fig.text(.5, .02, memory_sum_text, ha='center')
plt.savefig(SAVE_PLOT, dpi=100, bbox_inches = "tight")

# print("Save to: ", SAVE_PLOT)
    


