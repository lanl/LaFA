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
directory_path = 'pca_results_test'
# directory_path = 'pca_results'
output_file = 'pca_results_test/pca_wtsi_combined_re2.pkl'
# output_file = 'pca_results/pca_wtsi_combined.pkl'
combine_pickle_files(directory_path, output_file)


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

SAVE_PLOT = 'pca_results/pca_wtsi_recloss.png'
# SAVE_PLOT = 'pca_results/pca_wtsi_feloss.png'

# df_bp = pd.read_pickle(log_name_bp)
# mem_bp = df_bp['memory'].mean()/1000000
# df_im = pd.read_pickle(log_name_im)
# mem_im = df_im['memory'].mean()/1000000

# df = pd.concat([df_bp, df_im])
# print(df)

df['Epsilon'] = df['eps'].astype(float)
df['Feature error'] = df['out fe'].astype(float)
df['U error'] = df['out U'].astype(float)
df['V error'] = df['out V'].astype(float)
df['Recon error'] = df['out rec'].astype(float)

# def label_implicit(row):
#     if row['Implicit gradient'] == True:
#         return "Implicit gradient"
#     else:
#         return "Backpropagate"
    
# df['Method'] = df.apply(label_implicit, axis=1)

# memory_sum_text = f'Backpropagate: {mem_bp:.1f}Mbs/Implicit: {mem_im:.1f}Mbs'

fig = plt.figure(figsize = (8,8))
sns.scatterplot(data=df, x="Epsilon", y="Feature error", palette = 'tab10')
sns.scatterplot(data=df, x="Epsilon", y="U error", palette = 'tab10')
sns.scatterplot(data=df, x="Epsilon", y="V error", palette = 'tab10')
# sns.lineplot(data=df, x="Epsilon", y="V error", palette = 'tab10')÷
# sns.lineplot(data=df, x="Epsilon", y="Recon error", palette = 'tab10')
plt.grid()
# plt.ylim([0,0.3])
plt.title("Feature Errors of Adversarial Attacks.")
# fig.text(.5, .02, memory_sum_text, ha='center')
plt.savefig(SAVE_PLOT)

# print("Save to: ", SAVE_PLOT)
    


