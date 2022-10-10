
import os
import pandas as pd

print(os.getcwd())
import sys
sys.path.append('/home/t-kenzaamara/molucn/')
from molucn.utils.utils import read_list_targets
par_dir = '/home/t-kenzaamara/molucn/'

def read_list_targets(n_targets):
    list_targets = []
    list_file = '/home/t-kenzaamara/molucn/list_targets_{}.txt'.format(n_targets)
    if os.path.exists(list_file):
        with open(list_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                list_targets.append(line.strip())
    return list_targets

def get_stats(n_targets):
    list = read_list_targets(n_targets)
    df_stats = pd.DataFrame(columns=['n_compounds', 'n_pairs_init', 'n_pairs', 'n_pairs_train', 'n_pairs_test'])
    targets = []
    for folder in os.listdir('/home/t-kenzaamara/molucn/data'):
        print(folder in list)
        if len(folder)==8 and (folder in list):
            folder_path = os.path.join('/home/t-kenzaamara/molucn/data', folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('stats.csv'):
                        info_path = os.path.join(folder_path, file)
                        row = pd.read_csv(info_path)
                        df_stats = pd.concat([df_stats, row])
                        targets.append(folder)
    assert len(targets)==len(df_stats['n_pairs'])
    df_stats['target'] = targets
    df_stats.to_csv(os.path.join(par_dir,f'data/target_stats_{len(targets)}.csv'), index = False)
    return df_stats

