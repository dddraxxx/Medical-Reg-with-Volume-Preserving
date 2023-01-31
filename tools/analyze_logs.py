import pandas as pd
import numpy as np
from pathlib import Path as pa

def read_log2pd(log_path):
    l = pa(log_path)
    with open(l, 'r') as f:
        lines = f.readlines()
        keys = lines[0].split()
        data = []
        # read till summary
        for line in lines[1:]:
            if 'summary' in line.lower():
                break
            data.append(line.split())
        m_df = pd.DataFrame(data, columns=keys)
        for k in keys:
            if k not in ['id1', 'id2']:
                m_df[k] = m_df[k].astype(np.float64)
        return m_df

# log_path = ['/home/hynx/regis/recursive-cascaded-networks/evaluations/Jan02_033227_msk-ks1.5-vtn_2_.txt',\
#     '/home/hynx/regis/recursive-cascaded-networks/evaluations/Dec27_133859_normal_2_.txt']
log_path = (# ['/home/hynx/regis/recursive-cascaded-networks/evaluations/Jan08_180111_msk-ks1.5-vtn_2_.txt', 
    ['/home/hynx/regis/recursive-cascaded-networks/evaluations/Jan17_193345_msk-ks0.5-vtn_ep10_2_.txt', \
    '/home/hynx/regis/recursive-cascaded-networks/evaluations/Jan08_180325_normal-vtn_2_.txt'])
lmk_log_path = list(pa('/home/hynx/regis/recursive-cascaded-networks/evaluations/').glob('*_lm10*.txt'))
save_exc_dir = [pa('/home/hynx/regis/recursive-cascaded-networks/eval_excel') / lmk_log_path[i].name.split('_')[-2] for i in range(len(lmk_log_path))]
def save_log2excel(lmk_log_path, save_exc_dir):
    # create dir if not exist
    for sd in save_exc_dir: sd.mkdir(parents=True, exist_ok=True)
    for sd, l in zip(save_exc_dir, lmk_log_path):
        m_df = read_log2pd(l)
        # add a summary line
        m_df.loc[len(m_df)] = m_df.mean()
        # export pd to excel
        save_file = sd / pa(l).name.replace('.txt', '.xlsx')
        # filter columns that are neither nan or 0
        m_df = m_df.loc[:, ((m_df != 0) & (~m_df.isna())).any(axis=0)]
        m_df.to_excel(save_file, index=False, sheet_name=pa(l).stem)
        m_df.to_csv(sd / pa(l).name.replace('.txt', '.csv'), index=False)
        print('save to {}'.format(save_file))
# save_log2excel(lmk_log_path, save_exc_dir)
# exit()
log_path = [pa(l) for l in log_path]
# save all logs in a df
dfs = []
key = 'dice_liver'
key1 = 'sdice_liver'
for l in log_path:
    # read file
    print(l.name)
    m_df = read_log2pd(l)
    # find the id1, id2 with 5 smallest dice_liver
    m_df = m_df.sort_values(by=key)
    print(m_df.head(5))
    dfs.append(m_df)
print('\n')

# merge two df by their index, add suffix
m_df = pd.merge(dfs[0], dfs[1], left_index=True, right_index=True, suffixes=('_msk', '_normal'))
# sort by their difference in key
sorted_idx = m_df.apply(lambda x: x[key+'_msk']-x[key+'_normal'], axis=1).sort_values().index
m_df = m_df.loc[sorted_idx]
# show the difference
m_df['diff_{}'.format(key)] = m_df.apply(lambda x: x[key+'_msk']-x[key+'_normal'], axis=1)
m_df['diff_{}'.format(key1)] = m_df.apply(lambda x: x[key1+'_msk']-x[key1+'_normal'], axis=1)
# show correlation between
print(m_df[['diff_{}'.format(key), 'diff_{}'.format(key1)]].corr())
# culmulate the difference and divide it by the index
m_df['cum_diff_{}'.format(key)] = m_df['diff_{}'.format(key)].cumsum() / range(1, len(m_df)+1)
# select columns
print(m_df[['id1_msk', 'id2_msk', 'o_dice_liver_msk', 'sdice_liver_msk', 'sdice_liver_normal', 'dice_liver_msk', 'dice_liver_normal', 'tl1_ratio_msk', 'tl2_ratio_msk', 'to_ratio_msk', 'liver_ratio_msk', 'tumor_ratio_msk']].head(20))
# sort df by tl2_ratio_msk, and show the id2_msk and tl2_ratio_msk
print(m_df.sort_values(by='tl2_ratio_msk', ascending=False)[['id2_msk', 'tl2_ratio_msk']].drop_duplicates(subset='id2_msk').head(10))
# print(df[['diff_{}'.format(key), 'cum_diff_{}'.format(key)]][:20])

# rm rows with id2 lits_33 or lits_71, in dfs
print([df['dice_liver'].mean() for df in dfs])
rm_items = ['lits_33', 'lits_71']
for df in dfs:
    # for i in rm_items:
    #     df.drop(df[df['id2']==i].index, inplace=True)
    df.drop(df[df['tl2_ratio'].astype(np.float64)>0.1].index, inplace=True)
    # df.drop(df[df['tl1_ratio'].astype(np.float64)>0.1].index, inplace=True)
# check o_dice_liver in df
# m_df.drop(m_df[m_df['tl2_ratio_msk'].astype(np.float64)>0.1].index, inplace=True)
# print(m_df[['id1_msk', 'id2_msk', 'o_dice_liver_msk', 'sdice_liver_msk', 'sdice_liver_normal', 'dice_liver_msk', 'dice_liver_normal', 'tl1_ratio_msk', 'tl2_ratio_msk', 'to_ratio_msk', 'liver_ratio_msk', 'tumor_ratio_msk']].head(20)['o_dice_liver_msk'].astype(np.float64).mean())
# print(m_df['o_dice_liver_msk'].astype(np.float64).mean())

# print m_df sorted by o_dice_liver_msk


# print the mean of dice_liver of dfs
print([df['dice_liver'].mean() for df in dfs])
# print id2 by tl2_ratio in df
for df in dfs:
    print(df.sort_values(by='tl2_ratio', ascending=False)[['id2', 'tl2_ratio']].drop_duplicates(subset='id2').head(10))