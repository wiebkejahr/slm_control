'''
Project: deep-sted-autoalign
Created on: Thursday, 31st July 2020 10:47:12 am
--------
@author: hmcgovern
'''

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error



def read_data(paths=None):
    # print('made it here')
    # frames = []
    # for path in paths:
    #     frames.append(pd.read_json('../data_collection/' + path))
    
    # total = pd.concat(frames, ignore_index=True)
    # print(total.head(10))
    # exit()

    # result = total.to_json(orient="records", path_or_buf='../data_collection/200803_1D_centered_18k_norm_dist_eps_15_lr_0.001_bs_64_110_total.txt')
    # parsed = json.loads(result)
    # json.dumps(parsed, indent=4)
    # exit()

    for path in paths:
        print(path)
        df = pd.read_json('../data_collection/'+path)
        # print(df.head())



        df_preds = pd.DataFrame(np.abs(np.subtract(df["preds"].to_list(), df["gt"].to_list())), \
            columns = ["45 Astig", "90 Astig", "90 Trefoil", "90 Coma", "0 Coma", "45 Trefoil", "45 Quad", "45 Astig 2", "Spherical", "90 Astig 2", "90 Quad"])
        boxplot = df_preds.boxplot()
        plt.ylabel("Difference btw predicted and gt")
        plt.xlabel("Zernike Mode")
        plt.title(path)

        df2 = pd.DataFrame(df, columns=['init_corr', 'corr'])
        df2 = df2.rename(columns={"init_corr": "initial", "corr": "final"})
        df2.plot.bar()
        plt.title(path)
        plt.xlabel('Trial')
        plt.ylabel('Correlation Coefficient')
        


        df3 = pd.Series(df["corr"])
        df3.reset_index().plot.scatter(x = 'index', y = 'corr')
        plt.ylim(0.7,1)
        # df3.plot.scatter(x=df3.index(), y='corr')
#         # # plt.show()
#         # print(df.loc[:, 'gt'].head().to_numpy())
#         # mean_squared_error(df['gt'].to_numpy(), df['preds'].to_numpy())
#         # print(df.head())
#         # plt.figure(2)
#         # plt.plot(df['gt'][0])
#         # plt.plot(df['preds'][0])
#         # plt.ylim(-1,1)
#         # plt.show()

#         # plt.figure(3)
#         # plt.plot(df['gt'][9])
#         # plt.plot(df['preds'][9])
#         # plt.ylim(-1,1)
#         # plt.show()

        plt.show()
        



if __name__=="__main__":
    # read_data()
#     # read_data(['from0.txt', 'from5.txt', 'from11.txt', 'from19.txt',\
#     #      'from25.txt', 'from30.txt', 'from36.txt', 'from42.txt'])
#     # read_data(['from42.txt'])
#     # read_data([''])

#     # read_data(['20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64_noise_bg2poiss500_0.txt',
#     # '20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64_noise_bg2poiss500_128.txt',
#     # '20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64_noise_bg2poiss500_54.txt',
#     # '20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64_noise_bg2poiss500_30.txt'])
#     # read_data(['200803_1D_centered_18k_norm_dist_eps_15_lr_0.001_bs_64_0.txt',
#     # '200803_1D_centered_18k_norm_dist_eps_15_lr_0.001_bs_64_110.txt'])

    # read_data(['200803_1D_centered_18k_norm_dist_eps_15_lr_0.001_bs_64_0.txt'])
    # read_data(['20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64_noise_bg2poiss500_54 (1).txt'])
    read_data(['20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64_0.txt'])
    # read_data(['20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64_0.txt',\
    #     '20.07.22_multi_centered_11_dim_18k_eps_15_lr_0.001_bs_64_noise_bg2poiss500_total.txt',\
    #         '200712_no_defocus_1D_centered_20k_eps_15_lr_0.001_bs_64_noise_bg2poiss350.txt',
    #         '200726_1D_centered_offset_18k_eps_15_lr_0.001_bs_64_noise_bg2poiss350.txt',\
    #             '200803_1D_centered_18k_norm_dist_eps_15_lr_0.001_bs_64_total.txt'])