import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram,cut_tree
import ipywidgets as widgets
from IPython.display import display, clear_output

pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")
StartDate=20160101
EndDate=20230630

FundDetails=pd.read_csv('GuoYuanFundSector.csv')
# print(FundDetails)
FundDetails = FundDetails[FundDetails['分类3'] == '高仓位全市场']
FundDetails = FundDetails[['S_INFO_WINDCODE', 'F_INFO_FULLNAME']].drop_duplicates()
# FundDetails = FundDetails[(FundDetails['T'] <= EndDate) & (FundDetails['T'] >= StartDate)]
FundDetails=FundDetails.set_index('S_INFO_WINDCODE')


FundNav=pd.read_csv('基金净值20211231.csv')
# FundNav=FundNav[FundNav['PRICE_DATE']>20211231]
# FundNav.to_csv('基金净值20211231.csv')
FundNav = pd.merge(FundNav,FundDetails['F_INFO_FULLNAME'],left_on='F_INFO_WINDCODE',right_index=True,how='left')
FundNav = FundNav.pivot_table(index='PRICE_DATE', columns='F_INFO_FULLNAME', values='F_NAV_ADJUSTED')
ETFMRet=FundNav.pct_change()
ETFMRet=ETFMRet.iloc[2:,:]
ETFMRet=ETFMRet.dropna(axis=1)
ETFMRet=ETFMRet.T
plt.rcParams['font.family'] = 'Arial Unicode MS'
# print(ETFMRet)

distance_matrix = hierarchy.distance.pdist(ETFMRet, metric='euclidean')
linkage_matrix = hierarchy.linkage(distance_matrix, method='average')
dendrogram = hierarchy.dendrogram(linkage_matrix, labels=ETFMRet.index, truncate_mode='none')
clusters = hierarchy.cut_tree(linkage_matrix, height=0.2)
ETFMRet['Cluster_1'] = clusters
ETFMRet_re = ETFMRet.groupby('Cluster_1').apply(lambda x: ', '.join(x.index))
# print(ETFMRet_re)
# ETFMRet_re = ETFMRet.groupby('Cluster_1').agg({'F_INFO_FULLNAME': ', '.join})
ETFMRet_re.to_csv('基金分类结果1.csv')
# plt.show()