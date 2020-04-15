# 为 Arvato Financial Services 金融服务公司实现一个顾客分类报告

## 前言

作为优达数据科学微学位的毕业项目，我在这里选择了这个项目作为我的毕业报告。

## 数据预处理

### 数据的本信息

### 清除不需要的信息

### 

1. cleanup the differnt column from customers

## extract special columns from customers

customerSpecialColumns = ['CUSTOMER_GROUP', 'ONLINE_PURCHASE', 'PRODUCT_GROUP']

20000 sample:
clusters= 2 silhouette_score= -0.005550678780112048 inertia_= 3864191.8904432566
clusters= 3 silhouette_score= 0.003387561497095185 inertia_= 3768089.3112894897
clusters= 4 silhouette_score= 0.007278255762319557 inertia_= 3727637.6865278143
clusters= 5 silhouette_score= 0.010505625548645873 inertia_= 3693784.491780014
clusters= 6 silhouette_score= 0.012599622141218588 inertia_= 3672914.1697044154
clusters= 7 silhouette_score= 0.014458489443120543 inertia_= 3654783.171797119
clusters= 8 silhouette_score= 0.01640701522118684 inertia_= 3635866.820267003
Wall time: 20min 13s

all Data:
clusters= 2 silhouette_score= 0.9994315204944044 inertia_= 856.98862132494
clusters= 2 silhouette_score= 0.9994315204944044 inertia_= 856.98862132494
clusters= 3 silhouette_score= 0.9990344389749592 inertia_= 829.6255702529807
clusters= 4 silhouette_score= 0.9987878441445374 inertia_= 788.9907577670696

print('== cleanup_cameo_columns in azdias ==')
cleanup_cameo_columns(azdias)

print("== cleanup_cameo_columns in customers ==")
cleanup_cameo_columns(customers)

'ALTER_HH',
'ALTERSKATEGORIE_FEIN',
'D19_KONSUMTYP',
'FINANZ_VORSORGER',
'FINANZTYP',
'SEMIO_PFLICHT',
'SEMIO_SOZ',
'LEBENSPHASE_FEIN',
'PRAEGENDE_JUGENDJAHRE',


特征选择(Feature Selection)方法汇总
https://zhuanlan.zhihu.com/p/74198735

Feature Selection with sklearn and Pandas
https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b


customers_media_fillna_df, 
customers_unknown_filled_df, 
customers_numeric_df, 
customers_no_nan_df
customers_categorigcal_df, 
customers_no_nan_df]