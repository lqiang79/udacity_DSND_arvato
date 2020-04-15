###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

def detec_attributes_has_many_values(dataframe, valueSizeLimit=30):
    '''
    find attributes/columns in dataframe, they have continuous numerical values. 

    Args:
    dataframe {DataFrame} -- it could be customers and azdias
    valueSizeLimit {int} -- the limitation of value size you want to check, 
        default 10
    Returns:
    {set} -- all attributs, in dataframe has more than 10 values
    '''
    value_oversize_columns = set()
    for columnName, columnData in dataframe.iteritems():
        attr_unique_values = columnData.unique()
        if attr_unique_values.size >= valueSizeLimit:
            value_oversize_columns.add(columnName)
            print(f'{columnName} has value {attr_unique_values}')
    #return value_oversize_columns


def replace_unknown_values(dataframe, attribute_unknown_values):
    '''
    customers and azdias values has default value with meaning "unknown". some of they has two values for unknown. We use only the first one two replace the others
    we also fill the nan values with our default unknown value.
    Args:
        dataframe {DataFrame} -- customers or azdias
        attribute_unknown_values {DataFrame} -- a Attribute and its possible unknown values. it comes from "DIAS Attributes - Values 2017.xlsx"
    Returns:
        None
    '''
    for index, row in attribute_unknown_values.iterrows():
        attribute_name = row['Attribute'].replace('_RZ', '')
        unknown_values = list(map(int, str(row['Value']).split(',')))

        first_unknown_value = unknown_values[0]

        if attribute_name not in dataframe.columns:
            print(
                f'Attribute {attribute_name} can not be found in dataframe columns')
        else:
            dataframe[attribute_name].fillna(first_unknown_value, inplace=True)
            for val in unknown_values[1:]:
                dataframe[attribute_name].replace(
                    val, first_unknown_value, inplace=True)
                print(
                    f'replace unknown value {val} and NaN in {attribute_name} with {first_unknown_value}')


def detec_many_nan_attributes(dataframe, limitation=0.9):
    '''
    find attribute in dataframe with over 90% NaN value.
    '''
    nan_counts = pd.Series(dataframe.isnull().sum())
    line_count = dataframe.shape[0]
    too_many_nan_attributes = []
    for idx, col in nan_counts.items():
        if col/line_count > limitation:
            too_many_nan_attributes.append(idx)
    return too_many_nan_attributes


def drop_column_with_many_nan(dataframe):
    col_num_before = dataframe.shape[1]
    columns_to_drop = detec_many_nan_attributes(dataframe)
    print(detec_many_nan_attributes(dataframe),
          'in dataframe has over 90% NaN will be dropped')
    result_df = dataframe.drop(columns_to_drop, axis=1, errors='ignore')
    print(
        f'we drop the columns, before drop it has {col_num_before} columns, after drop {result_df.shape[1]} columns')
    return result_df

def split_customers_special_columns(customers_df):
    ''' split "CUSTOMER_GROUP", "ONLINE_PURCHASE", "PRODUCT_GROUP" in a special dataframe from customers
    Args:
        customers_df {DataFrame} customers data
    Retturns:
    
        {DataFrame} -- with just 3 special columns from the customers
    '''
    # extract special columns from customers
    customer_special_columns = ['CUSTOMER_GROUP', 'ONLINE_PURCHASE', 'PRODUCT_GROUP']
           
    customer_special_df = customers_df[customer_special_columns]
    print(f'columns {customer_special_columns} will be delete from customers')
    customers_df.drop(columns=customer_special_columns, axis=1, errors='ignore', inplace=True)
 
    return customer_special_df

