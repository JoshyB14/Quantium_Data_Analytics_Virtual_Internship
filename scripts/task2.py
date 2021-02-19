# %%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Pandas settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# seaborn settings
sns.set_style("darkgrid")

# %%
# Read in dataframe
chips_data = pd.read_csv('../data/chips_data.csv')
chips_data.head()

# %%
# Checking that all our data in encoded correctly
chips_data.dtypes

# %%
# Date is encoded as object - change to datetime
chips_data['DATE'] = pd.to_datetime(chips_data['DATE'])

# %%
 # Create column YEAR_MONTH to store year and month from the date
chips_data['YEAR_MONTH'] = chips_data['DATE'].dt.strftime('%Y%m').astype('int')
chips_data.head()

# %%
# Task asks us to select control stores
# Trial stores are store numbers: 77, 86 and 88
# Hence we need stores that match these trial stores in terms of:

#           - Same sales
#           - Same num of customers
#           - Same transaction numbers

# Group stores by STORE_NBR and YEAR_MONTH - showing:
#           - sum of sales - $
#           - unique loyal card numbers - num of customers
#           - unique transaction id - num of transactions
#           - sum of product qty - product sold

store_selection = chips_data.groupby(['STORE_NBR','YEAR_MONTH']).agg(
                {'TOT_SALES': 'sum', 'LYLTY_CARD_NBR': 'nunique',
                 'TXN_ID': 'nunique', 'PROD_QTY': 'sum'})

# Create UNIT_PRICE column for unit price of chips
store_selection['AVG_UNIT_PRICE'] = (store_selection['TOT_SALES']/
                                store_selection['PROD_QTY'])

# Transactions per customer
store_selection['TXN_PER_LOYALTY_CARD_NUM'] = (store_selection['TXN_ID']/
                                            store_selection['LYLTY_CARD_NBR'])

# Reset index
store_selection = store_selection.reset_index()
store_selection.head()

# %%
# Trial period start of Feb 19 - end of April 19
# Count num of times a store num appears (ie how many months of data)
stores_12_months = store_selection['STORE_NBR'].value_counts()
# Filter by stores with 12 months of data - store as index
stores_12_months = stores_12_months[stores_12_months == 12].index
# Filter index stores to store_selection df and map by store number
stores_12_months = (store_selection[store_selection['STORE_NBR']
                    .isin(stores_12_months)])

stores_12_months.info() # Stores with at least 12 months of data
# %%
# Correlation between trial and control stores based on chip sales

# Define function to calculate correlation
def storeCorr(metric_columns, trial_store_num, input_table=stores_12_months):

    """ Function calculates correction between stores
        Input: metric_columns - columns to compare
               trial_store_num - number of store that is being comapared to
               input_table - stores_12_months (df of stores with 12months sales)
        Output: df with correlation values (df)"""
# Extract control stores by taking inverse of the df where trial stores are (by store num)
    control_store_num = (input_table[~input_table['STORE_NBR'].isin([77,86.88])]
                        ['STORE_NBR'].unique())
# Create df with desired column names to store correlation
    correlation = pd.DataFrame(columns=['YEAR_MONTH', 'TRIAL_STORE',
                                         'CONTROL_STORE', 'CORR'])
# Extract the trial stores from the input table
    trial_stores = input_table[input_table['STORE_NBR']==
                                 trial_store_num][metric_columns].reset_index()
# Loop over the control stores
    for store in control_store_num:
        df = pd.DataFrame(columns= ['YEAR_MONTH', 'TRIAL_STORE', 'CONTROL_STORE'
                                    'CORR'])
# For each control store num extract all rows for that particular store
        control_store = input_table[input_table['STORE_NBR']== store][metric_columns].reset_index()
# Assign CORR column to the correlation with each trial store to new df
# (a row for each month and hence correlation for that month)
        df['CORR'] = trial_stores.corrwith(control_store, axis=1)
# Assign the trial store num to new df
        df['TRIAL_STORE'] = trial_store_num
# Assign the control store num to the df (a row for each month and hence correlation for that month)
        df['CONTROL_STORE'] = store
# Assign the year and month for the correlation based off the current input table row
        df['YEAR_MONTH'] = list(input_table[input_table['STORE_NBR']== trial_store_num]['YEAR_MONTH'])
# Combine the new df to our master correlation df 
        correlation = pd.concat([correlation, df])
    return correlation

# %%
# Create blank df for correlation
correlation_table = pd.DataFrame()
# Loop over each trial store
for trial_store in [77,86,88]:
    # For each trial store combine our df with the storeCorr function on selected columns
    correlation_table = pd.concat([correlation_table, storeCorr(['TOT_SALES',
                                                         'LYLTY_CARD_NBR',
                                                         'TXN_ID',
                                                         'TXN_PER_LOYALTY_CARD_NUM',
                                                         'AVG_UNIT_PRICE'], trial_store)])
correlation_table.head()
                                                    
# %%
# Create function to compute magnitiude distance
