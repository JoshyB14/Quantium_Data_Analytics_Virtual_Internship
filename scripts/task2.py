# %%

from numpy import core
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
def storeCorr(metric_columns, trial_store_num, stores_12_months=stores_12_months):

        """ Function calculates correction between stores

        Args:
            metric_columns ([list]): [columns to compare]
            trial_store_num ([int]): [store number of trial store]
            stores_12_months ([df], optional): [df of stores with 12 months of sales]. Defaults to stores_12_months.

        Returns:
            [correlation]: [df of the correlation between the trial store and
                                control stores]
        """

# Extract control stores by taking inverse of the df where trial stores are (by store num)
        control_store_num = (stores_12_months[~stores_12_months['STORE_NBR'].isin([77,86.88])]
                        ['STORE_NBR'].unique())
# Create df with desired column names to store correlation
        correlation = pd.DataFrame(columns=['YEAR_MONTH', 'TRIAL_STORE',
                                         'CONTROL_STORE', 'CORR'])
# Extract the trial stores from the input table
        trial_stores = stores_12_months[stores_12_months['STORE_NBR']==
                                 trial_store_num][metric_columns].reset_index()
# Loop over the control stores
        for store in control_store_num:
                df = pd.DataFrame(columns= ['YEAR_MONTH', 'TRIAL_STORE', 'CONTROL_STORE'
                                    'CORR'])
# For each control store num extract all rows for that particular store
        control_store = stores_12_months[stores_12_months['STORE_NBR']== store][metric_columns].reset_index()
# Assign CORR column to the correlation with each trial store to new df
# (a row for each month and hence correlation for that month)
        df['CORR'] = trial_stores.corrwith(control_store, axis=1)
# Assign the trial store num to new df
        df['TRIAL_STORE'] = trial_store_num
# Assign the control store num to the df (a row for each month and hence correlation for that month)
        df['CONTROL_STORE'] = store
# Assign the year and month for the correlation based off the current input table row
        df['YEAR_MONTH'] = list(stores_12_months[stores_12_months['STORE_NBR']== trial_store_num]['YEAR_MONTH'])
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
# Create function to compute magnitude distance
def store_magnitude (metric_columns, trial_store_num, stores_12_months=stores_12_months):
        """ Function calculates magnitude between stores

        Args:
            metric_columns ([list]): [columns to compare]
            trial_store_num ([int]): [store number of trial store]
            stores_12_months ([df], optional): [df of stores with 12 months of sales]. Defaults to stores_12_months.

        Returns: 
            [magnitude]: [df of the magnitude between the trial store and
                                control stores]
        """
# Extract control stores by taking inverse of the df where trial stores are (by store num)
        control_store_num = (stores_12_months[~stores_12_months['STORE_NBR']
                                .isin([77,86,88])]['STORE_NBR'].unique())
# Initialize empty df
        magnitude = pd.DataFrame()
# Extract the trial stores from the input table
        trial_stores = stores_12_months[stores_12_months['STORE_NBR']== trial_store_num][metric_columns]
# Loop over the control stores
        for store in control_store_num:
# Take absolute value of: (trial store metric columns of interest minus (-) the control store metric columns 
                df = abs(stores_12_months[stores_12_months['STORE_NBR']== trial_store_num].reset_index()[metric_columns]
                        - stores_12_months[stores_12_months['STORE_NBR']==store].reset_index()[metric_columns])
# Assign the trial store num to new df
                df['TRIAL_STORE'] = trial_store_num
# Assign the control store num to the df (a row for each month and hence magnitude for that month)
                df['CONTROL_STORE'] = store
# Assign the year and month for the magnitude based off the current input table row
                df['YEAR_MONTH'] = list(stores_12_months[stores_12_months['STORE_NBR']== trial_store_num]['YEAR_MONTH'])
# Concat the two df together
                magnitude = pd.concat([magnitude, df])
# Loop over each column of interest
        for column in metric_columns:
# Compute the magnitude:
# 1- (Observed distance – minimum distance)/(Maximum distance – minimum distance)
                magnitude[column] = 1 - ((magnitude[column] - magnitude[column].min()) /
                                        (magnitude[column].max() - magnitude[column].min()))
# Assign the mean value of the metric_columns for each store to a new column called magnitude
        magnitude['MAGNITUDE'] = magnitude[metric_columns].mean(axis=1)
# Return the df
        return magnitude

# %%
# Create blank df for magnitude
magnitude = pd.DataFrame()
# Loop over each trial store
for trial_store in [77,86,88]:
    # For each trial store combine our df with the store_magnitude function on selected columns
    magnitude = pd.concat([magnitude, store_magnitude(['TOT_SALES',
                                                         'LYLTY_CARD_NBR',
                                                         'TXN_ID',
                                                         'TXN_PER_LOYALTY_CARD_NUM',
                                                         'AVG_UNIT_PRICE'], trial_store)])
magnitude.head()

# %%
# Combine both the correlation and magnitude columns
def combine_mag_corr(metric_columns, trial_store_num, stores_12_months=stores_12_months):
        """ Function combines magnitude and correlation between stores

        Args:
            metric_columns ([list]): [columns to compare]
            trial_store_num ([int]): [store number of trial store]
            stores_12_months ([df], optional): [df of stores with 12 months of sales]. Defaults to stores_12_months.

        Returns: 
            [master]: [df of both the magnitude and correlation between the trial store and
                                control stores]
        """
        # Compute correlation with storeCorr function
        correlation = storeCorr(metric_columns, trial_store_num, stores_12_months)
        # Compute magnitude with store_magnitude function
        magnitude = store_magnitude(metric_columns, trial_store_num, stores_12_months)
        # Drop 
        magnitude = magnitude.drop(metric_columns, axis=1)
        master = pd.merge(correlation, magnitude, on=['CONTROL_STORE', 'TRIAL_STORE', 'YEAR_MONTH'])
        return master

# %%
# Compare stores based off total sales
master_sales = pd.DataFrame()
# Loop over trial stores
for trial_store in [77, 86, 88]:
        # Call combine_mag_corr
        master_sales = pd.concat([master_sales, combine_mag_corr(['TOT_SALES'], trial_store)])

# %%
# Group sales by trial and control stores and take mean
sales_compare = master_sales.groupby(['TRIAL_STORE', 'CONTROL_STORE']).mean().reset_index()
# Combine correlation and magnitude to obtain a new scoring method (0.5 weight for magnitude and 0.5 for correlation )
# This effectively takes each measure into equal consideration for analysis
sales_compare['master_score'] = (0.5*sales_compare['CORR']) + (0.5*sales_compare['MAGNITUDE'])
# For each trial store print the top store with the highest magnitude for total sales
for trial_store in [77, 86, 88]:
        print(sales_compare[sales_compare['TRIAL_STORE']==trial_store].sort_values(ascending=False, by='master_score'))

# %%
# Compare stores based off number of customers (loyalty card info)
master_customers = pd.DataFrame()
# Loop over trial stores
for trial_store in [77, 86, 88]:
        # Call combine_mag_corr
        master_customers = pd.concat([master_customers, combine_mag_corr(['LYLTY_CARD_NBR'], trial_store)])

# %%
customer_compare = master_customers.groupby(['TRIAL_STORE', 'CONTROL_STORE']).mean().reset_index()
# Combine correlation and magnitude to obtain a new scoring method (0.5 weight for magnitude and 0.5 for correlation )
# This effectively takes each measure into equal consideration for analysis
customer_compare['master_score'] = (0.5*customer_compare['CORR']) + (0.5*customer_compare['MAGNITUDE'])
# For each trial store print the top store with the highest magnitude for total sales
for trial_store in [77, 86, 88]:
        print(customer_compare[customer_compare['TRIAL_STORE']==trial_store].sort_values(ascending=False, by='master_score'))

# %%
# Loop over each trial store
for trial_store in [77, 86, 88]:
        # Set x to be where our sales match our trial stores the best by master score
        x = (sales_compare[sales_compare['TRIAL_STORE']== trial_store].sort_values(ascending=False, by='master_score')
                .set_index(['TRIAL_STORE', 'CONTROL_STORE'])['master_score'])

        # Set y to be where our customer numbers match our trial stores the most by master score
        y = (customer_compare[customer_compare['TRIAL_STORE']== trial_store].sort_values(ascending=False,
                 by='master_score').set_index(['TRIAL_STORE', 'CONTROL_STORE'])['master_score'])

        print((pd.concat([x,y],axis=1).sum(axis=1)/2).sort_values(ascending=False).head(3))

