# %% [markdown]
# # Quantium Data Analytics Virtual Experience - Task2 - Josh Bryden
# %%

from numpy import core
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.groupby.generic import ScalarResult
import seaborn as sns
from scipy.stats import ttest_ind, t

#Pandas settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# seaborn settings
sns.set_style("darkgrid")

# %% [markdown]
# ## Data importing

# %%
chips_data = pd.read_csv('../data/chips_data.csv')
chips_data.head()

# %% [markdown]
# ## Data cleaning
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
# Filter for data before the trial begun
pre_trial_stores = stores_12_months[stores_12_months['YEAR_MONTH']<201902]

# %%
# Correlation between trial and control stores based on chip sales

# Define function to calculate correlation
def storeCorr(metric_columns, trial_store_num, pre_trial_stores=pre_trial_stores):

        """ Function calculates correction between stores

        Args:
            metric_columns ([list]): [columns to compare]
            trial_store_num ([int]): [store number of trial store]
            pre_trial_stores ([df], optional): [df of stores before 201902 with 12 months of data]. 
                                                Defaults to pre_trial_stores.

        Returns:
            [correlation]: [df of the correlation between the trial store and
                                control stores]
        """

        # Extract control stores by taking inverse of the df where trial stores are (by store num)
        control_store_num = (pre_trial_stores[~pre_trial_stores['STORE_NBR'].isin([77,86.88])]['STORE_NBR'].unique())
        # Create df with desired column names to store correlation
        correlation = pd.DataFrame(columns=['YEAR_MONTH', 'TRIAL_STORE',
                                         'CONTROL_STORE', 'CORR'])
        # Extract the trial stores from the input table
        trial_stores = pre_trial_stores[pre_trial_stores['STORE_NBR']==trial_store_num][metric_columns].reset_index()
        # Loop over the control stores
        for store in control_store_num:
                df = pd.DataFrame(columns= ['YEAR_MONTH', 'TRIAL_STORE', 'CONTROL_STORE','CORR'])
                # For each control store num extract all rows for that particular store
                control_store = pre_trial_stores[pre_trial_stores['STORE_NBR']== store][metric_columns].reset_index()
                # Assign CORR column to the correlation with each trial store to new df
                # (a row for each month and hence correlation for that month)
                df['CORR'] = trial_stores.corrwith(control_store, axis=1)
                # Assign the trial store num to new df
                df['TRIAL_STORE'] = trial_store_num
                # Assign the control store num to the df (a row for each month and hence correlation for that month)
                df['CONTROL_STORE'] = store
                # Assign the year and month for the correlation based off the current input table row
                df['YEAR_MONTH'] = list(pre_trial_stores[pre_trial_stores['STORE_NBR']== trial_store_num]['YEAR_MONTH'])
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
def store_magnitude (metric_columns, trial_store_num, pre_trial_stores=pre_trial_stores):
        """ Function calculates magnitude between stores

        Args:
            metric_columns ([list]): [columns to compare]
            trial_store_num ([int]): [store number of trial store]
            pre_trial_stores ([df], optional): [df of stores before 201902 with 12 months of data]. 
                                                Defaults to pre_trial_stores.

        Returns: 
            [magnitude]: [df of the magnitude between the trial store and
                                control stores]
        """
        # Extract control stores by taking inverse of the df where trial stores are (by store num)
        control_store_num = (pre_trial_stores[~pre_trial_stores['STORE_NBR']
                                .isin([77,86,88])]['STORE_NBR'].unique())
        # Initialize empty df
        magnitude = pd.DataFrame()
        # Extract the trial stores from the input table
        trial_stores = pre_trial_stores[pre_trial_stores['STORE_NBR']== trial_store_num][metric_columns]
        # Loop over the control stores
        for store in control_store_num:
        # Take absolute value of: (trial store metric columns of interest minus (-) the control store metric columns 
                df = abs(pre_trial_stores[pre_trial_stores['STORE_NBR']== trial_store_num].reset_index()[metric_columns]
                        - pre_trial_stores[pre_trial_stores['STORE_NBR']==store].reset_index()[metric_columns])
        # Assign the trial store num to new df
                df['TRIAL_STORE'] = trial_store_num
        # Assign the control store num to the df (a row for each month and hence magnitude for that month)
                df['CONTROL_STORE'] = store
        # Assign the year and month for the magnitude based off the current input table row
                df['YEAR_MONTH'] = list(pre_trial_stores[pre_trial_stores['STORE_NBR']== trial_store_num]['YEAR_MONTH'])
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
def combine_mag_corr(metric_columns, trial_store_num, pre_trial_stores=pre_trial_stores):
        """ Function combines magnitude and correlation between stores

        Args:
            metric_columns ([list]): [columns to compare]
            trial_store_num ([int]): [store number of trial store]
            pre_trial_stores ([df], optional): [df of stores before 201902 with 12 months of data]. 
                                                Defaults to pre_trial_stores.

        Returns: 
            [master]: [df of both the magnitude and correlation between the trial store and
                                control stores]
        """
        # Compute correlation with storeCorr function
        correlation = storeCorr(metric_columns, trial_store_num, pre_trial_stores)
        # Compute magnitude with store_magnitude function
        magnitude = store_magnitude(metric_columns, trial_store_num, pre_trial_stores)
        # Drop all rows except magnitude before combining 
        magnitude = magnitude.drop(metric_columns, axis=1)
        #
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
        # Prints stores with highest master_score based off TOT_SALES alone
        print(sales_compare[sales_compare['TRIAL_STORE']==trial_store].sort_values(
                                                                ascending=False, by='master_score').head())

# %%
# Compare stores based off number of customers (loyalty card info)
master_customers = pd.DataFrame()
# Loop over trial stores
for trial_store in [77, 86, 88]:
        # Call combine_mag_corr
        master_customers = pd.concat([master_customers, combine_mag_corr(['LYLTY_CARD_NBR'], trial_store)])

# %%
# Group customers by trial and control stores and take mean
customer_compare = master_customers.groupby(['TRIAL_STORE', 'CONTROL_STORE']).mean().reset_index()
# Combine correlation and magnitude to obtain a new scoring method (0.5 weight for magnitude and 0.5 for correlation )
# This effectively takes each measure into equal consideration for analysis
customer_compare['master_score'] = (0.5*customer_compare['CORR']) + (0.5*customer_compare['MAGNITUDE'])
# For each trial store print the top store with the highest magnitude for total sales
for trial_store in [77, 86, 88]:
        # Prints stores with highest master_score based off LYLTY_CARD_NBR alone (num of customers)
        print(customer_compare[customer_compare['TRIAL_STORE']==trial_store].sort_values(
                                                        ascending=False, by='master_score').head())

# %%
# Combine master scores for TOT_SALES and LYLTY_CARD_NBR to obtain control/ trial store pairs
# Loop over each trial store
for trial_store in [77, 86, 88]:
        # Set x to be where our sales match our trial stores the best by master score
        sales = (sales_compare[sales_compare['TRIAL_STORE']== trial_store].sort_values(ascending=False, by='master_score')
                .set_index(['TRIAL_STORE', 'CONTROL_STORE'])['master_score'])

        # Set y to be where our customer numbers match our trial stores the most by master score
        customers = (customer_compare[customer_compare['TRIAL_STORE']== trial_store].sort_values(ascending=False,
                 by='master_score').set_index(['TRIAL_STORE', 'CONTROL_STORE'])['master_score'])
        # Print and combine sales/customers taking the average of the master scores and print top 3 pairs (control/trial)
        print((pd.concat([sales,customers],axis=1).sum(axis=1)/2).sort_values(ascending=False).head(3))
# Prints out:

# TRIAL_STORE  CONTROL_STORE
# 77           233              0.994554
#              46               0.983852
#              188              0.981705
# dtype: float64
# TRIAL_STORE  CONTROL_STORE
# 86           155              0.984800
#              109              0.976618
#              225              0.975346
# dtype: float64
# TRIAL_STORE  CONTROL_STORE
# 88           40               0.968176
#              26               0.957020
#              58               0.953097
# dtype: float64

# %% [markdown]
# ## Visualizations
# %%

# Create dic of trial/ control pairs to loop over
trial_control_dic = {77:233, 86:155, 88:40}
# Loop over keys and values
for trial, control in trial_control_dic.items():
        # Create plot for TOT_SALES
        sns.distplot(pre_trial_stores.loc[pre_trial_stores['STORE_NBR']==trial]['TOT_SALES'])
        sns.distplot(pre_trial_stores.loc[pre_trial_stores['STORE_NBR']==control]['TOT_SALES'])
        plt.legend(labels=[f'Store {trial} (trial)', f'Store {control} (control)'])
        plt.title('Trial/control store pairs on TOT_SALES')
        plt.xlabel('Total Sales')
        plt.show()
        # Create plot for LYLTY_CARD_NBR (num of customers)
        sns.distplot(pre_trial_stores.loc[pre_trial_stores['STORE_NBR']==trial]['LYLTY_CARD_NBR'])
        sns.distplot(pre_trial_stores.loc[pre_trial_stores['STORE_NBR']==control]['LYLTY_CARD_NBR'])
        plt.legend(labels=[f'Store {trial} (trial)', f'Store {control} (control)'])
        plt.title('Trial/control store pairs on LYLTY_CARD_NBR')
        plt.xlabel('Num. of Customers')
        plt.show()

# %%
# To compare performance of trial stores, we need to scale all control store
# performance to trial store performance for the pre-trial period for the sum of TOT_SALES

# Scale TOT_SALES for store 77 and store 233
sales_scale_77 = (pre_trial_stores[pre_trial_stores['STORE_NBR']==77]['TOT_SALES'].sum()/
                        pre_trial_stores[pre_trial_stores['STORE_NBR']==233]['TOT_SALES'].sum())

# Scale TOT_SALES for store 86 and store 155
sales_scale_86 = (pre_trial_stores[pre_trial_stores['STORE_NBR']==86]['TOT_SALES'].sum()/
                        pre_trial_stores[pre_trial_stores['STORE_NBR']==155]['TOT_SALES'].sum())

# Scale TOT_SALES for store 88 and store 40
sales_scale_88 = (pre_trial_stores[pre_trial_stores['STORE_NBR']==88]['TOT_SALES'].sum()/
                        pre_trial_stores[pre_trial_stores['STORE_NBR']==40]['TOT_SALES'].sum())

# %%
# Filter for trial period dates
trial_period = stores_12_months[(stores_12_months['YEAR_MONTH']>=201902) & (stores_12_months['YEAR_MONTH']<=201904)]
# Filter for control stores on 'STORE_NBR', 'YEAR_MONTH', 'TOT_SALES'
scaled_store_sales = stores_12_months[stores_12_months['STORE_NBR'].isin([233,155,40])][['STORE_NBR', 'YEAR_MONTH',
                                                                                        'TOT_SALES']]
#%%
# Create function to scale sales data
def scale(store):
        """[Function scales trial and control store pretrial data]

        Args:
            store ([row of data]): [input is row of df for a particular store num]

        Returns:
            [scaled data]: [scaled pretrial store data]
        """
        if store['STORE_NBR'] ==233:
                return store['TOT_SALES'] * sales_scale_77
        elif store['STORE_NBR']==155:
                return store['TOT_SALES'] * sales_scale_86
        elif store['STORE_NBR']==40:
                return store['TOT_SALES'] * sales_scale_88

# %%
# Apply scale function and create new column
scaled_store_sales['SALES_SCALED'] = scaled_store_sales.apply(lambda store: scale(store), axis=1)
# Filter out trial store period
trial_period_scaled_sales = scaled_store_sales[(scaled_store_sales['YEAR_MONTH']>=201902) &
                                                 (scaled_store_sales['YEAR_MONTH']<=201904)]
# Filter out pre-trial scaled data
pretrial_scaled_sales = scaled_store_sales[scaled_store_sales['YEAR_MONTH']<201902]

# %%
# Null hypothesis that there is no difference between store pre-trial and trial period performance on sales
# Use ttest_ind to test for null hypoth that two samples have identical average values
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
# Loop over each control store
for control_store in [233, 155, 40]:
        print(f'Store number: {control_store}')
        # Compute the t test statistic for the pretrial sales and the trial period sales for each control store
        print(ttest_ind(pretrial_scaled_sales[pretrial_scaled_sales['STORE_NBR']==control_store]['SALES_SCALED'],
                trial_period_scaled_sales[trial_period_scaled_sales['STORE_NBR']==control_store]['SALES_SCALED']))

# Outputs:

# Store number: 233
# Ttest_indResult(statistic=1.1432469352201307, pvalue=0.2859919469281543)
# Store number: 155
# Ttest_indResult(statistic=1.0217889604585213, pvalue=0.33678271820066796)
# Store number: 40
# Ttest_indResult(statistic=-0.30265739096672245, pvalue=0.7698710330791956)

# No significant difference between control stores pre-trial and trial scaled sales
