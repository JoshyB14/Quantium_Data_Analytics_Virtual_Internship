# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Quantium Data Analytics Virtual Experience - Task1 - Josh Bryden

# %%
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy import stats

#Pandas settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# seaborn settings
sns.set_style("darkgrid")

# %% [markdown]
# ## Data importing

# %%
purchase_data = pd.read_csv('../data/QVI_purchase_behaviour.csv')
purchase_data.head()


# %%
purchase_data.shape


# %%
purchase_data.dtypes 


# %%
purchase_data.nunique() 


# %%
purchase_data.isnull().sum() # missing values - none


# %%
transaction_data = pd.read_csv('../data/QVI_transaction_data.csv')
transaction_data.head()


# %%
transaction_data.shape 


# %%
transaction_data.dtypes 

# %% [markdown]
# Date encoded above as int - we will need to change to datetime

# %%
transaction_data.nunique()


# %%
transaction_data.isnull().sum() # missing values - none

# %% [markdown]
# ## Data cleaning

# %%
# merge the two datasets on the loyalty card number
chips_data = pd.merge(transaction_data, purchase_data, on='LYLTY_CARD_NBR', how='left')
chips_data.head()


# %%
# convert excel date time to python date time
chips_data['DATE'] = pd.to_datetime(chips_data['DATE'], origin = "1899-12-30", unit='D') # origin set based off excel time, units in days since epoch (for excel this is 1/1/1900)
# https://stackoverflow.com/a/64068366/11938664
chips_data.head()


# %%
chips_data.describe()

# %% [markdown]
# The describe table above shows that there was 200 items bought in a transaction and that the total sales for a product was 650. We will need to investigate

# %%
# where sales were over 5 - this is higher than the 75th percentile
chips_data[chips_data['PROD_QTY']>5]


# %%
# Drop the outliers - these sales may be real or not - however they are outliers
chips_data.drop(chips_data[chips_data['PROD_QTY']==200].index, inplace=True)


# %%
# sanity check - should be 0 rows 
chips_data[chips_data['PROD_QTY']>5]

# %% [markdown]
# Examine the products in the dataset

# %%
print(chips_data['PROD_NAME'].unique())

# %% [markdown]
# Count the number of times each word appears in the dataset

# %%
# split up string by spaces
product_words = chips_data["PROD_NAME"].str.split() # default is split on whitespace


# %%
def product_unique_words(list_of_words):
    """ Counts number of times a word
        appears in a list of words """
    for word in list_of_words:
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] +=1


# %%
# initalize word_count
word_count = {}
# apply product_unique_words function to each row of the product_words list
product_words.apply(lambda row: product_unique_words(row))
# change product words to series and sort values with largest at top
print(pd.Series(word_count).sort_values(ascending=False))

# %% [markdown]
# From the output above we can see that the datset contains products with 'Salsa' in it - we should remove these values

# %%
# drop salsa products
chips_data = chips_data[~chips_data['PROD_NAME'].str.contains('Salsa')]

# %% [markdown]
# ## Pack size/ weight

# %%
# extract weight/ size from product name

# function to determine if weight in product name
def product_weight(product_name):
    """ Function extracts weight
        from product name """
    weight='' # string
    for character in product_name: # loop over each character in product name
        if character.isdigit(): # if character is a digit
            weight+=character # add digit to the  string 'weight'
    return int(weight) # return the int of the string 'weight' for each product


# %%
# Apply product_weight to 'PROD_NAME' column to create 'PACK_WEIGHT' column
chips_data['PACK_WEIGHT']=chips_data['PROD_NAME'].apply(product_weight)
chips_data.head()


# %%
# Histogram of pack size
sns.histplot(x='PACK_WEIGHT', bins=20, data=chips_data)
plt.show()
# most packs around the 170g and 175g size

# %% [markdown]
# ## Brand names

# %%
# Create function to extract first word from product name - tends to be brand name
def brand_name(product):
    """ Function extracts first word
    from product name which is the Brand"""
    return product.split(' ')[0] # returns first word split by whitespace


# %%
# Apply brand_name to chips_data
chips_data['BRAND_NAME'] = chips_data['PROD_NAME'].apply(brand_name)
chips_data.head()


# %%
# Brands in BRAND_NAME column
chips_data['BRAND_NAME'].unique()


# %%
# replace abbrevations with actual brand names - used personal knowledge of Woolworths products here
# Woolworths products are somtimes abbreviated in system if product name is long. This is so it can fit onto a ticket instore (shelf label)
chips_data['BRAND_NAME'].replace('Natural', 'Natural Chip Co', inplace = True)
chips_data['BRAND_NAME'].replace('NCC', 'Natural Chip Co', inplace = True) 
chips_data['BRAND_NAME'].replace('Grain', 'Grain Waves', inplace = True)
chips_data['BRAND_NAME'].replace('WW', 'Woolworths', inplace = True)
chips_data['BRAND_NAME'].replace('Burger', 'Burger Rings', inplace = True)
chips_data['BRAND_NAME'].replace('Infzns', 'Infuzions', inplace = True)
chips_data['BRAND_NAME'].replace('Red', 'Red Rock Deli', inplace = True)
chips_data['BRAND_NAME'].replace('Dorito', 'Doritos', inplace = True)
chips_data['BRAND_NAME'].replace('Smith', 'Smiths', inplace = True)
chips_data['BRAND_NAME'].replace('GrainWves', 'Grain Waves', inplace = True)
chips_data['BRAND_NAME'].replace('French', 'French Fries', inplace = True)
chips_data['BRAND_NAME'].replace('Infzns', 'Infuzions', inplace = True)
chips_data['BRAND_NAME'].replace('RRD', 'Red Rock Deli', inplace = True)
chips_data['BRAND_NAME'].replace('Snbts', 'Sunbites', inplace = True)


# %%
# Heatmap of PROD_QTY by BRAND_NAME and PACK_WEIGHT - examine sales by item
plt.figure(figsize=(15,15))
sns.heatmap(pd.pivot_table(data=chips_data, index='BRAND_NAME', columns='PACK_WEIGHT', values='PROD_QTY'))
plt.show()

# %% [markdown]
# The heatmap above shows that (as seen in our pack_weight histogram) that a majority of chips have sizes of 170 and 175g. Furthermore we have a higher sales volume in Cobs popcorn (110g), Doritos (150g, 170g), Kettle chips (150g, 175g), Pringles (134g), Thins (175g) and Twisties (250g, 255g)  
# %% [markdown]
# ## Customers and their purchases

# %%
# histogram of customers at their various 'lifestage'
sns.histplot(x='LIFESTAGE', data=chips_data)
plt.xticks(rotation=75)
plt.show()

# %% [markdown]
# We see that we have high numbers of older families, older singles/ couples, retiriees and young families. With lower numbers of young singles/ couples and midage singles/couples.
# 
#  Interestingly new families bought chips the least in our dataset (we should investigate) - potentially due to small children and having different shopping items that are required (ie - baby formula, baby food, healthy school lunch snacks, etc)

# %%
# hisogram of how premium a customer is
sns.histplot(x='PREMIUM_CUSTOMER', data=chips_data)
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# We see that a majority of customers are mainstream, slightly less are budget and even less are premium customers buying the most expensive products. 
# 
# Woolworths stores are rated internally as: budget, mainstream, premium and super-premium. This is determined by the demographic of customer types based off loyalty card info.
# 
# Lets check that our dataset is not from just a couple stores...

# %%
# store numbers
sns.histplot(x='STORE_NBR', data=chips_data)
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# The above confirms that our dataset is not from a small handful of stores, hence we can assume that our dataset is an accurate representation of shoppers overall at Woolworths.

# %%
# Examine sales by how premium the customer is and by their lifestage
customer_sales = chips_data[['TOT_SALES', 'PREMIUM_CUSTOMER', 'LIFESTAGE']].groupby(['PREMIUM_CUSTOMER' ,'LIFESTAGE']).sum().sort_values('TOT_SALES',ascending=False)
customer_sales


# %%
plt.figure(figsize=(15,15))
sns.barplot(y=customer_sales.reset_index()['LIFESTAGE'], x=customer_sales.reset_index()['TOT_SALES'], hue=customer_sales.reset_index()['PREMIUM_CUSTOMER'], data=customer_sales)
plt.title('Customer sales by lifestage and premium category')
plt.xlabel('Total Sales')
plt.ylabel('Life Stage')
plt.show()


# %%
# Determine number of customer in each segment based off the loyalty card number
customer_numbers = chips_data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).count()


# %%
# Plot customer_numbers
plt.figure(figsize=(15,15))
sns.barplot(y=customer_numbers.reset_index()['LIFESTAGE'], x=customer_numbers.reset_index()['LYLTY_CARD_NBR'], hue=customer_numbers.reset_index()['PREMIUM_CUSTOMER'], data=customer_numbers)
plt.title('Number of customers per lifestage and permium catgeory')
plt.xlabel('Number of customers')
plt.ylabel('Lifestage')
plt.show()


# %%
# Compute average price of chips sales per customer
chips_data['AVG_CHIP_PRICE'] = chips_data['TOT_SALES']/chips_data['PROD_QTY']
chips_data.head()


# %%
# Group average sales by lifestage and market segment
average_sales = chips_data[['LIFESTAGE', 'PREMIUM_CUSTOMER', 'AVG_CHIP_PRICE']].groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).mean()
average_sales.sort_values('AVG_CHIP_PRICE',ascending=False)

# %% [markdown]
# The output above shows that both mainstream young and midage singles/couples are the higest spenders on chips. This may be due to the lack of children, hence more $ to spend as they see fit. This also may be due to the graph above showing much higher numbers of young and midage couples in the mainstream category compared to their respective premium and budget counterparts.
# 
# 
# To determine if the difference in average price is significant between the different premium levels of customer in these categories we will perform a T-test.
# 

# %%
# t - test
# First comparing mainstream against budget young and midage couples/ singles
# values taken from table above
stats.ttest_ind([4.065642,3.994241],[3.657366,3.743328])


# %%
# Comparing mainstream against premium young and midage couples/ singles
# values taken from table above
stats.ttest_ind([4.065642,3.994241],[3.665414,3.770698])

# %% [markdown]
# Based off the output above, we obtained a p value of 0.027 for the difference between mainstream and budget customers (young and midage couples/singles) - thus indicating a significant difference between these two groups.
# 
# We also found a significant p value of 0.039 for the difference between mainstream and premium customers.


# %%
