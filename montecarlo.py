import numpy as np
import pandas as pd
import random


def last_entry(entry):
    if len(entry) != 0:
        return entry[-1]


def create_sp_return_generator():

    # Import the historical S&P returns data
    sp = pd.read_csv(r'sp-500-historical-annual-returns.csv')
    # Create the array that stores the bucket bounds
    bucket_array = np.linspace(-50, 50, 101)
    # Assign each datapoint a bucket
    sp['bucket'] = pd.cut(sp['value'], bucket_array)
    # Groupby count the number of instances in each bucket
    sp = sp.groupby(['bucket'])[['value']].count()
    # Create pseudo-probabilities around the frequency of being in each bucket
    sp['probability'] = sp['value'] / sp['value'].sum()
    # Format the dataframe
    sp.reset_index(inplace=True)
    sp.rename(index=str, columns={"value": "count"}, inplace=True)

    def get_sp_return():
        # Randomly choose a bucket based off the distribution of probabilities
        bucket_choice = np.random.choice(len(sp), 1, p=sp['probability'])
        left, right = sp.iloc[bucket_choice]['bucket'][0].left, sp.iloc[bucket_choice]['bucket'][0].right
        # Randomly pick a return between the two bounds of the bucket (uniform)
        return random.uniform(left, right)/100

    # Return the function
    return get_sp_return


def create_housing_return_generator():

    # Import the historical housing index data for Montreal
    housing = pd.read_csv(r'montreal-historical-housing-index.csv', usecols=[0, 1], index_col='Date', parse_dates=True)
    # Resample the data on a yearly basis & take the last entry for that year
    housing = housing.resample(rule='A').apply(last_entry)
    # get the % change of the housing prices over each year
    housing['return'] = housing['Composite_HPI'].pct_change().fillna(106.3 / 100 - 1) * 100
    # Create the array that stores the bucket bounds
    bucket_array = np.linspace(0, 8, 9)
    # Assign each datapoint a bucket
    housing['bucket'] = pd.cut(housing['return'], bucket_array)
    # Groupby count the number of instances in each bucket
    housing = housing.groupby(['bucket'])[['return']].count()
    # Create pseudo-probabilities around the frequency of being in each bucket
    housing['probability'] = housing['return'] / housing['return'].sum()
    # Format the dataframe
    housing.reset_index(inplace=True)
    housing.rename(index=str, columns={"return": "count"}, inplace=True)

    def get_housing_return():
        # Randomly choose a bucket based off the distribution of probabilities
        bucket_choice = np.random.choice(len(housing), 1, p=housing['probability'])
        left, right = housing.iloc[bucket_choice]['bucket'][0].left, housing.iloc[bucket_choice]['bucket'][0].right
        # Randomly pick a return between the two bounds of the bucket (uniform)
        return random.uniform(left, right)/100

    # Return the function
    return get_housing_return
