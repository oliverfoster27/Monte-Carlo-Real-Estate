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


def one_sim(condo_cost, amortization_period, horizon, verbose=1):
    gross_salary = 68_000
    investment_rate = 0.45
    raise_rate = 0.03
    initial_investment = 20_000

    stock_asset = StockAsset(initial_investment, verbose=verbose)
    re_asset = REAsset(condo_cost, amortization_period=amortization_period, verbose=verbose)

    data = []
    for year in range(horizon):

        yearly_stock_investment = investment_rate * gross_salary - re_asset.yearly_payment

        if yearly_stock_investment < 0:
            raise ValueError("You can't afford this mortgage")

        data.append(
            {'Stock Asset: Market Value': stock_asset.market_value, 'Stock Asset: Book Value': stock_asset.book_value,
             'Stock Asset: Accrued Cost': stock_asset.cost,
             'RE Asset: Market Value': re_asset.market_value, 'RE Asset: Book Value': re_asset.book_value,
             'RE Asset: Accrued Cost': re_asset.cost})
        stock_asset.year_end_evaluation(yearly_stock_investment)
        re_asset.year_end_evaluation()

        gross_salary = (1 + raise_rate) * gross_salary

    # Add the cost of selling to your asset
    re_asset.sell_asset()

    total_market_value = stock_asset.market_value + re_asset.market_value
    total_cost = stock_asset.cost + re_asset.cost
    total_book_value = stock_asset.book_value + re_asset.book_value
    ROI = (total_market_value - (total_book_value + total_cost)) / (total_book_value + total_cost)

    return pd.DataFrame(data), ROI


class REAsset:

    def __init__(self, initial_investment, rent_cost=1100, condo_fees=300, startup_cost=6_000,
                 mortgate_fixed_rate=0.032,
                 property_tax_rate=0.01, school_tax_rate=0.0015, re_broker_fee=0.05, downpayment_rate=0.2,
                 amortization_period=20, verbose=1, rent_increase_rate=0.03):

        self.year_counter = 0

        self.amortization_period = amortization_period
        self.mortgage_rate = mortgate_fixed_rate
        self.property_tax_rate = property_tax_rate
        self.school_tax_rate = school_tax_rate
        self.market_value = initial_investment
        self.re_broker_fee = re_broker_fee
        self.rent_increase_rate = rent_increase_rate
        self.book_value = initial_investment
        self.taxes = self.market_value * (self.property_tax_rate + self.school_tax_rate)

        if initial_investment == 0:
            self.cost = 0
            self.rent_cost = rent_cost * 12
            self.condo_fees = 0
            self.mortgage_outstanding = 0
            self.mortgage_payment = 0
            self.principal_payment = 0
            self.interest_payment = 0
        else:
            self.cost = startup_cost
            self.rent_cost = 0
            self.condo_fees = condo_fees * 12
            self.mortgage_outstanding = (1 - downpayment_rate) * initial_investment
            self.r = self.mortgage_rate / 12
            self.n = amortization_period * 12
            self.mortgage_payment = 12 * self.mortgage_outstanding * \
                                    (self.r * (1 + self.r) ** self.n) / \
                                    (((1 + self.r) ** self.n) - 1)
            self.principal_payment = 12 * self.mortgage_outstanding / self.n
            self.interest_payment = self.mortgage_payment - self.principal_payment

        self.yearly_payment = self.mortgage_payment + self.condo_fees + self.rent_cost + self.taxes
        self.housing_return = create_housing_return_generator()

        if verbose:
            print("RE Initial Investment: ${}".format(initial_investment))
            print("RE Book Value: ${}".format(self.book_value))
            print("RE Downpayment: ${}".format(downpayment_rate * initial_investment))
            print("RE Amortization Period: {} years".format(amortization_period))
            print("RE Monthly Payment Due: ${:.2f}".format(self.yearly_payment/12))

    def year_end_evaluation(self):

        self.year_counter += 1

        self.market_value = self.market_value * (1 + self.housing_return())
        self.taxes = self.market_value * (self.property_tax_rate + self.school_tax_rate)

        if self.year_counter < self.amortization_period:
            self.mortgage_outstanding -= self.principal_payment
            self.cost += self.interest_payment + self.condo_fees + self.rent_cost + self.taxes
        elif self.year_counter == self.amortization_period:
            self.mortgage_outstanding -= self.principal_payment
            self.cost += self.interest_payment + self.condo_fees + self.rent_cost + self.taxes
            self.mortgage_payment = 0
            self.principal_payment = 0
            self.interest_payment = 0
        else:
            self.cost += self.condo_fees + self.rent_cost + self.taxes

        self.yearly_payment = self.mortgage_payment + self.condo_fees + self.rent_cost + self.taxes

        # Account for rent increase
        self.rent_cost = (1 + self.rent_increase_rate) * self.rent_cost

    def sell_asset(self):

        self.cost += self.re_broker_fee * self.market_value


class StockAsset:

    def __init__(self, initial_investment, verbose=1, yearly_transaction_cost=100):
        self.yearly_transaction_cost = yearly_transaction_cost
        self.market_value = initial_investment
        self.book_value = initial_investment
        self.cost = 0
        self.sp_return = create_sp_return_generator()

        if verbose:
            print("Stock Initial Investment: ${}".format(initial_investment))

    def year_end_evaluation(self, yearly_investment):
        # Calculate value based on a year of accruing along with a lump-sum investment at year-end
        self.book_value = self.book_value + yearly_investment
        self.market_value = self.market_value * (1 + self.sp_return()) + yearly_investment
        self.cost += self.yearly_transaction_cost
