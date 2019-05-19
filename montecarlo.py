from mortgage import Loan
import numpy as np
import pandas as pd


def last_entry(entry):
    """Returns the last entry of a series"""
    if len(entry) != 0:
        return entry[-1]


def create_sp_return_generator():
    """
    Returns a function that randomly returns S&P performance from it's historical distribution

    Parameters:
    None

    Returns:
    Function:get_sp_return
    """
    # Import the historical S&P returns data
    sp = pd.read_csv(r'sp-500-historical-annual-returns.csv', index_col='date', parse_dates=True)
    # Include data only after 1970
    sp = sp.loc[sp.index > '1970-01-01', :]

    mean = sp['value'].mean()
    stddev = sp['value'].std()

    def get_sp_return():

        return np.random.normal(mean, stddev) / 100

    # Return the function
    return get_sp_return


def create_housing_return_generator():
    """
    Returns a function that randomly returns Montreal composite HPI returns from it's historical distribution

    Parameters:
    None

    Returns:
    Function:get_housing_return
    """
    # Import the historical housing index data for Montreal
    housing = pd.read_csv(r'montreal-historical-housing-index.csv', usecols=[0, 1], index_col='Date', parse_dates=True)
    # Resample the data on a yearly basis & take the last entry for that year
    housing = housing.resample(rule='A').apply(last_entry)
    # get the % change of the housing prices over each year
    housing['return'] = housing['Composite_HPI'].pct_change().fillna(106.3 / 100 - 1) * 100

    mean = housing['return'].mean()
    stddev = housing['return'].std()

    def get_housing_return():

        return np.random.normal(mean, stddev) / 100

    # Return the function
    return get_housing_return


def one_sim(condo_cost, amortization_period, gross_salary, investment_rate, horizon, initial_investment,
            raise_rate=0.03, verbose=1):
    """
    Returns data from one simulation of a given investment strategy

    Parameters:
    condo_cost
    amortization_period
    gross_salary
    investment_rate
    horizon
    initial_investment
    raise_rate=0.03
    verbose=1

    Returns:
    tuple: (pd.DataFrame:performance_history, float:ROI)
    """
    stock_asset = StockAsset(initial_investment, verbose=verbose)
    re_asset = REAsset(condo_cost, amortization_period=amortization_period, verbose=verbose)

    data = []
    for year in range(horizon):

        yearly_stock_investment = investment_rate * gross_salary - re_asset.yearly_payment

        assert yearly_stock_investment >= 0, \
            "You can't afford this mortgage with monthly payment of {}".format(re_asset.yearly_payment/12)

        stock_asset.year_end_evaluation(yearly_stock_investment)
        re_asset.year_end_evaluation()
        data.append(
            {'Stock Asset: Market Value': stock_asset.market_value,
             'Stock Asset: Book Value': stock_asset.book_value,
             'Stock Asset: Accrued Cost': stock_asset.cost,
             'RE Asset: Market Value': re_asset.market_value,
             'RE Asset: Book Value': re_asset.book_value,
             'RE Asset: Accrued Cost': re_asset.cost,
             'RE Asset: Mortgage Payment': re_asset.mortgage_payment,
             'RE Asset: Taxes': re_asset.taxes,
             'RE Asset: Yearly Payment': re_asset.yearly_payment,
             'RE Asset: Interest Payment': re_asset.interest_payment,
             'RE Asset: Rent': re_asset.rent_cost})

        gross_salary = (1 + raise_rate) * gross_salary

    # Add the cost of selling to your asset
    re_asset.sell_asset()

    # Calculate ROI
    total_market_value = stock_asset.market_value + re_asset.market_value
    total_cost = stock_asset.cost + re_asset.cost
    total_book_value = stock_asset.book_value + re_asset.book_value
    ROI = (total_market_value - (total_book_value + total_cost)) / (total_book_value + total_cost)

    return pd.DataFrame(data), ROI


# noinspection PyProtectedMember
class REAsset:
    """
    Real-Estate asset object to track status of the investment over time

    Parameters:
    initial_investment
    rent_cost=1100
    condo_fees=300
    startup_cost=6_000
    mortgage_rate=0.032
    property_tax_rate=0.01
    school_tax_rate=0.0015
    re_broker_fee=0.05
    downpayment_rate=0.2
    amortization_period=20
    verbose=1
    rent_increase_rate=0.03
    """

    def __init__(self, initial_investment, rent_cost=1100, condo_fees=300, startup_cost=6_000,
                 mortgage_rate=0.032,
                 property_tax_rate=0.01, school_tax_rate=0.0015, re_broker_fee=0.05, downpayment_rate=0.2,
                 amortization_period=20, verbose=1, rent_increase_rate=0.03):

        self.year_counter = 0
        self.__dict__.update(locals())

        if initial_investment > 0:
            self.loan_schedule = Loan(principal=initial_investment,
                                      interest=mortgage_rate,
                                      term=amortization_period)._schedule
            self.cost = startup_cost
            self.rent_cost = 0
            self.condo_fees = condo_fees * 12
            self.book_value = initial_investment
            self.market_value = initial_investment
        else:
            self.loan_schedule = []
            self.cost = 0
            self.rent_cost = rent_cost * 12
            self.condo_fees = 0
            self.book_value = 0
            self.market_value = 0

        self.mortgage_outstanding = 0
        self.mortgage_payment = 0
        self.principal_payment = 0
        self.interest_payment = 0
        self.yearly_payment = 0
        self.taxes = self.market_value * (self.property_tax_rate + self.school_tax_rate)
        self.housing_return = create_housing_return_generator()

        if verbose:
            print("RE Initial Investment: ${}".format(initial_investment))
            print("RE Book Value: ${}".format(self.book_value))
            print("RE Downpayment: ${}".format(downpayment_rate * initial_investment))
            print("RE Amortization Period: {} years".format(amortization_period))
            print("RE Monthly Payment Due: ${:.2f}".format(self.yearly_payment/12))
            print("Mortgage Payment Due: ${:.2f}".format(self.mortgage_payment/12))

    def year_end_evaluation(self):
        """Calculate the updated value of the asset after each year"""

        self.mortgage_payment = 0
        self.principal_payment = 0
        self.interest_payment = 0
        for month in range(1, 13):
            try:
                self.mortgage_payment += float(self.loan_schedule[self.year_counter * 12 + month].payment)
                self.principal_payment += float(self.loan_schedule[self.year_counter * 12 + month].principal)
                self.interest_payment += float(self.loan_schedule[self.year_counter * 12 + month].interest)
                self.mortgage_outstanding = float(self.loan_schedule[self.year_counter * 12 + month].balance)
            except IndexError:
                break

        self.market_value = self.market_value * (1 + self.housing_return())
        self.taxes = self.market_value * (self.property_tax_rate + self.school_tax_rate)
        self.cost += self.interest_payment + self.condo_fees + self.rent_cost + self.taxes

        # Account for rent increase
        self.rent_cost = (1 + self.rent_increase_rate) * self.rent_cost
        # Sum the yearly payment
        self.yearly_payment = self.mortgage_payment + self.condo_fees + self.rent_cost + self.taxes

        self.year_counter += 1

    def sell_asset(self):

        self.cost += self.re_broker_fee * self.market_value


class StockAsset:
    """
    Stock asset object to track status of the investment over time

    Parameters:
    initial_investment
    verbose=1
    yearly_transaction_cost=100
    """

    def __init__(self, initial_investment, verbose=1, yearly_transaction_cost=100):
        self.yearly_transaction_cost = yearly_transaction_cost
        self.market_value = initial_investment
        self.book_value = initial_investment
        self.cost = 0
        self.sp_return = create_sp_return_generator()

        if verbose:
            print("Stock Initial Investment: ${}".format(initial_investment))

    def year_end_evaluation(self, yearly_investment):
        """
        Calculate the updated value of the asset after each year
        with the addition of a new yearly lump-sum investment
        """
        # Calculate value based on a year of accruing along with a lump-sum investment at year-end
        self.book_value = self.book_value + yearly_investment
        self.market_value = self.market_value * (1 + self.sp_return()) + yearly_investment
        self.cost += self.yearly_transaction_cost
