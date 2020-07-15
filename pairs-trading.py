#!/usr/bin/env python
# coding: utf-8
# Angelo Salton <gsalton4@hotmail.com>


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations


class PairsTrading:
    '''
    A class that performs pairs trading using a co-integration approach. Takes a pandas DataFrame with asset prices and a Timestamp index.
    '''

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __repr__(self):
        repr = f'Pairs trading object\n'
        repr += f'Assets: {self.data.shape[1]}'
        return repr

    def plot_relationship(self, asset1: str, asset2: str):
        '''
        Plot ratios between two assets using OLS for ratio calculation.

        :param asset1: ...
        :type asset1: str
        :param asset2: ...
        :type asset1: str
        '''
        # get pandas Series
        asset1 = self.data[asset1]
        asset2 = self.data[asset2]

        # estimate model
        model = sm.OLS(asset1, sm.tsa.stattools.add_trend(
            asset2, trend='ct')).fit()
        ratio = model.resid

        # plot
        plt.figure(figsize=(10, 6))
        plt.plot(ratio)
        plt.axhline(ratio.mean(), linestyle='--', color='gray')
        plt.fill_between(ratio.index, ratio.mean() - ratio.std(),
                         ratio.mean() + ratio.std(), alpha=0.15, color='red')
        plt.fill_between(ratio.index,
                         ratio.mean() - ratio.std() * 2,
                         ratio.mean() + ratio.std() * 2, alpha=0.1, color='red')
        plt.ylabel('Standard deviations')
        plt.xlabel('Date')
        plt.legend(['Ratio', 'Mean', '1 Std. dev.', '2 Std. dev.'])
        plt.title(
            f'Long-term relationship between {asset1.name} and {asset2.name}')

    def find_cointegrating_pairs(self, alpha=0.05):
        '''
        Grid search on data for co-integrating relationships between two assets.

        :param alpha: ...
        :type alpha: float
        :return pairs: ...
        :rtype alpha: list
        '''
        def cointegrate(x1, x2, **kwargs):
            test, pval, _ = sm.tsa.stattools.coint(x1, x2)
            return pval < alpha

        all_pairs = tuple(combinations(self.data.columns, 2))
        pairs = []

        for asset1, asset2 in all_pairs:
            result = cointegrate(self.data[asset1], self.data[asset2])
            if result:
                pairs.append((asset1, asset2))

        return pairs

    def asset_return_correlation(self):
        '''
        Return a correlation matrix of asset returns.
        '''
        return np.log(self.data).diff().corr().style.background_gradient(cmap='coolwarm', low=1.2, high=.8)

    def pairs_trade(self,
                    asset1: str,
                    asset2: str,
                    method: str = 'ols',
                    movavg: bool = False,
                    window: int = 14,
                    dev: float = 2,
                    plot: bool = False,
                    **kwargs):
        '''
        Simple pairs trading algorithm between two assets.

        :param asset1: ...
        :type asset1: str
        :param asset2: ...
        :type asset2: str
        :param method: ...
        :type method: str
        :param movavg: ...
        :type movavg: bool
        :param window: ...
        :type window: int
        :param dev: ...
        :type dev: float
        :param plot: ...
        :type plot: bool
        :return position: ...
        :rtype: :class:`pd.DataFrame`
        '''
        # get pandas series
        asset1 = self.data[asset1]
        asset2 = self.data[asset2]

        # calculate asset ratios:
        if method == 'ols':
            model = sm.OLS(asset1, sm.tsa.stattools.add_trend(
                asset2, trend='ct')).fit()
            ratio = model.resid

        else:
            raise ValueError("Methods available: 'ols'")

        if movavg:
            ratio = ratio.rolling(window=window, center=False, **kwargs).mean()

        # calculate standardized ratio:
        std_ratio = (ratio - ratio.mean()) / ratio.std()

        # initialize signals
        signal_asset1 = pd.Series(0, index=ratio.index)
        signal_asset2 = pd.Series(0, index=ratio.index)
        position = pd.Series('close', index=ratio.index)

        # count pair transactions
        transactions = 0

        # move along data
        for t in range(std_ratio.shape[0]):
            # short
            if std_ratio[t] > dev and position[t] != 'short':
                position[t] = 'short'
                signal_asset1[t] = -1
                signal_asset2[t] = ratio[t]
                transactions += 1

            # long
            elif std_ratio[t] < -dev and position[t] != 'long':
                position[t] = 'long'
                signal_asset1[t] = 1
                signal_asset2[t] = -ratio[t]
                transactions += 1

            # close
            elif abs(std_ratio[t]) < 0.2 and position[t] != 'close':
                position[t] = 'close'
                signal_asset1[t] = 0
                signal_asset2[t] = 0
                transactions += 1

        # plot
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(std_ratio, color='gray')
            plt.plot(signal_asset1, color='blue')
            plt.plot(signal_asset2, color='red')
            plt.legend([f'Ratio', f'{asset1.name} signal', f'{asset2.name} signal'])
            if movavg:
                plt.title(
                    f'Pairs trading strategy between {asset1.name} and {asset2.name}\n{dev} std. devs.\n{window}-day moving average')
            else:
                plt.title(
                    f'Pairs trading strategy between {asset1.name} and {asset2.name}\n{dev} std. devs.')

        # returns a DataFrame with positions
        result = pd.DataFrame({
            asset1.name: signal_asset1,
            asset2.name: signal_asset2
        })
        return result

    def optimal_portfolio(self, **kwargs):
        '''
        Estimate an optimal portfolio given asset data and a budget constraint. Returns a pandas DataFrame with quantities of assets.

        :return port: ...
        :rtype: :class:pd.DataFrame
        '''
        # output
        print('Estimating optimal portfolio')

        # initialize portfolio positions
        port = self.data.copy()
        port.values[:] = 0

        # find pairs
        pairs = self.find_cointegrating_pairs(**kwargs)

        # iterate over trading strategies and save positions
        for asset1, asset2 in pairs:
            tmp = self.pairs_trade(asset1, asset2, **kwargs)
            port[asset1] += tmp[asset1]
            port[asset2] += tmp[asset2]

        return port

    def portfolio_value(self):
        '''
        Get the value of portfolio: the optimal portfolio times asset prices.

        :return port_value:
        :rtype: :class:`pd.DataFrame`
        '''
        # output
        print('Calculating portfolio value')

        port_value = pd.DataFrame(
            self.optimal_portfolio() * self.data
        )
        return port_value

    def profits(self,
                total: bool = True,
                cost: float = 0):
        '''
        Get the profit of the optimal portfolio. Returns a pandas Series.

        :param total: ...
        :type total: bool
        :param cost: ...
        :type cost: float
        :rtype: :class:`pd.Series`
        '''
        # output
        print('Calculating profits')

        if total:
            profits = self.portfolio_value().cumsum(axis=0).sum(axis=1)
        else:
            profits = self.portfolio_value().cumsum(axis=0)

        # TODO: add transaction costs
        if cost > 0:
            pass

        return profits
