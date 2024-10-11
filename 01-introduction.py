# region imports
from AlgorithmImports import *

# endregion


class DeterminedFluorescentPinkViper(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2021, 1, 1)
        self.set_cash(100000)

        spy = self.add_equity("SPY", Resolution.MINUTE)
        spy.set_data_normalization_mode(DataNormalizationMode.RAW)

        self.spy = spy.Symbol

        self.SetBenchmark("SPY")
        self.set_brokerage_model(
            BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN
        )

        self.entryPrice = 0
        self.period = timedelta(31)
        self.nextEntryTime = self.Time

    def on_data(self, data: Slice):
        if not self.spy in data:
            return
        if data[self.spy] is None:
            return

        price = data[self.spy].Close

        if not self.portfolio.invested:
            if self.nextEntryTime <= self.Time:
                self.set_holdings(self.spy, 1)
                self.market_order(self.spy, int(self.portfolio.cash / price))
                self.log("Buy SPY @" + str(price))
                self.entryPrice = price
        elif self.entryPrice * 1.1 < price or self.entryPrice * 0.9 > price:
            self.liquidate()
            self.log("Sell SPY @" + str(price))
            self.nextEntryTime = self.Time + self.period
