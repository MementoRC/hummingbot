import asyncio
import os
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import ruamel.yaml

from hummingbot.client.settings import CONF_FILE_PATH, CONF_PREFIX
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.rate_oracle.rate_oracle import RateOracle
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.user.user_balances import UserBalances

from .ls_markets_init import LiteStrategyMarketsInit

yaml_parser = ruamel.yaml.YAML()
lsb_logger = None


class LiteStrategyFundAllocation(ScriptStrategyBase, LiteStrategyMarketsInit):
    """
    Trying to get a better sense of balances and inventory in a common currency (USDT)
    """

    config_filename: str = CONF_FILE_PATH + CONF_PREFIX + os.path.split(__file__)[1].split('.')[0] + ".yml"

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        #: Define markets to instruct Hummingbot to create connectors on the exchanges and markets you need
        super().__init__(connectors)

        self._last_async_refresh_ts = None
        self._prices = dict()
        self._data_ready = False
        self._balance_fut, self._pause_fut, self._prices_fut = None, None, dict()

    def stop(self, clock: Clock):
        pass

    def tick(self, timestamp: float):
        """
        Clock tick entry point, is run every second (on normal tick setting).
        Checks if all connectors are ready, if so the strategy is ready to trade.

        :param timestamp: current tick timestamp
        """
        self.logger().info("Entering tick()")
        if not self.ready_to_trade:
            self.ready_to_trade = all(ex.ready for ex in self.connectors.values())

            if not self.ready_to_trade:
                for con in [c for c in self.connectors.values() if not c.ready]:
                    self.logger().warning(f"{con.name} is not ready. Please wait...")
                return
        else:
            if self._last_async_refresh_ts < (self.current_timestamp - self._async_refresh):
                self._refresh_balances_prices()
                self._last_async_refresh_ts = self.current_timestamp

            if self._data_ready:
                self.on_tick()
            else:
                self.logger().warning("Strategy is not ready. Please wait...")
                return

    def on_tick(self):
        pass

    def get_strategy_exchange_assets(self, exchange: str) -> Set[str]:
        if exchange in self._config['markets']:
            return set(self._config['markets'][exchange]['assets'])
        else:
            return set()

    def get_balance_df(self) -> pd.DataFrame:
        """
        Returns a data frame for all asset balances for displaying purpose.
        """
        data: List[Any] = list()
        campaign_list = list()

        for exchange_name, connector in self.connectors.items():
            data += self._fetch_balances_prices(exchange_name)
            campaign_list += self._reorganize_campaign_info(exchange_name)

        # Combine balances, prices, campaign info into DataFrame for calculations
        df = self._combine_assets_campaigns(data, campaign_list)

        # Calculate the funds allocations
        funds_df = self._calculate_funds(df)

        df.groupby('Exchange').apply(self._calculate_asset_funding_per_exchange, 'Exchange', funds_df)

        # funds = campaigns.groupby(group.columns.str.extract("\w+ Funds$", expand=False), axis=1)
        # skew = group.groupby(group.columns.str.extract("\w+ Skew", expand=True), axis=1)
        # print(skew)
        # skewed = funds * skew
        # print(skewed)
        return df

    def format_status(self) -> str:
        """
            Returns status of the current strategy on user balances and current active orders. This function is called
            when status command is issued. Override this function to create custom status display output.
            """
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        lines = []
        csv = []
        warning_lines = []
        warning_lines.extend(self.network_warning(self.get_market_trading_pair_tuples()))

        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])
        csv.extend([line for line in balance_df.to_csv(index=True).split("\n")])

        try:
            df = self.active_orders_df()
            lines.extend(["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "  No active maker orders."])

        warning_lines.extend(self.balance_warning(self.get_market_trading_pair_tuples()))
        if len(warning_lines) > 0:
            lines.extend(["", "*** WARNINGS ***"] + warning_lines)
        return "\n".join(lines)

    @staticmethod
    def get_exchange_balances(connector: Optional[ConnectorBase] = None, exchange_name: Optional[str] = '') -> (
            Dict, Dict):
        # We prefer to pull the info from the connector if provided
        if connector:
            return [connector.get_all_balances(), connector.available_balances]
        elif exchange_name is not None:
            return [UserBalances.instance().all_balances(exchange_name),
                    UserBalances.instance().all_available_balances(exchange_name)]
        else:
            return dict(), dict()

    def _refresh_balances_prices(self) -> None:
        """
        Calls async methods for all balance & price
        """
        self._data_ready = False
        loop = asyncio.get_event_loop()
        # We need balances to be updated and prices for both exchange (rather than use the oracle)
        # Submit to the main Event loop  - We get the result after a few ticks and wait till then
        if self._pause_fut is None:
            for exchange_name, connector in self.connectors.items():
                self._balance_fut[exchange_name] = asyncio.run_coroutine_threadsafe(
                    UserBalances.instance().update_exchange_balance(exchange_name), loop)
                if exchange_name == 'kucoin':
                    self._prices_fut['kucoin'] = asyncio.run_coroutine_threadsafe(RateOracle.get_kucoin_prices(), loop)
                elif exchange_name == 'gate_io':
                    self._prices_fut['gate_io'] = asyncio.run_coroutine_threadsafe(RateOracle.get_gate_io_prices(), loop)
            self._pause_fut = asyncio.run_coroutine_threadsafe(asyncio.sleep(0.5))

        if all([[self._balance_fut[c].done(), self._prices_fut[c].done(), self._prices_fut[c].done()] for c in self.connectors.keys()]):
            self._prices['kucoin'] = self._prices_fut['kucoin'].result()
            self._prices['gate_io'] = self._prices_fut['gate_io'].result()
            self._data_ready = True

    def _fetch_balances_prices(self, exchange_name: str) -> List:
        """
        Fetches assets balance & price for the assets in the strategy

        param exchange_name; Name of the exchange
        """
        data = list()
        total_bal, avail_bal = self.get_exchange_balances(exchange_name=exchange_name)

        for asset in self.get_strategy_exchange_assets(exchange_name):
            if asset != self._config['base_currency']:
                pair = f"{asset}-{self._config['base_currency']}"
                if pair in self._prices[exchange_name] and self._prices[exchange_name][pair] > 0:
                    asset_price = self._prices[exchange_name][pair]
                else:
                    asset_price = 1e-6
            else:
                asset_price = 1

            total_bal[asset] = total_bal[asset] if asset in total_bal else 0
            avail_bal[asset] = avail_bal[asset] if asset in avail_bal else 0

            data.append([exchange_name,
                         asset,
                         float(total_bal[asset]),
                         float(avail_bal[asset]),
                         float(asset_price),
                         float(asset_price) * float(total_bal[asset])])
        return data

    def _reorganize_campaign_info(self, exchange_name: str) -> List[Dict]:
        """
        Gathers campaign information combing balances and prices

        param exchange_name; Name of the exchange
        """
        campaign_list = list()

        campaigns_sc = self._config['markets'][exchange_name]

        for quote in campaigns_sc['quotes']:
            it = iter(campaigns_sc['quotes'][quote])
            for asset in it:
                item = dict()

                # Update existing asset record
                for d in campaign_list:
                    if d['Exchange'] == exchange_name and d['Asset'] == asset:
                        item = d

                # No existing entry for this asset, create one
                if not item:
                    item = dict(Exchange=exchange_name, Asset=asset)
                    campaign_list.append(item)

                # Update the entry
                pair = f"{asset}-{quote}"
                item[f"{quote} Campaign"] = 1

                # Computing the weighting
                if 'weights' in campaigns_sc and pair in campaigns_sc['weights']:
                    item[f"{quote} Campaign"] *= campaigns_sc['weights'][pair]

                # HBOT campaign weights
                if 'hbot_weights' in campaigns_sc and pair in campaigns_sc['hbot_weights']:
                    item[f"{quote} Campaign"] *= campaigns_sc['hbot_weights'][pair]
                else:
                    item[f"{quote} Campaign"] *= self._config['hbot_weight']

                # Recording the inventory skew for the pair
                if 'inventory_skews' in campaigns_sc and pair in campaigns_sc['inventory_skews']:
                    item[f"{quote} Skew"] = campaigns_sc['inventory_skews'][pair]
                else:
                    item[f"{quote} Skew"] = self._config["inventory_skew"]

        return campaign_list

    @staticmethod
    def _combine_assets_campaigns(data: List[List[Any]], campaigns: List[Dict[str, Any]]) -> pd.DataFrame:
        columns: List[str] = \
            ["Exchange", "Asset", "Balance", "Available Balance", "Price (USDT)", "Capital (USDT)"]
        df = pd.DataFrame(data=data, columns=columns).replace(np.nan, '', regex=True)
        df = df.merge(pd.DataFrame(campaigns), how='outer').fillna(0)
        return df

    @staticmethod
    def _calculate_funds(data: pd.DataFrame) -> pd.DataFrame:

        # Select capital and campaigns
        funds_df = data.filter(regex='(^Exchange$|Capital.+|.+Campaign$)')

        # For each exchange, sum the campaigns (arranged per quotes) - reset the index to numbers
        funds_df = funds_df.groupby('Exchange').sum().reset_index()

        # Total the number of campaigns for normalized weigh
        funds_df['Total Campaigns'] = funds_df.filter(regex=r"\w+ Campaign$").sum(axis=1)

        # Base allocation of funds in USDT is the capital on the exchange divided by the total campaings
        funds_df['Unit Funding Base+Quote (USDT)'] = \
            funds_df['Capital (USDT)'] / funds_df['Total Campaigns']

        # Funds allocation per quote campaign, named 'Allocation'
        camp_funds_df = \
            funds_df.filter(regex=r"\w+ Campaign$").multiply(funds_df['Unit Funding Base+Quote (USDT)'], axis="index")
        camp_funds_df.columns = \
            camp_funds_df.columns.str.replace(pat='Campaign', repl='Allocation (USDT)')

        # Concatenate the allocations to the funds_df
        funds_df = pd.concat([funds_df, camp_funds_df], axis=1)

        return funds_df

    @staticmethod
    def _calculate_asset_funding_per_exchange(group: pd.DataFrame, grouping: str,
                                              funds_df: pd.DataFrame) -> pd.DataFrame:
        # Exchange selected by groupby
        selection = group[grouping].iat[0]

        # Reset the indexing for easy access to funds data
        funds_df_index = funds_df.set_index(grouping)

        # Unit funding for the exchange selected
        funds = funds_df_index.loc[selection]['Unit Funding Base+Quote (USDT)']

        # Calculate for each campaign in each quote
        campaigns = group.filter(regex=r"\w+ Campaign$") * funds
        campaigns.columns = campaigns.columns.str.replace(pat=r'(\w+) Campaign',
                                                          repl=lambda m: "Total " + m.group(1) + ' Funds (USDT)',
                                                          regex=True)

        # Total funds for every asset across campaigns
        group['Total Funding (USDT)'] = campaigns.sum(axis=1)

        # Funding gap wrt Capital in USDT
        group['Total Funding Gap (USDT)'] = group['Total Funding (USDT)'] - group['Capital (USDT)']

        return pd.concat([group, campaigns], axis=1, join='inner')

    @staticmethod
    def _calculate_base_quote_ratio(balance_sheet: pd.DataFrame) -> pd.DataFrame:
        # Apply inventory Skew
        # Filter the campaign funds
        funds = balance_sheet.filter(regex=r'^Total \S+ Funds \(USDT\)', axis=1)
        # Transform the columns title into the quote name
        funds.columns = funds.columns.str.rsplit(pat=' ', n=-1, expand=True).get_level_values(1)

        # Filter the Skew percent
        percent = balance_sheet.filter(regex=r'^\w+ Skew', axis=1) / 100
        # Transform the columns title into the quote name
        percent.columns = percent.columns.str.rsplit(pat=' Skew', n=1, expand=True).get_level_values(0)

        # Calculate the base/quote fund, set column title
        base_funds = funds.mul(percent).add_prefix('Base ').add_suffix(' Funds (USDT)')
        base_funds['Base Funding (USDT)'] = base_funds.sum(axis=1)
        quote_funds = funds.mul(1 - percent).add_prefix('Quote ').add_suffix(' Funds (USDT)')

        # Update the balance sheet
        balance_sheet = pd.concat([balance_sheet, base_funds, quote_funds], axis=1)

        # Calculate Asset Gap = Capital - sum(Base Funds per campaign)
        balance_sheet['Funding Gap (USDT)'] = balance_sheet['Base Funding (USDT)'] - balance_sheet['Capital (USDT)']
        balance_sheet['Funding Gap (asset)'] = balance_sheet['Funding Gap (USDT)'] / balance_sheet['Price (USDT)']

        # Examining the quote side
        quote_funding = balance_sheet.filter(regex=r'^Quote \w+ Funds \(USDT\)$', axis=1).sum()
        quote_funding.index = quote_funding.index.str.rsplit(pat=' ', expand=True).get_level_values(1)

        quote_funding = pd.DataFrame({'Quote Funding (USDT)': quote_funding}).rename_axis("Asset")
        quote_summary = balance_sheet.loc[balance_sheet['Asset'].isin(quote_funding.index.values)][['Exchange', 'Asset', 'Capital (USDT)', 'Price (USDT)']].set_index('Asset')
        quote_funding['Funding Gap (USDT)'] = quote_funding['Quote Funding (USDT)'] - quote_summary['Capital (USDT)']
        quote_funding['Funding Gap (asset)'] = quote_funding['Funding Gap (USDT)'] / quote_summary['Price (USDT)']

        # Update balance sheet, merge the Funding Gap
        balance_sheet = pd.merge(balance_sheet, quote_funding, on='Asset', how='left', suffixes=('', '_update'))
        replace_with_this = balance_sheet.loc[balance_sheet['Funding Gap (USDT)_update'].notnull(),
                                              ['Funding Gap (USDT)_update', 'Funding Gap (asset)_update']]
        balance_sheet.loc[balance_sheet['Funding Gap (USDT)_update'].notnull(),
                          ['Funding Gap (USDT)', 'Funding Gap (asset)']] = replace_with_this.values
        balance_sheet.drop(['Funding Gap (USDT)_update', 'Funding Gap (asset)_update'], axis=1, inplace=True)

        return balance_sheet

    @staticmethod
    def _implement_buys_sells(balance_sheet: pd.DataFrame) -> pd.DataFrame:
        # Separate Buys and Sells
        sells = balance_sheet[balance_sheet['Funding Gap (USDT)'] >= 0][
            ['Exchange', 'Asset', 'Price (USDT)', 'Funding Gap (USDT)']].T.to_dict('dict')
        buys = balance_sheet[balance_sheet['Funding Gap (USDT)'] < 0][
            ['Exchange', 'Asset', 'Price (USDT)', 'Funding Gap (USDT)']].T.to_dict('dict')

        sells = dict(sorted(sells.items(), key=lambda item: item[1]['Funding Gap (USDT)'], reverse=True))
        buys = dict(sorted(buys.items(), key=lambda item: item[1]['Funding Gap (USDT)']))

        orders = list()
        while True:

            f_sell = list(sells.values())[0]
            f_buy = list(buys.values())[0]

            if float(f_sell['Funding Gap (USDT)']) > abs(float(f_buy['Funding Gap (USDT)'])):
                orders.append(dict(asset=str(f_buy['Asset']),
                                   to=str(f_sell['Asset']),
                                   amount=- float(f_buy['Funding Gap (USDT)'])))
                f_sell['Funding Gap (USDT)'] += f_buy['Funding Gap (USDT)']
                f_buy['Funding Gap (USDT)'] -= f_buy['Funding Gap (USDT)']
            else:
                orders.append(dict(asset=str(f_buy['Asset']),
                                   to=str(f_sell['Asset']),
                                   amount=float(f_sell['Funding Gap (USDT)'])))
                f_buy['Funding Gap (USDT)'] += f_sell['Funding Gap (USDT)']
                f_sell['Funding Gap (USDT)'] -= f_sell['Funding Gap (USDT)']

            # Stopping criteria, one of sells or buys is empty
            sells = dict(sorted(sells.items(), key=lambda item: item[1]['Funding Gap (USDT)'], reverse=True))
            buys = dict(sorted(buys.items(), key=lambda item: item[1]['Funding Gap (USDT)']))

            if list(sells.values())[0]['Funding Gap (USDT)'] < 0.1 or list(sells.values())[0]['Funding Gap (USDT)'] < 0.1:
                # Normally both equal 0, but just in case, let's allow some slack and numerical inaccuracy
                break

        # Both sells and buys are sorted and their largest value should be numerical noise
        if list(sells.values())[0]['Funding Gap (USDT)'] >= 1 or list(sells.values())[0]['Funding Gap (USDT)'] > 1:
            # This is not supposed to happen - There's a bug in the code somewhere :(
            LiteStrategyFundAllocation.logger().error("There is an error in matching buys and sells")
            raise ValueError

        return orders
