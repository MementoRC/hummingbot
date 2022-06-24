import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import ruamel.yaml

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.logger import HummingbotLogger
from hummingbot.user.user_balances import UserBalances

from .markets_yml_config import MarketsYmlConfig

yaml_parser = ruamel.yaml.YAML()
lsb_logger = None


class FundsBalancer(object):
    """
    Trying to get a better sense of balances and inventory in a common currency (USDT)
    """

    def logger(cls) -> HummingbotLogger:
        global lsb_logger
        if lsb_logger is None:
            lsb_logger = logging.getLogger(__name__)
        return lsb_logger

    @staticmethod
    def balancing_proposal(market_init: MarketsYmlConfig,
                           prices: Dict,
                           connectors: Union[Dict[str, ConnectorBase], ConnectorBase]) -> List[Union[Dict[str, Union[str, float]], Dict[str, Union[str, float]]]]:

        data: List[Any] = list()
        campaign_list = list()

        if isinstance(connectors, ConnectorBase):
            data += FundsBalancer._fetch_balances_prices_to_list(market_init, prices, connectors.name)
            campaign_list += FundsBalancer._reorganize_campaign_info(market_init, connectors.name)
        else:
            for exchange_name, connector in connectors.items():
                data += FundsBalancer._fetch_balances_prices_to_list(market_init, prices, exchange_name)
                campaign_list += FundsBalancer._reorganize_campaign_info(market_init, exchange_name)

        # Combine balances, prices, campaign info into DataFrame for calculations
        df = FundsBalancer._combine_assets_campaigns(data, campaign_list)

        # Calculate the funds allocations
        funds_df = FundsBalancer._calculate_funds(df)

        # Calculate the funds allocations
        df = df.groupby('Exchange').apply(FundsBalancer._calculate_asset_funding_per_exchange, 'Exchange', funds_df)

        # Calculate Base/Quote ratio from campaign information, inventory skews, HBOT weights
        df = FundsBalancer._calculate_base_quote_ratio(df)

        # Compose Sells proposal
        list_sells = FundsBalancer._compose_sells(df)
        return list_sells

    @staticmethod
    def _get_exchange_balances(connector: Optional[ConnectorBase] = None, exchange_name: Optional[str] = '') -> (
            Dict, Dict):
        # We prefer to pull the info from the connector if provided
        if connector:
            return [connector.get_all_balances(), connector.available_balances]
        elif exchange_name is not None:
            return [UserBalances.instance().all_balances(exchange_name),
                    UserBalances.instance().all_available_balances_all_exchanges()[exchange_name]]
        else:
            return dict(), dict()

    @staticmethod
    def _fetch_balances_prices_to_list(market_init: MarketsYmlConfig,
                                       prices: Dict,
                                       exchange_name: str) -> List:
        """
        Fetches assets balance & price for the assets in the strategy

        param exchange_name; Name of the exchange
        """
        data = list()
        for asset in market_init.get_assets_from_config(exchange_name):
            if asset != market_init.base_currency:
                pair = f"{asset}-{market_init.base_currency}"
                opposite_pair = f"{market_init.base_currency}-{asset}"
                if pair in prices[exchange_name] and prices[exchange_name][pair] > 0:
                    asset_price = prices[exchange_name][pair]
                elif opposite_pair in prices[exchange_name] and prices[exchange_name][opposite_pair]:
                    asset_price = 1 / prices[exchange_name][opposite_pair]
                else:
                    # We could add a route finder before giving up
                    FundsBalancer.logger().error("Ill-configured Lite Strategy with asset not traded in quote "
                                                 "currency (no price found)")
                    raise ValueError
            else:
                asset_price = 1

            total, avail = FundsBalancer._get_exchange_balances(exchange_name=exchange_name)

            total_bal = total[asset] if total is not None and asset in total else 0
            avail_bal = avail[asset] if avail is not None and asset in avail else 0

            data.append([exchange_name,
                         asset,
                         float(total_bal),
                         float(avail_bal),
                         float(asset_price),
                         float(asset_price) * float(total_bal)])
        return data

    @staticmethod
    def _reorganize_campaign_info(market_init: MarketsYmlConfig, exchange_name: str) -> List[Dict]:
        """
        Gathers campaign information combing balances and prices

        param exchange_name; Name of the exchange
        """
        campaign_list = list()

        campaigns_sc = market_init.get_campaigns_from_config(exchange_name)

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

                # HBOT campaign
                if 'hbot' in campaigns_sc and pair in campaigns_sc['hbot']:
                    item[f"{quote} Campaign"] *= market_init.hbot_weight

                # HBOT campaign weights can have additional weights to emphasize particular pairs
                if 'hbot_weights' in campaigns_sc and pair in campaigns_sc['hbot_weights']:
                    item[f"{quote} Campaign"] *= market_init.hbot_weight * campaigns_sc['hbot_weights'][pair]

                # Recording the inventory skew for the pair
                if 'inventory_skews' in campaigns_sc and pair in campaigns_sc['inventory_skews']:
                    item[f"{quote} Skew"] = campaigns_sc['inventory_skews'][pair]
                else:
                    item[f"{quote} Skew"] = market_init.inventory_skew

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
        quote_summary = balance_sheet.loc[balance_sheet['Asset'].isin(quote_funding.index.values)][
            ['Exchange', 'Asset', 'Capital (USDT)', 'Price (USDT)']].set_index('Asset')
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
    def _compose_sells(balance_sheet: pd.DataFrame) -> List[Union[Dict[str, Union[str, float]], Dict[str, Union[str, float]]]]:

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
            FundsBalancer.logger().error("There is an error in matching buys and sells")
            raise ValueError

        return orders
