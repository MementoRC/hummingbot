import requests

from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

API_MINER_IO = "api.hummingbot.io"
MINER_IO = "miner.hummingbot.io"
CLIENT_ID = "IZ54MXFWYUYVDEMFY87R9UO65ZFCGI6D"

headers = {
    'authority': API_MINER_IO,
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'en-US,en;q=0.9',
    'authorization': 'Bearer <eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHBpcmF0aW9uIjoxNjQzMzI0ODA3LCJjbGllbnRfaWQiOiJJWjU0TVhGV1lVWVZERU1GWTg3UjlVTzY1WkZDR0k2RCJ9.bDPWenkNB5CaGzR7qPt5VW25V42BDpAfhiXgq8JwVtQ>',
    'origin': MINER_IO,
    'referer': MINER_IO,
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'sec-gpc': '1',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
    'x-device-id': 'zSM1HArcEJCIdWCzWtbPZa'
}

requests_cmds = {
    'markets': {'url': '/bounty/markets',
                'params': ()},
    'leaderboard': {'url': '/bounty/markets',
                    'params': ('start', 'market_id')
                    }
}

auth_requests_cmds = {
    'private': {
        'request': 'GET',
        'url': '/bounty/user/private',
        'params': ('client_id', 'chart_interval')},
    'recent_orders': {
        'request': 'GET',
        'url': '/bounty/user/recent_orders',
        'params': ('client_id', 'start', 'stop')},
    'single_snapshot': {
        'request': 'GET',
        'url': '/bounty/user/single_snapshot',
        'params': ('client_id', 'market_id', 'timestamp', 'aggregation_period')},
    'exchanges_and_wallets': {
        'request': 'GET',
        'url': '/bounty/user/private/exchanges_and_wallets',
        'params': 'client_id'},
    'order_book': {
        'request': 'OPTIONS',
        'url': '/bounty/charts/order_book',
        'params': ('start_from', 'market_id', 'aggregation_period', 'decimals', 'client_id')
    }
}


def request_miner_data():
    url = f"https://{API_MINER_IO}/bounty/markets"
    payload = {}
    return requests.request("GET", url, headers=headers, data=payload)


class ExampleMinerIO(ScriptStrategyBase):
    """
    Testing a reload method for Script Strategies
    """

    def format_status(self) -> str:
        """
        Returns status of the current strategy on user balances and current active orders. This function is called
        when status command is issued. Override this function to create custom status display output.
        """
        self.logger().info(f"{request_miner_data()}")
