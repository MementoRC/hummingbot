import unittest
from decimal import Decimal
from functools import partial

from scripts.trade_route_finder import TradeRouteFinder, WeightedGraph


class TestTradeRouteFinder(unittest.TestCase):
    def setUp(self) -> None:
        self.prices = {'ALGO-USDT': Decimal('1.01'),
                       'AVAX-USDT': Decimal('1.01'),
                       'ALGO-AVAX': Decimal('1.01'),
                       'BTC-ALGO': Decimal('10000'),
                       'BTC-ETH': Decimal('11'),
                       'ETH-ADA': Decimal('1000'),
                       'ADA-USDT': Decimal('1'),
                       }
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test__prices_to_graph_df(self):
        expected = {'ADA': ['ETH', 'USDT'],
                    'ALGO': ['USDT', 'AVAX', 'BTC'],
                    'AVAX': ['USDT', 'ALGO'],
                    'BTC': ['ALGO', 'ETH'],
                    'ETH': ['BTC', 'ADA'],
                    'USDT': ['ALGO', 'AVAX', 'ADA']}
        graph, df = TradeRouteFinder._prices_to_graph_df(self.prices)
        self.assertEqual(graph, expected)

    def test__get_price_w_fees(self):
        graph, df = TradeRouteFinder._prices_to_graph_df(self.prices)

        price = TradeRouteFinder._get_price_w_fees(df, "ALGO", "USDT", fee=0)
        self.assertEqual(price, (Decimal('1.01'), 'ALGO-USDT', 'buy_quote'))

    def test__is_valid_route(self):
        valid_routes = {'BTC', 'USDT', 'ETH', 'USDC', 'DAI'}
        valid = TradeRouteFinder._is_valid_route('BTC-USDT', valid_routes)
        invalid = TradeRouteFinder._is_valid_route('BTC-ADA', valid_routes)
        self.assertEqual(valid, True)
        self.assertEqual(invalid, False)

    def test_best_route_no_asset(self):
        valid_routes = {'BTC', 'USDT', 'ETH', 'USDC', 'DAI'}
        routes_inst = TradeRouteFinder(self.prices, valid_routes)
        br = routes_inst.best_route('ALGO', 'USDT')
        self.assertEqual(br, ([], {}))

    def test_best_route(self):
        valid_routes = {'BTC', 'USDT', 'ALGO', 'ETH', 'USDC', 'DAI'}
        routes_inst = TradeRouteFinder(self.prices, valid_routes)
        br, t = routes_inst.best_route('ALGO', 'USDT')
        # Fee defaulted to 0 gives 10, if set to 0.2 gives 8
        self.assertEqual(br, (['ALGO', 'USDT'], 10.0))

    def test_wg_from_dict_and_func(self):
        expected = {'ADA': ['ETH', 'USDT'],
                    'ALGO': ['USDT', 'AVAX', 'BTC'],
                    'AVAX': ['USDT', 'ALGO'],
                    'BTC': ['ALGO', 'ETH'],
                    'ETH': ['BTC', 'ADA'],
                    'USDT': ['ALGO', 'AVAX', 'ADA'],
                    }
        gh, df = TradeRouteFinder._prices_to_graph_df(self.prices)
        part_df = partial(TradeRouteFinder._get_price_w_fees, df, fee=0)
        graph = WeightedGraph(graph=gh, weights_func=part_df)
        self.assertEqual(graph._graph, gh)
        self.assertEqual(graph.get_weight, part_df)
        self.assertEqual(graph._graph, expected)

    def test_wg__walk_and_reduce(self):
        gh, df = TradeRouteFinder._prices_to_graph_df(self.prices)
        part_df = partial(TradeRouteFinder._get_price_w_fees, df, fee=0)
        graph = WeightedGraph(graph=gh, weights_func=part_df)
        expected = part_df('ALGO', 'BTC')[0] + part_df('BTC', 'ETH')[0] + part_df('ETH', 'ADA')[0] + part_df('ADA', 'USDT')[0]

        weight_addition, details = graph._walk_and_reduce(['ALGO', 'BTC', 'ETH', 'ADA', 'USDT'])
        self.assertEqual(weight_addition, expected)
        self.assertEqual(len(details['rate']), len(['ALGO', 'BTC', 'ETH', 'ADA', 'USDT']) - 1)
        self.assertEqual(details['pair'][1], 'BTC-ETH')
        self.assertEqual(details['order'][1], 'buy_quote')

    def test_wg__all_paths_bfs(self):
        expected = [['ALGO', 'USDT'], ['ALGO', 'AVAX', 'USDT'], ['ALGO', 'BTC', 'ETH', 'ADA', 'USDT']]
        gh, df = TradeRouteFinder._prices_to_graph_df(self.prices)
        part_df = partial(TradeRouteFinder._get_price_w_fees, df, fee=0)
        graph = WeightedGraph(graph=gh, weights_func=part_df)
        paths = graph._all_paths_bfs("ALGO", "USDT")
        self.assertEqual(paths, expected)

    def test_wg__sort_paths_by_reduced_weight(self):
        expected = [(['ALGO', 'BTC', 'ETH', 'ADA', 'USDT'], Decimal("1.1")),
                    (['ALGO', 'AVAX', 'USDT'], Decimal("1.0201")),
                    (['ALGO', 'USDT'], Decimal("1.01"))]
        gh, df = TradeRouteFinder._prices_to_graph_df(self.prices)
        part_df = partial(TradeRouteFinder._get_price_w_fees, df, fee=0)
        graph = WeightedGraph(graph=gh, weights_func=part_df)
        paths = graph._all_paths_bfs("ALGO", "USDT")
        paths = graph._sort_paths_by_reduced_weight(paths)
        self.assertEqual(paths[0][:2], expected[0])

    def test_wg_best_route(self):
        expected = (['ALGO', 'BTC', 'ETH', 'ADA', 'USDT'], Decimal("1.1"),
                    {'pairs': ('BTC-ALGO', 'BTC-ETH', 'ETH-ADA', 'ADA-USDT'),
                     'orders': ('sell_quote', 'buy_quote', 'buy_quote', 'buy_quote'),
                     'rates': (Decimal("0.0001"), Decimal("11"), Decimal("1000"), Decimal("1"))})
        gh, df = TradeRouteFinder._prices_to_graph_df(self.prices)
        part_df = partial(TradeRouteFinder._get_price_w_fees, df, fee=0)
        graph = WeightedGraph(graph=gh, weights_func=part_df)
        paths = graph.best_route("ALGO", "USDT")
        self.assertEqual(paths, expected)

    def test_wg_best_route_w_fee(self):
        fee = 0.001
        fee_impact = Decimal("1") - Decimal(fee)
        expected = (['ALGO', 'BTC', 'ETH', 'ADA', 'USDT'], Decimal("1.1") * fee_impact**4,
                    {'pairs': ('BTC-ALGO', 'BTC-ETH', 'ETH-ADA', 'ADA-USDT'),
                     'orders': ('sell_quote', 'buy_quote', 'buy_quote', 'buy_quote'),
                     'rates': (Decimal("0.0001") * fee_impact, Decimal("11") * fee_impact, Decimal("1000") * fee_impact, Decimal("1") * fee_impact)})
        gh, df = TradeRouteFinder._prices_to_graph_df(self.prices)
        part_df = partial(TradeRouteFinder._get_price_w_fees, df, fee=0.001)
        graph = WeightedGraph(graph=gh, weights_func=part_df)
        paths = graph.best_route("ALGO", "USDT")
        self.assertEqual(paths, expected)
