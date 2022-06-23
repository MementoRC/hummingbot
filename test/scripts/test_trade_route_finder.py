import unittest
from decimal import Decimal
from functools import partial

from scripts.trade_route_finder import TradeRouteFinder, WeightedGraph


class TestTradeRouteFinder(unittest.TestCase):
    def setUp(self) -> None:
        self.prices = {'ALGO-USDT': Decimal('10'),
                       'AVAX-USDT': Decimal('10'),
                       'ALGO-AVAX': Decimal('1900'),
                       'BTC-ALGO': Decimal('30000'),
                       'BTC-ETH': Decimal('30000'),
                       'ETH-ADA': Decimal('30000'),
                       'ADA-USDT': Decimal('30000'),
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
        self.assertEqual(price, Decimal('10'))

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
        self.assertEqual(br, [])

    def test_best_route(self):
        valid_routes = {'BTC', 'USDT', 'ALGO', 'ETH', 'USDC', 'DAI'}
        routes_inst = TradeRouteFinder(self.prices, valid_routes)
        br = routes_inst.best_route('ALGO', 'USDT')
        self.assertEqual(br, (['ALGO', 'USDT'], 8.0))

    def test_wg_from_dict_and_func(self):
        gh, df = TradeRouteFinder._prices_to_graph_df(self.prices)
        part_df = partial(TradeRouteFinder._get_price_w_fees, df)
        graph = WeightedGraph(graph=gh, weights_func=part_df)
        self.assertEqual(graph._graph, gh)
        self.assertEqual(graph.get_weight, part_df)

    def test_wg__walk_and_reduce(self):
        expected = {'ADA': ['ETH', 'USDT'],
                    'ALGO': ['USDT', 'AVAX', 'BTC'],
                    'AVAX': ['USDT', 'ALGO'],
                    'BTC': ['ALGO', 'ETH'],
                    'ETH': ['BTC', 'ADA'],
                    'USDT': ['ALGO', 'AVAX', 'ADA'],
                    }
        gh, df = TradeRouteFinder._prices_to_graph_df(self.prices)
        graph = WeightedGraph(graph=gh, weights_func=partial(TradeRouteFinder._get_price_w_fees, df))
        graph._walk_and_reduce(['ALGO', 'USDT'])
        self.assertEqual(graph._graph, expected)

    def test_wg__all_paths_bfs(self):
        expected = [['ALGO', 'USDT'], ['ALGO', 'AVAX', 'USDT'], ['ALGO', 'BTC', 'ETH', 'ADA', 'USDT']]
        gh, df = TradeRouteFinder._prices_to_graph_df(self.prices)
        graph = WeightedGraph(graph=gh, weights_func=partial(TradeRouteFinder._get_price_w_fees, df))
        paths = graph._all_paths_bfs("ALGO", "USDT")
        self.assertEqual(paths, expected)

    def test_wg__sort_paths_by_reduced_weight(self):
        expected = [(['ALGO', 'USDT'], 8.0),
                    (['ALGO', 'AVAX', 'USDT'], 12160.0),
                    (['ALGO', 'BTC', 'ETH', 'ADA', 'USDT'], 368640000.00000006)]
        gh, df = TradeRouteFinder._prices_to_graph_df(self.prices)
        graph = WeightedGraph(graph=gh, weights_func=partial(TradeRouteFinder._get_price_w_fees, df))
        paths = graph._all_paths_bfs("ALGO", "USDT")
        paths = graph._sort_paths_by_reduced_weight(paths)
        self.assertEqual(paths, expected)

    def test_wg_best_route(self):
        expected = (['ALGO', 'USDT'], 8.0)
        gh, df = TradeRouteFinder._prices_to_graph_df(self.prices)
        graph = WeightedGraph(graph=gh, weights_func=partial(TradeRouteFinder._get_price_w_fees, df))
        paths = graph.best_route("ALGO", "USDT")
        self.assertEqual(paths, expected)
