import operator
import queue
from functools import partial, reduce
from typing import Dict, List, Set, TypeVar

from pandas import DataFrame, concat

# Create a generic variable that can be 'WeightedGraph', or any subclass.
WG = TypeVar('WG', bound='WeightedGraph')


# Sourced from:
# https://iprokin.github.io/posts/2017-12-24-find-best-deal-with-BFS.html
class WeightedGraph(object):
    def __init__(self, graph: Dict = {}, weights_func=lambda f, t: 1) -> None:
        self.get_weight = weights_func
        self._graph = graph

    def best_route(self, start_asset: str, goal_asset: str) -> List:
        """
        Provides the best path available in the graph

        ;param start_asset: First asset of the trade
        ;param goal_asset: Second asset of the trade
        """
        if start_asset in self._graph.keys() and goal_asset in self._graph.keys():
            paths = self._all_paths_bfs(start_asset, goal_asset)
            paths = self._sort_paths_by_reduced_weight(paths)
            return paths[0]
        else:
            return []

    def _sort_paths_by_reduced_weight(self, paths, how: operator = operator.mul) -> List:
        """
        Provides the sorted path available in the graph based on the operator provided

        ;param paths: List of paths
        ;param how: operator to use to weigh the paths
        """
        path_reducer = partial(self._walk_and_reduce, how=how)
        path_conversion = list(zip(paths, map(path_reducer, paths)))
        # sort by conversion coefficient
        path_conversion_sorted = sorted(path_conversion, key=lambda x: x[1], reverse=False)
        return path_conversion_sorted

    def _get_neighbours(self, node: str) -> List:
        return self._graph[node]

    def _walk_and_reduce(self, path, how=operator.add):
        """
        Applies the operator to compute the weight of the path

        ;param path: Path to analyze
        ;param how: operator to use to weigh the path
        """
        costs = map(self.get_weight, path[:-1], path[1:])
        return reduce(how, costs)

    def _all_paths_bfs(self, start_asset: str, goal_asset: str) -> List:
        """
        Provides a list of all paths between 2 assets

        ;param start_asset: First asset of the trade
        ;param goal_asset: Second asset of the trade
        """
        # https://www.geeksforgeeks.org/print-paths-given-source-destination-using-bfs/
        paths = queue.Queue()
        paths.put([start_asset])
        good_paths = []

        while not paths.empty():
            path = paths.get()

            current_asset = path[-1]
            if current_asset == goal_asset:
                good_paths.append(path)
            else:
                for next_asset in self._get_neighbours(current_asset):
                    if next_asset not in path:
                        paths.put(path + [next_asset])
        return good_paths


class LiteStrategyOrderRoute(object):

    def __init__(self, prices: Dict, valid_assets: Set = None) -> None:
        if valid_assets is None:
            self.valid_assets = {'BTC', 'USDT', 'ETH', 'USDC', 'DAI'}
        else:
            self.valid_assets = valid_assets

        # Filters out the trading pairs that are not a valid asset to trade
        p_is_valid_route = partial(self._is_valid_route, valid_assets=self.valid_assets)
        routes = {k: v for k, v in prices.items() if p_is_valid_route(k)}

        # Transforms the dict of prices into its graph representation and a DataFrame with the prices
        gh, df = self._prices_to_graph_df(routes)

        # Combines the graph and prices into a weighted graph
        self._weighted_routes = WeightedGraph(graph=gh, weights_func=partial(self._get_price_w_fees, df))

    def best_route(self, start_asset: str, goal_asset: str) -> List:
        """
        Provides the lower cost route between 2 assets in the form of list of intermediate assets

        ;param start_asset: First asset of the trade
        ;param goal_asset: Second asset of the trade
        """
        return self._weighted_routes.best_route(start_asset, goal_asset)

    @staticmethod
    def _is_valid_route(pair, valid_assets):
        """
        Boolean method validating that the trade pair is between valid assets
        """
        base, quote = pair.split('-')
        return base in valid_assets and quote in valid_assets

    @staticmethod
    def _prices_to_graph_df(prices: Dict) -> [Dict, DataFrame]:
        """
        Transfers prices into a DataFrame with 2 levels for columns with all routes and 1 row
        """
        edges = list(map(lambda k: k.split('-'), prices.keys()))
        graph = {}

        def add_base_quote(g, f, t):
            if f in g.keys():
                g[f].append(t)
            else:
                g[f] = [t]

        for edge in edges:
            add_base_quote(graph, edge[0], edge[1])
            add_base_quote(graph, edge[1], edge[0])

        df = DataFrame.from_dict(prices, orient='index').T
        return graph, concat({'ask': df, 'bid': df}).unstack(level=0)

    @staticmethod
    def _get_price_w_fees(df: DataFrame, f: str, t: str, fee: float = 0.2) -> float:
        """
        Get price of a trade. Used to weigh the graph
        """
        tf = f"{f}-{t}"
        ft = f"{t}-{f}"
        ss = set(df.columns.get_level_values(0))
        if ft in ss:  # I am buying
            p = 1.0 / float(df[ft]['ask'])
        elif tf in ss:  # I am selling
            p = float(df[tf]['bid'])
        else:
            raise ValueError()
        return p * (1.0 - fee)
