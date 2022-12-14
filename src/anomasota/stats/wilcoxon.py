
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from pandas import DataFrame
from scipy import stats as spstats


@dataclass
class WilcoxonHolmResult:
    
    FRIEDMAN_PVALUE_ASSUMPTION = "chisquare"
    
    @dataclass
    class Parameters:
        alpha: float
        
    @dataclass
    class Metadata:
        num_datasets: int = field(init=False)
        num_methods: int = field(init=False)
        metric_name: str
        higher_is_better: bool
        datasets: List[str] = field(repr=False)
        methods: List[str] = field(repr=False)
        
        def __post_init__(self):
            self.num_datasets = len(self.datasets)
            self.num_methods = len(self.methods)
        
    @dataclass
    class PairwiseResult:
        statistic: float
        pvalue: float
        h0_is_rejected: bool
        
        pvalues_argsort_idx: int 
        corrected_alpha: float
    
    parameters: Parameters
    metadata: Metadata
    
    friedman_statistic: float = field(repr=False)
    friedman_pvalue: float = field(repr=False)
    friedman_h0_is_rejected: bool = field(repr=False)
    
    parwise_tests: Dict[Tuple[int, int], PairwiseResult] = field(repr=False)
    
    rankdata: np.ndarray = field(repr=False)
    rankavg: np.ndarray = field(repr=False)
    cliques_insignificant_difference: List[Tuple[int, ...]] = field(repr=False)

    pvalues_matrix: np.ndarray = field(repr=False, init=False)
    alphas_matrix: np.ndarray = field(repr=False, init=False)
    
    def __post_init__(self):
        self.rankavg = self.rankdata.mean(axis=0)
        self.pvalues_matrix = np.zeros((self.metadata.num_methods, self.metadata.num_methods))
        self.alphas_matrix = np.zeros((self.metadata.num_methods, self.metadata.num_methods))
        for (i, j), pairwise_result in self.parwise_tests.items():
            self.pvalues_matrix[i, j] = pairwise_result.pvalue
            self.pvalues_matrix[j, i] = pairwise_result.pvalue
            self.alphas_matrix[i, j] = pairwise_result.corrected_alpha
            self.alphas_matrix[j, i] = pairwise_result.corrected_alpha
        self._validate()
        
    def _validate(self):
        
        # metadata assumptions
        assert self.metadata.num_datasets > 1, f"num_datasets must be > 1, got {self.metadata.num_datasets=}"
        assert self.metadata.num_methods > 1, f"num_methods must be > 1, got {self.metadata.num_methods=}"
        assert self.metadata.metric_name, f"metric_name must not be empty, got {self.metadata.metric_name=}"
        
        method_method_matrix_shape = (self.metadata.num_methods, self.metadata.num_methods)
        rankdata_shape = (self.metadata.num_datasets, self.metadata.num_methods)
        expected_num_pairwise_tests = self.metadata.num_methods * (self.metadata.num_methods - 1) // 2
        
        # alpha / pvalue assumptions
        assert 0 < self.parameters.alpha < 1, f"alpha must be \\in (0, 1), got {self.parameters.alpha=}"
        assert 0 <= self.friedman_pvalue <= 1, f"pvalue must be \\in [0, 1], got {self.friedman_pvalue=}"
        
        assert self.pvalues_matrix.shape == method_method_matrix_shape, f"pvalues_matrix must have shape (num_methods, num_methods)=({method_method_matrix_shape}), got {self.pvalues_matrix.shape=}"
        assert ((self.pvalues_matrix >= 0) & (self.pvalues_matrix <= 1)).all(), f"pvalues_matrix must be \\in [0, self.parameters.alpha={self.parameters.alpha}], got {self.pvalues_matrix=}"
        assert (self.pvalues_matrix == self.pvalues_matrix.T).all(), f"pvalues_matrix must be symmetric, got {self.pvalues_matrix=}"
        
        assert self.alphas_matrix.shape == method_method_matrix_shape, f"alphas_matrix must have shape (num_methods, num_methods)=({method_method_matrix_shape}), got {self.alphas_matrix.shape=}"
        assert ((self.alphas_matrix >= 0) & (self.alphas_matrix <= self.parameters.alpha)).all(), f"alphas_matrix must be \\in [0, self.parameters.alpha={self.parameters.alpha}], got {self.alphas_matrix=}"
        assert (self.alphas_matrix == self.alphas_matrix.T).all(), f"alphas_matrix must be symmetric, got {self.alphas_matrix=}"
        
        # rank data assumptions
        assert self.rankdata.shape == rankdata_shape, f"rankdata must have shape (num_datasets, num_methods)=({rankdata_shape}), got {self.rankdata.shape=}"
        
        ranksum_perdataset = self.rankdata.sum(axis=1)
        sum_ranks_perdataset = self.metadata.num_methods * (self.metadata.num_methods + 1) / 2
        assert (ranksum_perdataset == sum_ranks_perdataset).all(), f"rankdata is inconsistent, got {ranksum_perdataset=}"
        
        assert self.rankavg.shape == (self.metadata.num_methods,), f"rankavg must have shape (num_methods,)=({self.num_methods},), got {self.rankavg.shape=}"
        assert (self.rankavg == self.rankdata.mean(axis=0)).all(), f"rankavg is inconsistent, got {self.rankavg=}"
        
        # pairwise tests assumptions
        assert len(self.parwise_tests) == expected_num_pairwise_tests, f"parwise_tests must have len == num_methods * (num_methods - 1) // 2, got {len(self.parwise_tests)=} and {expected_num_pairwise_tests=}"
       
        for cliq in self.cliques_insignificant_difference:
            assert 1 < len(cliq) <= self.metadata.num_methods, f"cliques_insignificant_difference must have len \\in [2, num_methods], got {cliq=}, {self.metadata.num_methods=}" 
            assert all(0 <= i < self.metadata.num_methods for i in cliq), f"cliques_insignificant_difference must have elements \\in [0, num_methods), got {cliq=}, {self.metadata.num_methods=}"
        

def wilcoxon_holm(dataset_method_df: DataFrame, alpha: float, higher_is_better: bool) -> WilcoxonHolmResult:
    """
    Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm
    to reject the null's hypothesis
    """
    
    dataset_method_df = dataset_method_df.copy()  # avoid bugs due to inplace modifications

    # ================================ VALIDATIONS ================================
        
    assert 0 < alpha < 1, f"alpha must be \\in (0, 1), got {alpha=}"
    
    num_columns = dataset_method_df.shape[1]
    assert num_columns == 3, f"dataset_method_df must have 3 columns, got {dataset_method_df.columns=}"
        
    EXPECTED_COLUMNS = {"dataset", "method",}
    columns: Set[str] = set(dataset_method_df.columns)
        
    assert columns.issuperset(EXPECTED_COLUMNS), f"dataset_method_df must have columns {sorted(EXPECTED_COLUMNS)}, got {sorted(columns)}"
    
    # deduce the metric name from the third column name
    metric_name = next(iter(columns - EXPECTED_COLUMNS))
    metric_dtype = dataset_method_df[metric_name].dtype

    assert metric_dtype == np.float64, f"dataset_method_df metric column ({metric_name=}) must be float64, got {metric_dtype=}"
    
    # if not higher_is_better:
    if higher_is_better:
        dataset_method_df.loc[:, metric_name] = (-1) * dataset_method_df.loc[:, metric_name]
    
    # all methods must be present in all datasets
    datasetcount_permethod = dataset_method_df.groupby("method").size()
    
    # dataset x method
    table_dataset_method = dataset_method_df.pivot_table(index="dataset", columns="method", values=metric_name)
    
    num_datasets, num_methods = table_dataset_method.shape
    
    # both conditions must be true because a (dataset, method) pair may be missing or duplicated
    assert len(datasetcount_permethod.unique()) == 1 and not table_dataset_method.isna().any().any(), f"all methods must be present in all datasets"
    # ================================ FRIEDMAN ================================

    # each ndarray corresponds to a model, whose each element is the performance on a dataset
    # [ndarray(...), ndarray(...), ndarray(...)]
    performances: List[np.ndarray] = list(map(np.array, table_dataset_method.T.values.tolist()))
    friedman_result = spstats.friedmanchisquare(*performances)
    friedman_statistic, friedman_pvalue = friedman_result

    friedman_h0_is_rejected = friedman_pvalue < alpha
    
    # ================================ WILCOXON ================================

    # the wilcoxon test considers this information as an apriori knowledge so we can use the sided alternative
    rank_data = _stats_py.rankdata(table_dataset_method.values, axis=1)
    rank_avg = rank_data.mean(axis=0)
    
    wilcoxon_parwise_tests: List[Dict[str, Any]] = []
    
    # heuristic to choose the method 'exact' or 'approx', copied from scipy
    # "exact" if n <= 50, "approx" otherwise
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html#r996422d5c98f-4
    EXACT_METHOD_SIZE_LIMIT = 50
    wilcoxon_method = "approx" if num_datasets > EXACT_METHOD_SIZE_LIMIT else "exact"
    
    # all combinations of methods
    for idx_i, idx_j in itertools.combinations(list(range(num_methods)), 2):
        
        method_i = table_dataset_method.columns[idx_i]
        method_j = table_dataset_method.columns[idx_j]
        
        perfs_i = table_dataset_method[method_i].values
        perfs_j = table_dataset_method[method_j].values
        
        avgrank_i = rank_avg[idx_i]
        avgrank_j = rank_avg[idx_j]
        
        # wilcoxon signed rank test
        
        # from the docs of scipy (version 1.9.3, 2022-12-07, adapted)
        #
        # def wilcoxon(x, y=None, ...)
        #  let ``d`` represent the difference between the paired samples: ``d = x - y``
        # alternative='greater': 
        #   the distribution underlying ``d`` is stochastically greater than a distribution symmetric about zero.
        
        # otherwise said, alternative='greater' is equivalent to assume x > y
        # therefore we chose x and y in the order of the performance (average rank) between them
        # x: higher average rank, y: lower average rank
        
        if avgrank_i > avgrank_j:
            x = perfs_i
            y = perfs_j
        else:
            x = perfs_j
            y = perfs_i

        # TODO check if the zero_method is correct, or try zsplit
        wilcoxon_result = spstats.wilcoxon(x, y, method=wilcoxon_method, zero_method='pratt', alternative='greater',)        
        wilcoxon_statistic, wilcoxon_pvalue = wilcoxon_result
        wilcoxon_parwise_tests.append(
            {
                "idx_i": idx_i, "idx_j": idx_j,
                "method_i": method_i, "method_j": method_j, 
                "statistic": wilcoxon_statistic,
                "pvalue": wilcoxon_pvalue, 
                "h0_is_rejected": False, 
            }
        )
        
    # ============================ ALPHA CORRECTION ============================
        
    num_hypothesis = len(wilcoxon_parwise_tests)  # num_methods * (num_methods - 1) / 2

    # sort acsending by p-value
    wilcoxon_parwise_tests.sort(key=lambda dic: dic['pvalue'])
    
    # the loop runs after the break point so all the tests will keep track of the use corrected alpha
    break_point_found = False
    
    # loop through the hypothesis
    for idx, wilcoxon_test in enumerate(wilcoxon_parwise_tests):

        pvalue = wilcoxon_test["pvalue"]
        wilcoxon_test["pvalues_argsort_idx"] = idx 
        
        # the correction starts at 1/num_hypothesis and ends at 1/(num_hypothesis - (num_hypothesis - 1)) = 1
        wilcoxon_test["corrected_alpha"] = corrected_alpha = alpha / (num_hypothesis - idx)

        if break_point_found:
            continue
        
        # test if significant after holm's correction of alpha
        if pvalue <= corrected_alpha:
            wilcoxon_test["h0_is_rejected"] = True
            continue

        break_point_found = True
    
    # ================================= CLIQUES ==================================
    # a clique is a subset of methods that are NOT pairwise-ly significantly different

    is_connected_matrix = np.zeros((num_methods, num_methods))
    
    for pairwise_test in wilcoxon_parwise_tests:
        idx_i, idx_j = pairwise_test["idx_i"], pairwise_test["idx_j"]
        is_rejected = pairwise_test["h0_is_rejected"]
        is_connected_matrix[idx_i, idx_j] = is_connected_matrix[idx_j, idx_i] = int(not is_rejected)
    
    graph = nx.Graph(is_connected_matrix)
    cliques = [
        cliq
        for cliq in nx.find_cliques(graph)
        if len(cliq) > 1
    ]
    cliques = sorted(map(tuple, map(sorted, cliques)))
    
    # ================================= RESULT ==================================
    
    def filter_pairwise_keys(dic: dict) -> dict:
        return {k: v for k, v in dic.items() if k not in ["idx_i", "idx_j", "method_i", "method_j"]}
    
    result = WilcoxonHolmResult(
        parameters=WilcoxonHolmResult.Parameters(alpha=alpha),
        metadata=WilcoxonHolmResult.Metadata(
            metric_name=metric_name,
            higher_is_better=higher_is_better,
            datasets=table_dataset_method.index.tolist(),
            methods=table_dataset_method.columns.tolist(),
        ),
        friedman_statistic=friedman_statistic,
        friedman_pvalue=friedman_pvalue,
        friedman_h0_is_rejected=friedman_h0_is_rejected,
        parwise_tests={
            (wt["idx_i"], wt["idx_j"]): WilcoxonHolmResult.PairwiseResult(**filter_pairwise_keys(wt))
            for wt in wilcoxon_parwise_tests
        },
        rankdata=rank_data,
        rankavg=rank_avg,
        cliques_insignificant_difference=cliques,
    )
    
    return result