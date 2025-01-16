## based on scanpy with minor modifications

# from ._deprecated.highly_variable_genes import filter_genes_dispersion#
# from ._highly_variable_genes import highly_variable_genes
# from ._simple import log1p, sqrt, scale, subsample#
# from ._simple import normalize_per_cell#
# from ._pca import pca#
# from ._normalization import normalize_total#
# from ..neighbors import neighbors#

# from ._simple import filter_cells, filter_genes#
# from ._qc import calculate_qc_metrics#
from functools import singledispatch
from typing import Optional, Tuple, Collection, Union, Iterable, Sequence, Dict, Any, Mapping, Callable, NamedTuple, Generator
from numbers import Number

# import numba
import numpy as np
import pandas as pd
import scipy as sp
import scipy
import scipy.sparse as sp_sparse
from scipy.sparse import issparse, isspmatrix_csr, isspmatrix_coo
from scipy.sparse import spmatrix, csr_matrix
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils import sparsefuncs, check_array
from pandas.api.types import CategoricalDtype


from .sc_utils import (
    sanitize_anndata,
    deprecated_arg_names,
    view_to_actual,
    AnyRandom,
    _check_array_function_arguments,
)
from .sc_utils import _get_obs_rep, _set_obs_rep
from .sc_utils import materialize_as_ndarray
from .sc_utils import _get_mean_var

from anndata import AnnData
from ._docs import (
    doc_expr_reps,
    doc_obs_qc_args,
    doc_qc_metric_naming,
    doc_obs_qc_returns,
    doc_var_qc_returns,
    doc_adata_basic,
)
from .sc_utils import _doc_params
from .sc_utils import Literal

# install dask if available
try:
    import dask.array as da
except ImportError:
    da = None

from .sc_utils import DaskArray
from .sc_utils import *


def _normalize_data(X, counts, after=None, copy=False):
    X = X.copy() if copy else X
    if issubclass(X.dtype.type, (int, np.integer)):
        X = X.astype(np.float32)  # TODO: Check if float64 should be used
    counts = np.asarray(counts)  # dask doesn't do medians
    after = np.median(counts[counts > 0], axis=0) if after is None else after
    counts += counts == 0
    counts = counts / after
    if issparse(X):
        sparsefuncs.inplace_row_scale(X, 1 / counts)
    else:
        np.divide(X, counts[:, None], out=X)
    return X


def normalize_total(
    adata: AnnData,
    target_sum: Optional[float] = None,
    exclude_highly_expressed: bool = False,
    max_fraction: float = 0.05,
    key_added: Optional[str] = None,
    layer: Optional[str] = None,
    layers: Union[Literal['all'], Iterable[str]] = None,
    layer_norm: Optional[str] = None,
    inplace: bool = True,
    copy: bool = False,
) -> Optional[Dict[str, np.ndarray]]:
    """\
    Normalize counts per cell.

    Normalize each cell by total counts over all genes,
    so that every cell has the same total count after normalization.
    If choosing `target_sum=1e6`, this is CPM normalization.

    If `exclude_highly_expressed=True`, very highly expressed genes are excluded
    from the computation of the normalization factor (size factor) for each
    cell. This is meaningful as these can strongly influence the resulting
    normalized values for all other genes [Weinreb17]_.

    Similar functions are used, for example, by Seurat [Satija15]_, Cell Ranger
    [Zheng17]_ or SPRING [Weinreb17]_.

    Params
    ------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    target_sum
        If `None`, after normalization, each observation (cell) has a total
        count equal to the median of total counts for observations (cells)
        before normalization.
    exclude_highly_expressed
        Exclude (very) highly expressed genes for the computation of the
        normalization factor (size factor) for each cell. A gene is considered
        highly expressed, if it has more than `max_fraction` of the total counts
        in at least one cell. The not-excluded genes will sum up to
        `target_sum`.
    max_fraction
        If `exclude_highly_expressed=True`, consider cells as highly expressed
        that have more counts than `max_fraction` of the original total counts
        in at least one cell.
    key_added
        Name of the field in `adata.obs` where the normalization factor is
        stored.
    layer
        Layer to normalize instead of `X`. If `None`, `X` is normalized.
    inplace
        Whether to update `adata` or return dictionary with normalized copies of
        `adata.X` and `adata.layers`.
    copy
        Whether to modify copied input object. Not compatible with inplace=False.

    Returns
    -------
    Returns dictionary with normalized copies of `adata.X` and `adata.layers`
    or updates `adata` with normalized version of the original
    `adata.X` and `adata.layers`, depending on `inplace`.

    Example
    --------
    >>> from anndata import AnnData
    >>> import scanpy as sc
    >>> sc.settings.verbosity = 2
    >>> np.set_printoptions(precision=2)
    >>> adata = AnnData(np.array([
    ...    [3, 3, 3, 6, 6],
    ...    [1, 1, 1, 2, 2],
    ...    [1, 22, 1, 2, 2],
    ... ]))
    >>> adata.X
    array([[ 3.,  3.,  3.,  6.,  6.],
           [ 1.,  1.,  1.,  2.,  2.],
           [ 1., 22.,  1.,  2.,  2.]], dtype=float32)
    >>> X_norm = sc.pp.normalize_total(adata, target_sum=1, inplace=False)['X']
    >>> X_norm
    array([[0.14, 0.14, 0.14, 0.29, 0.29],
           [0.14, 0.14, 0.14, 0.29, 0.29],
           [0.04, 0.79, 0.04, 0.07, 0.07]], dtype=float32)
    >>> X_norm = sc.pp.normalize_total(
    ...     adata, target_sum=1, exclude_highly_expressed=True,
    ...     max_fraction=0.2, inplace=False
    ... )['X']
    The following highly-expressed genes are not considered during normalization factor computation:
    ['1', '3', '4']
    >>> X_norm
    array([[ 0.5,  0.5,  0.5,  1. ,  1. ],
           [ 0.5,  0.5,  0.5,  1. ,  1. ],
           [ 0.5, 11. ,  0.5,  1. ,  1. ]], dtype=float32)
    """
    if copy:
        if not inplace:
            raise ValueError("`copy=True` cannot be used with `inplace=False`.")
        adata = adata.copy()

    if max_fraction < 0 or max_fraction > 1:
        raise ValueError('Choose max_fraction between 0 and 1.')

    # Deprecated features
    if layers is not None:
        print('The `layers` argument is deprecated. Instead, specify individual')
        # warn(
        #     FutureWarning(
        #         "The `layers` argument is deprecated. Instead, specify individual "
        #         "layers to normalize with `layer`."
        #     )
        # )
    if layer_norm is not None:
        print('The `layer_norm` argument is deprecated. Specify the target size')
        # warn(
        #     FutureWarning(
        #         "The `layer_norm` argument is deprecated. Specify the target size "
        #         "factor directly with `target_sum`."
        #     )
        # )

    if layers == 'all':
        layers = adata.layers.keys()
    elif isinstance(layers, str):
        raise ValueError(
            f"`layers` needs to be a list of strings or 'all', not {layers!r}"
        )

    view_to_actual(adata)

    X = _get_obs_rep(adata, layer=layer)

    gene_subset = None
    msg = 'normalizing counts per cell'
    if exclude_highly_expressed:
        counts_per_cell = X.sum(1)  # original counts per cell
        counts_per_cell = np.ravel(counts_per_cell)

        # at least one cell as more than max_fraction of counts per cell

        gene_subset = (X > counts_per_cell[:, None] * max_fraction).sum(0)
        gene_subset = np.ravel(gene_subset) == 0

        msg += (
            ' The following highly-expressed genes are not considered during '
            f'normalization factor computation:\n{adata.var_names[~gene_subset].tolist()}'
        )
        counts_per_cell = X[:, gene_subset].sum(1)
    else:
        counts_per_cell = X.sum(1)
    # start = logg.info(msg)
    counts_per_cell = np.ravel(counts_per_cell)

    cell_subset = counts_per_cell > 0
    if not np.all(cell_subset):
        print('Some cells have zero counts')
        # warn(UserWarning('Some cells have zero counts'))

    if inplace:
        if key_added is not None:
            adata.obs[key_added] = counts_per_cell
        _set_obs_rep(
            adata, _normalize_data(X, counts_per_cell, target_sum), layer=layer
        )
    else:
        # not recarray because need to support sparse
        dat = dict(
            X=_normalize_data(X, counts_per_cell, target_sum, copy=True),
            norm_factor=counts_per_cell,
        )

    # Deprecated features
    if layer_norm == 'after':
        after = target_sum
    elif layer_norm == 'X':
        after = np.median(counts_per_cell[cell_subset])
    elif layer_norm is None:
        after = None
    else:
        raise ValueError('layer_norm should be "after", "X" or None')

    for layer_to_norm in layers if layers is not None else ():
        res = normalize_total(
            adata, layer=layer_to_norm, target_sum=after, inplace=inplace
        )
        if not inplace:
            dat[layer_to_norm] = res["X"]

    # logg.info(
    #     '    finished ({time_passed})',
    #     time=start,
    # )
    if key_added is not None:
        print('counts per cell before normalization (adata.obs)')
        # logg.debug(
        #     f'and added {key_added!r}, counts per cell before normalization (adata.obs)'
        # )

    if copy:
        return adata
    elif not inplace:
        return dat

def _choose_mtx_rep(adata, use_raw=False, layer=None):
    is_layer = layer is not None
    if use_raw and is_layer:
        raise ValueError(
            "Cannot use expression from both layer and raw. You provided:"
            f"'use_raw={use_raw}' and 'layer={layer}'"
        )
    if is_layer:
        return adata.layers[layer]
    elif use_raw:
        return adata.raw.X
    else:
        return adata.X

@_doc_params(
    doc_adata_basic=doc_adata_basic,
    doc_expr_reps=doc_expr_reps,
    doc_obs_qc_args=doc_obs_qc_args,
    doc_qc_metric_naming=doc_qc_metric_naming,
    doc_obs_qc_returns=doc_obs_qc_returns,
)

def describe_obs(
    adata: AnnData,
    *,
    expr_type: str = "counts",
    var_type: str = "genes",
    qc_vars: Collection[str] = (),
    percent_top: Optional[Collection[int]] = (50, 100, 200, 500),
    layer: Optional[str] = None,
    use_raw: bool = False,
    log1p: Optional[str] = True,
    inplace: bool = False,
    X=None,
    parallel=None,
) -> Optional[pd.DataFrame]:
    """\
    Describe observations of anndata.

    Calculates a number of qc metrics for observations in AnnData object. See
    section `Returns` for a description of those metrics.

    Note that this method can take a while to compile on the first call. That
    result is then cached to disk to be used later.

    Params
    ------
    {doc_adata_basic}
    {doc_qc_metric_naming}
    {doc_obs_qc_args}
    {doc_expr_reps}
    log1p
        Add `log1p` transformed metrics.
    inplace
        Whether to place calculated metrics in `adata.obs`.
    X
        Matrix to calculate values on. Meant for internal usage.

    Returns
    -------
    QC metrics for observations in adata. If inplace, values are placed into
    the AnnData's `.obs` dataframe.

    {doc_obs_qc_returns}
    """
    if parallel is not None:
        pass
        # warn(
        #     "Argument `parallel` is deprecated, and currently has no effect.",
        #     FutureWarning,
        # )
    # Handle whether X is passed
    if X is None:
        X = _choose_mtx_rep(adata, use_raw, layer)
        if isspmatrix_coo(X):
            X = csr_matrix(X)  # COO not subscriptable
        if issparse(X):
            X.eliminate_zeros()
    obs_metrics = pd.DataFrame(index=adata.obs_names)
    if issparse(X):
        obs_metrics[f"n_{var_type}_by_{expr_type}"] = X.getnnz(axis=1)
    else:
        obs_metrics[f"n_{var_type}_by_{expr_type}"] = np.count_nonzero(X, axis=1)
    if log1p:
        obs_metrics[f"log1p_n_{var_type}_by_{expr_type}"] = np.log1p(
            obs_metrics[f"n_{var_type}_by_{expr_type}"]
        )
    obs_metrics[f"total_{expr_type}"] = np.ravel(X.sum(axis=1))
    if log1p:
        obs_metrics[f"log1p_total_{expr_type}"] = np.log1p(
            obs_metrics[f"total_{expr_type}"]
        )
    if percent_top:
        percent_top = sorted(percent_top)
        proportions = top_segment_proportions(X, percent_top)
        for i, n in enumerate(percent_top):
            obs_metrics[f"pct_{expr_type}_in_top_{n}_{var_type}"] = (
                proportions[:, i] * 100
            )
    for qc_var in qc_vars:
        obs_metrics[f"total_{expr_type}_{qc_var}"] = np.ravel(
            X[:, adata.var[qc_var].values].sum(axis=1)
        )
        if log1p:
            obs_metrics[f"log1p_total_{expr_type}_{qc_var}"] = np.log1p(
                obs_metrics[f"total_{expr_type}_{qc_var}"]
            )
        obs_metrics[f"pct_{expr_type}_{qc_var}"] = (
            obs_metrics[f"total_{expr_type}_{qc_var}"]
            / obs_metrics[f"total_{expr_type}"]
            * 100
        )
    if inplace:
        adata.obs[obs_metrics.columns] = obs_metrics
    else:
        return obs_metrics


@_doc_params(
    doc_adata_basic=doc_adata_basic,
    doc_expr_reps=doc_expr_reps,
    doc_qc_metric_naming=doc_qc_metric_naming,
    doc_var_qc_returns=doc_var_qc_returns,
)
def describe_var(
    adata: AnnData,
    *,
    expr_type: str = "counts",
    var_type: str = "genes",
    layer: Optional[str] = None,
    use_raw: bool = False,
    inplace=False,
    log1p=True,
    X=None,
) -> Optional[pd.DataFrame]:
    """\
    Describe variables of anndata.

    Calculates a number of qc metrics for variables in AnnData object. See
    section `Returns` for a description of those metrics.

    Params
    ------
    {doc_adata_basic}
    {doc_qc_metric_naming}
    {doc_expr_reps}
    inplace
        Whether to place calculated metrics in `adata.var`.
    X
        Matrix to calculate values on. Meant for internal usage.

    Returns
    -------
    QC metrics for variables in adata. If inplace, values are placed into the
    AnnData's `.var` dataframe.

    {doc_var_qc_returns}
    """
    # Handle whether X is passed
    if X is None:
        X = _choose_mtx_rep(adata, use_raw, layer)
        if isspmatrix_coo(X):
            X = csr_matrix(X)  # COO not subscriptable
        if issparse(X):
            X.eliminate_zeros()
    var_metrics = pd.DataFrame(index=adata.var_names)
    if issparse(X):
        # Current memory bottleneck for csr matrices:
        var_metrics["n_cells_by_{expr_type}"] = X.getnnz(axis=0)
        var_metrics["mean_{expr_type}"] = mean_variance_axis(X, axis=0)[0]
    else:
        var_metrics["n_cells_by_{expr_type}"] = np.count_nonzero(X, axis=0)
        var_metrics["mean_{expr_type}"] = X.mean(axis=0)
    if log1p:
        var_metrics["log1p_mean_{expr_type}"] = np.log1p(
            var_metrics["mean_{expr_type}"]
        )
    var_metrics["pct_dropout_by_{expr_type}"] = (
        1 - var_metrics["n_cells_by_{expr_type}"] / X.shape[0]
    ) * 100
    var_metrics["total_{expr_type}"] = np.ravel(X.sum(axis=0))
    if log1p:
        var_metrics["log1p_total_{expr_type}"] = np.log1p(
            var_metrics["total_{expr_type}"]
        )
    # Relabel
    new_colnames = []
    for col in var_metrics.columns:
        new_colnames.append(col.format(**locals()))
    var_metrics.columns = new_colnames
    if inplace:
        adata.var[var_metrics.columns] = var_metrics
    else:
        return var_metrics


@_doc_params(
    doc_adata_basic=doc_adata_basic,
    doc_expr_reps=doc_expr_reps,
    doc_obs_qc_args=doc_obs_qc_args,
    doc_qc_metric_naming=doc_qc_metric_naming,
    doc_obs_qc_returns=doc_obs_qc_returns,
    doc_var_qc_returns=doc_var_qc_returns,
)
def calculate_qc_metrics(
    adata: AnnData,
    *,
    expr_type: str = "counts",
    var_type: str = "genes",
    qc_vars: Collection[str] = (),
    percent_top: Optional[Collection[int]] = (50, 100, 200, 500),
    layer: Optional[str] = None,
    use_raw: bool = False,
    inplace: bool = False,
    log1p: bool = True,
    parallel: Optional[bool] = None,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """\
    Calculate quality control metrics.

    Calculates a number of qc metrics for an AnnData object, see section
    `Returns` for specifics. Largely based on `calculateQCMetrics` from scater
    [McCarthy17]_. Currently is most efficient on a sparse CSR or dense matrix.

    Note that this method can take a while to compile on the first call. That
    result is then cached to disk to be used later.

    Parameters
    ----------
    {doc_adata_basic}
    {doc_qc_metric_naming}
    {doc_obs_qc_args}
    {doc_expr_reps}
    inplace
        Whether to place calculated metrics in `adata`'s `.obs` and `.var`.
    log1p
        Set to `False` to skip computing `log1p` transformed annotations.

    Returns
    -------
    Depending on `inplace` returns calculated metrics
    (as :class:`~pandas.DataFrame`) or updates `adata`'s `obs` and `var`.

    {doc_obs_qc_returns}

    {doc_var_qc_returns}

    Example
    -------
    Calculate qc metrics for visualization.

    .. plot::
        :context: close-figs

        import scanpy as sc
        import seaborn as sns

        pbmc = sc.datasets.pbmc3k()
        pbmc.var["mito"] = pbmc.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(pbmc, qc_vars=["mito"], inplace=True)
        sns.jointplot(
            data=pbmc.obs,
            x="log1p_total_counts",
            y="log1p_n_genes_by_counts",
            kind="hex",
        )

    .. plot::
        :context: close-figs

        sns.histplot(pbmc.obs["pct_counts_mito"])
    """
    if parallel is not None:
        # warn(
        #     "Argument `parallel` is deprecated, and currently has no effect.",
        #     FutureWarning,
        # )
        pass
    # Pass X so I only have to do it once
    X = _choose_mtx_rep(adata, use_raw, layer)
    if isspmatrix_coo(X):
        X = csr_matrix(X)  # COO not subscriptable
    if issparse(X):
        X.eliminate_zeros()

    obs_metrics = describe_obs(
        adata,
        expr_type=expr_type,
        var_type=var_type,
        qc_vars=qc_vars,
        percent_top=percent_top,
        inplace=inplace,
        X=X,
        log1p=log1p,
    )
    var_metrics = describe_var(
        adata,
        expr_type=expr_type,
        var_type=var_type,
        inplace=inplace,
        X=X,
        log1p=log1p,
    )

    if not inplace:
        return obs_metrics, var_metrics

def top_proportions(mtx: Union[np.array, spmatrix], n: int):
    """\
    Calculates cumulative proportions of top expressed genes

    Parameters
    ----------
    mtx
        Matrix, where each row is a sample, each column a feature.
    n
        Rank to calculate proportions up to. Value is treated as 1-indexed,
        `n=50` will calculate cumulative proportions up to the 50th most
        expressed gene.
    """
    if issparse(mtx):
        if not isspmatrix_csr(mtx):
            mtx = csr_matrix(mtx)
        # Allowing numba to do more
        return top_proportions_sparse_csr(mtx.data, mtx.indptr, np.array(n))
    else:
        return top_proportions_dense(mtx, n)


def top_proportions_dense(mtx, n):
    sums = mtx.sum(axis=1)
    partitioned = np.apply_along_axis(np.argpartition, 1, -mtx, n - 1)
    partitioned = partitioned[:, :n]
    values = np.zeros_like(partitioned, dtype=np.float64)
    for i in range(partitioned.shape[0]):
        vec = mtx[i, partitioned[i, :]]  # Not a view
        vec[::-1].sort()  # Sorting on a reversed view (e.g. a descending sort)
        vec = np.cumsum(vec) / sums[i]
        values[i, :] = vec
    return values


def top_proportions_sparse_csr(data, indptr, n):
    values = np.zeros((indptr.size - 1, n), dtype=np.float64)
    for i in range(indptr.size - 1):#numba.p
        start, end = indptr[i], indptr[i + 1]
        vec = np.zeros(n, dtype=np.float64)
        if end - start <= n:
            vec[: end - start] = data[start:end]
            total = vec.sum()
        else:
            vec[:] = -(np.partition(-data[start:end], n - 1)[:n])
            total = (data[start:end]).sum()  # Is this not just vec.sum()?
        vec[::-1].sort()
        values[i, :] = vec.cumsum() / total
    return values


def top_segment_proportions(
    mtx: Union[np.array, spmatrix], ns: Collection[int]
) -> np.ndarray:
    """
    Calculates total percentage of counts in top ns genes.

    Parameters
    ----------
    mtx
        Matrix, where each row is a sample, each column a feature.
    ns
        Positions to calculate cumulative proportion at. Values are considered
        1-indexed, e.g. `ns=[50]` will calculate cumulative proportion up to
        the 50th most expressed gene.
    """
    # Pretty much just does dispatch
    if not (max(ns) <= mtx.shape[1] and min(ns) > 0):
        raise IndexError("Positions outside range of features.")
    if issparse(mtx):
        if not isspmatrix_csr(mtx):
            mtx = csr_matrix(mtx)
        return top_segment_proportions_sparse_csr(mtx.data, mtx.indptr, np.array(ns))
    else:
        return top_segment_proportions_dense(mtx, ns)


def top_segment_proportions_dense(
    mtx: Union[np.array, spmatrix], ns: Collection[int]
) -> np.ndarray:
    # Currently ns is considered to be 1 indexed
    ns = np.sort(ns)
    sums = mtx.sum(axis=1)
    partitioned = np.apply_along_axis(np.partition, 1, mtx, mtx.shape[1] - ns)[:, ::-1][
        :, : ns[-1]
    ]
    values = np.zeros((mtx.shape[0], len(ns)))
    acc = np.zeros(mtx.shape[0])
    prev = 0
    for j, n in enumerate(ns):
        acc += partitioned[:, prev:n].sum(axis=1)
        values[:, j] = acc
        prev = n
    return values / sums[:, None]


# @numba.njit(cache=True, parallel=True)
def top_segment_proportions_sparse_csr(data, indptr, ns):
    # work around https://github.com/numba/numba/issues/5056
    indptr = indptr.astype(np.int64)
    ns = ns.astype(np.int64)
    ns = np.sort(ns)
    maxidx = ns[-1]
    sums = np.zeros((indptr.size - 1), dtype=data.dtype)
    values = np.zeros((indptr.size - 1, len(ns)), dtype=np.float64)
    # Just to keep it simple, as a dense matrix
    partitioned = np.zeros((indptr.size - 1, maxidx), dtype=data.dtype)
    for i in range(indptr.size - 1):#numba.p
        start, end = indptr[i], indptr[i + 1]
        sums[i] = np.sum(data[start:end])
        if end - start <= maxidx:
            partitioned[i, : end - start] = data[start:end]
        elif (end - start) > maxidx:
            partitioned[i, :] = -(np.partition(-data[start:end], maxidx))[:maxidx]
        partitioned[i, :] = np.partition(partitioned[i, :], maxidx - ns)
    partitioned = partitioned[:, ::-1][:, : ns[-1]]
    acc = np.zeros((indptr.size - 1), dtype=data.dtype)
    prev = 0
    # can’t use enumerate due to https://github.com/numba/numba/issues/2625
    for j in range(ns.size):
        acc += partitioned[:, prev : ns[j]].sum(axis=1)
        values[:, j] = acc
        prev = ns[j]
    return values / sums.reshape((indptr.size - 1, 1))


def filter_cells(
    data: AnnData,
    min_counts: Optional[int] = None,
    min_genes: Optional[int] = None,
    max_counts: Optional[int] = None,
    max_genes: Optional[int] = None,
    inplace: bool = True,
    copy: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """\
    Filter cell outliers based on counts and numbers of genes expressed.

    For instance, only keep cells with at least `min_counts` counts or
    `min_genes` genes expressed. This is to filter measurement outliers,
    i.e. “unreliable” observations.

    Only provide one of the optional parameters `min_counts`, `min_genes`,
    `max_counts`, `max_genes` per call.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    min_counts
        Minimum number of counts required for a cell to pass filtering.
    min_genes
        Minimum number of genes expressed required for a cell to pass filtering.
    max_counts
        Maximum number of counts required for a cell to pass filtering.
    max_genes
        Maximum number of genes expressed required for a cell to pass filtering.
    inplace
        Perform computation inplace or return result.

    Returns
    -------
    Depending on `inplace`, returns the following arrays or directly subsets
    and annotates the data matrix:

    cells_subset
        Boolean index mask that does filtering. `True` means that the
        cell is kept. `False` means the cell is removed.
    number_per_cell
        Depending on what was thresholded (`counts` or `genes`),
        the array stores `n_counts` or `n_cells` per gene.

    Examples
    --------
    >>> import scanpy as sc
    >>> adata = sc.datasets.krumsiek11()
    >>> adata.n_obs
    640
    >>> adata.var_names.tolist()  # doctest: +NORMALIZE_WHITESPACE
    ['Gata2', 'Gata1', 'Fog1', 'EKLF', 'Fli1', 'SCL',
     'Cebpa', 'Pu.1', 'cJun', 'EgrNab', 'Gfi1']
    >>> # add some true zeros
    >>> adata.X[adata.X < 0.3] = 0
    >>> # simply compute the number of genes per cell
    >>> sc.pp.filter_cells(adata, min_genes=0)
    >>> adata.n_obs
    640
    >>> adata.obs['n_genes'].min()
    1
    >>> # filter manually
    >>> adata_copy = adata[adata.obs['n_genes'] >= 3]
    >>> adata_copy.n_obs
    554
    >>> adata_copy.obs['n_genes'].min()
    3
    >>> # actually do some filtering
    >>> sc.pp.filter_cells(adata, min_genes=3)
    >>> adata.n_obs
    554
    >>> adata.obs['n_genes'].min()
    3
    """
    if copy:
        # logg.warning("`copy` is deprecated, use `inplace` instead.")
        pass
    n_given_options = sum(
        option is not None for option in [min_genes, min_counts, max_genes, max_counts]
    )
    if n_given_options != 1:
        raise ValueError(
            "Only provide one of the optional parameters `min_counts`, "
            "`min_genes`, `max_counts`, `max_genes` per call."
        )
    if isinstance(data, AnnData):
        adata = data.copy() if copy else data
        cell_subset, number = materialize_as_ndarray(
            filter_cells(adata.X, min_counts, min_genes, max_counts, max_genes)
        )
        if not inplace:
            return cell_subset, number
        if min_genes is None and max_genes is None:
            adata.obs["n_counts"] = number
        else:
            adata.obs["n_genes"] = number
        adata._inplace_subset_obs(cell_subset)
        return adata if copy else None
    X = data  # proceed with processing the data matrix
    min_number = min_counts if min_genes is None else min_genes
    max_number = max_counts if max_genes is None else max_genes
    number_per_cell = np.sum(
        X if min_genes is None and max_genes is None else X > 0, axis=1
    )
    if issparse(X):
        number_per_cell = number_per_cell.A1
    if min_number is not None:
        cell_subset = number_per_cell >= min_number
    if max_number is not None:
        cell_subset = number_per_cell <= max_number

    s = np.sum(~cell_subset)
    if s > 0:
        msg = f"filtered out {s} cells that have "
        if min_genes is not None or min_counts is not None:
            msg += "less than "
            msg += (
                f"{min_genes} genes expressed"
                if min_counts is None
                else f"{min_counts} counts"
            )
        if max_genes is not None or max_counts is not None:
            msg += "more than "
            msg += (
                f"{max_genes} genes expressed"
                if max_counts is None
                else f"{max_counts} counts"
            )
        # logg.info(msg)
    return cell_subset, number_per_cell


def filter_genes(
    data: AnnData,
    min_counts: Optional[int] = None,
    min_cells: Optional[int] = None,
    max_counts: Optional[int] = None,
    max_cells: Optional[int] = None,
    inplace: bool = True,
    copy: bool = False,
) -> Union[AnnData, None, Tuple[np.ndarray, np.ndarray]]:
    """\
    Filter genes based on number of cells or counts.

    Keep genes that have at least `min_counts` counts or are expressed in at
    least `min_cells` cells or have at most `max_counts` counts or are expressed
    in at most `max_cells` cells.

    Only provide one of the optional parameters `min_counts`, `min_cells`,
    `max_counts`, `max_cells` per call.

    Parameters
    ----------
    data
        An annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    min_counts
        Minimum number of counts required for a gene to pass filtering.
    min_cells
        Minimum number of cells expressed required for a gene to pass filtering.
    max_counts
        Maximum number of counts required for a gene to pass filtering.
    max_cells
        Maximum number of cells expressed required for a gene to pass filtering.
    inplace
        Perform computation inplace or return result.

    Returns
    -------
    Depending on `inplace`, returns the following arrays or directly subsets
    and annotates the data matrix

    gene_subset
        Boolean index mask that does filtering. `True` means that the
        gene is kept. `False` means the gene is removed.
    number_per_gene
        Depending on what was thresholded (`counts` or `cells`), the array stores
        `n_counts` or `n_cells` per gene.
    """
    if copy:
        # logg.warning("`copy` is deprecated, use `inplace` instead.")
        pass
    n_given_options = sum(
        option is not None for option in [min_cells, min_counts, max_cells, max_counts]
    )
    if n_given_options != 1:
        raise ValueError(
            "Only provide one of the optional parameters `min_counts`, "
            "`min_cells`, `max_counts`, `max_cells` per call."
        )

    if isinstance(data, AnnData):
        adata = data.copy() if copy else data
        gene_subset, number = materialize_as_ndarray(
            filter_genes(
                adata.X,
                min_cells=min_cells,
                min_counts=min_counts,
                max_cells=max_cells,
                max_counts=max_counts,
            )
        )
        if not inplace:
            return gene_subset, number
        if min_cells is None and max_cells is None:
            adata.var["n_counts"] = number
        else:
            adata.var["n_cells"] = number
        adata._inplace_subset_var(gene_subset)
        return adata if copy else None

    X = data  # proceed with processing the data matrix
    min_number = min_counts if min_cells is None else min_cells
    max_number = max_counts if max_cells is None else max_cells
    number_per_gene = np.sum(
        X if min_cells is None and max_cells is None else X > 0, axis=0
    )
    if issparse(X):
        number_per_gene = number_per_gene.A1
    if min_number is not None:
        gene_subset = number_per_gene >= min_number
    if max_number is not None:
        gene_subset = number_per_gene <= max_number

    s = np.sum(~gene_subset)
    if s > 0:
        msg = f"filtered out {s} genes that are detected "
        if min_cells is not None or min_counts is not None:
            msg += "in less than "
            msg += (
                f"{min_cells} cells" if min_counts is None else f"{min_counts} counts"
            )
        if max_cells is not None or max_counts is not None:
            msg += "in more than "
            msg += (
                f"{max_cells} cells" if max_counts is None else f"{max_counts} counts"
            )
        # logg.info(msg)
    return gene_subset, number_per_gene


@singledispatch
def log1p(
    X: Union[AnnData, np.ndarray, spmatrix],
    *,
    base: Optional[Number] = None,
    copy: bool = False,
    chunked: bool = None,
    chunk_size: Optional[int] = None,
    layer: Optional[str] = None,
    obsm: Optional[str] = None,
):
    """\
    Logarithmize the data matrix.

    Computes :math:`X = \\log(X + 1)`,
    where :math:`log` denotes the natural logarithm unless a different base is given.

    Parameters
    ----------
    X
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    base
        Base of the logarithm. Natural logarithm is used by default.
    copy
        If an :class:`~anndata.AnnData` is passed, determines whether a copy
        is returned.
    chunked
        Process the data matrix in chunks, which will save memory.
        Applies only to :class:`~anndata.AnnData`.
    chunk_size
        `n_obs` of the chunks to process the data in.
    layer
        Entry of layers to transform.
    obsm
        Entry of obsm to transform.

    Returns
    -------
    Returns or updates `data`, depending on `copy`.
    """
    _check_array_function_arguments(
        chunked=chunked, chunk_size=chunk_size, layer=layer, obsm=obsm
    )
    return log1p_array(X, copy=copy, base=base)


@log1p.register(spmatrix)
def log1p_sparse(X, *, base: Optional[Number] = None, copy: bool = False):
    X = check_array(
        X, accept_sparse=("csr", "csc"), dtype=(np.float64, np.float32), copy=copy
    )
    X.data = log1p(X.data, copy=False, base=base)
    return X


@log1p.register(np.ndarray)
def log1p_array(X, *, base: Optional[Number] = None, copy: bool = False):
    # Can force arrays to be np.ndarrays, but would be useful to not
    # X = check_array(X, dtype=(np.float64, np.float32), ensure_2d=False, copy=copy)
    if copy:
        if not np.issubdtype(X.dtype, np.floating):
            X = X.astype(float)
        else:
            X = X.copy()
    elif not (np.issubdtype(X.dtype, np.floating) or np.issubdtype(X.dtype, complex)):
        X = X.astype(float)
    np.log1p(X, out=X)
    if base is not None:
        np.divide(X, np.log(base), out=X)
    return X


@log1p.register(AnnData)
def log1p_anndata(
    adata,
    *,
    base: Optional[Number] = None,
    copy: bool = False,
    chunked: bool = False,
    chunk_size: Optional[int] = None,
    layer: Optional[str] = None,
    obsm: Optional[str] = None,
) -> Optional[AnnData]:
    if "log1p" in adata.uns_keys():
        # logg.warning("adata.X seems to be already log-transformed.")
        pass

    adata = adata.copy() if copy else adata
    view_to_actual(adata)

    if chunked:
        if (layer is not None) or (obsm is not None):
            raise NotImplementedError(
                "Currently cannot perform chunked operations on arrays not stored in X."
            )
        for chunk, start, end in adata.chunked_X(chunk_size):
            adata.X[start:end] = log1p(chunk, base=base, copy=False)
    else:
        X = _get_obs_rep(adata, layer=layer, obsm=obsm)
        X = log1p(X, copy=False, base=base)
        _set_obs_rep(adata, X, layer=layer, obsm=obsm)

    adata.uns["log1p"] = {"base": base}
    if copy:
        return adata


def sqrt(
    data: AnnData,
    copy: bool = False,
    chunked: bool = False,
    chunk_size: Optional[int] = None,
) -> Optional[AnnData]:
    """\
    Square root the data matrix.

    Computes :math:`X = \\sqrt(X)`.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    copy
        If an :class:`~anndata.AnnData` object is passed,
        determines whether a copy is returned.
    chunked
        Process the data matrix in chunks, which will save memory.
        Applies only to :class:`~anndata.AnnData`.
    chunk_size
        `n_obs` of the chunks to process the data in.

    Returns
    -------
    Returns or updates `data`, depending on `copy`.
    """
    if isinstance(data, AnnData):
        adata = data.copy() if copy else data
        if chunked:
            for chunk, start, end in adata.chunked_X(chunk_size):
                adata.X[start:end] = sqrt(chunk)
        else:
            adata.X = sqrt(data.X)
        return adata if copy else None
    X = data  # proceed with data matrix
    if not issparse(X):
        return np.sqrt(X)
    else:
        return X.sqrt()


def normalize_per_cell(
    data: Union[AnnData, np.ndarray, spmatrix],
    counts_per_cell_after: Optional[float] = None,
    counts_per_cell: Optional[np.ndarray] = None,
    key_n_counts: str = "n_counts",
    copy: bool = False,
    layers: Union[Literal["all"], Iterable[str]] = (),
    use_rep: Optional[Literal["after", "X"]] = None,
    min_counts: int = 1,
) -> Optional[AnnData]:
    """\
    Normalize total counts per cell.

    .. warning::
        .. deprecated:: 1.3.7
            Use :func:`~scanpy.pp.normalize_total` instead.
            The new function is equivalent to the present
            function, except that

            * the new function doesn't filter cells based on `min_counts`,
              use :func:`~scanpy.pp.filter_cells` if filtering is needed.
            * some arguments were renamed
            * `copy` is replaced by `inplace`

    Normalize each cell by total counts over all genes, so that every cell has
    the same total count after normalization.

    Similar functions are used, for example, by Seurat [Satija15]_, Cell Ranger
    [Zheng17]_ or SPRING [Weinreb17]_.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    counts_per_cell_after
        If `None`, after normalization, each cell has a total count equal
        to the median of the *counts_per_cell* before normalization.
    counts_per_cell
        Precomputed counts per cell.
    key_n_counts
        Name of the field in `adata.obs` where the total counts per cell are
        stored.
    copy
        If an :class:`~anndata.AnnData` is passed, determines whether a copy
        is returned.
    min_counts
        Cells with counts less than `min_counts` are filtered out during
        normalization.

    Returns
    -------
    Returns or updates `adata` with normalized version of the original
    `adata.X`, depending on `copy`.

    Examples
    --------
    >>> import scanpy as sc
    >>> adata = AnnData(np.array([[1, 0], [3, 0], [5, 6]], dtype=np.float32))
    >>> print(adata.X.sum(axis=1))
    [ 1.  3. 11.]
    >>> sc.pp.normalize_per_cell(adata)
    >>> print(adata.obs)
       n_counts
    0       1.0
    1       3.0
    2      11.0
    >>> print(adata.X.sum(axis=1))
    [3. 3. 3.]
    >>> sc.pp.normalize_per_cell(
    ...     adata, counts_per_cell_after=1,
    ...     key_n_counts='n_counts2',
    ... )
    >>> print(adata.obs)
       n_counts  n_counts2
    0       1.0        3.0
    1       3.0        3.0
    2      11.0        3.0
    >>> print(adata.X.sum(axis=1))
    [1. 1. 1.]
    """
    if isinstance(data, AnnData):
        # start = logg.info("normalizing by total count per cell")
        adata = data.copy() if copy else data
        if counts_per_cell is None:
            cell_subset, counts_per_cell = materialize_as_ndarray(
                filter_cells(adata.X, min_counts=min_counts)
            )
            adata.obs[key_n_counts] = counts_per_cell
            adata._inplace_subset_obs(cell_subset)
            counts_per_cell = counts_per_cell[cell_subset]
        normalize_per_cell(adata.X, counts_per_cell_after, counts_per_cell)

        layers = adata.layers.keys() if layers == "all" else layers
        if use_rep == "after":
            after = counts_per_cell_after
        elif use_rep == "X":
            after = np.median(counts_per_cell[cell_subset])
        elif use_rep is None:
            after = None
        else:
            raise ValueError('use_rep should be "after", "X" or None')
        for layer in layers:
            subset, counts = filter_cells(adata.layers[layer], min_counts=min_counts)
            temp = normalize_per_cell(adata.layers[layer], after, counts, copy=True)
            adata.layers[layer] = temp

        # logg.info(
        #     "    finished ({time_passed}): normalized adata.X and added"
        #     f"    {key_n_counts!r}, counts per cell before normalization (adata.obs)",
        #     time=start,
        # )
        return adata if copy else None
    # proceed with data matrix
    X = data.copy() if copy else data
    if counts_per_cell is None:
        if not copy:
            raise ValueError("Can only be run with copy=True")
        cell_subset, counts_per_cell = filter_cells(X, min_counts=min_counts)
        X = X[cell_subset]
        counts_per_cell = counts_per_cell[cell_subset]
    if counts_per_cell_after is None:
        counts_per_cell_after = np.median(counts_per_cell)
    # with warnings.catch_warnings():
    if True:
        # warnings.simplefilter("ignore")
        counts_per_cell += counts_per_cell == 0
        counts_per_cell /= counts_per_cell_after
        if not issparse(X):
            X /= materialize_as_ndarray(counts_per_cell[:, np.newaxis])
        else:
            sparsefuncs.inplace_row_scale(X, 1 / counts_per_cell)
    return X if copy else None


def regress_out(
    adata: AnnData,
    keys: Union[str, Sequence[str]],
    layer: Optional[str] = None,
    n_jobs: Optional[int] = None,
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    Regress out (mostly) unwanted sources of variation.

    Uses simple linear regression. This is inspired by Seurat's `regressOut`
    function in R [Satija15]. Note that this function tends to overcorrect
    in certain circumstances as described in :issue:`526`.

    Parameters
    ----------
    adata
        The annotated data matrix.
    keys
        Keys for observation annotation on which to regress on.
    layer
        If provided, which element of layers to regress on.
    n_jobs
        Number of jobs for parallel computation.
        `None` means using :attr:`scanpy._settings.ScanpyConfig.n_jobs`.
    copy
        Determines whether a copy of `adata` is returned.

    Returns
    -------
    Depending on `copy` returns or updates `adata` with the corrected data matrix.
    """
    # start = logg.info(f"regressing out {keys}")
    adata = adata.copy() if copy else adata

    sanitize_anndata(adata)

    view_to_actual(adata)

    if isinstance(keys, str):
        keys = [keys]

    X = _get_obs_rep(adata, layer=layer)

    if issparse(X):
        # logg.info("    sparse input is densified and may " "lead to high memory use")
        X = X.A#toarray()

    sn_jobs = 1
    n_jobs = sn_jobs if n_jobs is None else n_jobs

    # regress on a single categorical variable
    variable_is_categorical = False
    if keys[0] in adata.obs_keys() and isinstance(
        adata.obs[keys[0]].dtype, CategoricalDtype
    ):
        if len(keys) > 1:
            raise ValueError(
                "If providing categorical variable, "
                "only a single one is allowed. For this one "
                "we regress on the mean for each category."
            )
        # logg.debug("... regressing on per-gene means within categories")
        regressors = np.zeros(X.shape, dtype="float32")
        for category in adata.obs[keys[0]].cat.categories:
            mask = (category == adata.obs[keys[0]]).values
            for ix, x in enumerate(X.T):
                regressors[mask, ix] = x[mask].mean()
        variable_is_categorical = True
    # regress on one or several ordinal variables
    else:
        # create data frame with selected keys (if given)
        if keys:
            regressors = adata.obs[keys]
        else:
            regressors = adata.obs.copy()

        # add column of ones at index 0 (first column)
        regressors.insert(0, "ones", 1.0)

    len_chunk = np.ceil(min(1000, X.shape[1]) / n_jobs).astype(int)
    n_chunks = np.ceil(X.shape[1] / len_chunk).astype(int)

    tasks = []
    # split the adata.X matrix by columns in chunks of size n_chunk
    # (the last chunk could be of smaller size than the others)
    chunk_list = np.array_split(X, n_chunks, axis=1)
    if variable_is_categorical:
        regressors_chunk = np.array_split(regressors, n_chunks, axis=1)
    for idx, data_chunk in enumerate(chunk_list):
        # each task is a tuple of a data_chunk eg. (adata.X[:,0:100]) and
        # the regressors. This data will be passed to each of the jobs.
        if variable_is_categorical:
            regres = regressors_chunk[idx]
        else:
            regres = regressors
        tasks.append(tuple((data_chunk, regres, variable_is_categorical)))

    from joblib import Parallel, delayed

    # TODO: figure out how to test that this doesn't oversubscribe resources
    res = Parallel(n_jobs=n_jobs)(delayed(_regress_out_chunk)(task) for task in tasks)

    # res is a list of vectors (each corresponding to a regressed gene column).
    # The transpose is needed to get the matrix in the shape needed
    _set_obs_rep(adata, np.vstack(res).T, layer=layer)
    # logg.info("    finished", time=start)
    return adata if copy else None


def _regress_out_chunk(data):
    # data is a tuple containing the selected columns from adata.X
    # and the regressors dataFrame
    data_chunk = data[0]
    regressors = data[1]
    variable_is_categorical = data[2]

    responses_chunk_list = []
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    for col_index in range(data_chunk.shape[1]):
        # if all values are identical, the statsmodel.api.GLM throws an error;
        # but then no regression is necessary anyways...
        if not (data_chunk[:, col_index] != data_chunk[0, col_index]).any():
            responses_chunk_list.append(data_chunk[:, col_index])
            continue

        if variable_is_categorical:
            regres = np.c_[np.ones(regressors.shape[0]), regressors[:, col_index]]
        else:
            regres = regressors
        try:
            result = sm.GLM(
                data_chunk[:, col_index], regres, family=sm.families.Gaussian()
            ).fit()
            new_column = result.resid_response
        except PerfectSeparationError:  # this emulates R's behavior
            # logg.warning("Encountered PerfectSeparationError, setting to 0 as in R.")
            new_column = np.zeros(data_chunk.shape[0])

        responses_chunk_list.append(new_column)

    return np.vstack(responses_chunk_list)


@singledispatch
def scale(
    X: Union[AnnData, spmatrix, np.ndarray],
    zero_center: bool = True,
    max_value: Optional[float] = None,
    copy: bool = False,
    layer: Optional[str] = None,
    obsm: Optional[str] = None,
    return_mean_std: bool = False,
):
    """\
    Scale data to unit variance and zero mean.

    .. note::
        Variables (genes) that do not display any variation (are constant across
        all observations) are retained and (for zero_center==True) set to 0
        during this operation. In the future, they might be set to NaNs.

    Parameters
    ----------
    X
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    zero_center
        If `False`, omit zero-centering variables, which allows to handle sparse
        input efficiently.
    max_value
        Clip (truncate) to this value after scaling. If `None`, do not clip.
    copy
        Whether this function should be performed inplace. If an AnnData object
        is passed, this also determines if a copy is returned.
    layer
        If provided, which element of layers to scale.
    obsm
        If provided, which element of obsm to scale.

    Returns
    -------
    Depending on `copy` returns or updates `adata` with a scaled `adata.X`,
    annotated with `'mean'` and `'std'` in `adata.var`.
    """
    _check_array_function_arguments(layer=layer, obsm=obsm)
    if layer is not None:
        raise ValueError(f"`layer` argument inappropriate for value of type {type(X)}")
    if obsm is not None:
        raise ValueError(f"`obsm` argument inappropriate for value of type {type(X)}")
    return scale_array(X, zero_center=zero_center, max_value=max_value, copy=copy)


@scale.register(np.ndarray)
def scale_array(
    X,
    *,
    zero_center: bool = True,
    max_value: Optional[float] = None,
    copy: bool = False,
    return_mean_std: bool = False,
):
    if copy:
        X = X.copy()
    if not zero_center and max_value is not None:
        # logg.info(  # Be careful of what? This should be more specific
        #     "... be careful when using `max_value` " "without `zero_center`."
        # )
        pass

    if np.issubdtype(X.dtype, np.integer):
        # logg.info(
        #     "... as scaling leads to float results, integer "
        #     "input is cast to float, returning copy."
        # )

        X = X.astype(float)

    mean, var = _get_mean_var(X)
    std = np.sqrt(var)
    std[std == 0] = 1
    if issparse(X):
        if zero_center:
            raise ValueError("Cannot zero-center sparse matrix.")
        sparsefuncs.inplace_column_scale(X, 1 / std)
    else:
        if zero_center:
            X -= mean
        X /= std

    # do the clipping
    if max_value is not None:
        # logg.debug(f"... clipping at max_value {max_value}")
        X[X > max_value] = max_value

    if return_mean_std:
        return X, mean, std
    else:
        return X


@scale.register(spmatrix)
def scale_sparse(
    X,
    *,
    zero_center: bool = True,
    max_value: Optional[float] = None,
    copy: bool = False,
    return_mean_std: bool = False,
):
    # need to add the following here to make inplace logic work
    if zero_center:
        # logg.info(
        #     "... as `zero_center=True`, sparse input is "
        #     "densified and may lead to large memory consumption"
        # )
        X = X.A#toarray()
        copy = False  # Since the data has been copied
    return scale_array(
        X,
        zero_center=zero_center,
        copy=copy,
        max_value=max_value,
        return_mean_std=return_mean_std,
    )


@scale.register(AnnData)
def scale_anndata(
    adata: AnnData,
    *,
    zero_center: bool = True,
    max_value: Optional[float] = None,
    copy: bool = False,
    layer: Optional[str] = None,
    obsm: Optional[str] = None,
) -> Optional[AnnData]:
    adata = adata.copy() if copy else adata
    view_to_actual(adata)
    X = _get_obs_rep(adata, layer=layer, obsm=obsm)
    X, adata.var["mean"], adata.var["std"] = scale(
        X,
        zero_center=zero_center,
        max_value=max_value,
        copy=False,  # because a copy has already been made, if it were to be made
        return_mean_std=True,
    )
    _set_obs_rep(adata, X, layer=layer, obsm=obsm)
    if copy:
        return adata


def subsample(
    data: Union[AnnData, np.ndarray, spmatrix],
    fraction: Optional[float] = None,
    n_obs: Optional[int] = None,
    random_state: AnyRandom = 0,
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    Subsample to a fraction of the number of observations.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    fraction
        Subsample to this `fraction` of the number of observations.
    n_obs
        Subsample to this number of observations.
    random_state
        Random seed to change subsampling.
    copy
        If an :class:`~anndata.AnnData` is passed,
        determines whether a copy is returned.

    Returns
    -------
    Returns `X[obs_indices], obs_indices` if data is array-like, otherwise
    subsamples the passed :class:`~anndata.AnnData` (`copy == False`) or
    returns a subsampled copy of it (`copy == True`).
    """
    np.random.seed(random_state)
    old_n_obs = data.n_obs if isinstance(data, AnnData) else data.shape[0]
    if n_obs is not None:
        new_n_obs = n_obs
    elif fraction is not None:
        if fraction > 1 or fraction < 0:
            raise ValueError(f"`fraction` needs to be within [0, 1], not {fraction}")
        new_n_obs = int(fraction * old_n_obs)
        # logg.debug(f"... subsampled to {new_n_obs} data points")
    else:
        raise ValueError("Either pass `n_obs` or `fraction`.")
    obs_indices = np.random.choice(old_n_obs, size=new_n_obs, replace=False)
    if isinstance(data, AnnData):
        if data.isbacked:
            if copy:
                return data[obs_indices].to_memory()
            else:
                raise NotImplementedError(
                    "Inplace subsampling is not implemented for backed objects."
                )
        else:
            if copy:
                return data[obs_indices].copy()
            else:
                data._inplace_subset_obs(obs_indices)
    else:
        X = data
        return X[obs_indices], obs_indices


@deprecated_arg_names({"target_counts": "counts_per_cell"})
def downsample_counts(
    adata: AnnData,
    counts_per_cell: Optional[Union[int, Collection[int]]] = None,
    total_counts: Optional[int] = None,
    *,
    random_state: AnyRandom = 0,
    replace: bool = False,
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    Downsample counts from count matrix.

    If `counts_per_cell` is specified, each cell will downsampled.
    If `total_counts` is specified, expression matrix will be downsampled to
    contain at most `total_counts`.

    Parameters
    ----------
    adata
        Annotated data matrix.
    counts_per_cell
        Target total counts per cell. If a cell has more than 'counts_per_cell',
        it will be downsampled to this number. Resulting counts can be specified
        on a per cell basis by passing an array.Should be an integer or integer
        ndarray with same length as number of obs.
    total_counts
        Target total counts. If the count matrix has more than `total_counts`
        it will be downsampled to have this number.
    random_state
        Random seed for subsampling.
    replace
        Whether to sample the counts with replacement.
    copy
        Determines whether a copy of `adata` is returned.

    Returns
    -------
    Depending on `copy` returns or updates an `adata` with downsampled `.X`.
    """
    # This logic is all dispatch
    total_counts_call = total_counts is not None
    counts_per_cell_call = counts_per_cell is not None
    if total_counts_call is counts_per_cell_call:
        raise ValueError(
            "Must specify exactly one of `total_counts` or `counts_per_cell`."
        )
    if copy:
        adata = adata.copy()
    if total_counts_call:
        adata.X = _downsample_total_counts(adata.X, total_counts, random_state, replace)
    elif counts_per_cell_call:
        adata.X = _downsample_per_cell(adata.X, counts_per_cell, random_state, replace)
    if copy:
        return adata


def _downsample_per_cell(X, counts_per_cell, random_state, replace):
    n_obs = X.shape[0]
    if isinstance(counts_per_cell, int):
        counts_per_cell = np.full(n_obs, counts_per_cell)
    else:
        counts_per_cell = np.asarray(counts_per_cell)
    # np.random.choice needs int arguments in numba code:
    counts_per_cell = counts_per_cell.astype(np.int_, copy=False)
    if not isinstance(counts_per_cell, np.ndarray) or len(counts_per_cell) != n_obs:
        raise ValueError(
            "If provided, 'counts_per_cell' must be either an integer, or "
            "coercible to an `np.ndarray` of length as number of observations"
            " by `np.asarray(counts_per_cell)`."
        )
    if issparse(X):
        original_type = type(X)
        if not isspmatrix_csr(X):
            X = csr_matrix(X)
        totals = np.ravel(X.sum(axis=1))  # Faster for csr matrix
        under_target = np.nonzero(totals > counts_per_cell)[0]
        rows = np.split(X.data, X.indptr[1:-1])
        for rowidx in under_target:
            row = rows[rowidx]
            _downsample_array(
                row,
                counts_per_cell[rowidx],
                random_state=random_state,
                replace=replace,
                inplace=True,
            )
        X.eliminate_zeros()
        if original_type is not csr_matrix:  # Put it back
            X = original_type(X)
    else:
        totals = np.ravel(X.sum(axis=1))
        under_target = np.nonzero(totals > counts_per_cell)[0]
        for rowidx in under_target:
            row = X[rowidx, :]
            _downsample_array(
                row,
                counts_per_cell[rowidx],
                random_state=random_state,
                replace=replace,
                inplace=True,
            )
    return X


def _downsample_total_counts(X, total_counts, random_state, replace):
    total_counts = int(total_counts)
    total = X.sum()
    if total < total_counts:
        return X
    if issparse(X):
        original_type = type(X)
        if not isspmatrix_csr(X):
            X = csr_matrix(X)
        _downsample_array(
            X.data,
            total_counts,
            random_state=random_state,
            replace=replace,
            inplace=True,
        )
        X.eliminate_zeros()
        if original_type is not csr_matrix:
            X = original_type(X)
    else:
        v = X.reshape(np.multiply(*X.shape))
        _downsample_array(v, total_counts, random_state, replace=replace, inplace=True)
    return X


# @numba.njit(cache=True)
def _downsample_array(
    col: np.ndarray,
    target: int,
    random_state: AnyRandom = 0,
    replace: bool = True,
    inplace: bool = False,
):
    """\
    Evenly reduce counts in cell to target amount.

    This is an internal function and has some restrictions:

    * total counts in cell must be less than target
    """
    np.random.seed(random_state)
    cumcounts = col.cumsum()
    if inplace:
        col[:] = 0
    else:
        col = np.zeros_like(col)
    total = np.int_(cumcounts[-1])
    sample = np.random.choice(total, target, replace=replace)
    sample.sort()
    geneptr = 0
    for count in sample:
        while count >= cumcounts[geneptr]:
            geneptr += 1
        col[geneptr] += 1
    return col


def filter_genes_dispersion(
    data: AnnData,
    flavor: Literal['seurat', 'cell_ranger'] = 'seurat',
    min_disp: Optional[float] = None,
    max_disp: Optional[float] = None,
    min_mean: Optional[float] = None,
    max_mean: Optional[float] = None,
    n_bins: int = 20,
    n_top_genes: Optional[int] = None,
    log: bool = True,
    subset: bool = True,
    copy: bool = False,
):
    """\
    Extract highly variable genes [Satija15]_ [Zheng17]_.

    .. warning::
        .. deprecated:: 1.3.6
            Use :func:`~scanpy.pp.highly_variable_genes`
            instead. The new function is equivalent to the present
            function, except that

            * the new function always expects logarithmized data
            * `subset=False` in the new function, it suffices to
              merely annotate the genes, tools like `pp.pca` will
              detect the annotation
            * you can now call: `sc.pl.highly_variable_genes(adata)`
            * `copy` is replaced by `inplace`

    If trying out parameters, pass the data matrix instead of AnnData.

    Depending on `flavor`, this reproduces the R-implementations of Seurat
    [Satija15]_ and Cell Ranger [Zheng17]_.

    The normalized dispersion is obtained by scaling with the mean and standard
    deviation of the dispersions for genes falling into a given bin for mean
    expression of genes. This means that for each bin of mean expression, highly
    variable genes are selected.

    Use `flavor='cell_ranger'` with care and in the same way as in
    :func:`~scanpy.pp.recipe_zheng17`.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    flavor
        Choose the flavor for computing normalized dispersion. If choosing
        'seurat', this expects non-logarithmized data – the logarithm of mean
        and dispersion is taken internally when `log` is at its default value
        `True`. For 'cell_ranger', this is usually called for logarithmized data
        – in this case you should set `log` to `False`. In their default
        workflows, Seurat passes the cutoffs whereas Cell Ranger passes
        `n_top_genes`.
    min_mean
    max_mean
    min_disp
    max_disp
        If `n_top_genes` unequals `None`, these cutoffs for the means and the
        normalized dispersions are ignored.
    n_bins
        Number of bins for binning the mean gene expression. Normalization is
        done with respect to each bin. If just a single gene falls into a bin,
        the normalized dispersion is artificially set to 1. You'll be informed
        about this if you set `settings.verbosity = 4`.
    n_top_genes
        Number of highly-variable genes to keep.
    log
        Use the logarithm of the mean to variance ratio.
    subset
        Keep highly-variable genes only (if True) else write a bool array for h
        ighly-variable genes while keeping all genes
    copy
        If an :class:`~anndata.AnnData` is passed, determines whether a copy
        is returned.

    Returns
    -------
    If an AnnData `adata` is passed, returns or updates `adata` depending on
    `copy`. It filters the `adata` and adds the annotations

    **means** : adata.var
        Means per gene. Logarithmized when `log` is `True`.
    **dispersions** : adata.var
        Dispersions per gene. Logarithmized when `log` is `True`.
    **dispersions_norm** : adata.var
        Normalized dispersions per gene. Logarithmized when `log` is `True`.

    If a data matrix `X` is passed, the annotation is returned as `np.recarray`
    with the same information stored in fields: `gene_subset`, `means`, `dispersions`, `dispersion_norm`.
    """
    if n_top_genes is not None and not all(
        x is None for x in [min_disp, max_disp, min_mean, max_mean]
    ):
        # logg.info('If you pass `n_top_genes`, all cutoffs are ignored.')
        pass
    if min_disp is None:
        min_disp = 0.5
    if min_mean is None:
        min_mean = 0.0125
    if max_mean is None:
        max_mean = 3
    if isinstance(data, AnnData):
        adata = data.copy() if copy else data
        result = filter_genes_dispersion(
            adata.X,
            log=log,
            min_disp=min_disp,
            max_disp=max_disp,
            min_mean=min_mean,
            max_mean=max_mean,
            n_top_genes=n_top_genes,
            flavor=flavor,
        )
        adata.var['means'] = result['means']
        adata.var['dispersions'] = result['dispersions']
        adata.var['dispersions_norm'] = result['dispersions_norm']
        if subset:
            adata._inplace_subset_var(result['gene_subset'])
        else:
            adata.var['highly_variable'] = result['gene_subset']
        return adata if copy else None
    # start = logg.info('extracting highly variable genes')
    X = data  # no copy necessary, X remains unchanged in the following
    mean, var = materialize_as_ndarray(_get_mean_var(X))
    # now actually compute the dispersion
    mean[mean == 0] = 1e-12  # set entries equal to zero to small value
    dispersion = var / mean
    if log:  # logarithmized mean as in Seurat
        dispersion[dispersion == 0] = np.nan
        dispersion = np.log(dispersion)
        mean = np.log1p(mean)
    # all of the following quantities are "per-gene" here
    df = pd.DataFrame()
    df['mean'] = mean
    df['dispersion'] = dispersion
    if flavor == 'seurat':
        df['mean_bin'] = pd.cut(df['mean'], bins=n_bins)
        disp_grouped = df.groupby('mean_bin')['dispersion']
        disp_mean_bin = disp_grouped.mean()
        disp_std_bin = disp_grouped.std(ddof=1)
        # retrieve those genes that have nan std, these are the ones where
        # only a single gene fell in the bin and implicitly set them to have
        # a normalized disperion of 1
        one_gene_per_bin = disp_std_bin.isnull()
        gen_indices = np.where(one_gene_per_bin[df['mean_bin'].values])[0].tolist()
        if len(gen_indices) > 0:
            pass
            # logg.debug(
            #     f'Gene indices {gen_indices} fell into a single bin: their '
            #     'normalized dispersion was set to 1.\n    '
            #     'Decreasing `n_bins` will likely avoid this effect.'
            # )
        # Circumvent pandas 0.23 bug. Both sides of the assignment have dtype==float32,
        # but there’s still a dtype error without “.value”.
        disp_std_bin[one_gene_per_bin] = disp_mean_bin[one_gene_per_bin.values].values
        disp_mean_bin[one_gene_per_bin] = 0
        # actually do the normalization
        df['dispersion_norm'] = (
            # use values here as index differs
            df['dispersion'].values
            - disp_mean_bin[df['mean_bin'].values].values
        ) / disp_std_bin[df['mean_bin'].values].values
    elif flavor == 'cell_ranger':
        from statsmodels import robust

        df['mean_bin'] = pd.cut(
            df['mean'],
            np.r_[-np.inf, np.percentile(df['mean'], np.arange(10, 105, 5)), np.inf],
        )
        disp_grouped = df.groupby('mean_bin')['dispersion']
        disp_median_bin = disp_grouped.median()
        # the next line raises the warning: "Mean of empty slice"
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')
        if True:
            disp_mad_bin = disp_grouped.apply(robust.mad)
        df['dispersion_norm'] = (
            np.abs(
                df['dispersion'].values - disp_median_bin[df['mean_bin'].values].values
            )
            / disp_mad_bin[df['mean_bin'].values].values
        )
    else:
        raise ValueError('`flavor` needs to be "seurat" or "cell_ranger"')
    dispersion_norm = df['dispersion_norm'].values.astype('float32')
    if n_top_genes is not None:
        dispersion_norm = dispersion_norm[~np.isnan(dispersion_norm)]
        dispersion_norm[
            ::-1
        ].sort()  # interestingly, np.argpartition is slightly slower
        disp_cut_off = dispersion_norm[n_top_genes - 1]
        gene_subset = df['dispersion_norm'].values >= disp_cut_off
        # logg.debug(
        #     f'the {n_top_genes} top genes correspond to a '
        #     f'normalized dispersion cutoff of {disp_cut_off}'
        # )
    else:
        max_disp = np.inf if max_disp is None else max_disp
        dispersion_norm[np.isnan(dispersion_norm)] = 0  # similar to Seurat
        gene_subset = np.logical_and.reduce(
            (
                mean > min_mean,
                mean < max_mean,
                dispersion_norm > min_disp,
                dispersion_norm < max_disp,
            )
        )
    # logg.info('    finished', time=start)
    return np.rec.fromarrays(
        (
            gene_subset,
            df['mean'].values,
            df['dispersion'].values,
            df['dispersion_norm'].values.astype('float32', copy=False),
        ),
        dtype=[
            ('gene_subset', bool),
            ('means', 'float32'),
            ('dispersions', 'float32'),
            ('dispersions_norm', 'float32'),
        ],
    )


def filter_genes_cv_deprecated(X, Ecutoff, cvFilter):
    """Filter genes by coefficient of variance and mean."""
    return _filter_genes(X, Ecutoff, cvFilter, np.std)


def filter_genes_fano_deprecated(X, Ecutoff, Vcutoff):
    """Filter genes by fano factor and mean."""
    return _filter_genes(X, Ecutoff, Vcutoff, np.var)


def _filter_genes(X, e_cutoff, v_cutoff, meth):
    """\
    See `filter_genes_dispersion`.

    Reference: Weinreb et al. (2017)."""
    if issparse(X):
        raise ValueError('Not defined for sparse input. See `filter_genes_dispersion`.')
    mean_filter = np.mean(X, axis=0) > e_cutoff
    var_filter = meth(X, axis=0) / (np.mean(X, axis=0) + 0.0001) > v_cutoff
    gene_subset = np.nonzero(np.all([mean_filter, var_filter], axis=0))[0]
    return gene_subset


# from .sc_utils import check_nonnegative_integers
def _highly_variable_genes_seurat_v3(
    adata: AnnData,
    layer: Optional[str] = None,
    n_top_genes: int = 2000,
    batch_key: Optional[str] = None,
    check_values: bool = True,
    span: float = 0.3,
    subset: bool = False,
    inplace: bool = True,
) -> Optional[pd.DataFrame]:
    """\
    See `highly_variable_genes`.

    For further implemenation details see https://www.overleaf.com/read/ckptrbgzzzpg

    Returns
    -------
    Depending on `inplace` returns calculated metrics (:class:`~pd.DataFrame`) or
    updates `.var` with the following fields

    highly_variable : bool
        boolean indicator of highly-variable genes
    **means**
        means per gene
    **variances**
        variance per gene
    **variances_norm**
        normalized variance per gene, averaged in the case of multiple batches
    highly_variable_rank : float
        Rank of the gene according to normalized variance, median rank in the case of multiple batches
    highly_variable_nbatches : int
        If batch_key is given, this denotes in how many batches genes are detected as HVG
    """

    try:
        from skmisc.loess import loess
    except ImportError:
        raise ImportError(
            'Please install skmisc package via `pip install --user scikit-misc'
        )
    df = pd.DataFrame(index=adata.var_names)
    X = adata.layers[layer] if layer is not None else adata.X

    # if check_values and not check_nonnegative_integers(X):
        # pass
        # warnings.warn(
        #     "`flavor='seurat_v3'` expects raw count data, but non-integers were found.",
        #     UserWarning,
        # )

    df['means'], df['variances'] = _get_mean_var(X)

    if batch_key is None:
        batch_info = pd.Categorical(np.zeros(adata.shape[0], dtype=int))
    else:
        batch_info = adata.obs[batch_key].values

    norm_gene_vars = []
    for b in np.unique(batch_info):
        X_batch = X[batch_info == b]

        mean, var = _get_mean_var(X_batch)
        not_const = var > 0
        estimat_var = np.zeros(X.shape[1], dtype=np.float64)

        y = np.log10(var[not_const])
        x = np.log10(mean[not_const])
        model = loess(x, y, span=span, degree=2)
        model.fit()
        estimat_var[not_const] = model.outputs.fitted_values
        reg_std = np.sqrt(10 ** estimat_var)

        batch_counts = X_batch.astype(np.float64).copy()
        # clip large values as in Seurat
        N = X_batch.shape[0]
        vmax = np.sqrt(N)
        clip_val = reg_std * vmax + mean
        if sp_sparse.issparse(batch_counts):
            batch_counts = sp_sparse.csr_matrix(batch_counts)
            mask = batch_counts.data > clip_val[batch_counts.indices]
            batch_counts.data[mask] = clip_val[batch_counts.indices[mask]]

            squared_batch_counts_sum = np.array(batch_counts.power(2).sum(axis=0))
            batch_counts_sum = np.array(batch_counts.sum(axis=0))
        else:
            clip_val_broad = np.broadcast_to(clip_val, batch_counts.shape)
            np.putmask(
                batch_counts,
                batch_counts > clip_val_broad,
                clip_val_broad,
            )

            squared_batch_counts_sum = np.square(batch_counts).sum(axis=0)
            batch_counts_sum = batch_counts.sum(axis=0)

        norm_gene_var = (1 / ((N - 1) * np.square(reg_std))) * (
            (N * np.square(mean))
            + squared_batch_counts_sum
            - 2 * batch_counts_sum * mean
        )
        norm_gene_vars.append(norm_gene_var.reshape(1, -1))

    norm_gene_vars = np.concatenate(norm_gene_vars, axis=0)
    # argsort twice gives ranks, small rank means most variable
    ranked_norm_gene_vars = np.argsort(np.argsort(-norm_gene_vars, axis=1), axis=1)

    # this is done in SelectIntegrationFeatures() in Seurat v3
    ranked_norm_gene_vars = ranked_norm_gene_vars.astype(np.float32)
    num_batches_high_var = np.sum(
        (ranked_norm_gene_vars < n_top_genes).astype(int), axis=0
    )
    ranked_norm_gene_vars[ranked_norm_gene_vars >= n_top_genes] = np.nan
    ma_ranked = np.ma.masked_invalid(ranked_norm_gene_vars)
    median_ranked = np.ma.median(ma_ranked, axis=0).filled(np.nan)

    df['highly_variable_nbatches'] = num_batches_high_var
    df['highly_variable_rank'] = median_ranked
    df['variances_norm'] = np.mean(norm_gene_vars, axis=0)

    sorted_index = (
        df[['highly_variable_rank', 'highly_variable_nbatches']]
        .sort_values(
            ['highly_variable_rank', 'highly_variable_nbatches'],
            ascending=[True, False],
            na_position='last',
        )
        .index
    )
    df['highly_variable'] = False
    df.loc[sorted_index[: int(n_top_genes)], 'highly_variable'] = True

    if inplace or subset:
        adata.uns['hvg'] = {'flavor': 'seurat_v3'}
        # logg.hint(
        #     'added\n'
        #     '    \'highly_variable\', boolean vector (adata.var)\n'
        #     '    \'highly_variable_rank\', float vector (adata.var)\n'
        #     '    \'means\', float vector (adata.var)\n'
        #     '    \'variances\', float vector (adata.var)\n'
        #     '    \'variances_norm\', float vector (adata.var)'
        # )
        adata.var['highly_variable'] = df['highly_variable'].values
        adata.var['highly_variable_rank'] = df['highly_variable_rank'].values
        adata.var['means'] = df['means'].values
        adata.var['variances'] = df['variances'].values
        adata.var['variances_norm'] = df['variances_norm'].values.astype(
            'float64', copy=False
        )
        if batch_key is not None:
            adata.var['highly_variable_nbatches'] = df[
                'highly_variable_nbatches'
            ].values
        if subset:
            adata._inplace_subset_var(df['highly_variable'].values)
    else:
        if batch_key is None:
            df = df.drop(['highly_variable_nbatches'], axis=1)
        return df


def _highly_variable_genes_single_batch(
    adata: AnnData,
    layer: Optional[str] = None,
    min_disp: Optional[float] = 0.5,
    max_disp: Optional[float] = np.inf,
    min_mean: Optional[float] = 0.0125,
    max_mean: Optional[float] = 3,
    n_top_genes: Optional[int] = None,
    n_bins: int = 20,
    flavor: Literal['seurat', 'cell_ranger'] = 'seurat',
) -> pd.DataFrame:
    """\
    See `highly_variable_genes`.

    Returns
    -------
    A DataFrame that contains the columns
    `highly_variable`, `means`, `dispersions`, and `dispersions_norm`.
    """
    X = adata.layers[layer] if layer is not None else adata.X
    if flavor == 'seurat':
        if 'log1p' in adata.uns_keys() and adata.uns['log1p']['base'] is not None:
            X *= np.log(adata.uns['log1p']['base'])
        X = np.expm1(X)

    mean, var = materialize_as_ndarray(_get_mean_var(X))
    # now actually compute the dispersion
    mean[mean == 0] = 1e-12  # set entries equal to zero to small value
    dispersion = var / mean
    if flavor == 'seurat':  # logarithmized mean as in Seurat
        dispersion[dispersion == 0] = np.nan
        dispersion = np.log(dispersion)
        mean = np.log1p(mean)
    # all of the following quantities are "per-gene" here
    df = pd.DataFrame()
    df['means'] = mean
    df['dispersions'] = dispersion
    if flavor == 'seurat':
        df['mean_bin'] = pd.cut(df['means'], bins=n_bins)
        disp_grouped = df.groupby('mean_bin')['dispersions']
        disp_mean_bin = disp_grouped.mean()
        disp_std_bin = disp_grouped.std(ddof=1)
        # retrieve those genes that have nan std, these are the ones where
        # only a single gene fell in the bin and implicitly set them to have
        # a normalized disperion of 1
        one_gene_per_bin = disp_std_bin.isnull()
        gen_indices = np.where(one_gene_per_bin[df['mean_bin'].values])[0].tolist()
        if len(gen_indices) > 0:
            pass
            # logg.debug(
            #     f'Gene indices {gen_indices} fell into a single bin: their '
            #     'normalized dispersion was set to 1.\n    '
            #     'Decreasing `n_bins` will likely avoid this effect.'
            # )
        # Circumvent pandas 0.23 bug. Both sides of the assignment have dtype==float32,
        # but there’s still a dtype error without “.value”.
        disp_std_bin[one_gene_per_bin.values] = disp_mean_bin[
            one_gene_per_bin.values
        ].values
        disp_mean_bin[one_gene_per_bin.values] = 0
        # actually do the normalization
        df['dispersions_norm'] = (
            df['dispersions'].values  # use values here as index differs
            - disp_mean_bin[df['mean_bin'].values].values
        ) / disp_std_bin[df['mean_bin'].values].values
    elif flavor == 'cell_ranger':
        from statsmodels import robust

        df['mean_bin'] = pd.cut(
            df['means'],
            np.r_[-np.inf, np.percentile(df['means'], np.arange(10, 105, 5)), np.inf],
        )
        disp_grouped = df.groupby('mean_bin')['dispersions']
        disp_median_bin = disp_grouped.median()
        # the next line raises the warning: "Mean of empty slice"
        # with warnings.catch_warnings():
        if True:
            # warnings.simplefilter('ignore')
            disp_mad_bin = disp_grouped.apply(robust.mad)
            df['dispersions_norm'] = (
                df['dispersions'].values - disp_median_bin[df['mean_bin'].values].values
            ) / disp_mad_bin[df['mean_bin'].values].values
    else:
        raise ValueError('`flavor` needs to be "seurat" or "cell_ranger"')
    dispersion_norm = df['dispersions_norm'].values
    if n_top_genes is not None:
        dispersion_norm = dispersion_norm[~np.isnan(dispersion_norm)]
        dispersion_norm[
            ::-1
        ].sort()  # interestingly, np.argpartition is slightly slower
        if n_top_genes > adata.n_vars:
            # logg.info('`n_top_genes` > `adata.n_var`, returning all genes.')
            n_top_genes = adata.n_vars
        disp_cut_off = dispersion_norm[n_top_genes - 1]
        gene_subset = np.nan_to_num(df['dispersions_norm'].values) >= disp_cut_off
        # logg.debug(
        #     f'the {n_top_genes} top genes correspond to a '
        #     f'normalized dispersion cutoff of {disp_cut_off}'
        # )
    else:
        dispersion_norm[np.isnan(dispersion_norm)] = 0  # similar to Seurat
        gene_subset = np.logical_and.reduce(
            (
                mean > min_mean,
                mean < max_mean,
                dispersion_norm > min_disp,
                dispersion_norm < max_disp,
            )
        )

    df['highly_variable'] = gene_subset
    return df


def highly_variable_genes(
    adata: AnnData,
    layer: Optional[str] = None,
    n_top_genes: Optional[int] = None,
    min_disp: Optional[float] = 0.5,
    max_disp: Optional[float] = np.inf,
    min_mean: Optional[float] = 0.0125,
    max_mean: Optional[float] = 3,
    span: Optional[float] = 0.3,
    n_bins: int = 20,
    flavor: Literal['seurat', 'cell_ranger', 'seurat_v3'] = 'seurat',
    subset: bool = False,
    inplace: bool = True,
    batch_key: Optional[str] = None,
    check_values: bool = True,
) -> Optional[pd.DataFrame]:
    """\
    Annotate highly variable genes [Satija15]_ [Zheng17]_ [Stuart19]_.

    Expects logarithmized data, except when `flavor='seurat_v3'` in which
    count data is expected.

    Depending on `flavor`, this reproduces the R-implementations of Seurat
    [Satija15]_, Cell Ranger [Zheng17]_, and Seurat v3 [Stuart19]_.

    For the dispersion-based methods ([Satija15]_ and [Zheng17]_), the normalized
    dispersion is obtained by scaling with the mean and standard deviation of
    the dispersions for genes falling into a given bin for mean expression of
    genes. This means that for each bin of mean expression, highly variable
    genes are selected.

    For [Stuart19]_, a normalized variance for each gene is computed. First, the data
    are standardized (i.e., z-score normalization per feature) with a regularized
    standard deviation. Next, the normalized variance is computed as the variance
    of each gene after the transformation. Genes are ranked by the normalized variance.

    Parameters
    ----------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    layer
        If provided, use `adata.layers[layer]` for expression values instead of `adata.X`.
    n_top_genes
        Number of highly-variable genes to keep. Mandatory if `flavor='seurat_v3'`.
    min_mean
        If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
        normalized dispersions are ignored. Ignored if `flavor='seurat_v3'`.
    max_mean
        If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
        normalized dispersions are ignored. Ignored if `flavor='seurat_v3'`.
    min_disp
        If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
        normalized dispersions are ignored. Ignored if `flavor='seurat_v3'`.
    max_disp
        If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
        normalized dispersions are ignored. Ignored if `flavor='seurat_v3'`.
    span
        The fraction of the data (cells) used when estimating the variance in the loess
        model fit if `flavor='seurat_v3'`.
    n_bins
        Number of bins for binning the mean gene expression. Normalization is
        done with respect to each bin. If just a single gene falls into a bin,
        the normalized dispersion is artificially set to 1. You'll be informed
        about this if you set `settings.verbosity = 4`.
    flavor
        Choose the flavor for identifying highly variable genes. For the dispersion
        based methods in their default workflows, Seurat passes the cutoffs whereas
        Cell Ranger passes `n_top_genes`.
    subset
        Inplace subset to highly-variable genes if `True` otherwise merely indicate
        highly variable genes.
    inplace
        Whether to place calculated metrics in `.var` or return them.
    batch_key
        If specified, highly-variable genes are selected within each batch separately and merged.
        This simple process avoids the selection of batch-specific genes and acts as a
        lightweight batch correction method. For all flavors, genes are first sorted
        by how many batches they are a HVG. For dispersion-based flavors ties are broken
        by normalized dispersion. If `flavor = 'seurat_v3'`, ties are broken by the median
        (across batches) rank based on within-batch normalized variance.
    check_values
        Check if counts in selected layer are integers. A Warning is returned if set to True.
        Only used if `flavor='seurat_v3'`.

    Returns
    -------
    Depending on `inplace` returns calculated metrics (:class:`~pandas.DataFrame`) or
    updates `.var` with the following fields

    highly_variable : bool
        boolean indicator of highly-variable genes
    **means**
        means per gene
    **dispersions**
        For dispersion-based flavors, dispersions per gene
    **dispersions_norm**
        For dispersion-based flavors, normalized dispersions per gene
    **variances**
        For `flavor='seurat_v3'`, variance per gene
    **variances_norm**
        For `flavor='seurat_v3'`, normalized variance per gene, averaged in
        the case of multiple batches
    highly_variable_rank : float
        For `flavor='seurat_v3'`, rank of the gene according to normalized
        variance, median rank in the case of multiple batches
    highly_variable_nbatches : int
        If batch_key is given, this denotes in how many batches genes are detected as HVG
    highly_variable_intersection : bool
        If batch_key is given, this denotes the genes that are highly variable in all batches

    Notes
    -----
    This function replaces :func:`~scanpy.pp.filter_genes_dispersion`.
    """

    if n_top_genes is not None and not all(
        m is None for m in [min_disp, max_disp, min_mean, max_mean]
    ):
        # logg.info('If you pass `n_top_genes`, all cutoffs are ignored.')
        pass

    # start = logg.info('extracting highly variable genes')

    if not isinstance(adata, AnnData):
        raise ValueError(
            '`pp.highly_variable_genes` expects an `AnnData` argument, '
            'pass `inplace=False` if you want to return a `pd.DataFrame`.'
        )

    if flavor == 'seurat_v3':
        return _highly_variable_genes_seurat_v3(
            adata,
            layer=layer,
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            check_values=check_values,
            span=span,
            subset=subset,
            inplace=inplace,
        )

    if batch_key is None:
        df = _highly_variable_genes_single_batch(
            adata,
            layer=layer,
            min_disp=min_disp,
            max_disp=max_disp,
            min_mean=min_mean,
            max_mean=max_mean,
            n_top_genes=n_top_genes,
            n_bins=n_bins,
            flavor=flavor,
        )
    else:
        sanitize_anndata(adata)
        batches = adata.obs[batch_key].cat.categories
        df = []
        gene_list = adata.var_names
        for batch in batches:
            adata_subset = adata[adata.obs[batch_key] == batch]

            # Filter to genes that are in the dataset
            # with settings.verbosity.override(Verbosity.error):
            if True:
                filt = filter_genes(adata_subset, min_cells=1, inplace=False)[0]

            adata_subset = adata_subset[:, filt]

            hvg = _highly_variable_genes_single_batch(
                adata_subset,
                min_disp=min_disp,
                max_disp=max_disp,
                min_mean=min_mean,
                max_mean=max_mean,
                n_top_genes=n_top_genes,
                n_bins=n_bins,
                flavor=flavor,
            )

            # Add 0 values for genes that were filtered out
            missing_hvg = pd.DataFrame(
                np.zeros((np.sum(~filt), len(hvg.columns))),
                columns=hvg.columns,
            )
            missing_hvg['highly_variable'] = missing_hvg['highly_variable'].astype(bool)
            missing_hvg['gene'] = gene_list[~filt]
            hvg['gene'] = adata_subset.var_names.values
            hvg = hvg.append(missing_hvg, ignore_index=True)

            # Order as before filtering
            idxs = np.concatenate((np.where(filt)[0], np.where(~filt)[0]))
            hvg = hvg.loc[np.argsort(idxs)]

            df.append(hvg)

        df = pd.concat(df, axis=0)
        df['highly_variable'] = df['highly_variable'].astype(int)
        df = df.groupby('gene').agg(
            dict(
                means=np.nanmean,
                dispersions=np.nanmean,
                dispersions_norm=np.nanmean,
                highly_variable=np.nansum,
            )
        )
        df.rename(
            columns=dict(highly_variable='highly_variable_nbatches'), inplace=True
        )
        df['highly_variable_intersection'] = df['highly_variable_nbatches'] == len(
            batches
        )

        if n_top_genes is not None:
            # sort genes by how often they selected as hvg within each batch and
            # break ties with normalized dispersion across batches
            df.sort_values(
                ['highly_variable_nbatches', 'dispersions_norm'],
                ascending=False,
                na_position='last',
                inplace=True,
            )
            df['highly_variable'] = False
            df.highly_variable.iloc[:n_top_genes] = True
            df = df.loc[adata.var_names]
        else:
            df = df.loc[adata.var_names]
            dispersion_norm = df.dispersions_norm.values
            dispersion_norm[np.isnan(dispersion_norm)] = 0  # similar to Seurat
            gene_subset = np.logical_and.reduce(
                (
                    df.means > min_mean,
                    df.means < max_mean,
                    df.dispersions_norm > min_disp,
                    df.dispersions_norm < max_disp,
                )
            )
            df['highly_variable'] = gene_subset

    # logg.info('    finished', time=start)

    if inplace or subset:
        adata.uns['hvg'] = {'flavor': flavor}
        # logg.hint(
        #     'added\n'
        #     '    \'highly_variable\', boolean vector (adata.var)\n'
        #     '    \'means\', float vector (adata.var)\n'
        #     '    \'dispersions\', float vector (adata.var)\n'
        #     '    \'dispersions_norm\', float vector (adata.var)'
        # )
        adata.var['highly_variable'] = df['highly_variable'].values
        adata.var['means'] = df['means'].values
        adata.var['dispersions'] = df['dispersions'].values
        adata.var['dispersions_norm'] = df['dispersions_norm'].values.astype(
            'float32', copy=False
        )
        if batch_key is not None:
            adata.var['highly_variable_nbatches'] = df[
                'highly_variable_nbatches'
            ].values
            adata.var['highly_variable_intersection'] = df[
                'highly_variable_intersection'
            ].values
        if subset:
            adata._inplace_subset_var(df['highly_variable'].values)
    else:
        return df


from scipy.sparse.linalg import LinearOperator, svds
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import svd_flip
# from .. import settings


def pca(
    data: Union[AnnData, np.ndarray, spmatrix],
    n_comps: Optional[int] = None,
    zero_center: Optional[bool] = True,
    svd_solver: str = 'arpack',
    random_state: AnyRandom = 0,
    return_info: bool = False,
    use_highly_variable: Optional[bool] = None,
    dtype: str = 'float32',
    copy: bool = False,
    chunked: bool = False,
    chunk_size: Optional[int] = None,
) -> Union[AnnData, np.ndarray, spmatrix]:
    """\
    Principal component analysis [Pedregosa11]_.

    Computes PCA coordinates, loadings and variance decomposition.
    Uses the implementation of *scikit-learn* [Pedregosa11]_.

    .. versionchanged:: 1.5.0

        In previous versions, computing a PCA on a sparse matrix would make a dense copy of
        the array for mean centering.
        As of scanpy 1.5.0, mean centering is implicit.
        While results are extremely similar, they are not exactly the same.
        If you would like to reproduce the old results, pass a dense array.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    n_comps
        Number of principal components to compute. Defaults to 50, or 1 - minimum
        dimension size of selected representation.
    zero_center
        If `True`, compute standard PCA from covariance matrix.
        If `False`, omit zero-centering variables
        (uses :class:`~sklearn.decomposition.TruncatedSVD`),
        which allows to handle sparse input efficiently.
        Passing `None` decides automatically based on sparseness of the data.
    svd_solver
        SVD solver to use:

        `'arpack'` (the default)
          for the ARPACK wrapper in SciPy (:func:`~scipy.sparse.linalg.svds`)
        `'randomized'`
          for the randomized algorithm due to Halko (2009).
        `'auto'`
          chooses automatically depending on the size of the problem.
        `'lobpcg'`
          An alternative SciPy solver.

        .. versionchanged:: 1.4.5
           Default value changed from `'auto'` to `'arpack'`.

        Efficient computation of the principal components of a sparse matrix
        currently only works with the `'arpack`' or `'lobpcg'` solvers.

    random_state
        Change to use different initial states for the optimization.
    return_info
        Only relevant when not passing an :class:`~anndata.AnnData`:
        see “**Returns**”.
    use_highly_variable
        Whether to use highly variable genes only, stored in
        `.var['highly_variable']`.
        By default uses them if they have been determined beforehand.
    dtype
        Numpy data type string to which to convert the result.
    copy
        If an :class:`~anndata.AnnData` is passed, determines whether a copy
        is returned. Is ignored otherwise.
    chunked
        If `True`, perform an incremental PCA on segments of `chunk_size`.
        The incremental PCA automatically zero centers and ignores settings of
        `random_seed` and `svd_solver`. If `False`, perform a full PCA.
    chunk_size
        Number of observations to include in each chunk.
        Required if `chunked=True` was passed.

    Returns
    -------
    X_pca : :class:`~scipy.sparse.spmatrix`, :class:`~numpy.ndarray`
        If `data` is array-like and `return_info=False` was passed,
        this function only returns `X_pca`…
    adata : anndata.AnnData
        …otherwise if `copy=True` it returns or else adds fields to `adata`:

        `.obsm['X_pca']`
             PCA representation of data.
        `.varm['PCs']`
             The principal components containing the loadings.
        `.uns['pca']['variance_ratio']`
             Ratio of explained variance.
        `.uns['pca']['variance']`
             Explained variance, equivalent to the eigenvalues of the
             covariance matrix.
    """
    # logg_start = logg.info('computing PCA')

    # chunked calculation is not randomized, anyways
    if svd_solver in {'auto', 'randomized'} and not chunked:
        # logg.info
        print(
            'Note that scikit-learn\'s randomized PCA might not be exactly '
            'reproducible across different computational platforms. For exact '
            'reproducibility, choose `svd_solver=\'arpack\'.`'
        )
    data_is_AnnData = isinstance(data, AnnData)
    if data_is_AnnData:
        adata = data.copy() if copy else data
    else:
        adata = AnnData(data, dtype=data.dtype)

    if use_highly_variable is True and 'highly_variable' not in adata.var.keys():
        raise ValueError(
            'Did not find adata.var[\'highly_variable\']. '
            'Either your data already only consists of highly-variable genes '
            'or consider running `pp.highly_variable_genes` first.'
        )
    if use_highly_variable is None:
        use_highly_variable = True if 'highly_variable' in adata.var.keys() else False
    if use_highly_variable:
        # logg.info
        print('    on highly variable genes')
    adata_comp = (
        adata[:, adata.var['highly_variable']] if use_highly_variable else adata
    )

    if n_comps is None:
        min_dim = min(adata_comp.n_vars, adata_comp.n_obs)
        N_PCS = 50
        if N_PCS >= min_dim:
            n_comps = min_dim - 1
        else:
            n_comps = N_PCS

    # logg.info(f'    with n_comps={n_comps}')

    random_state = check_random_state(random_state)

    X = adata_comp.X

    if chunked:
        if not zero_center or random_state or svd_solver != 'arpack':
            # logg.debug
            print('Ignoring zero_center, random_state, svd_solver')

        from sklearn.decomposition import IncrementalPCA

        X_pca = np.zeros((X.shape[0], n_comps), X.dtype)

        pca_ = IncrementalPCA(n_components=n_comps)

        for chunk, _, _ in adata_comp.chunked_X(chunk_size):
            chunk = chunk.A if issparse(chunk) else chunk#toarray()
            pca_.partial_fit(chunk)

        for chunk, start, end in adata_comp.chunked_X(chunk_size):
            chunk = chunk.A if issparse(chunk) else chunk#toarray()
            X_pca[start:end] = pca_.transform(chunk)
    elif (not issparse(X) or svd_solver == "randomized") and zero_center:
        from sklearn.decomposition import PCA

        if issparse(X) and svd_solver == "randomized":
            # This  is for backwards compat. Better behaviour would be to either error or use arpack.
            # logg.warning
            print(
                "svd_solver 'randomized' does not work with sparse input. Densifying the array. "
                "This may take a very large amount of memory."
            )
            X = X.A#toarray()
        pca_ = PCA(
            n_components=n_comps, svd_solver=svd_solver, random_state=random_state
        )
        X_pca = pca_.fit_transform(X)
    elif issparse(X) and zero_center:
        from sklearn.decomposition import PCA

        if svd_solver == "auto":
            svd_solver = "arpack"
        if svd_solver not in {'lobpcg', 'arpack'}:
            raise ValueError(
                'svd_solver: {svd_solver} can not be used with sparse input.\n'
                'Use "arpack" (the default) or "lobpcg" instead.'
            )

        output = _pca_with_sparse(
            X, n_comps, solver=svd_solver, random_state=random_state
        )
        # this is just a wrapper for the results
        X_pca = output['X_pca']
        pca_ = PCA(n_components=n_comps, svd_solver=svd_solver)
        pca_.components_ = output['components']
        pca_.explained_variance_ = output['variance']
        pca_.explained_variance_ratio_ = output['variance_ratio']
    elif not zero_center:
        from sklearn.decomposition import TruncatedSVD

        # logg.debug
        print(
            '    without zero-centering: \n'
            '    the explained variance does not correspond to the exact statistical defintion\n'
            '    the first component, e.g., might be heavily influenced by different means\n'
            '    the following components often resemble the exact PCA very closely'
        )
        pca_ = TruncatedSVD(
            n_components=n_comps, random_state=random_state, algorithm=svd_solver
        )
        X_pca = pca_.fit_transform(X)
    else:
        raise Exception("This shouldn't happen. Please open a bug report.")

    if X_pca.dtype.descr != np.dtype(dtype).descr:
        X_pca = X_pca.astype(dtype)

    if data_is_AnnData:
        adata.obsm['X_pca'] = X_pca
        adata.uns['pca'] = {}
        adata.uns['pca']['params'] = {
            'zero_center': zero_center,
            'use_highly_variable': use_highly_variable,
        }
        if use_highly_variable:
            adata.varm['PCs'] = np.zeros(shape=(adata.n_vars, n_comps))
            adata.varm['PCs'][adata.var['highly_variable']] = pca_.components_.T
        else:
            adata.varm['PCs'] = pca_.components_.T
        adata.uns['pca']['variance'] = pca_.explained_variance_
        adata.uns['pca']['variance_ratio'] = pca_.explained_variance_ratio_
        # logg.info('    finished', time=logg_start)
        # logg.debug(
        #     'and added\n'
        #     '    \'X_pca\', the PCA coordinates (adata.obs)\n'
        #     '    \'PC1\', \'PC2\', ..., the loadings (adata.var)\n'
        #     '    \'pca_variance\', the variance / eigenvalues (adata.uns)\n'
        #     '    \'pca_variance_ratio\', the variance ratio (adata.uns)'
        # )
        return adata if copy else None
    else:
        # logg.info('    finished', time=logg_start)
        if return_info:
            return (
                X_pca,
                pca_.components_,
                pca_.explained_variance_ratio_,
                pca_.explained_variance_,
            )
        else:
            return X_pca


def _pca_with_sparse(X, npcs, solver='arpack', mu=None, random_state=None):
    random_state = check_random_state(random_state)
    np.random.set_state(random_state.get_state())
    random_init = np.random.rand(np.min(X.shape))
    X = check_array(X, accept_sparse=['csr', 'csc'])

    if mu is None:
        mu = X.mean(0).A.flatten()[None, :]#toarray()
    mdot = mu.dot
    mmat = mdot
    mhdot = mu.T.dot
    mhmat = mu.T.dot
    Xdot = X.dot
    Xmat = Xdot
    XHdot = X.T.conj().dot
    XHmat = XHdot
    ones = np.ones(X.shape[0])[None, :].dot

    def matvec(x):
        return Xdot(x) - mdot(x)

    def matmat(x):
        return Xmat(x) - mmat(x)

    def rmatvec(x):
        return XHdot(x) - mhdot(ones(x))

    def rmatmat(x):
        return XHmat(x) - mhmat(ones(x))

    XL = LinearOperator(
        matvec=matvec,
        dtype=X.dtype,
        matmat=matmat,
        shape=X.shape,
        rmatvec=rmatvec,
        rmatmat=rmatmat,
    )

    u, s, v = svds(XL, solver=solver, k=npcs, v0=random_init)
    u, v = svd_flip(u, v)
    idx = np.argsort(-s)
    v = v[idx, :]

    X_pca = (u * s)[:, idx]
    ev = s[idx] ** 2 / (X.shape[0] - 1)

    total_var = _get_mean_var(X)[1].sum()
    ev_ratio = ev / total_var

    output = {
        'X_pca': X_pca,
        'variance': ev,
        'variance_ratio': ev_ratio,
        'components': v,
    }
    return output

from types import MappingProxyType
from scipy.sparse import coo_matrix
from .sc_utils import NeighborsView
# from .sc_utils import _choose_representation, doc_use_rep, doc_n_pcs
doc_use_rep = """\
use_rep
    Use the indicated representation. `'X'` or any key for `.obsm` is valid.
    If `None`, the representation is chosen automatically:
    For `.n_vars` < 50, `.X` is used, otherwise 'X_pca' is used.
    If 'X_pca' is not present, it’s computed with default parameters.\
"""

doc_n_pcs = """\
n_pcs
    Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`.\
"""

def _choose_graph(adata, obsp, neighbors_key):
    """Choose connectivities from neighbbors or another obsp column"""
    if obsp is not None and neighbors_key is not None:
        raise ValueError(
            'You can\'t specify both obsp, neighbors_key. ' 'Please select only one.'
        )

    if obsp is not None:
        return adata.obsp[obsp]
    else:
        neighbors = NeighborsView(adata, neighbors_key)
        if 'connectivities' not in neighbors:
            raise ValueError(
                'You need to run `pp.neighbors` first '
                'to compute a neighborhood graph.'
            )
        return neighbors['connectivities']


def _choose_representation(adata, use_rep=None, n_pcs=None, silent=False):
    # verbosity = settings.verbosity
    # if silent and settings.verbosity > 1:
    verbosity = 1
    if use_rep is None and n_pcs == 0:  # backwards compat for specifying `.X`
        use_rep = 'X'
    if use_rep is None:
        if adata.n_vars > 50:
            if 'X_pca' in adata.obsm.keys():
                if n_pcs is not None and n_pcs > adata.obsm['X_pca'].shape[1]:
                    raise ValueError(
                        '`X_pca` does not have enough PCs. Rerun `sc.pp.pca` with adjusted `n_comps`.'
                    )
                X = adata.obsm['X_pca'][:, :n_pcs]
                # logg.info(f'    using \'X_pca\' with n_pcs = {X.shape[1]}')
            else:
                # logg.warning(
                #     f'You’re trying to run this on {adata.n_vars} dimensions of `.X`, '
                #     'if you really want this, set `use_rep=\'X\'`.\n         '
                #     'Falling back to preprocessing with `sc.pp.pca` and default params.'
                # )
                X = pca(adata.X)
                adata.obsm['X_pca'] = X[:, :n_pcs]
        else:
            # logg.info('    using data matrix X directly')
            X = adata.X
    else:
        if use_rep in adata.obsm.keys():
            X = adata.obsm[use_rep]
            if use_rep == 'X_pca' and n_pcs is not None:
                X = adata.obsm[use_rep][:, :n_pcs]
        elif use_rep == 'X':
            X = adata.X
        else:
            raise ValueError(
                'Did not find {} in `.obsm.keys()`. '
                'You need to compute it first.'.format(use_rep)
            )
    # settings.verbosity = verbosity  # resetting verbosity
    return X


def preprocess_with_pca(adata, n_pcs: Optional[int] = None, random_state=0):
    """
    Parameters
    ----------
    n_pcs
        If `n_pcs=0`, do not preprocess with PCA.
        If `None` and there is a PCA version of the data, use this.
        If an integer, compute the PCA.
    """
    if n_pcs == 0:
        # logg.info('    using data matrix X directly (no PCA)')
        return adata.X
    elif n_pcs is None and 'X_pca' in adata.obsm_keys():
        # logg.info(f'    using \'X_pca\' with n_pcs = {adata.obsm["X_pca"].shape[1]}')
        return adata.obsm['X_pca']
    elif 'X_pca' in adata.obsm_keys() and adata.obsm['X_pca'].shape[1] >= n_pcs:
        # logg.info(f'    using \'X_pca\' with n_pcs = {n_pcs}')
        return adata.obsm['X_pca'][:, :n_pcs]
    else:
        n_pcs = 50 if n_pcs is None else n_pcs
        if adata.X.shape[1] > n_pcs:
            # logg.info(f'    computing \'X_pca\' with n_pcs = {n_pcs}')
            # logg.hint('avoid this by setting n_pcs = 0')
            X = pca(adata.X, n_comps=n_pcs, random_state=random_state)
            adata.obsm['X_pca'] = X
            return X
        else:
            pass
            # logg.info('    using data matrix X directly (no PCA)')
            return adata.X


def get_init_pos_from_paga(
    adata, adjacency=None, random_state=0, neighbors_key=None, obsp=None
):
    np.random.seed(random_state)
    if adjacency is None:
        adjacency = _choose_graph(adata, obsp, neighbors_key)
    if 'paga' in adata.uns and 'pos' in adata.uns['paga']:
        groups = adata.obs[adata.uns['paga']['groups']]
        pos = adata.uns['paga']['pos']
        connectivities_coarse = adata.uns['paga']['connectivities']
        init_pos = np.ones((adjacency.shape[0], 2))
        for i, group_pos in enumerate(pos):
            subset = (groups == groups.cat.categories[i]).values
            neighbors = connectivities_coarse[i].nonzero()
            if len(neighbors[1]) > 0:
                connectivities = connectivities_coarse[i][neighbors]
                nearest_neighbor = neighbors[1][np.argmax(connectivities)]
                noise = np.random.random((len(subset[subset]), 2))
                dist = pos[i] - pos[nearest_neighbor]
                noise = noise * dist
                init_pos[subset] = group_pos - 0.5 * dist + noise
            else:
                init_pos[subset] = group_pos
    else:
        raise ValueError(
            'Plot PAGA first, so that adata.uns[\'paga\']' 'with key \'pos\'.'
        )
    return init_pos

def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""
    import igraph as ig

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except KeyError:
        pass
    if g.vcount() != adjacency.shape[0]:
        pass
        # logg.warning(
        #     f'The constructed graph has only {g.vcount()} nodes. '
        #     'Your adjacency matrix contained redundant nodes.'
        # )
    return g


def get_sparse_from_igraph(graph, weight_attr=None):
    from scipy.sparse import csr_matrix

    edges = graph.get_edgelist()
    if weight_attr is None:
        weights = [1] * len(edges)
    else:
        weights = graph.es[weight_attr]
    if not graph.is_directed():
        edges.extend([(v, u) for u, v in edges])
        weights.extend(weights)
    shape = graph.vcount()
    shape = (shape, shape)
    if len(edges) > 0:
        return csr_matrix((weights, zip(*edges)), shape=shape)
    else:
        return csr_matrix(shape)

def check_nonnegative_integers(X: Union[np.ndarray, sparse.spmatrix]):
    """Checks values of X to ensure it is count data"""
    from numbers import Integral

    data = X if isinstance(X, np.ndarray) else X.data
    # Check no negatives
    if np.signbit(data).any():
        return False
    # Check all are integers
    elif issubclass(data.dtype.type, Integral):
        return True
    elif np.any(~np.equal(np.mod(data, 1), 0)):
        return False
    else:
        return True

_InitPos = Literal['paga', 'spectral', 'random']

def umap(
    adata: AnnData,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: Optional[int] = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos: Union[_InitPos, np.ndarray, None] = 'spectral',
    random_state: AnyRandom = 0,
    a: Optional[float] = None,
    b: Optional[float] = None,
    copy: bool = False,
    method: Literal['umap', 'rapids'] = 'umap',
    neighbors_key: Optional[str] = None,
) -> Optional[AnnData]:
    """\
    Embed the neighborhood graph using UMAP [McInnes18]_.

    UMAP (Uniform Manifold Approximation and Projection) is a manifold learning
    technique suitable for visualizing high-dimensional data. Besides tending to
    be faster than tSNE, it optimizes the embedding such that it best reflects
    the topology of the data, which we represent throughout Scanpy using a
    neighborhood graph. tSNE, by contrast, optimizes the distribution of
    nearest-neighbor distances in the embedding such that these best match the
    distribution of distances in the high-dimensional space.  We use the
    implementation of `umap-learn <https://github.com/lmcinnes/umap>`__
    [McInnes18]_. For a few comparisons of UMAP with tSNE, see this `preprint
    <https://doi.org/10.1101/298430>`__.

    Parameters
    ----------
    adata
        Annotated data matrix.
    min_dist
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points on
        the manifold are drawn closer together, while larger values will result
        on a more even dispersal of points. The value should be set relative to
        the ``spread`` value, which determines the scale at which embedded
        points will be spread out. The default of in the `umap-learn` package is
        0.1.
    spread
        The effective scale of embedded points. In combination with `min_dist`
        this determines how clustered/clumped the embedded points are.
    n_components
        The number of dimensions of the embedding.
    maxiter
        The number of iterations (epochs) of the optimization. Called `n_epochs`
        in the original UMAP.
    alpha
        The initial learning rate for the embedding optimization.
    gamma
        Weighting applied to negative samples in low dimensional embedding
        optimization. Values higher than one will result in greater weight
        being given to negative samples.
    negative_sample_rate
        The number of negative edge/1-simplex samples to use per positive
        edge/1-simplex sample in optimizing the low dimensional embedding.
    init_pos
        How to initialize the low dimensional embedding. Called `init` in the
        original UMAP. Options are:

        * Any key for `adata.obsm`.
        * 'paga': positions from :func:`~scanpy.pl.paga`.
        * 'spectral': use a spectral embedding of the graph.
        * 'random': assign initial embedding positions at random.
        * A numpy array of initial embedding positions.
    random_state
        If `int`, `random_state` is the seed used by the random number generator;
        If `RandomState` or `Generator`, `random_state` is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    a
        More specific parameters controlling the embedding. If `None` these
        values are set automatically as determined by `min_dist` and
        `spread`.
    b
        More specific parameters controlling the embedding. If `None` these
        values are set automatically as determined by `min_dist` and
        `spread`.
    copy
        Return a copy instead of writing to adata.
    method
        Use the original 'umap' implementation, or 'rapids' (experimental, GPU only)
    neighbors_key
        If not specified, umap looks .uns['neighbors'] for neighbors settings
        and .obsp['connectivities'] for connectivities
        (default storage places for pp.neighbors).
        If specified, umap looks .uns[neighbors_key] for neighbors settings and
        .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.

    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.

    **X_umap** : `adata.obsm` field
        UMAP coordinates of data.
    """
    adata = adata.copy() if copy else adata

    if neighbors_key is None:
        neighbors_key = 'neighbors'

    if neighbors_key not in adata.uns:
        raise ValueError(
            f'Did not find .uns["{neighbors_key}"]. Run `sc.pp.neighbors` first.'
        )
    # start = logg.info('computing UMAP')

    neighbors = NeighborsView(adata, neighbors_key)

    if 'params' not in neighbors or neighbors['params']['method'] != 'umap':
        pass
        # logg.warning(
        #     f'.obsp["{neighbors["connectivities_key"]}"] have not been computed using umap'
        # )

    # # Compat for umap 0.4 -> 0.5
    # with warnings.catch_warnings():
    #     # umap 0.5.0
    #     warnings.filterwarnings("ignore", message=r"Tensorflow not installed")
    import umap

    # if version.parse(umap.__version__) >= version.parse("0.5.0"):
    if True:
        def simplicial_set_embedding(*args, **kwargs):
            from umap.umap_ import simplicial_set_embedding

            X_umap, _ = simplicial_set_embedding(
                *args,
                densmap=False,
                densmap_kwds={},
                output_dens=False,
                **kwargs,
            )
            return X_umap

    # else:
    #     from umap.umap_ import simplicial_set_embedding
    from umap.umap_ import find_ab_params

    if a is None or b is None:
        a, b = find_ab_params(spread, min_dist)
    else:
        a = a
        b = b
    adata.uns['umap'] = {'params': {'a': a, 'b': b}}
    if isinstance(init_pos, str) and init_pos in adata.obsm.keys():
        init_coords = adata.obsm[init_pos]
    elif isinstance(init_pos, str) and init_pos == 'paga':
        init_coords = get_init_pos_from_paga(
            adata, random_state=random_state, neighbors_key=neighbors_key
        )
    else:
        init_coords = init_pos  # Let umap handle it
    if hasattr(init_coords, "dtype"):
        init_coords = check_array(init_coords, dtype=np.float32, accept_sparse=False)

    if random_state != 0:
        adata.uns['umap']['params']['random_state'] = random_state
    random_state = check_random_state(random_state)

    neigh_params = neighbors['params']
    X = _choose_representation(
        adata,
        neigh_params.get('use_rep', None),
        neigh_params.get('n_pcs', None),
        silent=True,
    )
    if method == 'umap':
        # the data matrix X is really only used for determining the number of connected components
        # for the init condition in the UMAP embedding
        default_epochs = 500 if neighbors['connectivities'].shape[0] <= 10000 else 200
        n_epochs = default_epochs if maxiter is None else maxiter
        verbosity = 1
        X_umap = simplicial_set_embedding(
            X,
            neighbors['connectivities'].tocoo(),
            n_components,
            alpha,
            a,
            b,
            gamma,
            negative_sample_rate,
            n_epochs,
            init_coords,
            random_state,
            neigh_params.get('metric', 'euclidean'),
            neigh_params.get('metric_kwds', {}),
            verbose=verbosity > 3,
        )
    elif method == 'rapids':
        metric = neigh_params.get('metric', 'euclidean')
        if metric != 'euclidean':
            raise ValueError(
                f'`sc.pp.neighbors` was called with `metric` {metric!r}, '
                "but umap `method` 'rapids' only supports the 'euclidean' metric."
            )
        from cuml import UMAP

        n_neighbors = neighbors['params']['n_neighbors']
        n_epochs = (
            500 if maxiter is None else maxiter
        )  # 0 is not a valid value for rapids, unlike original umap
        X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
        verbosity = 1
        umap = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            n_epochs=n_epochs,
            learning_rate=alpha,
            init=init_pos,
            min_dist=min_dist,
            spread=spread,
            negative_sample_rate=negative_sample_rate,
            a=a,
            b=b,
            verbose=verbosity > 3,
            random_state=random_state,
        )
        X_umap = umap.fit_transform(X_contiguous)
    adata.obsm['X_umap'] = X_umap  # annotate samples with UMAP coordinates
    # logg.info(
    #     '    finished',
    #     time=start,
    #     deep=('added\n' "    'X_umap', UMAP coordinates (adata.obsm)"),
    # )
    return adata if copy else None

N_DCS = 15  # default number of diffusion components
N_PCS = (
    50
)  # Backwards compat, constants should be defined in only one place.

_Method = Literal['umap', 'gauss', 'rapids']
_MetricFn = Callable[[np.ndarray, np.ndarray], float]
# from sklearn.metrics.pairwise_distances.__doc__:
_MetricSparseCapable = Literal[
    'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'
]
_MetricScipySpatial = Literal[
    'braycurtis',
    'canberra',
    'chebyshev',
    'correlation',
    'dice',
    'hamming',
    'jaccard',
    'kulsinski',
    'mahalanobis',
    'minkowski',
    'rogerstanimoto',
    'russellrao',
    'seuclidean',
    'sokalmichener',
    'sokalsneath',
    'sqeuclidean',
    'yule',
]
_Metric = Union[_MetricSparseCapable, _MetricScipySpatial]


@_doc_params(n_pcs=doc_n_pcs, use_rep=doc_use_rep)
def neighbors(
    adata: AnnData,
    n_neighbors: int = 15,
    n_pcs: Optional[int] = None,
    use_rep: Optional[str] = None,
    knn: bool = True,
    random_state: AnyRandom = 0,
    method: Optional[_Method] = 'umap',
    metric: Union[_Metric, _MetricFn] = 'euclidean',
    metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    key_added: Optional[str] = None,
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    Compute a neighborhood graph of observations [McInnes18]_.

    The neighbor search efficiency of this heavily relies on UMAP [McInnes18]_,
    which also provides a method for estimating connectivities of data points -
    the connectivity of the manifold (`method=='umap'`). If `method=='gauss'`,
    connectivities are computed according to [Coifman05]_, in the adaption of
    [Haghverdi16]_.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_neighbors
        The size of local neighborhood (in terms of number of neighboring data
        points) used for manifold approximation. Larger values result in more
        global views of the manifold, while smaller values result in more local
        data being preserved. In general values should be in the range 2 to 100.
        If `knn` is `True`, number of nearest neighbors to be searched. If `knn`
        is `False`, a Gaussian kernel width is set to the distance of the
        `n_neighbors` neighbor.
    {n_pcs}
    {use_rep}
    knn
        If `True`, use a hard threshold to restrict the number of neighbors to
        `n_neighbors`, that is, consider a knn graph. Otherwise, use a Gaussian
        Kernel to assign low weights to neighbors more distant than the
        `n_neighbors` nearest neighbor.
    random_state
        A numpy random seed.
    method
        Use 'umap' [McInnes18]_ or 'gauss' (Gauss kernel following [Coifman05]_
        with adaptive width [Haghverdi16]_) for computing connectivities.
        Use 'rapids' for the RAPIDS implementation of UMAP (experimental, GPU
        only).
    metric
        A known metric’s name or a callable that returns a distance.
    metric_kwds
        Options for the metric.
    key_added
        If not specified, the neighbors data is stored in .uns['neighbors'],
        distances and connectivities are stored in .obsp['distances'] and
        .obsp['connectivities'] respectively.
        If specified, the neighbors data is added to .uns[key_added],
        distances are stored in .obsp[key_added+'_distances'] and
        connectivities in .obsp[key_added+'_connectivities'].
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    Depending on `copy`, updates or returns `adata` with the following:

    See `key_added` parameter description for the storage path of
    connectivities and distances.

    **connectivities** : sparse matrix of dtype `float32`.
        Weighted adjacency matrix of the neighborhood graph of data
        points. Weights should be interpreted as connectivities.
    **distances** : sparse matrix of dtype `float32`.
        Instead of decaying weights, this stores distances for each pair of
        neighbors.

    Notes
    -----
    If `method='umap'`, it's highly recommended to install pynndescent ``pip install pynndescent``.
    Installing `pynndescent` can significantly increase performance,
    and in later versions it will become a hard dependency.
    """
    # start = logg.info('computing neighbors')
    adata = adata.copy() if copy else adata
    if adata.is_view:  # we shouldn't need this here...
        adata._init_as_actual(adata.copy())
    neighbors = Neighbors(adata)
    neighbors.compute_neighbors(
        n_neighbors=n_neighbors,
        knn=knn,
        n_pcs=n_pcs,
        use_rep=use_rep,
        method=method,
        metric=metric,
        metric_kwds=metric_kwds,
        random_state=random_state,
    )

    if key_added is None:
        key_added = 'neighbors'
        conns_key = 'connectivities'
        dists_key = 'distances'
    else:
        conns_key = key_added + '_connectivities'
        dists_key = key_added + '_distances'

    adata.uns[key_added] = {}

    neighbors_dict = adata.uns[key_added]

    neighbors_dict['connectivities_key'] = conns_key
    neighbors_dict['distances_key'] = dists_key

    neighbors_dict['params'] = {'n_neighbors': neighbors.n_neighbors, 'method': method}
    neighbors_dict['params']['random_state'] = random_state
    neighbors_dict['params']['metric'] = metric
    if metric_kwds:
        neighbors_dict['params']['metric_kwds'] = metric_kwds
    if use_rep is not None:
        neighbors_dict['params']['use_rep'] = use_rep
    if n_pcs is not None:
        neighbors_dict['params']['n_pcs'] = n_pcs

    adata.obsp[dists_key] = neighbors.distances
    adata.obsp[conns_key] = neighbors.connectivities

    if neighbors.rp_forest is not None:
        neighbors_dict['rp_forest'] = neighbors.rp_forest
    # logg.info(
    #     '    finished',
    #     time=start,
    #     deep=(
    #         f'added to `.uns[{key_added!r}]`\n'
    #         f'    `.obsp[{dists_key!r}]`, distances for each pair of neighbors\n'
    #         f'    `.obsp[{conns_key!r}]`, weighted adjacency matrix'
    #     ),
    # )
    return adata if copy else None


class FlatTree(NamedTuple):
    hyperplanes: None
    offsets: None
    children: None
    indices: None


RPForestDict = Mapping[str, Mapping[str, np.ndarray]]


def _rp_forest_generate(
    rp_forest_dict: RPForestDict,
) -> Generator[FlatTree, None, None]:
    props = FlatTree._fields
    num_trees = len(rp_forest_dict[props[0]]['start']) - 1

    for i in range(num_trees):
        tree = []
        for prop in props:
            start = rp_forest_dict[prop]['start'][i]
            end = rp_forest_dict[prop]['start'][i + 1]
            tree.append(rp_forest_dict[prop]['data'][start:end])
        yield FlatTree(*tree)

    tree = []
    for prop in props:
        start = rp_forest_dict[prop]['start'][num_trees]
        tree.append(rp_forest_dict[prop]['data'][start:])
    yield FlatTree(*tree)


def compute_neighbors_umap(
    X: Union[np.ndarray, csr_matrix],
    n_neighbors: int,
    random_state: AnyRandom = None,
    metric: Union[_Metric, _MetricFn] = 'euclidean',
    metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    angular: bool = False,
    verbose: bool = False,
):
    """This is from umap.fuzzy_simplicial_set [McInnes18]_.

    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The data to be modelled as a fuzzy simplicial set.
    n_neighbors
        The number of neighbors to use to approximate geodesic distance.
        Larger numbers induce more global estimates of the manifold that can
        miss finer detail, while smaller values will focus on fine manifold
        structure to the detriment of the larger picture.
    random_state
        A state capable being used as a numpy random state.
    metric
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.
    metric_kwds
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance.
    angular
        Whether to use angular/cosine distance for the random projection
        forest for seeding NN-descent to determine approximate nearest
        neighbors.
    verbose
        Whether to report information on the current progress of the algorithm.

    Returns
    -------
    **knn_indices**, **knn_dists** : np.arrays of shape (n_observations, n_neighbors)
    """
    # with warnings.catch_warnings():
    #     # umap 0.5.0
    #     warnings.filterwarnings("ignore", message=r"Tensorflow not installed")
    from umap.umap_ import nearest_neighbors

    random_state = check_random_state(random_state)

    knn_indices, knn_dists, forest = nearest_neighbors(
        X,
        n_neighbors,
        random_state=random_state,
        metric=metric,
        metric_kwds=metric_kwds,
        angular=angular,
        verbose=verbose,
    )

    return knn_indices, knn_dists, forest


def compute_neighbors_rapids(X: np.ndarray, n_neighbors: int):
    """Compute nearest neighbors using RAPIDS cuml.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The data to compute nearest neighbors for.
    n_neighbors
        The number of neighbors to use.

        Returns
    -------
    **knn_indices**, **knn_dists** : np.arrays of shape (n_observations, n_neighbors)
    """
    from cuml.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
    nn.fit(X_contiguous)
    knn_distsq, knn_indices = nn.kneighbors(X_contiguous)
    return knn_indices, np.sqrt(knn_distsq)  # cuml uses sqeuclidean metric so take sqrt


def _get_sparse_matrix_from_indices_distances_umap(
    knn_indices, knn_dists, n_obs, n_neighbors
):
    rows = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_obs * n_neighbors), dtype=np.float64)

    for i in range(knn_indices.shape[0]):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            else:
                val = knn_dists[i, j]

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    result = coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
    result.eliminate_zeros()
    return result.tocsr()


def _compute_connectivities_umap(
    knn_indices,
    knn_dists,
    n_obs,
    n_neighbors,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
):
    """\
    This is from umap.fuzzy_simplicial_set [McInnes18]_.

    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    """
    # with warnings.catch_warnings():
    #     # umap 0.5.0
    #     warnings.filterwarnings("ignore", message=r"Tensorflow not installed")
    from umap.umap_ import fuzzy_simplicial_set

    X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
    connectivities = fuzzy_simplicial_set(
        X,
        n_neighbors,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )

    if isinstance(connectivities, tuple):
        # In umap-learn 0.4, this returns (result, sigmas, rhos)
        connectivities = connectivities[0]

    distances = _get_sparse_matrix_from_indices_distances_umap(
        knn_indices, knn_dists, n_obs, n_neighbors
    )

    return distances, connectivities.tocsr()


def _get_sparse_matrix_from_indices_distances_numpy(
    indices, distances, n_obs, n_neighbors
):
    n_nonzero = n_obs * n_neighbors
    indptr = np.arange(0, n_nonzero + 1, n_neighbors)
    D = csr_matrix(
        (
            distances.copy().ravel(),  # copy the data, otherwise strange behavior here
            indices.copy().ravel(),
            indptr,
        ),
        shape=(n_obs, n_obs),
    )
    D.eliminate_zeros()
    return D


def _get_indices_distances_from_sparse_matrix(D, n_neighbors: int):
    indices = np.zeros((D.shape[0], n_neighbors), dtype=int)
    distances = np.zeros((D.shape[0], n_neighbors), dtype=D.dtype)
    n_neighbors_m1 = n_neighbors - 1
    for i in range(indices.shape[0]):
        neighbors = D[i].nonzero()  # 'true' and 'spurious' zeros
        indices[i, 0] = i
        distances[i, 0] = 0
        # account for the fact that there might be more than n_neighbors
        # due to an approximate search
        # [the point itself was not detected as its own neighbor during the search]
        if len(neighbors[1]) > n_neighbors_m1:
            sorted_indices = np.argsort(D[i][neighbors].A1)[:n_neighbors_m1]
            indices[i, 1:] = neighbors[1][sorted_indices]
            distances[i, 1:] = D[i][
                neighbors[0][sorted_indices], neighbors[1][sorted_indices]
            ]
        else:
            indices[i, 1:] = neighbors[1]
            distances[i, 1:] = D[i][neighbors]
    return indices, distances


def _get_indices_distances_from_dense_matrix(D, n_neighbors: int):
    sample_range = np.arange(D.shape[0])[:, None]
    indices = np.argpartition(D, n_neighbors - 1, axis=1)[:, :n_neighbors]
    indices = indices[sample_range, np.argsort(D[sample_range, indices])]
    distances = D[sample_range, indices]
    return indices, distances


def _backwards_compat_get_full_X_diffmap(adata: AnnData) -> np.ndarray:
    if 'X_diffmap0' in adata.obs:
        return np.c_[adata.obs['X_diffmap0'].values[:, None], adata.obsm['X_diffmap']]
    else:
        return adata.obsm['X_diffmap']


def _backwards_compat_get_full_eval(adata: AnnData):
    if 'X_diffmap0' in adata.obs:
        return np.r_[1, adata.uns['diffmap_evals']]
    else:
        return adata.uns['diffmap_evals']


def _make_forest_dict(forest):
    d = {}
    props = ('hyperplanes', 'offsets', 'children', 'indices')
    for prop in props:
        d[prop] = {}
        sizes = np.fromiter(
            (getattr(tree, prop).shape[0] for tree in forest), dtype=int
        )
        d[prop]['start'] = np.zeros_like(sizes)
        if prop == 'offsets':
            dims = sizes.sum()
        else:
            dims = (sizes.sum(), getattr(forest[0], prop).shape[1])
        dtype = getattr(forest[0], prop).dtype
        dat = np.empty(dims, dtype=dtype)
        start = 0
        for i, size in enumerate(sizes):
            d[prop]['start'][i] = start
            end = start + size
            dat[start:end] = getattr(forest[i], prop)
            start = end
        d[prop]['data'] = dat
    return d


class OnFlySymMatrix:
    """Emulate a matrix where elements are calculated on the fly."""

    def __init__(
        self,
        get_row: Callable[[Any], np.ndarray],
        shape: Tuple[int, int],
        DC_start: int = 0,
        DC_end: int = -1,
        rows: Optional[Mapping[Any, np.ndarray]] = None,
        restrict_array: Optional[np.ndarray] = None,
    ):
        self.get_row = get_row
        self.shape = shape
        self.DC_start = DC_start
        self.DC_end = DC_end
        self.rows = {} if rows is None else rows
        self.restrict_array = restrict_array  # restrict the array to a subset

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            if self.restrict_array is None:
                glob_index = index
            else:
                # map the index back to the global index
                glob_index = self.restrict_array[index]
            if glob_index not in self.rows:
                self.rows[glob_index] = self.get_row(glob_index)
            row = self.rows[glob_index]
            if self.restrict_array is None:
                return row
            else:
                return row[self.restrict_array]
        else:
            if self.restrict_array is None:
                glob_index_0, glob_index_1 = index
            else:
                glob_index_0 = self.restrict_array[index[0]]
                glob_index_1 = self.restrict_array[index[1]]
            if glob_index_0 not in self.rows:
                self.rows[glob_index_0] = self.get_row(glob_index_0)
            return self.rows[glob_index_0][glob_index_1]

    def restrict(self, index_array):
        """Generate a view restricted to a subset of indices."""
        new_shape = index_array.shape[0], index_array.shape[0]
        return OnFlySymMatrix(
            self.get_row,
            new_shape,
            DC_start=self.DC_start,
            DC_end=self.DC_end,
            rows=self.rows,
            restrict_array=index_array,
        )


class Neighbors:
    """\
    Data represented as graph of nearest neighbors.

    Represent a data matrix as a graph of nearest neighbor relations (edges)
    among data points (nodes).

    Parameters
    ----------
    adata
        Annotated data object.
    n_dcs
        Number of diffusion components to use.
    neighbors_key
        Where to look in .uns and .obsp for neighbors data
    """

    def __init__(
        self,
        adata: AnnData,
        n_dcs: Optional[int] = None,
        neighbors_key: Optional[str] = None,
    ):
        self._adata = adata
        self._init_iroot()
        # use the graph in adata
        info_str = ''
        self.knn: Optional[bool] = None
        self._distances: Union[np.ndarray, csr_matrix, None] = None
        self._connectivities: Union[np.ndarray, csr_matrix, None] = None
        self._transitions_sym: Union[np.ndarray, csr_matrix, None] = None
        self._number_connected_components: Optional[int] = None
        self._rp_forest: Optional[RPForestDict] = None
        if neighbors_key is None:
            neighbors_key = 'neighbors'
        if neighbors_key in adata.uns:
            neighbors = NeighborsView(adata, neighbors_key)
            if 'distances' in neighbors:
                self.knn = issparse(neighbors['distances'])
                self._distances = neighbors['distances']
            if 'connectivities' in neighbors:
                self.knn = issparse(neighbors['connectivities'])
                self._connectivities = neighbors['connectivities']
            if 'rp_forest' in neighbors:
                self._rp_forest = neighbors['rp_forest']
            if 'params' in neighbors:
                self.n_neighbors = neighbors['params']['n_neighbors']
            else:

                def count_nonzero(a: Union[np.ndarray, csr_matrix]) -> int:
                    return a.count_nonzero() if issparse(a) else np.count_nonzero(a)

                # estimating n_neighbors
                if self._connectivities is None:
                    self.n_neighbors = int(
                        count_nonzero(self._distances) / self._distances.shape[0]
                    )
                else:
                    self.n_neighbors = int(
                        count_nonzero(self._connectivities)
                        / self._connectivities.shape[0]
                        / 2
                    )
            info_str += '`.distances` `.connectivities` '
            self._number_connected_components = 1
            if issparse(self._connectivities):
                from scipy.sparse.csgraph import connected_components

                self._connected_components = connected_components(self._connectivities)
                self._number_connected_components = self._connected_components[0]
        if 'X_diffmap' in adata.obsm_keys():
            self._eigen_values = _backwards_compat_get_full_eval(adata)
            self._eigen_basis = _backwards_compat_get_full_X_diffmap(adata)
            if n_dcs is not None:
                if n_dcs > len(self._eigen_values):
                    raise ValueError(
                        'Cannot instantiate using `n_dcs`={}. '
                        'Compute diffmap/spectrum with more components first.'.format(
                            n_dcs
                        )
                    )
                self._eigen_values = self._eigen_values[:n_dcs]
                self._eigen_basis = self._eigen_basis[:, :n_dcs]
            self.n_dcs = len(self._eigen_values)
            info_str += '`.eigen_values` `.eigen_basis` `.distances_dpt`'
        else:
            self._eigen_values = None
            self._eigen_basis = None
            self.n_dcs = None
        if info_str != '':
            pass
            # logg.debug(f'    initialized {info_str}')

    @property
    def rp_forest(self) -> Optional[RPForestDict]:
        return self._rp_forest

    @property
    def distances(self) -> Union[np.ndarray, csr_matrix, None]:
        """Distances between data points (sparse matrix)."""
        return self._distances

    @property
    def connectivities(self) -> Union[np.ndarray, csr_matrix, None]:
        """Connectivities between data points (sparse matrix)."""
        return self._connectivities

    @property
    def transitions(self) -> Union[np.ndarray, csr_matrix]:
        """Transition matrix (sparse matrix).

        Is conjugate to the symmetrized transition matrix via::

            self.transitions = self.Z *  self.transitions_sym / self.Z

        where ``self.Z`` is the diagonal matrix storing the normalization of the
        underlying kernel matrix.

        Notes
        -----
        This has not been tested, in contrast to `transitions_sym`.
        """
        if issparse(self.Z):
            Zinv = self.Z.power(-1)
        else:
            Zinv = np.diag(1.0 / np.diag(self.Z))
        return self.Z @ self.transitions_sym @ Zinv

    @property
    def transitions_sym(self) -> Union[np.ndarray, csr_matrix, None]:
        """Symmetrized transition matrix (sparse matrix).

        Is conjugate to the transition matrix via::

            self.transitions_sym = self.Z /  self.transitions * self.Z

        where ``self.Z`` is the diagonal matrix storing the normalization of the
        underlying kernel matrix.
        """
        return self._transitions_sym

    @property
    def eigen_values(self):
        """Eigen values of transition matrix (numpy array)."""
        return self._eigen_values

    @property
    def eigen_basis(self):
        """Eigen basis of transition matrix (numpy array)."""
        return self._eigen_basis

    @property
    def distances_dpt(self):
        """DPT distances (on-fly matrix).

        This is yields [Haghverdi16]_, Eq. 15 from the supplement with the
        extensions of [Wolf19]_, supplement on random-walk based distance
        measures.
        """
        return OnFlySymMatrix(self._get_dpt_row, shape=self._adata.shape)

    def to_igraph(self):
        """Generate igraph from connectiviies."""
        # from .sc_utils import get_igraph_from_adjacency
        return get_igraph_from_adjacency(self.connectivities)

    @_doc_params(n_pcs=doc_n_pcs, use_rep=doc_use_rep)
    def compute_neighbors(
        self,
        n_neighbors: int = 30,
        knn: bool = True,
        n_pcs: Optional[int] = None,
        use_rep: Optional[str] = None,
        method: _Method = 'umap',
        random_state: AnyRandom = 0,
        write_knn_indices: bool = False,
        metric: _Metric = 'euclidean',
        metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    ) -> None:
        """\
        Compute distances and connectivities of neighbors.

        Parameters
        ----------
        n_neighbors
             Use this number of nearest neighbors.
        knn
             Restrict result to `n_neighbors` nearest neighbors.
        {n_pcs}
        {use_rep}

        Returns
        -------
        Writes sparse graph attributes `.distances` and `.connectivities`.
        Also writes `.knn_indices` and `.knn_distances` if
        `write_knn_indices==True`.
        """
        from sklearn.metrics import pairwise_distances

        # start_neighbors = logg.debug('computing neighbors')
        if n_neighbors > self._adata.shape[0]:  # very small datasets
            n_neighbors = 1 + int(0.5 * self._adata.shape[0])
            # logg.warning(f'n_obs too small: adjusting to `n_neighbors = {n_neighbors}`')
        if method == 'umap' and not knn:
            raise ValueError('`method = \'umap\' only with `knn = True`.')
        if method == 'rapids' and metric != 'euclidean':
            raise ValueError(
                "`method` 'rapids' only supports the 'euclidean' `metric`."
            )
        if method not in {'umap', 'gauss', 'rapids'}:
            raise ValueError("`method` needs to be 'umap', 'gauss', or 'rapids'.")
        if self._adata.shape[0] >= 10000 and not knn:
            pass
            # logg.warning('Using high n_obs without `knn=True` takes a lot of memory...')
        # do not use the cached rp_forest
        self._rp_forest = None
        self.n_neighbors = n_neighbors
        self.knn = knn
        X = _choose_representation(self._adata, use_rep=use_rep, n_pcs=n_pcs)
        # neighbor search
        use_dense_distances = (metric == 'euclidean' and X.shape[0] < 8192) or not knn
        if use_dense_distances:
            _distances = pairwise_distances(X, metric=metric, **metric_kwds)
            knn_indices, knn_distances = _get_indices_distances_from_dense_matrix(
                _distances, n_neighbors
            )
            if knn:
                self._distances = _get_sparse_matrix_from_indices_distances_numpy(
                    knn_indices, knn_distances, X.shape[0], n_neighbors
                )
            else:
                self._distances = _distances
        elif method == 'rapids':
            knn_indices, knn_distances = compute_neighbors_rapids(X, n_neighbors)
        else:
            # non-euclidean case and approx nearest neighbors
            if X.shape[0] < 4096:
                X = pairwise_distances(X, metric=metric, **metric_kwds)
                metric = 'precomputed'
            knn_indices, knn_distances, forest = compute_neighbors_umap(
                X, n_neighbors, random_state, metric=metric, metric_kwds=metric_kwds
            )
            # very cautious here
            try:
                if forest:
                    self._rp_forest = _make_forest_dict(forest)
            except Exception:  # TODO catch the correct exception
                pass
        # write indices as attributes
        if write_knn_indices:
            self.knn_indices = knn_indices
            self.knn_distances = knn_distances
        # start_connect = logg.debug('computed neighbors', time=start_neighbors)
        if not use_dense_distances or method in {'umap', 'rapids'}:
            # we need self._distances also for method == 'gauss' if we didn't
            # use dense distances
            self._distances, self._connectivities = _compute_connectivities_umap(
                knn_indices,
                knn_distances,
                self._adata.shape[0],
                self.n_neighbors,
            )
        # overwrite the umap connectivities if method is 'gauss'
        # self._distances is unaffected by this
        if method == 'gauss':
            self._compute_connectivities_diffmap()
        # logg.debug('computed connectivities', time=start_connect)
        self._number_connected_components = 1
        if issparse(self._connectivities):
            from scipy.sparse.csgraph import connected_components

            self._connected_components = connected_components(self._connectivities)
            self._number_connected_components = self._connected_components[0]

    def _compute_connectivities_diffmap(self, density_normalize=True):
        # init distances
        if self.knn:
            Dsq = self._distances.power(2)
            indices, distances_sq = _get_indices_distances_from_sparse_matrix(
                Dsq, self.n_neighbors
            )
        else:
            Dsq = np.power(self._distances, 2)
            indices, distances_sq = _get_indices_distances_from_dense_matrix(
                Dsq, self.n_neighbors
            )

        # exclude the first point, the 0th neighbor
        indices = indices[:, 1:]
        distances_sq = distances_sq[:, 1:]

        # choose sigma, the heuristic here doesn't seem to make much of a difference,
        # but is used to reproduce the figures of Haghverdi et al. (2016)
        if self.knn:
            # as the distances are not sorted
            # we have decay within the n_neighbors first neighbors
            sigmas_sq = np.median(distances_sq, axis=1)
        else:
            # the last item is already in its sorted position through argpartition
            # we have decay beyond the n_neighbors neighbors
            sigmas_sq = distances_sq[:, -1] / 4
        sigmas = np.sqrt(sigmas_sq)

        # compute the symmetric weight matrix
        if not issparse(self._distances):
            Num = 2 * np.multiply.outer(sigmas, sigmas)
            Den = np.add.outer(sigmas_sq, sigmas_sq)
            W = np.sqrt(Num / Den) * np.exp(-Dsq / Den)
            # make the weight matrix sparse
            if not self.knn:
                mask = W > 1e-14
                W[~mask] = 0
            else:
                # restrict number of neighbors to ~k
                # build a symmetric mask
                mask = np.zeros(Dsq.shape, dtype=bool)
                for i, row in enumerate(indices):
                    mask[i, row] = True
                    for j in row:
                        if i not in set(indices[j]):
                            W[j, i] = W[i, j]
                            mask[j, i] = True
                # set all entries that are not nearest neighbors to zero
                W[~mask] = 0
        else:
            W = (
                Dsq.copy()
            )  # need to copy the distance matrix here; what follows is inplace
            for i in range(len(Dsq.indptr[:-1])):
                row = Dsq.indices[Dsq.indptr[i] : Dsq.indptr[i + 1]]
                num = 2 * sigmas[i] * sigmas[row]
                den = sigmas_sq[i] + sigmas_sq[row]
                W.data[Dsq.indptr[i] : Dsq.indptr[i + 1]] = np.sqrt(num / den) * np.exp(
                    -Dsq.data[Dsq.indptr[i] : Dsq.indptr[i + 1]] / den
                )
            W = W.tolil()
            for i, row in enumerate(indices):
                for j in row:
                    if i not in set(indices[j]):
                        W[j, i] = W[i, j]
            W = W.tocsr()

        self._connectivities = W

    def compute_transitions(self, density_normalize: bool = True):
        """\
        Compute transition matrix.

        Parameters
        ----------
        density_normalize
            The density rescaling of Coifman and Lafon (2006): Then only the
            geometry of the data matters, not the sampled density.

        Returns
        -------
        Makes attributes `.transitions_sym` and `.transitions` available.
        """
        # start = logg.info('computing transitions')
        W = self._connectivities
        # density normalization as of Coifman et al. (2005)
        # ensures that kernel matrix is independent of sampling density
        if density_normalize:
            # q[i] is an estimate for the sampling density at point i
            # it's also the degree of the underlying graph
            q = np.asarray(W.sum(axis=0))
            if not issparse(W):
                Q = np.diag(1.0 / q)
            else:
                Q = scipy.sparse.spdiags(1.0 / q, 0, W.shape[0], W.shape[0])
            K = Q @ W @ Q
        else:
            K = W

        # z[i] is the square root of the row sum of K
        z = np.sqrt(np.asarray(K.sum(axis=0)))
        if not issparse(K):
            self.Z = np.diag(1.0 / z)
        else:
            self.Z = scipy.sparse.spdiags(1.0 / z, 0, K.shape[0], K.shape[0])
        self._transitions_sym = self.Z @ K @ self.Z
        # logg.info('    finished', time=start)

    def compute_eigen(
        self,
        n_comps: int = 15,
        sym: Optional[bool] = None,
        sort: Literal['decrease', 'increase'] = 'decrease',
        random_state: AnyRandom = 0,
    ):
        """\
        Compute eigen decomposition of transition matrix.

        Parameters
        ----------
        n_comps
            Number of eigenvalues/vectors to be computed, set `n_comps = 0` if
            you need all eigenvectors.
        sym
            Instead of computing the eigendecomposition of the assymetric
            transition matrix, computed the eigendecomposition of the symmetric
            Ktilde matrix.
        random_state
            A numpy random seed

        Returns
        -------
        Writes the following attributes.

        eigen_values : numpy.ndarray
            Eigenvalues of transition matrix.
        eigen_basis : numpy.ndarray
             Matrix of eigenvectors (stored in columns).  `.eigen_basis` is
             projection of data matrix on right eigenvectors, that is, the
             projection on the diffusion components.  these are simply the
             components of the right eigenvectors and can directly be used for
             plotting.
        """
        np.set_printoptions(precision=10)
        if self._transitions_sym is None:
            raise ValueError('Run `.compute_transitions` first.')
        matrix = self._transitions_sym
        # compute the spectrum
        if n_comps == 0:
            evals, evecs = scipy.linalg.eigh(matrix)
        else:
            n_comps = min(matrix.shape[0] - 1, n_comps)
            # ncv = max(2 * n_comps + 1, int(np.sqrt(matrix.shape[0])))
            ncv = None
            which = 'LM' if sort == 'decrease' else 'SM'
            # it pays off to increase the stability with a bit more precision
            matrix = matrix.astype(np.float64)

            # Setting the random initial vector
            random_state = check_random_state(random_state)
            v0 = random_state.standard_normal((matrix.shape[0]))
            evals, evecs = scipy.sparse.linalg.eigsh(
                matrix, k=n_comps, which=which, ncv=ncv, v0=v0
            )
            evals, evecs = evals.astype(np.float32), evecs.astype(np.float32)
        if sort == 'decrease':
            evals = evals[::-1]
            evecs = evecs[:, ::-1]
        # logg.info(
        #     '    eigenvalues of transition matrix\n'
        #     '    {}'.format(str(evals).replace('\n', '\n    '))
        # )
        if self._number_connected_components > len(evals) / 2:
            pass
            # logg.warning('Transition matrix has many disconnected components!')
        self._eigen_values = evals
        self._eigen_basis = evecs

    def _init_iroot(self):
        self.iroot = None
        # set iroot directly
        if 'iroot' in self._adata.uns:
            if self._adata.uns['iroot'] >= self._adata.n_obs:
                pass
                # logg.warning(
                #     f'Root cell index {self._adata.uns["iroot"]} does not '
                #     f'exist for {self._adata.n_obs} samples. It’s ignored.'
                # )
            else:
                self.iroot = self._adata.uns['iroot']
            return
        # set iroot via xroot
        xroot = None
        if 'xroot' in self._adata.uns:
            xroot = self._adata.uns['xroot']
        elif 'xroot' in self._adata.var:
            xroot = self._adata.var['xroot']
        # see whether we can set self.iroot using the full data matrix
        if xroot is not None and xroot.size == self._adata.shape[1]:
            self._set_iroot_via_xroot(xroot)

    def _get_dpt_row(self, i):
        mask = None
        if self._number_connected_components > 1:
            label = self._connected_components[1][i]
            mask = self._connected_components[1] == label
        row = sum(
            (
                self.eigen_values[j]
                / (1 - self.eigen_values[j])
                * (self.eigen_basis[i, j] - self.eigen_basis[:, j])
            )
            ** 2
            # account for float32 precision
            for j in range(0, self.eigen_values.size)
            if self.eigen_values[j] < 0.9994
        )
        # thanks to Marius Lange for pointing Alex to this:
        # we will likely remove the contributions from the stationary state below when making
        # backwards compat breaking changes, they originate from an early implementation in 2015
        # they never seem to have deteriorated results, but also other distance measures (see e.g.
        # PAGA paper) don't have it, which makes sense
        row += sum(
            (self.eigen_basis[i, k] - self.eigen_basis[:, k]) ** 2
            for k in range(0, self.eigen_values.size)
            if self.eigen_values[k] >= 0.9994
        )
        if mask is not None:
            row[~mask] = np.inf
        return np.sqrt(row)

    def _set_pseudotime(self):
        """Return pseudotime with respect to root point."""
        self.pseudotime = self.distances_dpt[self.iroot].copy()
        self.pseudotime /= np.max(self.pseudotime[self.pseudotime < np.inf])

    def _set_iroot_via_xroot(self, xroot):
        """Determine the index of the root cell.

        Given an expression vector, find the observation index that is closest
        to this vector.

        Parameters
        ----------
        xroot : np.ndarray
            Vector that marks the root cell, the vector storing the initial
            condition, only relevant for computing pseudotime.
        """
        if self._adata.shape[1] != xroot.size:
            raise ValueError(
                'The root vector you provided does not have the ' 'correct dimension.'
            )
        # this is the squared distance
        dsqroot = 1e10
        iroot = 0
        for i in range(self._adata.shape[0]):
            diff = self._adata.X[i, :] - xroot
            dsq = diff @ diff
            if dsq < dsqroot:
                dsqroot = dsq
                iroot = i
                if np.sqrt(dsqroot) < 1e-10:
                    break
        # logg.debug(f'setting root index to {iroot}')
        if self.iroot is not None and iroot != self.iroot:
            pass
            # logg.warning(f'Changing index of iroot from {self.iroot} to {iroot}.')
        self.iroot = iroot


# --------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------


def _pca_fallback(data, n_comps=2):
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    C = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since C is symmetric,
    # the performance gain is substantial
    # evals, evecs = np.linalg.eigh(C)
    evals, evecs = sp.sparse.linalg.eigsh(C, k=n_comps)
    # sort eigenvalues in decreasing order
    idcs = np.argsort(evals)[::-1]
    evecs = evecs[:, idcs]
    evals = evals[idcs]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or n_comps)
    evecs = evecs[:, :n_comps]
    # project data points on eigenvectors
    return np.dot(evecs.T, data.T).T



def magic(
    adata: AnnData,
    name_list: Union[Literal['all_genes', 'pca_only'], Sequence[str], None] = None,
    *,
    knn: int = 5,
    decay: Optional[float] = 1,
    knn_max: Optional[int] = None,
    t: Union[Literal['auto'], int] = 3,
    n_pca: Optional[int] = 100,
    solver: Literal['exact', 'approximate'] = 'exact',
    knn_dist: str = 'euclidean',
    random_state: AnyRandom = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    copy: Optional[bool] = None,
    **kwargs,
) -> Optional[AnnData]:
    """\
    Markov Affinity-based Graph Imputation of Cells (MAGIC) API [vanDijk18]_.

    MAGIC is an algorithm for denoising and transcript recover of single cells
    applied to single-cell sequencing data. MAGIC builds a graph from the data
    and uses diffusion to smooth out noise and recover the data manifold.

    The algorithm implemented here has changed primarily in two ways
    compared to the algorithm described in [vanDijk18]_. Firstly, we use
    the adaptive kernel described in Moon et al, 2019 [Moon17]_ for
    improved stability. Secondly, data diffusion is applied
    in the PCA space, rather than the data space, for speed and
    memory improvements.

    More information and bug reports
    `here <https://github.com/KrishnaswamyLab/MAGIC>`__. For help, visit
    <https://krishnaswamylab.org/get-help>.

    Parameters
    ----------
    adata
        An anndata file with `.raw` attribute representing raw counts.
    name_list
        Denoised genes to return. The default `'all_genes'`/`None`
        may require a large amount of memory if the input data is sparse.
        Another possibility is `'pca_only'`.
    knn
        number of nearest neighbors on which to build kernel.
    decay
        sets decay rate of kernel tails.
        If None, alpha decaying kernel is not used.
    knn_max
        maximum number of nearest neighbors with nonzero connection.
        If `None`, will be set to 3 * `knn`.
    t
        power to which the diffusion operator is powered.
        This sets the level of diffusion. If 'auto', t is selected
        according to the Procrustes disparity of the diffused data.
    n_pca
        Number of principal components to use for calculating
        neighborhoods. For extremely large datasets, using
        n_pca < 20 allows neighborhoods to be calculated in
        roughly log(n_samples) time. If `None`, no PCA is performed.
    solver
        Which solver to use. "exact" uses the implementation described
        in van Dijk et al. (2018) [vanDijk18]_. "approximate" uses a faster
        implementation that performs imputation in the PCA space and then
        projects back to the gene space. Note, the "approximate" solver may
        return negative values.
    knn_dist
        recommended values: 'euclidean', 'cosine', 'precomputed'
        Any metric from `scipy.spatial.distance` can be used
        distance metric for building kNN graph. If 'precomputed',
        `data` should be an n_samples x n_samples distance or
        affinity matrix.
    random_state
        Random seed. Defaults to the global `numpy` random number generator.
    n_jobs
        Number of threads to use in training. All cores are used by default.
    verbose
        If `True` or an integer `>= 2`, print status messages.
        If `None`, `sc.settings.verbosity` is used.
    copy
        If true, a copy of anndata is returned. If `None`, `copy` is True if
        `genes` is not `'all_genes'` or `'pca_only'`. `copy` may only be False
        if `genes` is `'all_genes'` or `'pca_only'`, as the resultant data
        will otherwise have different column names from the input data.
    kwargs
        Additional arguments to `magic.MAGIC`.

    Returns
    -------
    If `copy` is True, AnnData object is returned.

    If `subset_genes` is not `all_genes`, PCA on MAGIC values of cells are
    stored in `adata.obsm['X_magic']` and `adata.X` is not modified.

    The raw counts are stored in `.raw` attribute of AnnData object.

    Examples
    --------
    >>> import scanpy as sc
    >>> import scanpy.external as sce
    >>> adata = sc.datasets.paul15()
    >>> sc.pp.normalize_per_cell(adata)
    >>> sc.pp.sqrt(adata)  # or sc.pp.log1p(adata)
    >>> adata_magic = sce.pp.magic(adata, name_list=['Mpo', 'Klf1', 'Ifitm1'], knn=5)
    >>> adata_magic.shape
    (2730, 3)
    >>> sce.pp.magic(adata, name_list='pca_only', knn=5)
    >>> adata.obsm['X_magic'].shape
    (2730, 100)
    >>> sce.pp.magic(adata, name_list='all_genes', knn=5)
    >>> adata.X.shape
    (2730, 3451)
    """
    MIN_VERSION = "2.0"
    try:
        from packaging import version
    except ImportError:
        raise ImportError('Please install packaging package')

    try:
        from magic import MAGIC, __version__
    except ImportError:
        raise ImportError(
            'Please install magic package via `pip install --user '
            'git+git://github.com/KrishnaswamyLab/MAGIC.git#subdirectory=python`'
        )
    else:
        if not version.parse(__version__) >= version.parse(MIN_VERSION):
            raise ImportError(
                'scanpy requires magic-impute >= '
                f'v{MIN_VERSION} (detected: v{__version__}). '
                'Please update magic package via `pip install --user '
                '--upgrade magic-impute`'
            )

    # start = logg.info('computing MAGIC')
    all_or_pca = isinstance(name_list, (str, type(None)))
    if all_or_pca and name_list not in {"all_genes", "pca_only", None}:
        raise ValueError(
            "Invalid string value for `name_list`: "
            "Only `'all_genes'` and `'pca_only'` are allowed."
        )
    if copy is None:
        copy = not all_or_pca
    elif not all_or_pca and not copy:
        raise ValueError(
            "Can only perform MAGIC in-place with `name_list=='all_genes' or "
            f"`name_list=='pca_only'` (got {name_list}). Consider setting "
            "`copy=True`"
        )
    adata = adata.copy() if copy else adata
    n_jobs = 1 if n_jobs is None else n_jobs

    X_magic = MAGIC(
        knn=knn,
        decay=decay,
        knn_max=knn_max,
        t=t,
        n_pca=n_pca,
        solver=solver,
        knn_dist=knn_dist,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        **kwargs,
    ).fit_transform(adata, genes=name_list)
    # logg.info(
    #     '    finished',
    #     time=start,
    #     deep=(
    #         "added\n    'X_magic', PCA on MAGIC coordinates (adata.obsm)"
    #         if name_list == "pca_only"
    #         else ''
    #     ),
    # )
    # update AnnData instance
    if name_list == "pca_only":
        # special case – update adata.obsm with smoothed values
        adata.obsm["X_magic"] = X_magic.X
    elif copy:
        # just return X_magic
        X_magic.raw = adata
        adata = X_magic
    else:
        # replace data with smoothed data
        adata.raw = adata
        adata.X = X_magic.X

    if copy:
        return adata