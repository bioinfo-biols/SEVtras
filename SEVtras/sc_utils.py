## based on scanpy with minor modifications

from textwrap import dedent
def _doc_params(**kwds):
    """\
    Docstrings should start with "\" in the first line for proper formatting.
    """

    def dec(obj):
        obj.__orig_doc__ = obj.__doc__
        obj.__doc__ = dedent(obj.__doc__).format_map(kwds)
        return obj

    return dec

try:
    from typing import Literal
except ImportError:
    try:
        from typing_extensions import Literal
    except ImportError:

        class LiteralMeta(type):
            def __getitem__(cls, values):
                if not isinstance(values, tuple):
                    values = (values,)
                return type('Literal_', (Literal,), dict(__args__=values))

        class Literal(metaclass=LiteralMeta):
            pass

import sys
import inspect
# import importlib.util
from enum import Enum
from pathlib import Path
# from weakref import WeakSet
from collections import namedtuple
from functools import partial, singledispatch, wraps
from types import ModuleType, MethodType, MappingProxyType
from typing import Union, Callable, Optional, Mapping, Any, Dict, Tuple, Iterable, List#, Literal
from sklearn.utils import check_random_state, check_array

import numpy as np
import pandas as pd
from numpy import random
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import spmatrix
from anndata import AnnData, __version__ as anndata_version
from dataclasses import dataclass, field

# from .sc_utils import (
#     sanitize_anndata,
#     deprecated_arg_names,
#     view_to_actual,
#     AnyRandom,
#     _check_array_function_arguments,
# )
# from .sc_utils import _get_obs_rep, _set_obs_rep
# from .sc_utils import materialize_as_ndarray
# from .sc_utils import _get_mean_var
# 

# e.g. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
AnyRandom = Union[None, int, random.RandomState]  # maybe in the future random.Generator

EPS = 1e-15

# install dask if available
try:
    import dask.array as da
except ImportError:
    da = None

try:
    from dask.array import Array as DaskArray
except ImportError:

    class DaskArray:
        pass


_SparseMatrix = Union[sparse.csr_matrix, sparse.csc_matrix]
_MemoryArray = Union[NDArray, _SparseMatrix]
_SupportedArray = Union[_MemoryArray, DaskArray]


@singledispatch
def elem_mul(x: _SupportedArray, y: _SupportedArray) -> _SupportedArray:
    raise NotImplementedError



def materialize_as_ndarray(a):
    """Convert distributed arrays to ndarrays."""
    if type(a) in (list, tuple):
        if da is not None and any(isinstance(arr, da.Array) for arr in a):
            return da.compute(*a, sync=True)
        return tuple(np.asarray(arr) for arr in a)
    return np.asarray(a)

# backwards compat... remove this in the future
def sanitize_anndata(adata: AnnData) -> None:
    """Transform string annotations to categoricals."""
    adata._sanitize()

def view_to_actual(adata: AnnData) -> None:
    if adata.is_view:
        # warnings.warn(
        #     "Received a view of an AnnData. Making a copy.",
        #     stacklevel=2,
        # )
        adata._init_as_actual(adata.copy())


def _check_array_function_arguments(**kwargs):
    """Checks for invalid arguments when an array is passed.

    Helper for functions that work on either AnnData objects or array-likes.
    """
    # TODO: Figure out a better solution for documenting dispatched functions
    invalid_args = [k for k, v in kwargs.items() if v is not None]
    if len(invalid_args) > 0:
        raise TypeError(
            f"Arguments {invalid_args} are only valid if an AnnData object is passed."
        )


def deprecated_arg_names(arg_mapping: Mapping[str, str]):
    """
    Decorator which marks a functions keyword arguments as deprecated. It will
    result in a warning being emitted when the deprecated keyword argument is
    used, and the function being called with the new argument.

    Parameters
    ----------
    arg_mapping
        Mapping from deprecated argument name to current argument name.
    """

    def decorator(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            # warnings.simplefilter("always", DeprecationWarning)  # turn off filter
            for old, new in arg_mapping.items():
                if old in kwargs:
                    # warnings.warn(
                    #     f"Keyword argument '{old}' has been "
                    #     f"deprecated in favour of '{new}'. "
                    #     f"'{old}' will be removed in a future version.",
                    #     category=DeprecationWarning,
                    #     stacklevel=2,
                    # )
                    val = kwargs.pop(old)
                    kwargs[new] = val
            # reset filter
            # warnings.simplefilter("default", DeprecationWarning)
            return func(*args, **kwargs)

        return func_wrapper

    return decorator

# --------------------------------------------------------------------------------
# Plotting data helpers
# --------------------------------------------------------------------------------


# TODO: implement diffxpy method, make singledispatch
def rank_genes_groups_df(
    adata: AnnData,
    group: Union[str, Iterable[str]],
    *,
    key: str = "rank_genes_groups",
    pval_cutoff: Optional[float] = None,
    log2fc_min: Optional[float] = None,
    log2fc_max: Optional[float] = None,
    gene_symbols: Optional[str] = None,
) -> pd.DataFrame:
    """\
    :func:`scanpy.tl.rank_genes_groups` results in the form of a
    :class:`~pandas.DataFrame`.

    Params
    ------
    adata
        Object to get results from.
    group
        Which group (as in :func:`scanpy.tl.rank_genes_groups`'s `groupby`
        argument) to return results from. Can be a list. All groups are
        returned if groups is `None`.
    key
        Key differential expression groups were stored under.
    pval_cutoff
        Return only adjusted p-values below the  cutoff.
    log2fc_min
        Minimum logfc to return.
    log2fc_max
        Maximum logfc to return.
    gene_symbols
        Column name in `.var` DataFrame that stores gene symbols. Specifying
        this will add that column to the returned dataframe.

    Example
    -------
    >>> import scanpy as sc
    >>> pbmc = sc.datasets.pbmc68k_reduced()
    >>> sc.tl.rank_genes_groups(pbmc, groupby="louvain", use_raw=True)
    >>> dedf = sc.get.rank_genes_groups_df(pbmc, group="0")
    """
    if isinstance(group, str):
        group = [group]
    if group is None:
        group = list(adata.uns[key]["names"].dtype.names)
    method = adata.uns[key]["params"]["method"]
    if method == "logreg":
        colnames = ["names", "scores"]
    else:
        colnames = ["names", "scores", "logfoldchanges", "pvals", "pvals_adj"]

    d = [pd.DataFrame(adata.uns[key][c])[group] for c in colnames]
    d = pd.concat(d, axis=1, names=[None, "group"], keys=colnames)
    d = d.stack(level=1).reset_index()
    d["group"] = pd.Categorical(d["group"], categories=group)
    d = d.sort_values(["group", "level_0"]).drop(columns="level_0")

    if method != "logreg":
        if pval_cutoff is not None:
            d = d[d["pvals_adj"] < pval_cutoff]
        if log2fc_min is not None:
            d = d[d["logfoldchanges"] > log2fc_min]
        if log2fc_max is not None:
            d = d[d["logfoldchanges"] < log2fc_max]
    if gene_symbols is not None:
        d = d.join(adata.var[gene_symbols], on="names")

    for pts, name in {"pts": "pct_nz_group", "pts_rest": "pct_nz_reference"}.items():
        if pts in adata.uns[key]:
            pts_df = (
                adata.uns[key][pts][group]
                .rename_axis(index="names")
                .reset_index()
                .melt(id_vars="names", var_name="group", value_name=name)
            )
            d = d.merge(pts_df)

    # remove group column for backward compat if len(group) == 1
    if len(group) == 1:
        d.drop(columns="group", inplace=True)

    return d.reset_index(drop=True)


def _check_indices(
    dim_df: pd.DataFrame,
    alt_index: pd.Index,
    dim: Literal["obs", "var"],
    keys: List[str],
    alias_index: Optional[pd.Index] = None,
    use_raw: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """Common logic for checking indices for obs_df and var_df."""
    if use_raw:
        alt_repr = "adata.raw"
    else:
        alt_repr = "adata"

    alt_dim = ("obs", "var")[dim == "obs"]

    alias_name = None
    if alias_index is not None:
        alt_names = pd.Series(alt_index, index=alias_index)
        alias_name = alias_index.name
        alt_search_repr = f"{alt_dim}['{alias_name}']"
    else:
        alt_names = pd.Series(alt_index, index=alt_index)
        alt_search_repr = f"{alt_dim}_names"

    col_keys = []
    index_keys = []
    index_aliases = []
    not_found = []

    # check that adata.obs does not contain duplicated columns
    # if duplicated columns names are present, they will
    # be further duplicated when selecting them.
    if not dim_df.columns.is_unique:
        dup_cols = dim_df.columns[dim_df.columns.duplicated()].tolist()
        raise ValueError(
            f"adata.{dim} contains duplicated columns. Please rename or remove "
            "these columns first.\n`"
            f"Duplicated columns {dup_cols}"
        )

    if not alt_index.is_unique:
        raise ValueError(
            f"{alt_repr}.{alt_dim}_names contains duplicated items\n"
            f"Please rename these {alt_dim} names first for example using "
            f"`adata.{alt_dim}_names_make_unique()`"
        )

    # use only unique keys, otherwise duplicated keys will
    # further duplicate when reordering the keys later in the function
    for key in np.unique(keys):
        if key in dim_df.columns:
            col_keys.append(key)
            if key in alt_names.index:
                raise KeyError(
                    f"The key '{key}' is found in both adata.{dim} and {alt_repr}.{alt_search_repr}."
                )
        elif key in alt_names.index:
            val = alt_names[key]
            if isinstance(val, pd.Series):
                # while var_names must be unique, adata.var[gene_symbols] does not
                # It's still ambiguous to refer to a duplicated entry though.
                assert alias_index is not None
                raise KeyError(
                    f"Found duplicate entries for '{key}' in {alt_repr}.{alt_search_repr}."
                )
            index_keys.append(val)
            index_aliases.append(key)
        else:
            not_found.append(key)
    if len(not_found) > 0:
        raise KeyError(
            f"Could not find keys '{not_found}' in columns of `adata.{dim}` or in"
            f" {alt_repr}.{alt_search_repr}."
        )

    return col_keys, index_keys, index_aliases


def _get_array_values(
    X,
    dim_names: pd.Index,
    keys: List[str],
    axis: Literal[0, 1],
    backed: bool,
):
    # TODO: This should be made easier on the anndata side
    mutable_idxer = [slice(None), slice(None)]
    idx = dim_names.get_indexer(keys)

    # for backed AnnData is important that the indices are ordered
    if backed:
        idx_order = np.argsort(idx)
        rev_idxer = mutable_idxer.copy()
        mutable_idxer[axis] = idx[idx_order]
        rev_idxer[axis] = np.argsort(idx_order)
        matrix = X[tuple(mutable_idxer)][tuple(rev_idxer)]
    else:
        mutable_idxer[axis] = idx
        matrix = X[tuple(mutable_idxer)]

    from scipy.sparse import issparse

    if issparse(matrix):
        matrix = matrix.toarray()

    return matrix


def obs_df(
    adata: AnnData,
    keys: Iterable[str] = (),
    obsm_keys: Iterable[Tuple[str, int]] = (),
    *,
    layer: str = None,
    gene_symbols: str = None,
    use_raw: bool = False,
) -> pd.DataFrame:
    """\
    Return values for observations in adata.

    Params
    ------
    adata
        AnnData object to get values from.
    keys
        Keys from either `.var_names`, `.var[gene_symbols]`, or `.obs.columns`.
    obsm_keys
        Tuple of `(key from obsm, column index of obsm[key])`.
    layer
        Layer of `adata` to use as expression values.
    gene_symbols
        Column of `adata.var` to search for `keys` in.
    use_raw
        Whether to get expression values from `adata.raw`.

    Returns
    -------
    A dataframe with `adata.obs_names` as index, and values specified by `keys`
    and `obsm_keys`.

    Examples
    --------
    Getting value for plotting:

    >>> import scanpy as sc
    >>> pbmc = sc.datasets.pbmc68k_reduced()
    >>> plotdf = sc.get.obs_df(
    ...     pbmc,
    ...     keys=["CD8B", "n_genes"],
    ...     obsm_keys=[("X_umap", 0), ("X_umap", 1)]
    ... )
    >>> plotdf.columns
    Index(['CD8B', 'n_genes', 'X_umap-0', 'X_umap-1'], dtype='object')
    >>> plotdf.plot.scatter("X_umap-0", "X_umap-1", c="CD8B")
    <Axes: xlabel='X_umap-0', ylabel='X_umap-1'>

    Calculating mean expression for marker genes by cluster:

    >>> pbmc = sc.datasets.pbmc68k_reduced()
    >>> marker_genes = ['CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ']
    >>> genedf = sc.get.obs_df(
    ...     pbmc,
    ...     keys=["louvain", *marker_genes]
    ... )
    >>> grouped = genedf.groupby("louvain")
    >>> mean, var = grouped.mean(), grouped.var()
    """
    if use_raw:
        assert (
            layer is None
        ), "Cannot specify use_raw=True and a layer at the same time."
        var = adata.raw.var
    else:
        var = adata.var
    if gene_symbols is not None:
        alias_index = pd.Index(var[gene_symbols])
    else:
        alias_index = None

    obs_cols, var_idx_keys, var_symbols = _check_indices(
        adata.obs,
        var.index,
        "obs",
        keys,
        alias_index=alias_index,
        use_raw=use_raw,
    )

    # Make df
    df = pd.DataFrame(index=adata.obs_names)

    # add var values
    if len(var_idx_keys) > 0:
        matrix = _get_array_values(
            _get_obs_rep(adata, layer=layer, use_raw=use_raw),
            var.index,
            var_idx_keys,
            axis=1,
            backed=adata.isbacked,
        )
        df = pd.concat(
            [df, pd.DataFrame(matrix, columns=var_symbols, index=adata.obs_names)],
            axis=1,
        )

    # add obs values
    if len(obs_cols) > 0:
        df = pd.concat([df, adata.obs[obs_cols]], axis=1)

    # reorder columns to given order (including duplicates keys if present)
    if keys:
        df = df[keys]

    for k, idx in obsm_keys:
        added_k = f"{k}-{idx}"
        val = adata.obsm[k]
        if isinstance(val, np.ndarray):
            df[added_k] = np.ravel(val[:, idx])
        elif isinstance(val, spmatrix):
            df[added_k] = np.ravel(val[:, idx].toarray())
        elif isinstance(val, pd.DataFrame):
            df[added_k] = val.loc[:, idx]

    return df


def var_df(
    adata: AnnData,
    keys: Iterable[str] = (),
    varm_keys: Iterable[Tuple[str, int]] = (),
    *,
    layer: str = None,
) -> pd.DataFrame:
    """\
    Return values for observations in adata.

    Params
    ------
    adata
        AnnData object to get values from.
    keys
        Keys from either `.obs_names`, or `.var.columns`.
    varm_keys
        Tuple of `(key from varm, column index of varm[key])`.
    layer
        Layer of `adata` to use as expression values.

    Returns
    -------
    A dataframe with `adata.var_names` as index, and values specified by `keys`
    and `varm_keys`.
    """
    # Argument handling
    var_cols, obs_idx_keys, _ = _check_indices(adata.var, adata.obs_names, "var", keys)

    # initialize df
    df = pd.DataFrame(index=adata.var.index)

    if len(obs_idx_keys) > 0:
        matrix = _get_array_values(
            _get_obs_rep(adata, layer=layer),
            adata.obs_names,
            obs_idx_keys,
            axis=0,
            backed=adata.isbacked,
        ).T
        df = pd.concat(
            [df, pd.DataFrame(matrix, columns=obs_idx_keys, index=adata.var_names)],
            axis=1,
        )

    # add obs values
    if len(var_cols) > 0:
        df = pd.concat([df, adata.var[var_cols]], axis=1)

    # reorder columns to given order
    if keys:
        df = df[keys]

    for k, idx in varm_keys:
        added_k = f"{k}-{idx}"
        val = adata.varm[k]
        if isinstance(val, np.ndarray):
            df[added_k] = np.ravel(val[:, idx])
        elif isinstance(val, spmatrix):
            df[added_k] = np.ravel(val[:, idx].toarray())
        elif isinstance(val, pd.DataFrame):
            df[added_k] = val.loc[:, idx]
    return df


def _get_obs_rep(adata, *, use_raw=False, layer=None, obsm=None, obsp=None):
    """
    Choose array aligned with obs annotation.
    """
    # https://github.com/scverse/scanpy/issues/1546
    if not isinstance(use_raw, bool):
        raise TypeError(f"use_raw expected to be bool, was {type(use_raw)}.")

    is_layer = layer is not None
    is_raw = use_raw is not False
    is_obsm = obsm is not None
    is_obsp = obsp is not None
    choices_made = sum((is_layer, is_raw, is_obsm, is_obsp))
    assert choices_made <= 1
    if choices_made == 0:
        return adata.X
    elif is_layer:
        return adata.layers[layer]
    elif use_raw:
        return adata.raw.X
    elif is_obsm:
        return adata.obsm[obsm]
    elif is_obsp:
        return adata.obsp[obsp]
    else:
        assert False, (
            "That was unexpected. Please report this bug at:\n\n\t"
            " https://github.com/scverse/scanpy/issues"
        )


def _set_obs_rep(adata, val, *, use_raw=False, layer=None, obsm=None, obsp=None):
    """
    Set value for observation rep.
    """
    is_layer = layer is not None
    is_raw = use_raw is not False
    is_obsm = obsm is not None
    is_obsp = obsp is not None
    choices_made = sum((is_layer, is_raw, is_obsm, is_obsp))
    assert choices_made <= 1
    if choices_made == 0:
        adata.X = val
    elif is_layer:
        adata.layers[layer] = val
    elif use_raw:
        adata.raw.X = val
    elif is_obsm:
        adata.obsm[obsm] = val
    elif is_obsp:
        adata.obsp[obsp] = val
    else:
        assert False, (
            "That was unexpected. Please report this bug at:\n\n\t"
            " https://github.com/scverse/scanpy/issues"
        )

def _get_mean_var(X, *, axis=0):
    if sparse.issparse(X):
        mean, var = sparse_mean_variance_axis(X, axis=axis)
    else:
        mean = np.mean(X, axis=axis, dtype=np.float64)
        mean_sq = np.multiply(X, X).mean(axis=axis, dtype=np.float64)
        var = mean_sq - mean ** 2
    # enforce R convention (unbiased estimator) for variance
    var *= X.shape[axis] / (X.shape[axis] - 1)
    return mean, var


def sparse_mean_variance_axis(mtx: sparse.spmatrix, axis: int):
    """
    This code and internal functions are based on sklearns
    `sparsefuncs.mean_variance_axis`.

    Modifications:
    * allow deciding on the output type, which can increase accuracy when calculating the mean and variance of 32bit floats.
    * This doesn't currently implement support for null values, but could.
    * Uses numba not cython
    """
    assert axis in (0, 1)
    if isinstance(mtx, sparse.csr_matrix):
        ax_minor = 1
        shape = mtx.shape
    elif isinstance(mtx, sparse.csc_matrix):
        ax_minor = 0
        shape = mtx.shape[::-1]
    else:
        raise ValueError("This function only works on sparse csr and csc matrices")
    if axis == ax_minor:
        return sparse_mean_var_major_axis(
            mtx.data, mtx.indices, mtx.indptr, *shape, np.float64
        )
    else:
        return sparse_mean_var_minor_axis(mtx.data, mtx.indices, *shape, np.float64)


# @numba.njit(cache=True)
def sparse_mean_var_minor_axis(data, indices, major_len, minor_len, dtype):
    """
    Computes mean and variance for a sparse matrix for the minor axis.

    Given arrays for a csr matrix, returns the means and variances for each
    column back.
    """
    non_zero = indices.shape[0]

    means = np.zeros(minor_len, dtype=dtype)
    variances = np.zeros_like(means, dtype=dtype)

    counts = np.zeros(minor_len, dtype=np.int64)

    for i in range(non_zero):
        col_ind = indices[i]
        means[col_ind] += data[i]

    for i in range(minor_len):
        means[i] /= major_len

    for i in range(non_zero):
        col_ind = indices[i]
        diff = data[i] - means[col_ind]
        variances[col_ind] += diff * diff
        counts[col_ind] += 1

    for i in range(minor_len):
        variances[i] += (major_len - counts[i]) * means[i] ** 2
        variances[i] /= major_len

    return means, variances


# @numba.njit(cache=True)
def sparse_mean_var_major_axis(data, indices, indptr, major_len, minor_len, dtype):
    """
    Computes mean and variance for a sparse array for the major axis.

    Given arrays for a csr matrix, returns the means and variances for each
    row back.
    """
    means = np.zeros(major_len, dtype=dtype)
    variances = np.zeros_like(means, dtype=dtype)

    for i in range(major_len):
        startptr = indptr[i]
        endptr = indptr[i + 1]
        counts = endptr - startptr

        for j in range(startptr, endptr):
            means[i] += data[j]
        means[i] /= minor_len

        for j in range(startptr, endptr):
            diff = data[j] - means[i]
            variances[i] += diff * diff

        variances[i] += (minor_len - counts) * means[i] ** 2
        variances[i] /= minor_len

    return means, variances

def _fallback_to_uns(dct, conns, dists, conns_key, dists_key):
    if conns is None and conns_key in dct:
        conns = dct[conns_key]
    if dists is None and dists_key in dct:
        dists = dct[dists_key]

    return conns, dists

class NeighborsView:
    """Convenience class for accessing neighbors graph representations.

    Allows to access neighbors distances, connectivities and settings
    dictionary in a uniform manner.

    Parameters
    ----------

    adata
        AnnData object.
    key
        This defines where to look for neighbors dictionary,
        connectivities, distances.

        neigh = NeighborsView(adata, key)
        neigh['distances']
        neigh['connectivities']
        neigh['params']
        'connectivities' in neigh
        'params' in neigh

        is the same as

        adata.obsp[adata.uns[key]['distances_key']]
        adata.obsp[adata.uns[key]['connectivities_key']]
        adata.uns[key]['params']
        adata.uns[key]['connectivities_key'] in adata.obsp
        'params' in adata.uns[key]
    """

    def __init__(self, adata, key=None):
        self._connectivities = None
        self._distances = None

        if key is None or key == 'neighbors':
            if 'neighbors' not in adata.uns:
                raise KeyError('No "neighbors" in .uns')
            self._neighbors_dict = adata.uns['neighbors']
            self._conns_key = 'connectivities'
            self._dists_key = 'distances'
        else:
            if key not in adata.uns:
                raise KeyError(f'No "{key}" in .uns')
            self._neighbors_dict = adata.uns[key]
            self._conns_key = self._neighbors_dict['connectivities_key']
            self._dists_key = self._neighbors_dict['distances_key']

        if self._conns_key in adata.obsp:
            self._connectivities = adata.obsp[self._conns_key]
        if self._dists_key in adata.obsp:
            self._distances = adata.obsp[self._dists_key]

        # fallback to uns
        self._connectivities, self._distances = _fallback_to_uns(
            self._neighbors_dict,
            self._connectivities,
            self._distances,
            self._conns_key,
            self._dists_key,
        )

    def __getitem__(self, key):
        if key == 'distances':
            if 'distances' not in self:
                raise KeyError(f'No "{self._dists_key}" in .obsp')
            return self._distances
        elif key == 'connectivities':
            if 'connectivities' not in self:
                raise KeyError(f'No "{self._conns_key}" in .obsp')
            return self._connectivities
        else:
            return self._neighbors_dict[key]

    def __contains__(self, key):
        if key == 'distances':
            return self._distances is not None
        elif key == 'connectivities':
            return self._connectivities is not None
        else:
            return key in self._neighbors_dict


_doc_org = """\
org
    Organism to query. Must be an organism in ensembl biomart. "hsapiens",
    "mmusculus", "drerio", etc.\
"""

@singledispatch
@_doc_params(doc_org=_doc_org)
def enrich(
    container: Union[Iterable[str], Mapping[str, Iterable[str]]],
    *,
    org: str = "hsapiens",
    gprofiler_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> pd.DataFrame:
    """\
    Get enrichment for DE results.

    This is a thin convenience wrapper around the very useful gprofiler_.

    This method dispatches on the first argument, leading to the following two
    signatures::

        enrich(container, ...)
        enrich(adata: AnnData, group, key: str, ...)

    Where::

        enrich(adata, group, key, ...) = enrich(adata.uns[key]["names"][group], ...)

    .. _gprofiler: https://pypi.org/project/gprofiler-official/#description

    Parameters
    ----------
    container
        Contains list of genes you'd like to search. If container is a `dict` all
        enrichment queries are made at once.
    adata
        AnnData object whose group will be looked for.
    group
        The group whose genes should be used for enrichment.
    key
        Key in `uns` to find group under.
    {doc_org}
    gprofiler_kwargs
        Keyword arguments to pass to `GProfiler.profile`, see gprofiler_. Some
        useful options are `no_evidences=False` which reports gene intersections,
        `sources=['GO:BP']` which limits gene sets to only GO biological processes and
        `all_results=True` which returns all results including the non-significant ones.
    **kwargs
        All other keyword arguments are passed to `sc.get.rank_genes_groups_df`. E.g.
        pval_cutoff, log2fc_min.

    Returns
    -------
    Dataframe of enrichment results.

    Examples
    --------
    Using `sc.queries.enrich` on a list of genes:

    >>> import scanpy as sc
    >>> sc.queries.enrich(['KLF4', 'PAX5', 'SOX2', 'NANOG'], org="hsapiens")
    >>> sc.queries.enrich({{'set1':['KLF4', 'PAX5'], 'set2':['SOX2', 'NANOG']}}, org="hsapiens")

    Using `sc.queries.enrich` on an :class:`anndata.AnnData` object:

    >>> pbmcs = sc.datasets.pbmc68k_reduced()
    >>> sc.tl.rank_genes_groups(pbmcs, "bulk_labels")
    >>> sc.queries.enrich(pbmcs, "CD34+")
    """
    try:
        from gprofiler import GProfiler
    except ImportError:
        raise ImportError(
            "This method requires the `gprofiler-official` module to be installed."
        )
    gprofiler = GProfiler(user_agent="scanpy", return_dataframe=True)
    gprofiler_kwargs = dict(gprofiler_kwargs)
    for k in ["organism"]:
        if gprofiler_kwargs.get(k) is not None:
            raise ValueError(
                f"Argument `{k}` should be passed directly through `enrich`, "
                "not through `gprofiler_kwargs`"
            )
    return gprofiler.profile(container, organism=org, **gprofiler_kwargs)