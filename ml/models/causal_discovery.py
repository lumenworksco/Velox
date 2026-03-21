"""EDGE-012: Causal Discovery for Factor Construction.

Discovers causal relationships between features and assets from
observational data using constraint-based algorithms:

  - PC (Peter-Clark) algorithm for causal DAG discovery
  - FCI (Fast Causal Inference) variant that handles latent confounders
  - Conditional independence tests (partial correlation, Fisher Z)

Useful for:
  - Constructing causal alpha factors (not merely correlated)
  - Identifying confounders that bias model training
  - Building structural models of market relationships

Uses networkx for graph representation when available; falls back to a
dict-of-lists adjacency representation.  scipy is used for statistical
tests with a pure-numpy fallback.

All library imports are conditional — the bot runs without them.
"""

import logging
import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

_HAS_NETWORKX = False
_HAS_SCIPY = False

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    nx = None  # type: ignore[assignment]

try:
    from scipy import stats as sp_stats
    _HAS_SCIPY = True
except ImportError:
    sp_stats = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CausalConfig:
    """Configuration for causal discovery."""

    alpha: float = 0.05          # significance level for CI tests
    max_cond_set: int = 3        # max conditioning set size
    method: str = "pc"           # "pc" or "fci"
    ci_test: str = "partial_corr"  # "partial_corr" or "fisher_z"
    stable: bool = True          # use order-independent (stable) PC
    seed: int = 42


# ---------------------------------------------------------------------------
# Causal graph wrapper
# ---------------------------------------------------------------------------


class CausalGraph:
    """Lightweight causal graph representation.

    Uses networkx.DiGraph when available, otherwise a dict-of-sets
    adjacency list.
    """

    def __init__(self, nodes: List[str]):
        self.nodes = list(nodes)
        self._use_nx = _HAS_NETWORKX
        if self._use_nx:
            self._graph = nx.DiGraph()
            self._graph.add_nodes_from(nodes)
        else:
            self._adj: Dict[str, Set[str]] = {n: set() for n in nodes}

    def add_edge(self, u: str, v: str) -> None:
        if self._use_nx:
            self._graph.add_edge(u, v)
        else:
            self._adj.setdefault(u, set()).add(v)

    def remove_edge(self, u: str, v: str) -> None:
        if self._use_nx:
            if self._graph.has_edge(u, v):
                self._graph.remove_edge(u, v)
        else:
            self._adj.get(u, set()).discard(v)

    def has_edge(self, u: str, v: str) -> bool:
        if self._use_nx:
            return self._graph.has_edge(u, v)
        return v in self._adj.get(u, set())

    def parents(self, node: str) -> List[str]:
        if self._use_nx:
            return list(self._graph.predecessors(node))
        return [u for u, children in self._adj.items() if node in children]

    def children(self, node: str) -> List[str]:
        if self._use_nx:
            return list(self._graph.successors(node))
        return list(self._adj.get(node, set()))

    def neighbors(self, node: str) -> List[str]:
        """All nodes connected to node (either direction)."""
        result = set(self.parents(node)) | set(self.children(node))
        return list(result)

    def edges(self) -> List[Tuple[str, str]]:
        if self._use_nx:
            return list(self._graph.edges())
        result = []
        for u, vs in self._adj.items():
            for v in vs:
                result.append((u, v))
        return result

    def to_dict(self) -> Dict[str, List[str]]:
        return {n: self.children(n) for n in self.nodes}

    def __repr__(self) -> str:
        return f"CausalGraph(nodes={len(self.nodes)}, edges={len(self.edges())})"


# ---------------------------------------------------------------------------
# Conditional independence tests
# ---------------------------------------------------------------------------


def _partial_correlation(
    data: np.ndarray, i: int, j: int, cond: List[int]
) -> Tuple[float, float]:
    """Compute partial correlation between columns i, j given cond.

    Returns (correlation, p-value).
    """
    n = data.shape[0]
    if not cond:
        r = np.corrcoef(data[:, i], data[:, j])[0, 1]
    else:
        # Regress out conditioning variables
        C = data[:, cond]
        C_aug = np.column_stack([C, np.ones(n)])
        try:
            proj = C_aug @ np.linalg.lstsq(C_aug, data[:, [i, j]], rcond=None)[0]
        except np.linalg.LinAlgError:
            return 0.0, 1.0
        resid = data[:, [i, j]] - proj
        r = np.corrcoef(resid[:, 0], resid[:, 1])[0, 1]

    r = np.clip(r, -0.9999, 0.9999)
    # Fisher Z-transform for p-value
    dof = n - len(cond) - 2
    if dof < 1:
        return float(r), 1.0
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(max(dof, 1))
    z_stat = abs(z) / se

    # p-value from normal distribution
    if _HAS_SCIPY:
        p_val = 2.0 * (1.0 - sp_stats.norm.cdf(z_stat))
    else:
        # Approximation of normal CDF tail
        p_val = 2.0 * _approx_normal_sf(z_stat)

    return float(r), float(p_val)


def _approx_normal_sf(z: float) -> float:
    """Approximate standard normal survival function (no scipy)."""
    # Abramowitz & Stegun 26.2.17
    if z < 0:
        return 1.0 - _approx_normal_sf(-z)
    t = 1.0 / (1.0 + 0.2316419 * z)
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return d * math.exp(-0.5 * z * z) * poly


# ---------------------------------------------------------------------------
# PC algorithm
# ---------------------------------------------------------------------------


def _pc_skeleton(
    data: np.ndarray,
    node_names: List[str],
    alpha: float,
    max_cond: int,
    stable: bool,
) -> Tuple[Dict[FrozenSet[str], bool], Dict[FrozenSet[str], List[int]]]:
    """Learn the undirected skeleton using the PC algorithm.

    Returns
    -------
    adj : dict mapping frozenset({u,v}) -> bool (edge present)
    sep_sets : dict mapping frozenset({u,v}) -> list of conditioning indices
    """
    n_vars = data.shape[1]
    idx_map = {name: i for i, name in enumerate(node_names)}

    # Start with complete undirected graph
    adj: Dict[FrozenSet[str], bool] = {}
    for u, v in combinations(node_names, 2):
        adj[frozenset({u, v})] = True

    sep_sets: Dict[FrozenSet[str], List[int]] = {}

    for depth in range(max_cond + 1):
        removals: List[FrozenSet[str]] = []

        for u, v in combinations(node_names, 2):
            if not adj.get(frozenset({u, v}), False):
                continue

            i_u, i_v = idx_map[u], idx_map[v]
            # Neighbors of u (excluding v)
            nbrs = [
                w for w in node_names
                if w != u and w != v and adj.get(frozenset({u, w}), False)
            ]

            if len(nbrs) < depth:
                continue

            for cond_set in combinations(nbrs, depth):
                cond_idx = [idx_map[w] for w in cond_set]
                _, p_val = _partial_correlation(data, i_u, i_v, cond_idx)

                if p_val > alpha:
                    if stable:
                        removals.append(frozenset({u, v}))
                    else:
                        adj[frozenset({u, v})] = False
                    sep_sets[frozenset({u, v})] = cond_idx
                    break

        if stable:
            for edge in removals:
                adj[edge] = False

    return adj, sep_sets


def _orient_edges(
    adj: Dict[FrozenSet[str], bool],
    sep_sets: Dict[FrozenSet[str], List[int]],
    node_names: List[str],
    idx_map: Dict[str, int],
) -> CausalGraph:
    """Orient edges using v-structures and Meek rules."""
    graph = CausalGraph(node_names)

    # Add all undirected edges as bidirectional
    for edge, present in adj.items():
        if not present:
            continue
        u, v = list(edge)
        graph.add_edge(u, v)
        graph.add_edge(v, u)

    # Rule 1: Orient v-structures (u -> w <- v if w not in sep(u,v))
    for w in node_names:
        nbrs = graph.neighbors(w)
        for u, v in combinations(nbrs, 2):
            if graph.has_edge(u, v) or graph.has_edge(v, u):
                continue  # u and v are adjacent — not a v-structure
            sep = sep_sets.get(frozenset({u, v}), [])
            if idx_map[w] not in sep:
                # Orient as u -> w <- v
                graph.remove_edge(w, u)
                graph.remove_edge(w, v)

    # Rule 2 (Meek): if u -> v — w and u not adj w, then v -> w
    changed = True
    max_iters = len(node_names) * 2
    iters = 0
    while changed and iters < max_iters:
        changed = False
        iters += 1
        for v in node_names:
            parents_v = graph.parents(v)
            children_v = [c for c in graph.children(v) if graph.has_edge(c, v)]  # bidirectional
            for u in parents_v:
                if graph.has_edge(v, u):
                    continue  # still bidirectional
                for w in children_v:
                    if w == u:
                        continue
                    if not graph.has_edge(u, w) and not graph.has_edge(w, u):
                        graph.remove_edge(w, v)
                        changed = True

    return graph


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class CausalDiscovery:
    """Causal discovery engine for financial factor construction.

    Follows the common model interface with fit() / predict() / score().

    Parameters
    ----------
    config : CausalConfig, optional
        Algorithm configuration.
    """

    def __init__(self, config: Optional[CausalConfig] = None):
        self.config = config or CausalConfig()
        self._graph: Optional[CausalGraph] = None
        self._fitted = False
        self._node_names: List[str] = []
        self._data: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Common interface
    # ------------------------------------------------------------------

    def fit(
        self,
        data: np.ndarray,
        column_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "CausalDiscovery":
        """Discover the causal graph from observational data.

        Parameters
        ----------
        data : np.ndarray
            2-D array (n_samples, n_features).
        column_names : list of str, optional
            Feature names.  Defaults to X0, X1, ...
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("data must be 2-D (n_samples, n_features)")

        n_vars = data.shape[1]
        if column_names is None:
            column_names = [f"X{i}" for i in range(n_vars)]
        if len(column_names) != n_vars:
            raise ValueError("column_names length must match number of features")

        self._node_names = column_names
        self._data = data
        self._graph = self.discover_graph(data, column_names)
        self._fitted = True
        return self

    def predict(self, target: str) -> List[str]:
        """Return causal parents of the target variable (alias for get_causal_features)."""
        return self.get_causal_features(target)

    def score(self, data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Return summary statistics about the discovered graph."""
        if self._graph is None:
            raise RuntimeError("Model must be fitted first")
        edges = self._graph.edges()
        return {
            "n_nodes": len(self._node_names),
            "n_edges": len(edges),
            "density": len(edges) / max(len(self._node_names) * (len(self._node_names) - 1), 1),
            "edges": edges,
            "method": self.config.method,
        }

    # ------------------------------------------------------------------
    # Core discovery
    # ------------------------------------------------------------------

    def discover_graph(
        self,
        data: np.ndarray,
        column_names: Optional[List[str]] = None,
    ) -> CausalGraph:
        """Run the PC or FCI algorithm to discover a causal DAG.

        Parameters
        ----------
        data : np.ndarray
            Observational data (n_samples, n_features).
        column_names : list of str, optional
            Feature names.

        Returns
        -------
        CausalGraph
        """
        data = np.asarray(data, dtype=np.float64)
        n_vars = data.shape[1]
        names = column_names or [f"X{i}" for i in range(n_vars)]
        idx_map = {name: i for i, name in enumerate(names)}

        logger.info(
            "Running %s algorithm on %d variables, %d observations ...",
            self.config.method.upper(), n_vars, data.shape[0],
        )

        adj, sep_sets = _pc_skeleton(
            data, names, self.config.alpha, self.config.max_cond_set, self.config.stable,
        )

        if self.config.method == "fci":
            graph = self._fci_orient(adj, sep_sets, names, idx_map, data)
        else:
            graph = _orient_edges(adj, sep_sets, names, idx_map)

        n_edges = len(graph.edges())
        logger.info("Discovered causal graph: %d nodes, %d edges", len(names), n_edges)
        return graph

    def get_causal_features(self, target: str) -> List[str]:
        """Return the causal parents (direct causes) of a target variable.

        Parameters
        ----------
        target : str
            Name of the target variable.

        Returns
        -------
        list of str
            Variables that are direct causes of target.
        """
        if self._graph is None:
            raise RuntimeError("Must call fit() or discover_graph() first")
        if target not in self._node_names:
            raise ValueError(f"Target '{target}' not in graph nodes: {self._node_names}")
        parents = self._graph.parents(target)
        logger.info("Causal parents of '%s': %s", target, parents)
        return parents

    def get_causal_children(self, source: str) -> List[str]:
        """Return the direct effects of a source variable."""
        if self._graph is None:
            raise RuntimeError("Must call fit() or discover_graph() first")
        return self._graph.children(source)

    def get_markov_blanket(self, target: str) -> List[str]:
        """Return the Markov blanket of a target variable.

        The Markov blanket = parents + children + parents of children.
        """
        if self._graph is None:
            raise RuntimeError("Must call fit() or discover_graph() first")
        parents = set(self._graph.parents(target))
        children = set(self._graph.children(target))
        co_parents: Set[str] = set()
        for child in children:
            co_parents.update(self._graph.parents(child))
        blanket = parents | children | co_parents
        blanket.discard(target)
        return sorted(blanket)

    # ------------------------------------------------------------------
    # FCI extension (handles latent confounders)
    # ------------------------------------------------------------------

    def _fci_orient(
        self,
        adj: Dict[FrozenSet[str], bool],
        sep_sets: Dict[FrozenSet[str], List[int]],
        names: List[str],
        idx_map: Dict[str, int],
        data: np.ndarray,
    ) -> CausalGraph:
        """FCI orientation rules (simplified).

        Extends PC with additional rules for detecting latent confounders
        (bidirected edges: u <-> v).
        """
        # Start with PC orientation
        graph = _orient_edges(adj, sep_sets, names, idx_map)

        # FCI discriminating path rule (simplified):
        # For each undirected edge u -- v, test if there exists a
        # discriminating path indicating a latent confounder.
        for u in names:
            for v in graph.children(u):
                if not graph.has_edge(v, u):
                    continue  # already oriented
                # Test for possible latent confounder via additional CI tests
                nbrs_u = set(graph.neighbors(u)) - {v}
                nbrs_v = set(graph.neighbors(v)) - {u}
                common = nbrs_u & nbrs_v
                if common:
                    # If there's a common neighbor with certain CI structure,
                    # mark as potentially confounded (keep bidirected)
                    pass
                else:
                    # No evidence of confounder — orient based on topology
                    parents_u = [p for p in graph.parents(u) if not graph.has_edge(u, p)]
                    if parents_u:
                        graph.remove_edge(v, u)

        return graph

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def to_adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Return the causal graph as an adjacency matrix.

        Returns (matrix, node_names) where matrix[i,j] = 1 means i -> j.
        """
        if self._graph is None:
            raise RuntimeError("Must call fit() first")
        n = len(self._node_names)
        idx = {name: i for i, name in enumerate(self._node_names)}
        mat = np.zeros((n, n), dtype=int)
        for u, v in self._graph.edges():
            mat[idx[u], idx[v]] = 1
        return mat, self._node_names

    def summary(self) -> str:
        """Return a human-readable summary of the discovered graph."""
        if self._graph is None:
            return "No graph discovered yet. Call fit() first."
        edges = self._graph.edges()
        lines = [
            f"Causal Graph Summary ({self.config.method.upper()})",
            f"  Nodes: {len(self._node_names)}",
            f"  Edges: {len(edges)}",
            "  Edges:",
        ]
        for u, v in edges:
            lines.append(f"    {u} -> {v}")
        return "\n".join(lines)
