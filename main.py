import math
import random
import heapq
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Set

import numpy as np
import networkx as nx
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GINConv, global_add_pool
    from torch_geometric.datasets import TUDataset
except Exception as e:
    raise ImportError("PyTorch / PyG not installed") from e


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def attach_indices(dataset: List[Data]) -> List[Data]:
    for i, d in enumerate(dataset):
        d.idx = torch.tensor([i], dtype=torch.long)
    return dataset


def ensure_node_features(dataset: List[Data]):
    if dataset[0].x is None:
        for d in dataset:
            d.x = torch.ones((d.num_nodes, 1), dtype=torch.float32)


def stratified_split(
    dataset: List[Data],
    test_ratio: float,
    seed: int,
) -> Tuple[List[Data], List[Data]]:
    y = np.array([int(d.y.item()) for d in dataset], dtype=np.int64)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    idx_all = np.arange(len(dataset))
    tr_idx, te_idx = next(splitter.split(idx_all, y))
    train_set = [dataset[i] for i in tr_idx]
    test_set = [dataset[i] for i in te_idx]
    return train_set, test_set


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for b in loader:
        b = b.to(device)
        logits = model(b)
        pred = logits.argmax(dim=-1)
        y = b.y.view(-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return (correct / total) if total > 0 else 0.0


def to_undirected_edge_set(edge_index: torch.Tensor) -> Set[Tuple[int, int]]:
    edges = set()
    if edge_index.numel() == 0:
        return edges
    ei = edge_index.detach().cpu().numpy()
    for u, v in ei.T:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))
    return edges


def edge_set_to_edge_index(edges: Set[Tuple[int, int]], num_nodes: int) -> torch.Tensor:
    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    rows = []
    cols = []
    for u, v in edges:
        rows.append(u)
        cols.append(v)
        rows.append(v)
        cols.append(u)
    return torch.tensor([rows, cols], dtype=torch.long)


def data_to_nx_undirected(data: Data) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(int(data.num_nodes)))
    edges = to_undirected_edge_set(data.edge_index)
    g.add_edges_from(list(edges))
    return g


def nx_to_data_with_same_x(base: Data, g: nx.Graph) -> Data:
    edges = set()
    for u, v in g.edges():
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))
    edge_index = edge_set_to_edge_index(edges, int(base.num_nodes))
    out = Data(
        x=base.x,
        edge_index=edge_index,
        y=base.y,
        num_nodes=base.num_nodes
    )
    if hasattr(base, "idx"):
        out.idx = base.idx
    return out


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1)
    return 0.5 * (kl_pm + kl_qm)


class Predicate:
    def canon_id(self) -> str:
        raise NotImplementedError

    def applicable(self, g: nx.Graph) -> bool:
        return True

    def __call__(self, g: nx.Graph) -> int:
        raise NotImplementedError

    def scope_nodes(self, g: nx.Graph) -> Optional[Set[int]]:
        return None


class TriangleExists(Predicate):
    def canon_id(self) -> str:
        return "triangle_exists"

    def __call__(self, g: nx.Graph) -> int:
        tri = nx.triangles(g)
        return 1 if any(v > 0 for v in tri.values()) else 0

    def scope_nodes(self, g: nx.Graph) -> Optional[Set[int]]:
        return {n for n, d in g.degree() if d >= 2}


class MaxDegreeAtLeastK(Predicate):
    def __init__(self, k: int):
        self.k = int(k)

    def canon_id(self) -> str:
        return f"max_degree_ge_{self.k}"

    def __call__(self, g: nx.Graph) -> int:
        if g.number_of_nodes() == 0:
            return 0
        md = max((d for _, d in g.degree()), default=0)
        return 1 if md >= self.k else 0

    def scope_nodes(self, g: nx.Graph) -> Optional[Set[int]]:
        degs = dict(g.degree())
        if not degs:
            return None
        md = max(degs.values())
        return {n for n, d in degs.items() if d == md}


class EdgeCountAtLeastK(Predicate):
    def __init__(self, k: int):
        self.k = int(k)

    def canon_id(self) -> str:
        return f"edge_count_ge_{self.k}"

    def __call__(self, g: nx.Graph) -> int:
        return 1 if g.number_of_edges() >= self.k else 0


class PredicateLibrary:
    def __init__(self, degree_thresholds=(2, 3, 4), edge_thresholds=(10, 20, 30), include_triangle=True):
        self.degree_thresholds = list(degree_thresholds)
        self.edge_thresholds = list(edge_thresholds)
        self.include_triangle = include_triangle

    def propose(self, g: nx.Graph) -> List[Predicate]:
        preds: List[Predicate] = []
        if self.include_triangle:
            preds.append(TriangleExists())
        for k in self.degree_thresholds:
            preds.append(MaxDegreeAtLeastK(k))
        for k in self.edge_thresholds:
            preds.append(EdgeCountAtLeastK(k))
        return preds


class DomainValidator:
    def validate(self, g: nx.Graph) -> bool:
        if g.is_directed():
            return False
        if any(u == v for u, v in g.edges()):
            return False
        return True


@dataclass(frozen=True)
class EditCost:
    alpha_e: float = 1.0
    alpha_v: float = 1.0
    alpha_s: float = 1.0


class GraphEditSpace:
    def __init__(self, cost: EditCost, max_add_candidates: int = 80):
        self.cost = cost
        self.max_add_candidates = max_add_candidates

    def neighbors(self, g: nx.Graph, scope: Optional[Set[int]] = None) -> Iterable[Tuple[nx.Graph, float, str]]:
        nodes = list(g.nodes())
        if scope is not None and len(scope) > 0:
            nodes = [n for n in nodes if n in scope]
            if len(nodes) == 0:
                nodes = list(g.nodes())

        edges = list(g.edges())
        if scope is not None and len(scope) > 0:
            edges = [(u, v) for (u, v) in edges if (u in scope or v in scope)] or list(g.edges())

        for u, v in edges:
            g2 = g.copy()
            g2.remove_edge(u, v)
            yield g2, self.cost.alpha_e, f"remove_edge({u},{v})"

        cand = []
        all_nodes = list(g.nodes())
        if len(all_nodes) <= 1:
            return

        tries = 0
        while len(cand) < self.max_add_candidates and tries < self.max_add_candidates * 10:
            tries += 1
            u = random.choice(nodes)
            v = random.choice(all_nodes)
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if g.has_edge(a, b):
                continue
            cand.append((a, b))

        for u, v in cand:
            g2 = g.copy()
            g2.add_edge(u, v)
            yield g2, self.cost.alpha_e, f"add_edge({u},{v})"


@dataclass
class ToggleResult:
    found: bool
    graph: Optional[nx.Graph]
    ub: float
    lb: float
    edits: List[str]


class BestFirstToggleSearch:
    def __init__(
        self,
        validator: DomainValidator,
        edit_space: GraphEditSpace,
        budget_B: float,
        eps_gap: float = 0.0,
        max_expansions: int = 2000,
        dedup_limit: int = 200000,
    ):
        self.validator = validator
        self.edit_space = edit_space
        self.B = float(budget_B)
        self.eps_gap = float(eps_gap)
        self.max_expansions = int(max_expansions)
        self.dedup_limit = int(dedup_limit)

    @staticmethod
    def _graph_key(g: nx.Graph) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
        edges = []
        for u, v in g.edges():
            a, b = (u, v) if u < v else (v, u)
            edges.append((a, b))
        edges.sort()
        return (g.number_of_nodes(), tuple(edges))

    def search(self, g0: nx.Graph, predicate: Predicate, target_b: int) -> ToggleResult:
        assert target_b in (0, 1)

        if not predicate.applicable(g0):
            return ToggleResult(False, None, ub=float("inf"), lb=float("inf"), edits=[])

        if not self.validator.validate(g0):
            return ToggleResult(False, None, ub=float("inf"), lb=float("inf"), edits=[])

        if int(predicate(g0)) == target_b:
            return ToggleResult(True, g0, ub=0.0, lb=0.0, edits=[])

        scope = predicate.scope_nodes(g0)

        pq: List[Tuple[float, int, nx.Graph, List[str]]] = []
        tie = 0
        heapq.heappush(pq, (0.0, tie, g0, []))

        seen: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], float] = {self._graph_key(g0): 0.0}
        ub = float("inf")
        best_graph = None
        best_edits: List[str] = []

        expansions = 0
        while pq and expansions < self.max_expansions:
            lb = pq[0][0]

            if lb > self.B:
                return ToggleResult(False, None, ub=ub, lb=lb, edits=[])

            if ub < float("inf") and (ub - lb) <= self.eps_gap:
                return ToggleResult(True, best_graph, ub=ub, lb=lb, edits=best_edits)

            g_cost, _, g, edits = heapq.heappop(pq)
            expansions += 1

            if g_cost > self.B:
                continue
            if ub < float("inf") and g_cost >= ub:
                continue

            for g2, step_cost, desc in self.edit_space.neighbors(g, scope=scope):
                new_cost = g_cost + step_cost
                if new_cost > self.B:
                    continue
                if not self.validator.validate(g2):
                    continue

                key = self._graph_key(g2)
                old = seen.get(key, float("inf"))
                if new_cost >= old:
                    continue
                if len(seen) > self.dedup_limit:
                    continue

                seen[key] = new_cost
                new_edits = edits + [desc]

                if int(predicate(g2)) == target_b:
                    if new_cost < ub:
                        ub = new_cost
                        best_graph = g2
                        best_edits = new_edits
                    continue

                tie += 1
                heapq.heappush(pq, (new_cost, tie, g2, new_edits))

        lb = pq[0][0] if pq else float("inf")
        if best_graph is None:
            return ToggleResult(False, None, ub=float("inf"), lb=lb, edits=[])
        return ToggleResult(True, best_graph, ub=ub, lb=lb, edits=best_edits)


@dataclass
class InterventionPair:
    g1: nx.Graph
    g0: nx.Graph
    ub1: float
    lb1: float
    ub0: float
    lb0: float
    gap: float
    max_cost: float
    weight: float
    edits1: List[str]
    edits0: List[str]


class InterventionGenerator:
    def __init__(
        self,
        validator: DomainValidator,
        edit_space: GraphEditSpace,
        B: float,
        gamma: float = 1.0,
        eta: float = 1.0,
        eps_gap: float = 0.0,
        max_expansions: int = 2000,
    ):
        self.searcher = BestFirstToggleSearch(
            validator=validator,
            edit_space=edit_space,
            budget_B=B,
            eps_gap=eps_gap,
            max_expansions=max_expansions,
        )
        self.gamma = float(gamma)
        self.eta = float(eta)

    def generate_pair(self, g: nx.Graph, predicate: Predicate) -> Optional[InterventionPair]:
        r1 = self.searcher.search(g, predicate, target_b=1)
        r0 = self.searcher.search(g, predicate, target_b=0)
        if (not r1.found) or (not r0.found) or (r1.graph is None) or (r0.graph is None):
            return None

        gap = max((r1.ub - r1.lb), (r0.ub - r0.lb))
        max_cost = max(r1.ub, r0.ub)
        weight = math.exp(-self.gamma * gap - self.eta * max_cost)

        return InterventionPair(
            g1=r1.graph, g0=r0.graph,
            ub1=r1.ub, lb1=r1.lb, ub0=r0.ub, lb0=r0.lb,
            gap=gap, max_cost=max_cost, weight=weight,
            edits1=r1.edits, edits0=r0.edits
        )


@dataclass
class StabilityOutputs:
    s_dir: np.ndarray
    r_rel: np.ndarray
    u_list: List[str]


class StabilityEstimator:
    def __init__(
        self,
        epsilon: float = 1e-6,
        Emin: int = 2,
        Pmin: int = 2,
        tau_delta: float = 0.05,
        r_max: float = 1.0,
        support_min: int = 5,
        pseudo_env_E: int = 4,
        pseudo_env_P: int = 3,
        crossfit_folds: int = 3,
        warm_epochs: int = 10,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.epsilon = float(epsilon)
        self.Emin = int(Emin)
        self.Pmin = int(Pmin)
        self.tau_delta = float(tau_delta)
        self.r_max = float(r_max)
        self.support_min = int(support_min)

        self.pseudo_env_E = int(pseudo_env_E)
        self.pseudo_env_P = int(pseudo_env_P)
        self.crossfit_folds = int(crossfit_folds)
        self.warm_epochs = int(warm_epochs)

        self.device = device
        self.seed = seed

    def build_exec_table(
        self,
        graphs_nx: List[nx.Graph],
        y: np.ndarray,
        u_keep: List[str],
        u_to_predicate: Dict[str, Predicate],
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(graphs_nx)
        U = len(u_keep)
        m = np.zeros((n, U), dtype=np.int32)
        val = np.zeros((n, U), dtype=np.int32)

        for j, uid in enumerate(u_keep):
            pred = u_to_predicate[uid]
            for i, g in enumerate(graphs_nx):
                app = 1 if pred.applicable(g) else 0
                m[i, j] = app
                if app:
                    val[i, j] = int(pred(g))
        return m, val

    def _train_warm_model(self, train_loader: DataLoader, num_classes: int, num_node_features: int) -> nn.Module:
        model = GINClassifier(num_node_features=num_node_features, num_classes=num_classes)
        model.to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        model.train()
        for _ in range(self.warm_epochs):
            for batch in train_loader:
                batch = batch.to(self.device)
                logits = model(batch)
                loss = F.cross_entropy(logits, batch.y.view(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
        return model

    @torch.no_grad()
    def _get_logits(self, model: nn.Module, loader: DataLoader) -> np.ndarray:
        model.eval()
        outs = []
        for batch in loader:
            batch = batch.to(self.device)
            logits = model(batch)
            outs.append(logits.detach().cpu().numpy())
        return np.concatenate(outs, axis=0)

    def infer_pseudo_envs(self, dataset: List[Data], y: np.ndarray, num_classes: int) -> List[np.ndarray]:
        partitions = []
        n = len(dataset)

        if dataset[0].x is None:
            num_node_features = 1
        else:
            num_node_features = int(dataset[0].x.size(-1))

        for p in range(self.pseudo_env_P):
            seed = self.seed + 1000 * p
            skf = StratifiedKFold(n_splits=self.crossfit_folds, shuffle=True, random_state=seed)
            logits_all = np.zeros((n, num_classes), dtype=np.float32)

            for _, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(n), y)):
                tr_set = [dataset[i] for i in tr_idx]
                te_set = [dataset[i] for i in te_idx]
                tr_loader = DataLoader(tr_set, batch_size=64, shuffle=True)
                te_loader = DataLoader(te_set, batch_size=128, shuffle=False)

                model = self._train_warm_model(tr_loader, num_classes=num_classes, num_node_features=num_node_features)
                logits = self._get_logits(model, te_loader)
                logits_all[te_idx] = logits

            E = min(self.pseudo_env_E, logits_all.shape[0])
            km = KMeans(n_clusters=E, random_state=seed, n_init=10)
            env = km.fit_predict(logits_all)
            partitions.append(env.astype(np.int32))

        return partitions

    def compute_stability(
        self,
        y: np.ndarray,
        m_exec: np.ndarray,
        val_exec: np.ndarray,
        partitions: List[np.ndarray],
        num_classes: int,
        support_threshold: int = 10,
    ) -> StabilityOutputs:
        n, U = m_exec.shape
        C = num_classes
        P = len(partitions)

        s_part = np.ones((P, C, U), dtype=np.int32)
        r_part = np.zeros((P, C, U), dtype=np.float32)

        for p, env in enumerate(partitions):
            E = int(env.max()) + 1

            for ycls in range(C):
                y_mask = (y == ycls)
                ny = int(y_mask.sum())
                nnot = int((~y_mask).sum())
                if ny == 0 or nnot == 0:
                    continue

                for u in range(U):
                    delta_list = []

                    for e in range(E):
                        idx_e = (env == e)
                        den_y = int(np.sum(idx_e & y_mask & (m_exec[:, u] == 1)))
                        den_ny = int(np.sum(idx_e & (~y_mask) & (m_exec[:, u] == 1)))
                        if den_y < support_threshold or den_ny < support_threshold:
                            continue

                        num_y = int(np.sum(idx_e & y_mask & (m_exec[:, u] == 1) & (val_exec[:, u] == 1)))
                        num_ny = int(np.sum(idx_e & (~y_mask) & (m_exec[:, u] == 1) & (val_exec[:, u] == 1)))

                        pi_y = num_y / (den_y + self.epsilon)
                        pi_ny = num_ny / (den_ny + self.epsilon)
                        delta_list.append(pi_y - pi_ny)

                    if len(delta_list) < self.Emin:
                        r_part[p, ycls, u] = 0.0
                        s_part[p, ycls, u] = +1
                        continue

                    delta_arr = np.array(delta_list, dtype=np.float32)
                    bar_delta = float(delta_arr.mean())
                    sgn = +1 if bar_delta >= 0 else -1
                    s_part[p, ycls, u] = sgn

                    agree = np.mean((np.sign(delta_arr + 1e-12) == sgn).astype(np.float32))
                    dmin = float(np.min(np.abs(delta_arr)))
                    r_part[p, ycls, u] = float(agree * min(1.0, dmin / self.tau_delta))

        s_cons = np.ones((C, U), dtype=np.int32)
        r_cons = np.zeros((C, U), dtype=np.float32)

        for ycls in range(C):
            for u in range(U):
                p_support = [p for p in range(P) if r_part[p, ycls, u] > 0]
                if len(p_support) < self.Pmin:
                    s_cons[ycls, u] = +1
                    r_cons[ycls, u] = 0.0
                    continue

                dirs = [s_part[p, ycls, u] for p in p_support]
                cnt_pos = sum(1 for d in dirs if d == +1)
                cnt_neg = len(dirs) - cnt_pos
                s_mode = +1 if cnt_pos >= cnt_neg else -1

                b_agree = np.mean([(1.0 if s_part[p, ycls, u] == s_mode else 0.0) for p in p_support]).item()
                r_min = min(r_part[p, ycls, u] for p in p_support if s_part[p, ycls, u] == s_mode)
                r_val = float(b_agree * r_min)
                r_val = min(r_val, self.r_max)

                s_cons[ycls, u] = s_mode
                r_cons[ycls, u] = r_val

        return StabilityOutputs(s_dir=s_cons, r_rel=r_cons, u_list=[])


class GINClassifier(nn.Module):
    def __init__(self, num_node_features: int, num_classes: int, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.num_node_features = num_node_features
        self.num_classes = num_classes

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_dim = num_node_features
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden))
            in_dim = hidden

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        hg = global_add_pool(x, batch)
        return self.head(hg)


@dataclass
class PredicateInstance:
    uid: str
    predicate: Predicate
    pair: InterventionPair


def compute_intervention_loss(
    model: nn.Module,
    batch_items: List[Tuple[Data, int, List[PredicateInstance]]],
    stability: StabilityOutputs,
    uid_to_index: Dict[str, int],
    margin_m: float,
    device: str,
) -> torch.Tensor:
    if len(batch_items) == 0:
        return torch.tensor(0.0, device=device)

    all_graphs: List[Data] = []
    pos_map: List[List[Tuple[int, int, float]]] = []
    uid_map: List[List[str]] = []

    for _, (Gi, _, phis) in enumerate(batch_items):
        per_item = []
        per_uid = []

        for inst in phis:
            g1_data = nx_to_data_with_same_x(Gi, inst.pair.g1)
            g0_data = nx_to_data_with_same_x(Gi, inst.pair.g0)

            pos_g1 = len(all_graphs)
            all_graphs.append(g1_data)
            pos_g0 = len(all_graphs)
            all_graphs.append(g0_data)

            per_item.append((pos_g1, pos_g0, inst.pair.weight))
            per_uid.append(inst.uid)

        pos_map.append(per_item)
        uid_map.append(per_uid)

    if len(all_graphs) == 0:
        return torch.tensor(0.0, device=device)

    loader = DataLoader(all_graphs, batch_size=256, shuffle=False)
    logits_list = []
    for b in loader:
        b = b.to(device)
        logits_list.append(model(b))
    logits_all = torch.cat(logits_list, dim=0)
    probs_all = F.softmax(logits_all, dim=-1)

    total = 0.0
    count_items = 0

    for item_idx, (_, yi, phis) in enumerate(batch_items):
        if len(phis) == 0:
            continue

        yi_int = int(yi)
        item_loss = 0.0

        for k, inst in enumerate(phis):
            uid = inst.uid
            uidx = uid_to_index[uid]
            s_yu = int(stability.s_dir[yi_int, uidx])
            r_yu = float(stability.r_rel[yi_int, uidx])
            r_yu = max(0.0, min(1.0, r_yu))

            pos_g1, pos_g0, w = pos_map[item_idx][k]
            w_t = torch.tensor(w, device=device, dtype=torch.float32)

            if s_yu == +1:
                pos_up, pos_down = pos_g1, pos_g0
            else:
                pos_up, pos_down = pos_g0, pos_g1

            delta = logits_all[pos_up, yi_int] - logits_all[pos_down, yi_int]
            hinge = F.relu(torch.tensor(margin_m, device=device) - delta)

            js = js_divergence(
                probs_all[pos_g1:pos_g1 + 1, :],
                probs_all[pos_g0:pos_g0 + 1, :],
            )[0]

            item_loss = item_loss + w_t * (r_yu * hinge + (1.0 - r_yu) * js)

        item_loss = item_loss / float(len(phis))
        total = total + item_loss
        count_items += 1

    if count_items == 0:
        return torch.tensor(0.0, device=device)
    return total / float(count_items)


@dataclass
class MPTConfig:
    B: float = 3.0
    alpha_e: float = 1.0
    gamma: float = 1.0
    eta: float = 1.0
    eps_gap: float = 0.0
    max_expansions: int = 1500
    K_per_graph: int = 4

    support_min: int = 5
    pseudo_env_E: int = 4
    pseudo_env_P: int = 3
    crossfit_folds: int = 3
    warm_epochs: int = 8
    Emin: int = 2
    Pmin: int = 2
    tau_delta: float = 0.05
    r_max: float = 1.0

    margin_m: float = 0.2
    lam: float = 1.0

    lr: float = 1e-3
    wd: float = 1e-4
    epochs: int = 50
    batch_size: int = 64


def preprocess_mpt(
    train_dataset: List[Data],
    predicate_library: PredicateLibrary,
    config: MPTConfig,
    device: str,
    env_labels: Optional[np.ndarray] = None,
    seed: int = 0,
) -> Tuple[
    List[List[PredicateInstance]],
    StabilityOutputs,
    Dict[str, int],
]:
    set_seed(seed)

    graphs_nx = [data_to_nx_undirected(g) for g in train_dataset]
    y = np.array([int(g.y.item()) for g in train_dataset], dtype=np.int32)
    num_classes = int(y.max()) + 1

    Phi_tilde: List[Dict[str, Predicate]] = []
    support: Dict[str, int] = {}
    uid_to_predicate: Dict[str, Predicate] = {}

    for g in graphs_nx:
        preds = predicate_library.propose(g)
        per_graph: Dict[str, Predicate] = {}
        for p in preds:
            uid = p.canon_id()
            if uid not in per_graph:
                per_graph[uid] = p
                uid_to_predicate.setdefault(uid, p)
        Phi_tilde.append(per_graph)

    for per_graph in Phi_tilde:
        for uid in per_graph.keys():
            support[uid] = support.get(uid, 0) + 1

    u_keep = [uid for uid, s in support.items() if s >= config.support_min]
    u_keep.sort()
    uid_to_index = {uid: j for j, uid in enumerate(u_keep)}

    stab = StabilityEstimator(
        Emin=config.Emin,
        Pmin=config.Pmin,
        tau_delta=config.tau_delta,
        r_max=config.r_max,
        support_min=config.support_min,
        pseudo_env_E=config.pseudo_env_E,
        pseudo_env_P=config.pseudo_env_P,
        crossfit_folds=config.crossfit_folds,
        warm_epochs=config.warm_epochs,
        device=device,
        seed=seed,
    )
    m_exec, val_exec = stab.build_exec_table(
        graphs_nx=graphs_nx,
        y=y,
        u_keep=u_keep,
        u_to_predicate=uid_to_predicate,
    )

    if env_labels is not None:
        partitions = [env_labels.astype(np.int32)]
    else:
        partitions = stab.infer_pseudo_envs(dataset=train_dataset, y=y, num_classes=num_classes)

    stability_out = stab.compute_stability(
        y=y,
        m_exec=m_exec,
        val_exec=val_exec,
        partitions=partitions,
        num_classes=num_classes,
        support_threshold=10,
    )
    stability_out.u_list = u_keep

    validator = DomainValidator()
    edit_space = GraphEditSpace(cost=EditCost(alpha_e=config.alpha_e))
    generator = InterventionGenerator(
        validator=validator,
        edit_space=edit_space,
        B=config.B,
        gamma=config.gamma,
        eta=config.eta,
        eps_gap=config.eps_gap,
        max_expansions=config.max_expansions,
    )

    Phi_instances: List[List[PredicateInstance]] = []

    for i, g in enumerate(tqdm(graphs_nx, desc="Generating toggle pairs (Module I)")):
        candidates = [(uid, pred) for uid, pred in Phi_tilde[i].items() if uid in uid_to_index]
        random.shuffle(candidates)
        candidates = candidates[: config.K_per_graph]

        insts: List[PredicateInstance] = []
        for uid, pred in candidates:
            if not pred.applicable(g):
                continue
            pair = generator.generate_pair(g, pred)
            if pair is None:
                continue
            insts.append(PredicateInstance(uid=uid, predicate=pred, pair=pair))

        Phi_instances.append(insts)

    return Phi_instances, stability_out, uid_to_index


def train_mpt(
    train_dataset: List[Data],
    Phi_instances: List[List[PredicateInstance]],
    stability: StabilityOutputs,
    uid_to_index: Dict[str, int],
    config: MPTConfig,
    device: str,
    seed: int = 0,
) -> nn.Module:
    set_seed(seed)

    if train_dataset[0].x is None:
        num_node_features = 1
    else:
        num_node_features = int(train_dataset[0].x.size(-1))
    num_classes = int(max(int(d.y.item()) for d in train_dataset)) + 1

    model = GINClassifier(num_node_features=num_node_features, num_classes=num_classes)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)

    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model.train()
    for epoch in range(1, config.epochs + 1):
        total_loss = 0.0
        total_ce = 0.0
        total_int = 0.0
        steps = 0

        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            ce = F.cross_entropy(logits, batch.y.view(-1))

            data_list = batch.to_data_list()
            batch_items = []
            for gi in data_list:
                i = int(gi.idx.item())
                yi = int(gi.y.item())
                phis = Phi_instances[i]
                batch_items.append((gi, yi, phis))

            int_loss = compute_intervention_loss(
                model=model,
                batch_items=batch_items,
                stability=stability,
                uid_to_index=uid_to_index,
                margin_m=config.margin_m,
                device=device,
            )

            loss = ce + config.lam * int_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            total_ce += float(ce.item())
            total_int += float(int_loss.item())
            steps += 1

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={total_loss/steps:.4f} ce={total_ce/steps:.4f} int={total_int/steps:.4f}"
        )

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MUTAG", help="Any TUDataset name, e.g., MUTAG / NCI1 / PROTEINS / IMDB-BINARY ...")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full = TUDataset(root=args.data_root, name=args.dataset)
    full = list(full)
    ensure_node_features(full)

    train_set, test_set = stratified_split(full, test_ratio=args.test_ratio, seed=args.seed)

    train_set = attach_indices(train_set)
    test_set = attach_indices(test_set)

    pred_lib = PredicateLibrary(
        degree_thresholds=(2, 3, 4),
        edge_thresholds=(10, 15, 20),
        include_triangle=True
    )

    cfg = MPTConfig(
        B=2.0,
        alpha_e=1.0,
        gamma=1.0,
        eta=0.7,
        eps_gap=0.0,
        max_expansions=1200,
        K_per_graph=4,

        support_min=5,
        pseudo_env_E=4,
        pseudo_env_P=3,
        crossfit_folds=3,
        warm_epochs=6,
        Emin=2,
        Pmin=2,
        tau_delta=0.05,
        r_max=1.0,

        margin_m=0.2,
        lam=1.0,

        lr=1e-3,
        wd=1e-4,
        epochs=30,
        batch_size=64
    )

    Phi_instances, stability, uid_to_index = preprocess_mpt(
        train_dataset=train_set,
        predicate_library=pred_lib,
        config=cfg,
        device=device,
        env_labels=None,
        seed=args.seed,
    )

    model = train_mpt(
        train_dataset=train_set,
        Phi_instances=Phi_instances,
        stability=stability,
        uid_to_index=uid_to_index,
        config=cfg,
        device=device,
        seed=args.seed,
    )

    train_loader = DataLoader(train_set, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    train_acc = evaluate_accuracy(model, train_loader, device)
    test_acc = evaluate_accuracy(model, test_loader, device)

    print(f"\nDataset: {args.dataset}")
    print(f"Train size: {len(train_set)} | Test size: {len(test_set)}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy : {test_acc:.4f}")


if __name__ == "__main__":
    main()
