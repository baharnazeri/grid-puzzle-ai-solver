import heapq
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from environment import BOX

def boxes_count(grid: np.ndarray) -> int:
    return int((grid == BOX).sum())

def stamina_bin(stamina: float, q: int = 25) -> int:
    return int(float(stamina) // q)

def state_key(obs: Dict[str, Any], stamina_q: int = 25) -> Tuple[bytes, int]:
    return (obs["grid"].tobytes(), stamina_bin(float(obs["stamina"]), stamina_q))

def heuristic(obs: Dict[str, Any]) -> float:
    grid = obs["grid"]
    n_rows, n_cols = grid.shape
    rs, cs = np.where(grid == BOX)
    if len(rs) == 0:
        return 0.0
    d_top = rs
    d_bottom = (n_rows - 1) - rs
    d_left = cs
    d_right = (n_cols - 1) - cs
    d = np.minimum(np.minimum(d_top, d_bottom), np.minimum(d_left, d_right))
    return float(np.sum(d))

def candidate_actions(obs: Dict[str, Any], top_k: int = 40) -> List[Tuple[Tuple[int, int], int]]:
    grid = obs["grid"]
    n_rows, n_cols = grid.shape
    coords = []
    for r in range(n_rows):
        for c in range(n_cols):
            if grid[r, c] == BOX:
                d = min(r, n_rows - 1 - r, c, n_cols - 1 - c)
                coords.append((d, r, c))
    if not coords:
        return []
    coords.sort(key=lambda x: x[0])
    coords = coords[: min(top_k, len(coords))]

    dir_map = {(-1, 0): 1, (0, 1): 2, (1, 0): 3, (0, -1): 4}
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    acts: List[Tuple[Tuple[int, int], int]] = []
    for _, r, c in coords:
        best_d = None
        best_dist = 10**9
        for dr, dc in dirs:
            if dr == -1:
                dist = r
            elif dr == 1:
                dist = (n_rows - 1 - r)
            elif dc == -1:
                dist = c
            else:
                dist = (n_cols - 1 - c)
            if dist < best_dist:
                best_dist = dist
                best_d = (dr, dc)
        prefer_act = dir_map[best_d]
        acts.append(((r, c), prefer_act))
        for a in (1, 2, 3, 4):
            if a != prefer_act:
                acts.append(((r, c), a))
    return acts

def step_clone(env, action):
    if hasattr(env, "fast_clone"):
        env2 = env.fast_clone()
    else:
        import copy
        env2 = copy.deepcopy(env)
    obs2, rew2, done2, info2 = env2.step(action)
    return env2, obs2, rew2, done2, info2


class PlayerAI:
    def __init__(
        self,
        max_expansions: int = 60000,
        w: float = 2.5,
        stamina_q: int = 25,
        top_k_actions: int = 80,
        max_seconds: float = 120.0,
        log_every: int = 500,
        max_open: int = 40000,
    ):
        self.max_expansions = int(max_expansions)
        self.w = float(w)
        self.stamina_q = int(stamina_q)
        self.top_k_actions = int(top_k_actions)
        self.max_seconds = float(max_seconds)
        self.log_every = int(log_every)
        self.max_open = int(max_open)

    def solve(self, env, verbose: bool = True) -> bool:
        t_start = time.time()

        start_obs = env._get_obs()
        if boxes_count(start_obs["grid"]) == 0:
            return True

        start_k = state_key(start_obs, self.stamina_q)

        best_g: Dict[Tuple[bytes, int], float] = {start_k: 0.0}
        parent: Dict[Tuple[bytes, int], Tuple[Optional[Tuple[bytes, int]], Optional[Any]]] = {start_k: (None, None)}

        env_store: Dict[Tuple[bytes, int], Any]
        if hasattr(env, "fast_clone"):
            env_store = {start_k: env.fast_clone()}
        else:
            env_store = {start_k: env}

        pq: List[Tuple[float, float, int, Tuple[bytes, int]]] = []
        tie = 0
        h0 = heuristic(start_obs)
        heapq.heappush(pq, (0.0 + self.w * h0, 0.0, tie, start_k))

        expansions = 0
        goal_k: Optional[Tuple[bytes, int]] = None

        while pq and expansions < self.max_expansions:
            if time.time() - t_start > self.max_seconds:
                if verbose:
                    print("[A*] timeout reached, stopping search.", flush=True)
                return False

            f, g, _, k = heapq.heappop(pq)

            if g != best_g.get(k, None):
                continue

            cur_env = env_store.get(k)
            if cur_env is None:
                continue

            cur_obs = cur_env._get_obs()
            b = boxes_count(cur_obs["grid"])
            if b == 0:
                goal_k = k
                break

            expansions += 1
            if verbose and self.log_every > 0 and expansions % self.log_every == 0:
                print(f"[A*] expansions={expansions} open={len(pq)} boxes={b} stamina={float(cur_obs['stamina']):.1f}", flush=True)

            actions = candidate_actions(cur_obs, top_k=self.top_k_actions)
            if not actions:
                continue

            for action in actions:
                env2, obs2, rew2, done2, info2 = step_clone(cur_env, action)

                if not info2.get("last_action_valid?", False):
                    continue
                if info2.get("lost", False):
                    continue
                if float(obs2.get("stamina", 0.0)) < 0.0:
                    continue

                k2 = state_key(obs2, self.stamina_q)
                g2 = g + 1.0
                if g2 >= best_g.get(k2, 1e18):
                    continue

                best_g[k2] = g2
                parent[k2] = (k, action)
                env_store[k2] = env2

                h2 = heuristic(obs2)
                f2 = g2 + self.w * h2
                tie += 1
                heapq.heappush(pq, (f2, g2, tie, k2))

            # ---- مهم: وقتی open بزرگ شد prune کن (به جای pass کردن حالت‌های جدید) ----
            if len(pq) > self.max_open:
                pq.sort(key=lambda x: x[0])  # sort by f
                pq = pq[: self.max_open]
                heapq.heapify(pq)

        if goal_k is None:
            if verbose:
                print("[A*] failed: no solution within limits", flush=True)
            return False

        plan: List[Any] = []
        k = goal_k
        while True:
            pk, act = parent[k]
            if pk is None:
                break
            plan.append(act)
            k = pk
        plan.reverse()

        if verbose:
            print(f"[A*] solved. plan_len={len(plan)} expansions={expansions}", flush=True)

        for a in plan:
            obs, rew, done, info = env.step(a)
            if info.get("lost", False):
                if verbose:
                    print("[A*] execution lost unexpectedly", flush=True)
                return False
            if boxes_count(obs["grid"]) == 0 or info.get("won", False):
                return True

        return boxes_count(env.grid) == 0
