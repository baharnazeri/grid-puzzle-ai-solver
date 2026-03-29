from typing import Optional, Tuple, List, Dict, Any
import random
import numpy as np

try:
    import gym
    from gym import spaces
except Exception:
    import gymnasium as gym
    from gymnasium import spaces


LAVA = -100
EMPTY = 0
BOX = 10
BARRIER = 100
OBSTACLE = 200

DIRS = {1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1)}


class ShoverWorldEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        n_rows: int = 9,
        n_cols: int = 13,
        max_timestep: int = 400,
        initial_stamina: float = 1000.0,
        initial_force: float = 40.0,
        unit_force: float = 10.0,
        perf_sq_initial_age: int = 10,
        seed: Optional[int] = None,
        max_levels: int = 20,
        auto_advance_levels: bool = True,  # مهم: برای حالت چالش خاموش می‌کنیم
    ):
        super().__init__()

        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.max_timestep = int(max_timestep)

        self.initial_stamina = float(initial_stamina)
        self.initial_force = float(initial_force)
        self.unit_force = float(unit_force)

        self.perf_sq_initial_age = int(perf_sq_initial_age)
        self.max_levels = int(max_levels)

        self.auto_advance_levels = bool(auto_advance_levels)

        self._seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # state
        self.level = 1
        self.grid = np.zeros((self.n_rows, self.n_cols), dtype=np.int32)
        self.stamina = float(self.initial_stamina)
        self.timestep = 0
        self.number_destroyed = 0
        self.score = 0.0

        self.last_action_valid = True
        self.previous_selected_position = (-1, -1)
        self.previous_action = 0

        self.stationary_flags: Dict[Tuple[int, int], Dict[int, bool]] = {}
        self.perf_squares: List[Dict[str, Any]] = []
        self.obstacles: List[Tuple[int, int]] = []

        self.num_actions_per_cell = 6
        self._update_spaces()

        self.last_won_info: Optional[Dict[str, Any]] = None
        self.win_flash_timer = 0
        self.win_flash_duration = 60

        self.last_lost_info: Optional[Dict[str, Any]] = None
        self.lost_flash_timer = 0
        self.lost_flash_duration = 90

        # default random init
        self._random_fill_for_level(self.level)
        self._apply_border_lava()
        self._rebuild_stationary_flags()
        self._scan_perf_squares()

    # ---------------- Spaces ----------------
    def _update_spaces(self):
        self.action_space = spaces.Discrete(self.n_rows * self.n_cols * self.num_actions_per_cell)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=-100000, high=100000, shape=(self.n_rows, self.n_cols), dtype=np.int32),
            "stamina": spaces.Box(low=-1e9, high=1e9, shape=(), dtype=np.float32),
            "previous_selected_position": spaces.Box(low=-1, high=max(self.n_rows, self.n_cols), shape=(2,), dtype=np.int32),
            "previous_action": spaces.Box(low=0, high=10, shape=(), dtype=np.int32),
            "level": spaces.Box(low=0, high=999, shape=(), dtype=np.int32),
            "score": spaces.Box(low=-1e9, high=1e9, shape=(), dtype=np.float32),
        })

    # ---------------- Helpers ----------------
    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def _is_box(self, v: int) -> bool:
        return int(v) == BOX

    def _apply_border_lava(self):
        if self.n_rows <= 0 or self.n_cols <= 0:
            return
        self.grid[0, :] = LAVA
        self.grid[self.n_rows - 1, :] = LAVA
        self.grid[:, 0] = LAVA
        self.grid[:, self.n_cols - 1] = LAVA

    # ---------------- Random Fill ----------------
    def _random_fill_with_count(self, count: int):
        self.grid = np.zeros((self.n_rows, self.n_cols), dtype=np.int32)
        capacity = max(0, (self.n_rows - 2) * (self.n_cols - 2))
        count = max(0, min(count, capacity))
        positions = [(r, c) for r in range(1, self.n_rows - 1) for c in range(1, self.n_cols - 1)]
        random.shuffle(positions)
        for i in range(count):
            r, c = positions[i]
            self.grid[r, c] = BOX
        self._apply_border_lava()

    def _random_fill_for_level(self, level: int):
        base = max(1, (self.n_rows * self.n_cols) // 12)
        increment = 2
        count = base + (level - 1) * increment
        cap = max(0, (self.n_rows - 2) * (self.n_cols - 2))
        count = min(count, cap)
        self._random_fill_with_count(count)

        self.obstacles = []
        if level >= 3:
            self._place_obstacles_for_level(level)

    def _place_obstacles_for_level(self, level: int):
        base_obs = 4
        extra = max(0, (level - 3) // 2)
        obs_count = base_obs + extra
        positions = [(r, c) for r in range(1, self.n_rows - 1) for c in range(1, self.n_cols - 1)]
        random.shuffle(positions)
        placed = 0
        for (r, c) in positions:
            if placed >= obs_count:
                break
            if self.grid[r, c] == EMPTY:
                self.grid[r, c] = OBSTACLE
                self.obstacles.append((r, c))
                placed += 1

    # ---------------- Perfect Squares ----------------
    def _rebuild_stationary_flags(self):
        self.stationary_flags.clear()
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self._is_box(self.grid[r, c]):
                    self.stationary_flags[(r, c)] = {d: True for d in DIRS.keys()}

    def _detect_all_perf_squares(self) -> List[Tuple[int, Tuple[int, int]]]:
        res = []
        max_n = min(self.n_rows, self.n_cols)
        for n in range(2, max_n + 1):
            for r in range(0, self.n_rows - n + 1):
                for c in range(0, self.n_cols - n + 1):
                    block = self.grid[r:r + n, c:c + n]
                    if np.all(block == BOX):
                        # ensure it is "maximal" (not attached to other boxes around)
                        ok = True
                        for rr in range(r - 1, r + n + 1):
                            for cc in range(c - 1, c + n + 1):
                                if r <= rr < r + n and c <= cc < c + n:
                                    continue
                                if not self._in_bounds(rr, cc):
                                    continue
                                if self._is_box(self.grid[rr, cc]):
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            res.append((n, (r, c)))
        return res

    def _scan_perf_squares(self):
        found = self._detect_all_perf_squares()
        existing = {(ps["n"], tuple(ps["top_left"])): ps for ps in self.perf_squares}
        new_list = []
        for n, top_left in found:
            key = (n, tuple(top_left))
            if key in existing:
                new_list.append(existing[key])
            else:
                new_list.append({"n": n, "top_left": top_left, "created": self.timestep, "last_changed": self.timestep})
        self.perf_squares = new_list

    def _age_and_dissolve_perf_squares(self):
        remaining = []
        for ps in self.perf_squares:
            created = ps["created"]
            last_changed = ps.get("last_changed", created)
            if (self.timestep - created) >= self.perf_sq_initial_age and last_changed == created:
                r, c = ps["top_left"]
                n = ps["n"]
                self.grid[r:r + n, c:c + n] = EMPTY
            else:
                remaining.append(ps)
        self.perf_squares = remaining

    def _mark_perf_squares_changed_by_cells(self, changed_cells: List[Tuple[int, int]]):
        if not changed_cells:
            return
        for ps in self.perf_squares:
            r0, c0 = ps["top_left"]
            n = ps["n"]
            for (cr, cc) in changed_cells:
                if r0 <= cr < r0 + n and c0 <= cc < c0 + n:
                    ps["last_changed"] = self.timestep
                    break

    # ---------------- Reset / Obs ----------------
    def reset(self, seed: Optional[int] = None, return_info: bool = False):
        if seed is not None:
            self._seed = seed
            random.seed(seed)
            np.random.seed(seed)

        self.timestep = 0
        self.stamina = float(self.initial_stamina)
        self.number_destroyed = 0
        self.score = 0.0

        self.last_action_valid = True
        self.previous_selected_position = (-1, -1)
        self.previous_action = 0

        self.last_won_info = None
        self.win_flash_timer = 0
        self.last_lost_info = None
        self.lost_flash_timer = 0

        self._rebuild_stationary_flags()
        self._scan_perf_squares()

        obs = self._get_obs()
        if return_info:
            return obs, {}
        return obs

    def _get_obs(self):
        return {
            "grid": self.grid.copy(),
            "stamina": np.float32(self.stamina),
            "previous_selected_position": np.array(self.previous_selected_position, dtype=np.int32),
            "previous_action": np.int32(self.previous_action),
            "level": np.int32(self.level),
            "score": np.float32(self.score),
        }

    # ---------------- Challenge Loader ----------------
    def load_challenge_txt(self, path: str):
        """
        File format: 9 lines of 13 chars (for 13x9).
        Symbols:
          '.' -> EMPTY
          'B' -> BOX
          '#' -> OBSTACLE
          'X' -> OBSTACLE (optional)
          'L' -> LAVA (optional inside map; border lava is applied anyway)
        """
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f.readlines() if ln.strip("\n") != ""]

        if len(lines) != self.n_rows:
            raise ValueError(f"Map must have exactly {self.n_rows} rows, got {len(lines)}")

        for i, ln in enumerate(lines):
            if len(ln) != self.n_cols:
                raise ValueError(f"Row {i} must have exactly {self.n_cols} cols, got {len(ln)}")

        g = np.zeros((self.n_rows, self.n_cols), dtype=np.int32)
        obs_list = []
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                ch = lines[r][c]
                if ch == ".":
                    g[r, c] = EMPTY
                elif ch == "B":
                    g[r, c] = BOX
                elif ch == "#" or ch == "X":
                    g[r, c] = OBSTACLE
                    obs_list.append((r, c))
                elif ch in ("L","~"):
                    g[r, c] = LAVA
                else:
                    raise ValueError(f"Unknown char '{ch}' at (r={r}, c={c})")

        self.grid = g
        self.obstacles = obs_list

        # apply border lava regardless
        self._apply_border_lava()

        # reset counters but keep parameters
        self.level = 1
        self.timestep = 0
        self.number_destroyed = 0
        self.score = 0.0
        self.stamina = float(self.initial_stamina)

        self.last_action_valid = True
        self.previous_selected_position = (-1, -1)
        self.previous_action = 0

        self._rebuild_stationary_flags()
        self.perf_squares = []
        self._scan_perf_squares()

        # important for challenge: do NOT auto-advance after win
        self.auto_advance_levels = False
        self.max_levels = 1

    # ---------------- Level Advance ----------------
    def _advance_level(self):
        prev = self.level
        if self.level < self.max_levels:
            self.level += 1
        else:
            self.level = 1

        self._random_fill_for_level(self.level)
        self._apply_border_lava()

        self.stamina = float(self.initial_stamina)
        self.timestep = 0
        self.number_destroyed = 0
        self.score = 0.0

        self.last_action_valid = True
        self.previous_selected_position = (-1, -1)
        self.previous_action = 0

        self._rebuild_stationary_flags()
        self._scan_perf_squares()
        return prev, self.level

    # ---------------- fast_clone ----------------
    def fast_clone(self):
        """
        Lightweight clone for search: much faster than deepcopy.
        Copies fields needed by step().
        """
        new = ShoverWorldEnv(
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            max_timestep=self.max_timestep,
            initial_stamina=self.initial_stamina,
            initial_force=self.initial_force,
            unit_force=self.unit_force,
            perf_sq_initial_age=self.perf_sq_initial_age,
            seed=self._seed,
            max_levels=self.max_levels,
            auto_advance_levels=self.auto_advance_levels,
        )

        new.level = self.level
        new.grid = self.grid.copy()
        new.stamina = float(self.stamina)
        new.timestep = int(self.timestep)
        new.number_destroyed = int(self.number_destroyed)
        new.score = float(self.score)

        new.last_action_valid = bool(self.last_action_valid)
        new.previous_selected_position = tuple(self.previous_selected_position)
        new.previous_action = int(self.previous_action)

        new.obstacles = list(self.obstacles)
        new.perf_squares = [dict(ps) for ps in self.perf_squares]
        new.stationary_flags = {k: dict(v) for k, v in self.stationary_flags.items()}

        new.last_won_info = None
        new.win_flash_timer = 0
        new.last_lost_info = None
        new.lost_flash_timer = 0

        return new

    # ---------------- Step ----------------
    def step(self, action):
        """
        Accepts either encoded int or ((r,c), act)
        acts: 1..4 push, 5 Barrier Maker, 6 Hellify
        """
        self.timestep += 1
        self.last_action_valid = False

        self._scan_perf_squares()

        # decode action
        if isinstance(action, int):
            idx = int(action)
            cell_idx = idx // self.num_actions_per_cell
            act = (idx % self.num_actions_per_cell) + 1
            r = cell_idx // self.n_cols
            c = cell_idx % self.n_cols
        elif isinstance(action, (list, tuple)) and len(action) == 2:
            (r, c), act = action
            r = int(r)
            c = int(c)
            act = int(act)
        else:
            raise ValueError("Unsupported action format")

        self.previous_selected_position = (r, c)
        self.previous_action = int(act)

        reward = 0.0
        chain_length_k = 0
        initial_force_charged = False
        lava_destroyed_this_step = 0

        # push actions
        if act in (1, 2, 3, 4):
            dr, dc = DIRS[act]
            if not self._in_bounds(r, c) or not self._is_box(self.grid[r, c]):
                self.last_action_valid = False
            else:
                # build chain in direction
                chain = []
                rr, cc = r, c
                while self._in_bounds(rr, cc) and self._is_box(self.grid[rr, cc]):
                    chain.append((rr, cc))
                    rr += dr
                    cc += dc

                moving_set = list(chain)
                moving_set_set = set(moving_set)

                # closure: include any boxes that would get pushed due to collisions
                changed = True
                iter_guard = 0
                invalid_due_barrier = False
                while changed and iter_guard < (self.n_rows * self.n_cols + 5):
                    iter_guard += 1
                    changed = False
                    for (cr, cc) in list(moving_set):
                        nr, nc = cr + dr, cc + dc
                        if self._in_bounds(nr, nc):
                            val = int(self.grid[nr, nc])
                            if val in (BARRIER, OBSTACLE):
                                invalid_due_barrier = True
                                break
                            if val == BOX and (nr, nc) not in moving_set_set:
                                moving_set.append((nr, nc))
                                moving_set_set.add((nr, nc))
                                changed = True
                        else:
                            pass
                    if invalid_due_barrier:
                        break

                if invalid_due_barrier:
                    self.last_action_valid = False
                else:
                    orig_head = chain[-1]
                    head_stationary = self.stationary_flags.get(orig_head, {d: True for d in DIRS.keys()}).get(act, True)
                    chain_length_k = len(moving_set)
                    initial_force_charged = bool(head_stationary)

                    cost = (self.initial_force if head_stationary else 0.0) + (self.unit_force * chain_length_k)

                    new_grid = self.grid.copy()
                    for (cr, cc) in moving_set:
                        new_grid[cr, cc] = EMPTY

                    destroyed_this_step = 0
                    moved_positions = []
                    changed_cells = []

                    def key_proj(pos):
                        return pos[0] * dr + pos[1] * dc

                    ordered = sorted(moving_set, key=key_proj, reverse=True)

                    for (cr, cc) in ordered:
                        nr, nc = cr + dr, cc + dc
                        if not self._in_bounds(nr, nc):
                            destroyed_this_step += 1
                            changed_cells.append((cr, cc))
                            continue
                        if new_grid[nr, nc] == LAVA:
                            destroyed_this_step += 1
                            changed_cells.append((cr, cc))
                            continue
                        if new_grid[nr, nc] in (BARRIER, OBSTACLE):
                            destroyed_this_step += 1
                            changed_cells.append((cr, cc))
                            continue
                        new_grid[nr, nc] = BOX
                        moved_positions.append((nr, nc))
                        changed_cells.append((cr, cc))
                        changed_cells.append((nr, nc))

                    self.grid = new_grid
                    self._rebuild_stationary_flags()
                    for pos in moved_positions:
                        if pos in self.stationary_flags:
                            self.stationary_flags[pos][act] = False

                    self.stamina -= cost
                    if destroyed_this_step > 0:
                        refund = destroyed_this_step * self.initial_force
                        self.stamina += refund
                        self.number_destroyed += destroyed_this_step
                        lava_destroyed_this_step = destroyed_this_step
                        self.score += 5 * destroyed_this_step
                        reward += 5 * destroyed_this_step

                    self._mark_perf_squares_changed_by_cells(changed_cells)
                    self.last_action_valid = True

        # special actions
        elif act in (5, 6):
            avail = any(ps.get("n", 0) >= 2 for ps in self.perf_squares)
            if not avail:
                self.last_action_valid = False
            else:
                oldest = min(self.perf_squares, key=lambda x: x["created"])
                n = oldest["n"]
                sr, sc = oldest["top_left"]

                if act == 5:
                    # turn perfect square into barrier
                    for rr in range(sr, sr + n):
                        for cc in range(sc, sc + n):
                            self.grid[rr, cc] = BARRIER

                    self.stamina += float(10 * (n * n))
                    added = 40 * (n * n)
                    self.score += added
                    reward += added

                    self.perf_squares = [p for p in self.perf_squares if not (p["top_left"] == (sr, sc) and p["n"] == n)]
                    changed_cells = [(rr, cc) for rr in range(sr, sr + n) for cc in range(sc, sc + n)]
                    self._mark_perf_squares_changed_by_cells(changed_cells)
                    self._rebuild_stationary_flags()
                    self.last_action_valid = True

                else:
                    # hellify: turn into lava, count destroyed boxes
                    affected_boxes = 0
                    for rr in range(sr, sr + n):
                        for cc in range(sc, sc + n):
                            if self._is_box(self.grid[rr, cc]):
                                affected_boxes += 1

                    for rr in range(sr, sr + n):
                        for cc in range(sc, sc + n):
                            self.grid[rr, cc] = LAVA

                    self.number_destroyed += affected_boxes
                    self.score += 5 * affected_boxes
                    reward += 5 * affected_boxes

                    self.perf_squares = [p for p in self.perf_squares if not (p["top_left"] == (sr, sc) and p["n"] == n)]
                    changed_cells = [(rr, cc) for rr in range(sr, sr + n) for cc in range(sc, sc + n)]
                    self._mark_perf_squares_changed_by_cells(changed_cells)
                    self._rebuild_stationary_flags()
                    self.last_action_valid = True

        else:
            self.last_action_valid = False

        # age perfect squares
        self._age_and_dissolve_perf_squares()
        self._scan_perf_squares()

        # time cost
        self.stamina -= 1.0

        # goal / terminal
        boxes_left = int(np.sum(self.grid == BOX))
        info_extra = {}

        if boxes_left == 0:
            # win
            self.last_won_info = {"message": "you win :)"}
            self.win_flash_timer = self.win_flash_duration
            info_extra["won"] = True
            info_extra["won_message"] = "you win :)"

            # IMPORTANT: if auto_advance_levels is False, do NOT change the grid
            if self.auto_advance_levels:
                prev_level = self.level
                self._advance_level()
                info_extra["prev_level"] = prev_level
                info_extra["new_level"] = self.level

        if self.stamina < 0 and not info_extra.get("won", False):
            self.last_lost_info = {"message": "you lose :(", "reason": "stamina_negative"}
            self.lost_flash_timer = self.lost_flash_duration
            info_extra["lost"] = True
            info_extra["lost_message"] = "you lose :("

        done = False
        if self.timestep >= self.max_timestep:
            done = True
        if info_extra.get("lost", False):
            done = True
        if info_extra.get("won", False) and not self.auto_advance_levels:
            # for challenge mode, end episode on win
            done = True

        perf_list = [(ps["n"], tuple(ps["top_left"])) for ps in self.perf_squares]
        info = {
            "timestep": self.timestep,
            "stamina": float(self.stamina),
            "number_of_boxes": int(np.sum(self.grid == BOX)),
            "number_destroyed": int(self.number_destroyed),
            "last_action_valid?": bool(self.last_action_valid),
            "chain_length_k": int(chain_length_k),
            "initial_force_charged?": bool(initial_force_charged),
            "lava_destroyed_this_step": int(lava_destroyed_this_step),
            "perfect_squares_available": perf_list,
            "level": int(self.level),
            "score": float(self.score),
        }
        info.update(info_extra)

        obs = self._get_obs()
        return obs, float(reward), bool(done), info

    # ---------- rendering ----------
    def render(self, mode="human"):
        chars = {LAVA: "L", EMPTY: ".", BOX: "B", BARRIER: "#", OBSTACLE: "X"}
        print("=" * (self.n_cols))
        for r in range(self.n_rows):
            row = "".join(chars.get(int(self.grid[r, c]), "?") for c in range(self.n_cols))
            print(row)
        print(
            f"Level: {self.level}  t={self.timestep} stamina={self.stamina:.2f} "
            f"boxes={int(np.sum(self.grid == BOX))} score={self.score:.1f}"
        )

    def close(self):
        pass
