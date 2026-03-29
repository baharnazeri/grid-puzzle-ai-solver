# run_ai.py (diagnostic + safe A* run)
import time
import faulthandler
faulthandler.enable()

print("[BOOT] run_ai.py started", flush=True)

from environment import ShoverWorldEnv, BOX
print("[BOOT] imported environment", flush=True)

from player_ai import PlayerAI
print("[BOOT] imported player_ai", flush=True)


# --------------------------------------------------
# QUICK UNSOLVABLE CHECK
# --------------------------------------------------
def quick_unsolvable_check(env):
    grid = env.grid
    n_rows, n_cols = grid.shape

    def is_block(v):
        v = int(v)
        # هر چیزی غیر از EMPTY / BOX / LAVA مانع حساب می‌شود
        return v not in (0, 10, -100)

    stuck_boxes = []
    for r in range(n_rows):
        for c in range(n_cols):
            if int(grid[r, c]) != 10:  # BOX
                continue

            blocked = 0
            for dr, dc in [(-1,0),(0,1),(1,0),(0,-1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                    continue
                if is_block(grid[nr, nc]):
                    blocked += 1

            if blocked == 4:
                stuck_boxes.append((r, c))

    if stuck_boxes:
        print("[CHECK] UNSOLVABLE MAP: stuck boxes at", stuck_boxes, flush=True)
        return True

    print("[CHECK] Map passed quick solvability check.", flush=True)
    return False


def main():
    print("[MAIN] building env...", flush=True)
    env = ShoverWorldEnv(
        n_rows=9, n_cols=13,
        initial_stamina=1000.0,
        initial_force=40.0,
        unit_force=10.0,
        seed=42
    )
    print("[MAIN] env built", flush=True)

    env.reset()
    print("[MAIN] env reset ok", flush=True)

    # ----------- load map safely -----------
    map_path = "maps/challenge_03.txt"
    loaded = False
    try:
        print(f"[MAIN] loading map: {map_path}", flush=True)
        env.load_challenge_txt(map_path)
        loaded = True
        print("[MAIN] map loaded", flush=True)
    except Exception as e:
        print("[MAIN] map load failed:", e, flush=True)

    if not loaded:
        print("[MAIN] STOP: map not loaded. Fix the map file/format and try again.", flush=True)
        return

    # render map
    try:
        print("[MAIN] rendering loaded map:", flush=True)
        env.render()
    except Exception as e:
        print("[MAIN] render failed:", e, flush=True)

    print("[MAIN] initial boxes:", int((env.grid == BOX).sum()), flush=True)

    # ----------- QUICK CHECK -----------
    if quick_unsolvable_check(env):
        print("[MAIN] STOP: Map is structurally unsolvable.", flush=True)
        return

    # ----------- A* / Weighted A* -----------
    ai = PlayerAI(
        max_expansions=60000,
        w=2.5,
        top_k_actions=80,
        stamina_q=25,
        max_seconds=120,
        log_every=500,
        max_open=40000
    )

    t0 = time.time()
    solved = ai.solve(env, verbose=True)
    t1 = time.time()

    print("[RESULT] solved:", solved, flush=True)
    print("[RESULT] time:", (t1 - t0), "sec", flush=True)
    print(
        "[RESULT] final boxes:", int((env.grid == BOX).sum()),
        "stamina:", env.stamina,
        "score:", env.score,
        flush=True
    )


# --------------------------------------------------
# IMPORTANT: این خط حتما باید همین باشد
# --------------------------------------------------
if __name__ == "__main__":
    print("[BOOT] entering main", flush=True)
    main()