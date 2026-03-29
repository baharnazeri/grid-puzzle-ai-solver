import sys
import glob
from environment import ShoverWorldEnv
from gui import run_gui


def print_help():
    print("Usage:")
    print("  python run_gui.py                   -> random board")
    print("  python run_gui.py maps/challenge_03.txt -> load a map")
    print("  python run_gui.py --list            -> list maps")


def list_maps():
    maps = sorted(glob.glob("maps/*.txt"))
    if not maps:
        print("No maps found in maps/*.txt")
        return
    print("Available maps:")
    for m in maps:
        print(" -", m)


def main():
    args = sys.argv[1:]
    if args and args[0] in ("-h", "--help"):
        print_help()
        return
    if args and args[0] == "--list":
        list_maps()
        return

    env = ShoverWorldEnv(
        n_rows=9, n_cols=13,
        initial_stamina=1000.0,
        initial_force=40.0,
        unit_force=10.0,
        seed=42
    )
    env.reset()

    if args:
        map_path = args[0]
        print(f"[GUI] loading map: {map_path}")
        env.load_challenge_txt(map_path)
    else:
        print("[GUI] random mode")

    run_gui(env)


if __name__ == "__main__":
    main()
