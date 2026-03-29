def encode_action(r: int, c: int, act: int, n_rows: int, n_cols: int, num_actions_per_cell: int = 6) -> int:
    """
    Encode ((r,c),act) -> single int for action_space compatibility.
    act in 1..6
    """
    cell_idx = int(r) * int(n_cols) + int(c)
    return int(cell_idx) * int(num_actions_per_cell) + (int(act) - 1)


def decode_action(idx: int, n_rows: int, n_cols: int, num_actions_per_cell: int = 6):
    cell_idx = int(idx) // int(num_actions_per_cell)
    act = (int(idx) % int(num_actions_per_cell)) + 1
    r = cell_idx // int(n_cols)
    c = cell_idx % int(n_cols)
    return (r, c), act
