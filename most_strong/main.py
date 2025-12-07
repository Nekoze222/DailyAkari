from collections import deque
from copy import deepcopy
import json
import sys

UNKNOWN, BULB, EMPTY = 0, 1, -1

def solve_akari_json(width, height, grid_tokens):
    if not grid_tokens:
        return None

    if isinstance(grid_tokens[0], list):
        grid2d = grid_tokens
        if len(grid2d) != height or any(len(row) != width for row in grid2d):
            raise ValueError("grid_tokens(2D) と width/height のサイズが一致していません")
        original_shape = "2d"
    else:
        if len(grid_tokens) != width * height:
            raise ValueError("grid_tokens の長さが width*height と一致していません")
        grid2d = [
            [grid_tokens[r * width + c] for c in range(width)]
            for r in range(height)
        ]
        original_shape = "1d"

    AkariSolverJSON.episode_counter = 0
    AkariSolverJSON.forced_move_counter = 0

    solver = AkariSolverJSON(width, height, grid2d)
    solved_2d = solver.solve()
    if solved_2d is None:
        return None

    if original_shape == "2d":
        return solved_2d
    else:
        flat = []
        for r in range(height):
            for c in range(width):
                flat.append(solved_2d[r][c])
        return flat


def to_2d(width, height, grid_tokens):
    if not grid_tokens:
        return []

    if isinstance(grid_tokens[0], list):
        return grid_tokens
    if len(grid_tokens) != width * height:
        raise ValueError("grid_tokens の長さが width*height と一致していません")
    return [
        [grid_tokens[r * width + c] for c in range(width)]
        for r in range(height)
    ]


def render_board(tokens_2d):
    h = len(tokens_2d)
    if h == 0:
        return ""
    w = len(tokens_2d[0])

    def cell_str(tok: str) -> str:
        if tok == ".":
            return "."
        if tok == "L":
            return "L"
        if tok == "x":
            return "x"  
        if tok == "?":
            return "?" 
        if tok == "#":
            return "■" 
        if tok.startswith("#") and len(tok) > 1 and tok[1].isdigit():
            return tok[1]  # "#2" → "2"
        return tok[:1]

    border = "+" + "+".join("---" for _ in range(w)) + "+"
    lines = [border]
    for r in range(h):
        cells = [cell_str(tok) for tok in tokens_2d[r]]
        line = "|" + "|".join(f" {c} " for c in cells) + "|"
        lines.append(line)
        lines.append(border)
    return "\n".join(lines)

class AkariSolverJSON:
    episode_counter = 0
    forced_move_counter = 0

    def __init__(self, width, height, tokens_2d):
        self.w = width
        self.h = height

        if len(tokens_2d) != height or any(len(row) != width for row in tokens_2d):
            raise ValueError("tokens_2d のサイズが width/height と一致していません")

        self.tokens = [[str(ch) for ch in row] for row in tokens_2d]

        self.white_index = [[-1] * width for _ in range(height)]
        self.white_pos = []
        self.num_whites = 0

        self.digit_value = {}

        for r in range(height):
            for c in range(width):
                t = self.tokens[r][c]
                if t == ".":
                    idx = self.num_whites
                    self.num_whites += 1
                    self.white_index[r][c] = idx
                    self.white_pos.append((r, c))
                else:
                    if t.startswith("#") and len(t) > 1 and t[1] in "01234":
                        self.digit_value[(r, c)] = int(t[1])

        self.state = [UNKNOWN] * self.num_whites

        self.hseg_of = [-1] * self.num_whites
        self.vseg_of = [-1] * self.num_whites
        self.hsegments = []
        self.vsegments = []
        self._build_segments()

        self.digits = []
        self.digit_for_white = [[] for _ in range(self.num_whites)]
        self._build_digits()

        self.vis = [[] for _ in range(self.num_whites)]
        self.lighted_by = [[] for _ in range(self.num_whites)]
        self._build_visibility()

        self.lit_count = [0] * self.num_whites
        self.cand_count = [len(self.vis[i]) for i in range(self.num_whites)]
        self.last_cand = [None] * self.num_whites

        for i in range(self.num_whites):
            if self.cand_count[i] == 0:
                raise ValueError(f"白マス {i} が誰からも照らされ得ない（初期状態から矛盾）")

        self.queue = deque()
        self.in_queue_digit = [False] * len(self.digits)
        self.in_queue_hseg = [False] * len(self.hsegments)
        self.in_queue_vseg = [False] * len(self.vsegments)
        self.in_queue_cover = [False] * self.num_whites

        for b_idx in range(len(self.digits)):
            self.enqueue_digit(b_idx)
        for s in range(len(self.hsegments)):
            self.enqueue_hseg(s)
        for s in range(len(self.vsegments)):
            self.enqueue_vseg(s)
        for p in range(self.num_whites):
            self.enqueue_cover(p)

    def debug_tokens_with_state(self, show_unknown=True):
        out = [[self.tokens[r][c] for c in range(self.w)] for r in range(self.h)]
        for i, (r, c) in enumerate(self.white_pos):
            st = self.state[i]
            if st == BULB:
                out[r][c] = "L"
            elif st == EMPTY:
                out[r][c] = "x"
            else:
                if show_unknown:
                    out[r][c] = "?" 
                else:
                    out[r][c] = "."
        return out

    def debug_print_board(self, title=None):
        if title:
            print(title)
        print(render_board(self.debug_tokens_with_state(show_unknown=True)))

    def _build_segments(self):
        for r in range(self.h):
            c = 0
            while c < self.w:
                if self.white_index[r][c] != -1:
                    seg = []
                    while c < self.w and self.white_index[r][c] != -1:
                        idx = self.white_index[r][c]
                        seg.append(idx)
                        c += 1
                    if seg:
                        sid = len(self.hsegments)
                        self.hsegments.append(seg)
                        for i in seg:
                            self.hseg_of[i] = sid
                else:
                    c += 1

        for c in range(self.w):
            r = 0
            while r < self.h:
                if self.white_index[r][c] != -1:
                    seg = []
                    while r < self.h and self.white_index[r][c] != -1:
                        idx = self.white_index[r][c]
                        seg.append(idx)
                        r += 1
                    if seg:
                        sid = len(self.vsegments)
                        self.vsegments.append(seg)
                        for i in seg:
                            self.vseg_of[i] = sid
                else:
                    r += 1

    def _build_digits(self):
        for (r, c), n in self.digit_value.items():
            cells = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.h and 0 <= cc < self.w:
                    idx = self.white_index[rr][cc]
                    if idx != -1:
                        cells.append(idx)
            d_idx = len(self.digits)
            self.digits.append({"cells": cells, "n": n})
            for i in cells:
                self.digit_for_white[i].append(d_idx)

    def _build_visibility(self):
        for i in range(self.num_whites):
            hseg = self.hsegments[self.hseg_of[i]]
            vseg = self.vsegments[self.vseg_of[i]]
            vs = set(hseg) | set(vseg)
            self.vis[i] = list(vs)

        for j in range(self.num_whites):
            for p in self.vis[j]:
                self.lighted_by[p].append(j)

    def enqueue_digit(self, b_idx):
        if not self.in_queue_digit[b_idx]:
            self.in_queue_digit[b_idx] = True
            self.queue.append(("digit", b_idx))

    def enqueue_hseg(self, s_idx):
        if not self.in_queue_hseg[s_idx]:
            self.in_queue_hseg[s_idx] = True
            self.queue.append(("hseg", s_idx))

    def enqueue_vseg(self, s_idx):
        if not self.in_queue_vseg[s_idx]:
            self.in_queue_vseg[s_idx] = True
            self.queue.append(("vseg", s_idx))

    def enqueue_cover(self, p_idx):
        if not self.in_queue_cover[p_idx]:
            self.in_queue_cover[p_idx] = True
            self.queue.append(("cover", p_idx))

    def set_state(self, i, new_state):
        old = self.state[i]
        if old == new_state:
            return True
        if old != UNKNOWN:
            return False

        self.state[i] = new_state

        self.enqueue_hseg(self.hseg_of[i])
        self.enqueue_vseg(self.vseg_of[i])

        for b_idx in self.digit_for_white[i]:
            self.enqueue_digit(b_idx)

        if new_state == BULB:
            for p in self.lighted_by[i]:
                self.lit_count[p] += 1
                self.enqueue_cover(p)
        elif new_state == EMPTY:
            for p in self.lighted_by[i]:
                self.cand_count[p] -= 1
                self.enqueue_cover(p)

        return True

    def process_digit(self, b_idx):
        d = self.digits[b_idx]
        cells = d["cells"]
        n = d["n"]
        placed = 0
        unknowns = []

        for i in cells:
            if self.state[i] == BULB:
                placed += 1
            elif self.state[i] == UNKNOWN:
                unknowns.append(i)

        need = n - placed
        if need < 0 or need > len(unknowns):
            return False

        if need == 0:
            for i in unknowns:
                if not self.set_state(i, EMPTY):
                    return False
        elif need == len(unknowns):
            for i in unknowns:
                if not self.set_state(i, BULB):
                    return False

        return True

    def process_segment(self, seg_idx, is_h):
        cells = self.hsegments[seg_idx] if is_h else self.vsegments[seg_idx]
        placed = 0
        unknowns = []

        for i in cells:
            if self.state[i] == BULB:
                placed += 1
            elif self.state[i] == UNKNOWN:
                unknowns.append(i)

        if placed >= 2:
            return False

        if placed == 1:
            for i in unknowns:
                if not self.set_state(i, EMPTY):
                    return False

        return True

    def process_cover(self, p):
        cand = self.cand_count[p]

        if cand == 1:
            last = None
            for j in self.vis[p]:
                if self.state[j] != EMPTY:
                    last = j
                    break
            self.last_cand[p] = last
        elif cand <= 0:
            self.last_cand[p] = None

        lit = self.lit_count[p]
        if lit == 0:
            if cand == 0:
                return False
            if cand == 1:
                j = self.last_cand[p]
                if j is None:
                    for k in self.vis[p]:
                        if self.state[k] != EMPTY:
                            j = k
                            break
                if j is None:
                    return False
                if not self.set_state(j, BULB):
                    return False

        return True

    def propagate(self):
        while self.queue:
            kind, idx = self.queue.popleft()
            if kind == "digit":
                self.in_queue_digit[idx] = False
                if not self.process_digit(idx):
                    return False
            elif kind == "hseg":
                self.in_queue_hseg[idx] = False
                if not self.process_segment(idx, True):
                    return False
            elif kind == "vseg":
                self.in_queue_vseg[idx] = False
                if not self.process_segment(idx, False):
                    return False
            elif kind == "cover":
                self.in_queue_cover[idx] = False
                if not self.process_cover(idx):
                    return False
        return True

    def is_complete(self):
        return all(s != UNKNOWN for s in self.state)

    def verify_solution(self):
        for i in range(self.num_whites):
            lit = 0
            for j in self.vis[i]:
                if self.state[j] == BULB:
                    lit += 1
            if lit == 0:
                return False

        for seg in self.hsegments:
            if sum(1 for i in seg if self.state[i] == BULB) > 1:
                return False
        for seg in self.vsegments:
            if sum(1 for i in seg if self.state[i] == BULB) > 1:
                return False

        for d in self.digits:
            n = d["n"]
            cells = d["cells"]
            if sum(1 for i in cells if self.state[i] == BULB) != n:
                return False

        return True

    def find_branch_var(self):
        for i, s in enumerate(self.state):
            if s == UNKNOWN:
                return i
        return None

    def _search_exists(self):
        if not self.propagate():
            return False

        if self.is_complete():
            return self.verify_solution()

        i = self.find_branch_var()
        if i is None:
            return False

        s1 = deepcopy(self)
        can_bulb = False
        if s1.set_state(i, BULB):
            can_bulb = s1._search_exists()
        if can_bulb:
            return True

        s2 = deepcopy(self)
        can_empty = False
        if s2.set_state(i, EMPTY):
            can_empty = s2._search_exists()
        if can_empty:
            return True

        return False

    def strengthen_all_forced_moves(self):
        if not self.propagate():
            return False

        changed = True
        while changed:
            changed = False

            for i, s in enumerate(self.state):
                if s != UNKNOWN:
                    continue

                pos = self.white_pos[i]
                AkariSolverJSON.episode_counter += 1
                ep_id = AkariSolverJSON.episode_counter

                print("\n" + "=" * 60)
                print(f"[Episode {ep_id}] Test cell index={i}, pos={pos}")
                self.debug_print_board("Current partial board:")

                print(f"\n  - Hypothesis A: place BULB at {pos}")
                s_bulb = deepcopy(self)
                can_bulb = False
                if s_bulb.set_state(i, BULB):
                    can_bulb = s_bulb._search_exists()
                print(f"    => feasible? {can_bulb}")

                print(f"\n  - Hypothesis B: forbid BULB at {pos} (EMPTY)")
                s_empty = deepcopy(self)
                can_empty = False
                if s_empty.set_state(i, EMPTY):
                    can_empty = s_empty._search_exists()
                print(f"    => feasible? {can_empty}")

                if not can_bulb and not can_empty:
                    print("  !! Both hypotheses infeasible: current state is inconsistent.")
                    return False

                if can_bulb and not can_empty:
                    print(f"\n  => Forced move: BULB at {pos} (EMPTY would kill all solutions)")
                    AkariSolverJSON.forced_move_counter += 1
                    if not self.set_state(i, BULB):
                        return False
                    self.debug_print_board("Board after forced BULB:")
                    changed = True

                elif can_empty and not can_bulb:
                    print(f"\n  => Forced move: EMPTY at {pos} (BULB would kill all solutions)")
                    AkariSolverJSON.forced_move_counter += 1
                    if not self.set_state(i, EMPTY):
                        return False
                    self.debug_print_board("Board after forced EMPTY:")
                    changed = True

            if changed:
                print("\n[Propagation] Run local constraint propagation after forced moves.")
                if not self.propagate():
                    print("  !! Propagation after forced moves found inconsistency.")
                    return False
                self.debug_print_board("Board after propagation:")

        print("\n[Info] No more forced moves deducible by contradiction.")
        return True


    def _search(self):
        if not self.propagate():
            return None

        if not self.strengthen_all_forced_moves():
            return None

        if self.is_complete():
            if self.verify_solution():
                return list(self.state)
            return None

        i = self.find_branch_var()
        if i is None:
            return None

        pos = self.white_pos[i]
        print("\n" + "=" * 60)
        print(f"[Search branching] Trying branch at cell index={i}, pos={pos}")
        self.debug_print_board("Board before branching:")

        print(f"\n  -> Branch 1: BULB at {pos}")
        s1 = deepcopy(self)
        if s1.set_state(i, BULB):
            res1 = s1._search()
            if res1 is not None:
                return res1

        print(f"\n  -> Branch 2: EMPTY at {pos}")
        s2 = deepcopy(self)
        if s2.set_state(i, EMPTY):
            res2 = s2._search()
            if res2 is not None:
                return res2

        print("  -> Both branches failed; backtrack.")
        return None


    def solve(self):
        print("[Start] Initial propagation")
        if not self.propagate():
            print("  !! Initial propagation found inconsistency.")
            return None
        self.debug_print_board("Board after initial propagation:")

        print("\n[Start] Strengthening forced moves by contradiction")
        if not self.strengthen_all_forced_moves():
            print("  !! strengthen_all_forced_moves found inconsistency.")
            return None

        if self.is_complete():
            print("\n[Info] Board is complete after forced moves only.")
            if not self.verify_solution():
                print("  !! Completed state does not satisfy all rules.")
                return None
            bulbs_state = list(self.state)
        else:
            print("\n[Info] Some UNKNOWN remain; start full search with branching.")
            bulbs_state = self._search()
            if bulbs_state is None:
                print("  !! Search failed: no solution.")
                return None

        out = [[self.tokens[r][c] for c in range(self.w)] for r in range(self.h)]
        for r in range(self.h):
            for c in range(self.w):
                idx = self.white_index[r][c]
                if idx != -1:
                    out[r][c] = "L" if bulbs_state[idx] == BULB else "."

        print("\n" + "#" * 60)
        print("[Summary]")
        print(f"  Episodes (hypothetical checks) : {AkariSolverJSON.episode_counter}")
        print(f"  Forced moves deduced           : {AkariSolverJSON.forced_move_counter}")
        print("#" * 60 + "\n")

        return out

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 main.py puzzle.json")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    width = data["width"]
    height = data["height"]
    grid = data["grid"] 

    print("=== Original board ===")
    print(render_board(to_2d(width, height, grid)))
    print()

    ans = solve_akari_json(width, height, grid)

    if ans is None:
        print("No solution")
    else:
        print("=== Solved board ===")
        print(render_board(to_2d(width, height, ans)))