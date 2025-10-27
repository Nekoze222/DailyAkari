#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Akari (Light Up) solver using a Monte Carlo Tree Search (MCTS) guided by
constraint propagation (sound hand-tuned rules).

- Input format: JSON (flexible, see examples below)
- Run example:  python akari_mcts_solver.py sample_puzzle.json --iters 5000

JSON grid formats supported:
1) Flat list (row-major):
{
  "width": 5,
  "height": 5,
  "grid": [".", "#2", ".", "#", ".",  "..."]
}
2) 2D list:
{
  "width": 5,
  "height": 5,
  "grid": [
    [".", "#2", ".", "#", "."],
    [".", ".", ".", ".", "."],
    ["#", ".", ".", ".", "."],
    [".", ".", ".", ".", "."],
    [".", ".", ".", ".", "."]
  ]
}

Cell token variants supported:
- "."              => white
- "#" or -1        => black (no number)
- "#n" or "n#"     => numeric black with n in 0..4 (string)
- integer n (0..4) => numeric black with n
- object forms:
    {"num": n}               => numeric black with n (0..4)
    {"black": true} or {"b": true} => black
    {"white": true} or {"w": true} => white
"""

from __future__ import annotations
import json, math, random, argparse, time
from typing import List, Tuple, Dict, Optional, Set, Any

WHITE = 0
BLACK = 1
NUM   = 2

def _flatten_grid(grid: Any, W: int, H: int) -> List[Any]:
    if isinstance(grid, list) and len(grid) == W*H:
        return grid
    if isinstance(grid, list) and len(grid) == H and all(isinstance(row, list) and len(row)==W for row in grid):
        flat = []
        for r in range(H):
            flat.extend(grid[r])
        return flat
    raise ValueError("grid must be a flat list of length W*H or a 2D list of shape [H][W]")

def _parse_token(tok: Any) -> Tuple[int, int]:
    if isinstance(tok, dict):
        if "num" in tok:
            n = int(tok["num"])
            if n < 0 or n > 4:
                raise ValueError(f"numeric value out of range: {tok}")
            return NUM, n
        if tok.get("black") or tok.get("b"):
            return BLACK, -1
        if tok.get("white") or tok.get("w"):
            return WHITE, -1
        raise ValueError(f"unknown cell object: {tok}")
    if isinstance(tok, (int, float)):
        n = int(tok)
        if n == -1:
            return BLACK, -1
        if 0 <= n <= 4:
            return NUM, n
        raise ValueError(f"unknown numeric token: {tok}")
    if not isinstance(tok, str):
        raise ValueError(f"unknown token type: {tok!r}")
    s = tok.strip()
    if s == ".":
        return WHITE, -1
    if s == "#":
        return BLACK, -1
    if s.startswith("#"):
        rest = s[1:]
        if rest.isdigit():
            n = int(rest)
            if 0 <= n <= 4:
                return NUM, n
    if s.endswith("#"):
        rest = s[:-1]
        if rest.isdigit():
            n = int(rest)
            if 0 <= n <= 4:
                return NUM, n
    raise ValueError(f"Unknown token: {tok} (supported: '.', '#', '#n', 'n#', 0..4, -1, or object forms)")

class Board:
    def __init__(self, width:int, height:int, grid_tokens:List[Any]):
        assert width*height == len(grid_tokens), "grid size mismatch"
        self.W = width
        self.H = height
        self.N = width*height

        self.cell_type = [WHITE]*self.N
        self.num_value = [-1]*self.N  # only for NUM cells
        self.white_indices: List[int] = []
        self.num_indices: List[int] = []

        for idx, tok in enumerate(grid_tokens):
            ct, nv = _parse_token(tok)
            self.cell_type[idx] = ct
            if ct == WHITE:
                self.white_indices.append(idx)
            elif ct == BLACK:
                pass
            elif ct == NUM:
                self.num_value[idx] = nv
                self.num_indices.append(idx)
            else:
                raise ValueError(f"bad cell type parsed: {tok}")

        self.adj: Dict[int, List[int]] = {b: [] for b in self.num_indices}
        for b in self.num_indices:
            r, c = divmod(b, self.W)
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                rr, cc = r+dr, c+dc
                if 0 <= rr < self.H and 0 <= cc < self.W:
                    idx2 = rr*self.W + cc
                    if self.cell_type[idx2] == WHITE:
                        self.adj[b].append(idx2)

        self.vis: Dict[int, List[int]] = {}
        for i in self.white_indices:
            r,c = divmod(i, self.W)
            lst = [i]
            rr = r-1
            while rr >= 0:
                j = rr*self.W + c
                if self.cell_type[j] in (BLACK, NUM):
                    break
                lst.append(j)
                rr -= 1
            rr = r+1
            while rr < self.H:
                j = rr*self.W + c
                if self.cell_type[j] in (BLACK, NUM):
                    break
                lst.append(j)
                rr += 1
            cc = c-1
            while cc >= 0:
                j = r*self.W + cc
                if self.cell_type[j] in (BLACK, NUM):
                    break
                lst.append(j)
                cc -= 1
            cc = c+1
            while cc < self.W:
                j = r*self.W + cc
                if self.cell_type[j] in (BLACK, NUM):
                    break
                lst.append(j)
                cc += 1
            self.vis[i] = lst

        self.segments: List[List[int]] = []
        for r in range(self.H):
            start = None
            for c in range(self.W+1):
                end_of_row = (c == self.W)
                is_block = end_of_row or (self.cell_type[r*self.W+c] in (BLACK, NUM))
                if not is_block and start is None:
                    start = c
                if (is_block or end_of_row) and start is not None:
                    seg = [r*self.W+cc for cc in range(start, c)]
                    if seg:
                        self.segments.append(seg)
                    start = None
        for c in range(self.W):
            start = None
            for r in range(self.H+1):
                end_of_col = (r == self.H)
                is_block = end_of_col or (self.cell_type[r*self.W+c] in (BLACK, NUM))
                if not is_block and start is None:
                    start = r
                if (is_block or end_of_col) and start is not None:
                    seg = [rr*self.W+c for rr in range(start, r)]
                    if seg:
                        self.segments.append(seg)
                    start = None

UNDECIDED = 0
FORBID    = 1
BULB      = 2

class State:
    def __init__(self, board:Board):
        self.b = board
        self.status = {i: UNDECIDED for i in board.white_indices} 
        self.lit: Set[int] = set()  
        self.rem_need: Dict[int,int] = {}
        self.adj_undec: Dict[int, Set[int]] = {}
        for b in board.num_indices:
            need = board.num_value[b]
            neigh = [i for i in board.adj[b] if i in self.status]
            self.rem_need[b] = need
            self.adj_undec[b] = set(neigh)

        for b in board.num_indices:
            if board.num_value[b] == 0:
                for i in list(self.adj_undec[b]):
                    self._set_forbid(i)

        ok = self.propagate()
        self.valid = ok

    def clone(self) -> "State":
        s = State.__new__(State)
        s.b = self.b
        s.status = dict(self.status)
        s.lit = set(self.lit)
        s.rem_need = dict(self.rem_need)
        s.adj_undec = {k:set(v) for k,v in self.adj_undec.items()}
        s.valid = self.valid
        return s

    def _place_bulb(self, i:int) -> bool:
        if i not in self.status or self.status[i] == FORBID:
            return False
        if self.status[i] == BULB:
            return True
        self.status[i] = BULB

        for j in self.b.vis[i]:
            self.lit.add(j)
            if j != i and j in self.status and self.status[j] == UNDECIDED:
                self.status[j] = FORBID

        r,c = divmod(i, self.b.W)
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            rr,cc = r+dr, c+dc
            if 0 <= rr < self.b.H and 0 <= cc < self.b.W:
                nb = rr*self.b.W + cc
                if nb in self.rem_need:  
                    self.rem_need[nb] -= 1
                    if self.rem_need[nb] < 0:
                        return False
                    self.adj_undec[nb].discard(i)
        return True

    def _set_forbid(self, i:int) -> bool:
        if i not in self.status:
            return False
        if self.status[i] == BULB:
            return False
        if self.status[i] == FORBID:
            return True
        self.status[i] = FORBID
        r,c = divmod(i, self.b.W)
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            rr,cc = r+dr, c+dc
            if 0 <= rr < self.b.H and 0 <= cc < self.b.W:
                nb = rr*self.b.W + cc
                if nb in self.adj_undec:
                    self.adj_undec[nb].discard(i)
        return True

    def _check_unlightable(self) -> bool:
        for i in self.b.white_indices:
            if i in self.lit:
                continue
            possible = False
            for j in self.b.vis[i]:
                st = self.status.get(j, None)
                if st == UNDECIDED or st == BULB:
                    possible = True
                    break
            if not possible:
                return False
        return True

    def _check_segment_conflict(self) -> bool:
        for seg in self.b.segments:
            count = 0
            for i in seg:
                if i in self.status and self.status[i] == BULB:
                    count += 1
                    if count > 1:
                        return False
        return True

    def propagate(self) -> bool:
        changed = True
        while changed:
            changed = False
            for b in self.b.num_indices:
                need = self.rem_need[b]
                undec = self.adj_undec[b]
                if need < 0 or need > len(undec):
                    return False
                if need == 0 and len(undec) > 0:
                    for i in list(undec):
                        if self._set_forbid(i):
                            changed = True
                    undec.clear()
                elif need == len(undec) and need > 0:
                    for i in list(undec):
                        if not self._place_bulb(i):
                            return False
                        changed = True
                    undec.clear()
            if not self._check_segment_conflict():
                return False
            if not self._check_unlightable():
                return False
        return True

    def is_solved(self) -> bool:
        if not self.valid:
            return False
        for i in self.b.white_indices:
            if i not in self.lit:
                return False
        for b in self.b.num_indices:
            placed = 0
            for i in self.b.adj[b]:
                if i in self.status and self.status[i] == BULB:
                    placed += 1
            if placed != self.b.num_value[b]:
                return False
        return self._check_segment_conflict()

    def legal_actions(self) -> List[int]:
        return [i for i, st in self.status.items() if st == UNDECIDED]

    def do_action(self, i:int) -> bool:
        if not self._place_bulb(i):
            return False
        return self.propagate()

    def heuristic_candidates(self) -> List[int]:
        cand = self.legal_actions()
        if not cand:
            return []
        scores = []
        unlit = set(j for j in self.b.white_indices if j not in self.lit)
        for i in cand:
            newcov = len(unlit.intersection(self.b.vis[i]))
            tight = 0
            r,c = divmod(i, self.b.W)
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                rr,cc = r+dr,c+dc
                if 0<=rr<self.b.H and 0<=cc<self.b.W:
                    nb = rr*self.b.W+cc
                    if nb in self.rem_need:
                        remaining = self.rem_need[nb]
                        slack = len(self.adj_undec[nb]) - remaining
                        if remaining>0:
                            tight += (3 - min(3, slack))
            scores.append((newcov*3 + tight, i))
        scores.sort(reverse=True)
        top = [i for _,i in scores[:max(1, len(scores)//2)]]
        return top

    def score(self) -> float:
        lit_ratio = sum(1 for i in self.b.white_indices if i in self.lit) / max(1,len(self.b.white_indices))
        num_prog = 0.0
        for b in self.b.num_indices:
            need = self.b.num_value[b]
            placed = 0
            undec = 0
            for i in self.b.adj[b]:
                if i in self.status and self.status[i]==BULB:
                    placed += 1
                elif i in self.status and self.status[i]==UNDECIDED:
                    undec += 1
            if placed > need:
                return 0.0
            if need == 0:
                num_prog += 1.0
            else:
                num_prog += min(1.0, placed/need)
        if self.b.num_indices:
            num_prog /= len(self.b.num_indices)
        else:
            num_prog = 1.0
        return 0.6*lit_ratio + 0.4*num_prog

    def key(self) -> Tuple[Tuple[int,...], Tuple[int,...]]:
        bulbs = tuple(sorted(i for i,st in self.status.items() if st==BULB))
        forb  = tuple(sorted(i for i,st in self.status.items() if st==FORBID))
        return (bulbs, forb)


class MCTSNode:
    def __init__(self, state_key, parent=None):
        self.key = state_key
        self.parent = parent
        self.children: Dict[int, MCTSNode] = {}
        self.N = 0
        self.W = 0.0  
        self.untried_actions: List[int] = []

    def uct(self, c=1.2):
        if self.N == 0:
            return float('inf')
        return (self.W / self.N) + c * math.sqrt(math.log(self.parent.N + 1) / (self.N))

def mcts_solve(board:Board, iters:int=5000, seed:int=0, time_limit_sec:Optional[float]=None) -> Tuple[Optional[State], float]:
    random.seed(seed)
    root_state = State(board)
    if not root_state.valid:
        return None, 0.0
    if root_state.is_solved():
        return root_state, 1.0

    root = MCTSNode(root_state.key(), parent=None)
    root.untried_actions = root_state.heuristic_candidates() or root_state.legal_actions()
    TT: Dict[Tuple, MCTSNode] = {root.key: root}

    best_state = None
    best_score = 0.0

    t0 = time.time()
    for it in range(iters):
        if time_limit_sec is not None and (time.time() - t0) > time_limit_sec:
            break
        node = root
        state = root_state.clone()

        while node.untried_actions == [] and node.children:
            a, node = max(node.children.items(), key=lambda kv: kv[1].uct())
            ok = state.do_action(a)
            if not ok:
                break

        if not state.valid:
            reward = 0.0
            while node is not None:
                node.N += 1
                node.W += reward
                node = node.parent
            continue

        if state.is_solved():
            best_state = state
            best_score = 1.0
            reward = 1.0
            nd = node
            while nd is not None:
                nd.N += 1
                nd.W += reward
                nd = nd.parent
            break

        if node.untried_actions:
            a = random.choice(node.untried_actions)
            node.untried_actions.remove(a)
            child_state = state.clone()
            ok = child_state.do_action(a)
            child_key = child_state.key() if ok else ("dead", a, state.key())
            child_node = MCTSNode(child_key, parent=node)
            if ok:
                child_node.untried_actions = child_state.heuristic_candidates() or child_state.legal_actions()
            else:
                child_node.untried_actions = []
            node.children[a] = child_node
            node = child_node
            state = child_state

        rollout_steps = 0
        while state.valid and (not state.is_solved()):
            acts = state.heuristic_candidates()
            if not acts:
                acts = state.legal_actions()
            if not acts:
                break
            a = random.choice(acts[:max(1, min(3, len(acts)))])
            if not state.do_action(a):
                break
            rollout_steps += 1
            if rollout_steps > (board.N * 2):
                break

        reward = 1.0 if (state.valid and state.is_solved()) else state.score()
        if reward > best_score and state.valid:
            best_score = reward
            best_state = state.clone()

        nd = node
        while nd is not None:
            nd.N += 1
            nd.W += reward
            nd = nd.parent

    return best_state, best_score

def load_json(path:str) -> Board:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    W = int(obj["width"])
    H = int(obj["height"])
    grid = _flatten_grid(obj["grid"], W, H)
    return Board(W,H,grid)

def render(board:Board, state:Optional[State]) -> str:
    W,H = board.W, board.H
    out = []
    for r in range(H):
        row = []
        for c in range(W):
            idx = r*W+c
            t = board.cell_type[idx]
            if t == WHITE:
                if state is None:
                    ch = "."
                else:
                    st = state.status.get(idx, UNDECIDED)
                    if st == BULB:
                        ch = "O"
                    elif idx in state.lit:
                        ch = "*"
                    elif st == FORBID:
                        ch = "x"
                    else:
                        ch = "."
            elif t == BLACK:
                ch = "#"
            else:
                ch = "#" + str(board.num_value[idx])
            row.append(ch)
        out.append(" ".join(row))
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="path to puzzle JSON")
    ap.add_argument("--iters", type=int, default=5000, help="MCTS iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time_limit", type=float, default=None, help="seconds")
    args = ap.parse_args()

    board = load_json(args.json_path)
    best, score = mcts_solve(board, iters=args.iters, seed=args.seed, time_limit_sec=args.time_limit)
    if best is None or not best.valid:
        print("No valid state found (contradiction at root).")
        return
    print("Score:", round(score, 3))
    print(render(board, best))
    if best.is_solved():
        print("Solved ✅")
    else:
        print("Partial (try more iterations)")

if __name__ == "__main__":
    main()
