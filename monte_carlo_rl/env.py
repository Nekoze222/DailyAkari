#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AkariEnv: OpenAI Gym 互換の Akari (Light Up) パズル環境

JSON フォーマット仕様:
----------------------
{
  "width": int,      # 盤面の幅 (列数)
  "height": int,     # 盤面の高さ (行数)
  "grid": [...]      # 長さ width*height のフラットリスト (row-major)
                     # または height 行 × width 列の 2D リスト
}

grid の各セル:
- "."              => 白マス (UNKNOWN 状態から開始)
- "#" or -1        => 黒マス (数字なし)
- "#n" or "n#"     => 数字付き黒マス (n = 0..4)
- int n (0..4)     => 数字付き黒マス
- {"num": n}       => 数字付き黒マス
- {"black": true}  => 黒マス

行動空間:
---------
action = cell_index * 2 + action_type
- action_type = 0: BLOCK (×を置く)
- action_type = 1: BULB (電球を置く)
- cell_index は白マスのインデックス (0-indexed, row-major)

報酬設計:
---------
- 成功 (パズル完了): +100
- 矛盾/失敗: -50
- 各ステップ: -1
- シェーピング報酬: 数字マスが条件を満たした瞬間に +1
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

# セルタイプ定数
WHITE = 0  # 白マス
BLACK = 1  # 黒マス (数字なし)
NUM = 2    # 数字付き黒マス

# 白マスの状態
UNKNOWN = 0  # 未確定
BLOCK = 1    # × (BULBを置けない)
BULB = 2     # 電球

# 行動タイプ
ACTION_BLOCK = 0
ACTION_BULB = 1


def flatten_grid(grid: Any, width: int, height: int) -> List[Any]:
    """2D または flat な grid を flat list に変換"""
    if isinstance(grid, list) and len(grid) == width * height:
        return grid
    if isinstance(grid, list) and len(grid) == height:
        if all(isinstance(row, list) and len(row) == width for row in grid):
            return [cell for row in grid for cell in row]
    raise ValueError("grid must be flat list of W*H or 2D list [H][W]")


def parse_token(tok: Any) -> Tuple[int, int]:
    """
    セルトークンを (cell_type, num_value) に変換
    
    Returns:
        (WHITE, -1), (BLACK, -1), or (NUM, n) where n in 0..4
    """
    # オブジェクト形式
    if isinstance(tok, dict):
        if "num" in tok:
            n = int(tok["num"])
            if not (0 <= n <= 4):
                raise ValueError(f"num out of range: {tok}")
            return NUM, n
        if tok.get("black") or tok.get("b"):
            return BLACK, -1
        if tok.get("white") or tok.get("w"):
            return WHITE, -1
        raise ValueError(f"unknown cell object: {tok}")
    
    # 数値形式
    if isinstance(tok, (int, float)):
        n = int(tok)
        if n == -1:
            return BLACK, -1
        if 0 <= n <= 4:
            return NUM, n
        raise ValueError(f"unknown numeric token: {tok}")
    
    # 文字列形式
    if not isinstance(tok, str):
        raise ValueError(f"unknown token type: {tok!r}")
    
    s = tok.strip()
    if s == ".":
        return WHITE, -1
    if s == "#":
        return BLACK, -1
    # "#n" 形式
    if s.startswith("#") and s[1:].isdigit():
        n = int(s[1:])
        if 0 <= n <= 4:
            return NUM, n
    # "n#" 形式
    if s.endswith("#") and s[:-1].isdigit():
        n = int(s[:-1])
        if 0 <= n <= 4:
            return NUM, n
    
    raise ValueError(f"unknown token: {tok}")


class Board:
    """
    パズル盤面の静的構造を保持
    
    Attributes:
        W, H: 盤面サイズ
        cell_type: 各セルのタイプ (WHITE/BLACK/NUM)
        num_value: 数字マスの値 (-1 if not NUM)
        white_indices: 白マスのインデックスリスト
        num_indices: 数字マスのインデックスリスト
        adj: 各数字マスに隣接する白マスのリスト
        vis: 各白マスから見える白マスのリスト (十字方向、自分含む)
    """
    
    def __init__(self, width: int, height: int, grid_tokens: List[Any]):
        self.W = width
        self.H = height
        self.N = width * height
        
        assert len(grid_tokens) == self.N, "grid size mismatch"
        
        self.cell_type: List[int] = [WHITE] * self.N
        self.num_value: List[int] = [-1] * self.N
        self.white_indices: List[int] = []
        self.num_indices: List[int] = []
        
        # セルをパース
        for idx, tok in enumerate(grid_tokens):
            ct, nv = parse_token(tok)
            self.cell_type[idx] = ct
            if ct == WHITE:
                self.white_indices.append(idx)
            elif ct == NUM:
                self.num_value[idx] = nv
                self.num_indices.append(idx)
        
        # 各数字マスに隣接する白マス
        self.adj: Dict[int, List[int]] = {b: [] for b in self.num_indices}
        for b in self.num_indices:
            r, c = divmod(b, self.W)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.H and 0 <= cc < self.W:
                    idx = rr * self.W + cc
                    if self.cell_type[idx] == WHITE:
                        self.adj[b].append(idx)
        
        # 各白マスから見える白マス (電球の光が届く範囲)
        self.vis: Dict[int, List[int]] = {}
        for i in self.white_indices:
            r, c = divmod(i, self.W)
            visible = [i]  # 自分自身
            
            # 上方向
            for rr in range(r - 1, -1, -1):
                j = rr * self.W + c
                if self.cell_type[j] in (BLACK, NUM):
                    break
                visible.append(j)
            
            # 下方向
            for rr in range(r + 1, self.H):
                j = rr * self.W + c
                if self.cell_type[j] in (BLACK, NUM):
                    break
                visible.append(j)
            
            # 左方向
            for cc in range(c - 1, -1, -1):
                j = r * self.W + cc
                if self.cell_type[j] in (BLACK, NUM):
                    break
                visible.append(j)
            
            # 右方向
            for cc in range(c + 1, self.W):
                j = r * self.W + cc
                if self.cell_type[j] in (BLACK, NUM):
                    break
                visible.append(j)
            
            self.vis[i] = visible


class AkariState:
    """
    パズルの動的状態を保持
    
    Attributes:
        status: 各白マスの状態 (UNKNOWN/BLOCK/BULB)
        lit: 照らされている白マスの集合
        rem_need: 各数字マスの残り必要BULB数
        adj_undec: 各数字マスに隣接する未確定白マス
        valid: 矛盾が発生していないか
        satisfied_nums: 条件を満たした数字マスの集合 (シェーピング報酬用)
    """
    
    def __init__(self, board: Board):
        self.board = board
        self.status: Dict[int, int] = {i: UNKNOWN for i in board.white_indices}
        self.lit: Set[int] = set()
        
        # 数字マスの制約追跡
        self.rem_need: Dict[int, int] = {}
        self.adj_undec: Dict[int, Set[int]] = {}
        for b in board.num_indices:
            self.rem_need[b] = board.num_value[b]
            self.adj_undec[b] = set(board.adj[b])
        
        self.valid = True
        self.satisfied_nums: Set[int] = set()
        
        # 初期の制約伝播 (数字 0 のマス周辺を BLOCK に)
        for b in board.num_indices:
            if board.num_value[b] == 0:
                for i in list(self.adj_undec[b]):
                    self._set_block(i)
        
        self.valid = self._propagate()
    
    def clone(self) -> "AkariState":
        """状態の深いコピーを作成"""
        s = AkariState.__new__(AkariState)
        s.board = self.board
        s.status = dict(self.status)
        s.lit = set(self.lit)
        s.rem_need = dict(self.rem_need)
        s.adj_undec = {k: set(v) for k, v in self.adj_undec.items()}
        s.valid = self.valid
        s.satisfied_nums = set(self.satisfied_nums)
        return s
    
    def _place_bulb(self, i: int) -> bool:
        """
        白マス i に電球を置く
        
        Returns:
            成功なら True、矛盾なら False
        """
        if i not in self.status:
            return False
        if self.status[i] == BLOCK:
            return False  # 既に BLOCK なら置けない
        if self.status[i] == BULB:
            return True  # 既に置いてある
        
        self.status[i] = BULB
        
        # 見えるマスを照らす
        for j in self.board.vis[i]:
            self.lit.add(j)
            # 他の UNKNOWN マスは BLOCK になる (電球同士が見えてはいけない)
            if j != i and j in self.status and self.status[j] == UNKNOWN:
                self.status[j] = BLOCK
        
        # 隣接する数字マスの残り必要数を減らす
        r, c = divmod(i, self.board.W)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.board.H and 0 <= cc < self.board.W:
                nb = rr * self.board.W + cc
                if nb in self.rem_need:
                    self.rem_need[nb] -= 1
                    if self.rem_need[nb] < 0:
                        return False  # 数字マスの制約違反
                    self.adj_undec[nb].discard(i)
        
        return True
    
    def _set_block(self, i: int) -> bool:
        """
        白マス i を BLOCK に設定
        
        Returns:
            成功なら True
        """
        if i not in self.status:
            return False
        if self.status[i] == BULB:
            return False  # 既に BULB なら変更不可
        if self.status[i] == BLOCK:
            return True  # 既に BLOCK
        
        self.status[i] = BLOCK
        
        # 隣接する数字マスの未確定リストから削除
        r, c = divmod(i, self.board.W)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.board.H and 0 <= cc < self.board.W:
                nb = rr * self.board.W + cc
                if nb in self.adj_undec:
                    self.adj_undec[nb].discard(i)
        
        return True
    
    def _check_can_light_all(self) -> bool:
        """全白マスが照らせる可能性があるかチェック"""
        for i in self.board.white_indices:
            if i in self.lit:
                continue
            # まだ照らされていない → 見える範囲に UNKNOWN か BULB があるか
            can_light = False
            for j in self.board.vis[i]:
                st = self.status.get(j, None)
                if st == UNKNOWN or st == BULB:
                    can_light = True
                    break
            if not can_light:
                return False
        return True
    
    def _check_no_bulb_conflict(self) -> bool:
        """電球同士が見えていないかチェック"""
        bulbs = [i for i, st in self.status.items() if st == BULB]
        for i in bulbs:
            for j in self.board.vis[i]:
                if j != i and j in self.status and self.status[j] == BULB:
                    return False
        return True
    
    def _propagate(self) -> bool:
        """
        ハードルール (A1/B1) を適用して状態を簡約
        
        A1: 数字マスの残り必要数 == 隣接未確定数 → 全て BULB
        B1: 数字マスの残り必要数 == 0 → 隣接未確定は全て BLOCK
        
        Returns:
            矛盾がなければ True
        """
        prev_satisfied = len(self.satisfied_nums)
        
        changed = True
        while changed:
            changed = False
            
            for b in self.board.num_indices:
                need = self.rem_need[b]
                undec = self.adj_undec[b]
                
                # 制約チェック
                if need < 0 or need > len(undec):
                    return False
                
                # B1: 残り必要数 0 → 全て BLOCK
                if need == 0 and len(undec) > 0:
                    for i in list(undec):
                        self._set_block(i)
                        changed = True
                    undec.clear()
                
                # A1: 残り必要数 == 未確定数 → 全て BULB
                elif need == len(undec) and need > 0:
                    for i in list(undec):
                        if not self._place_bulb(i):
                            return False
                        changed = True
                    undec.clear()
                
                # 数字マスが条件を満たしたかチェック
                if need == 0 and b not in self.satisfied_nums:
                    self.satisfied_nums.add(b)
            
            # 整合性チェック
            if not self._check_no_bulb_conflict():
                return False
            if not self._check_can_light_all():
                return False
        
        return True
    
    def get_legal_actions(self) -> List[int]:
        """
        合法行動のリストを返す
        
        Returns:
            行動リスト (action = cell_index * 2 + action_type)
        """
        actions = []
        for i, st in self.status.items():
            if st == UNKNOWN:
                # BLOCK を置く行動
                actions.append(i * 2 + ACTION_BLOCK)
                # BULB を置く行動
                actions.append(i * 2 + ACTION_BULB)
        return actions
    
    def apply_action(self, action: int) -> Tuple[bool, int, int]:
        """
        行動を適用
        
        Args:
            action: cell_index * 2 + action_type
        
        Returns:
            (success, shaping_reward, newly_lit)
            - success: 矛盾なく適用できたか
            - shaping_reward: シェーピング報酬 (満たした数字マス数)
            - newly_lit: 新たに照らしたマス数
        """
        cell_index = action // 2
        action_type = action % 2
        
        prev_satisfied = len(self.satisfied_nums)
        prev_lit = len(self.lit)
        
        if action_type == ACTION_BULB:
            if not self._place_bulb(cell_index):
                self.valid = False
                return False, 0, 0
        else:  # ACTION_BLOCK
            if not self._set_block(cell_index):
                self.valid = False
                return False, 0, 0
        
        # ハードルール適用
        if not self._propagate():
            self.valid = False
            return False, 0, 0
        
        # シェーピング報酬
        new_satisfied = len(self.satisfied_nums) - prev_satisfied
        newly_lit = len(self.lit) - prev_lit
        return True, new_satisfied, newly_lit
    
    def is_solved(self) -> bool:
        """パズルが完全に解けたかチェック"""
        if not self.valid:
            return False
        
        # 全白マスが照らされているか
        for i in self.board.white_indices:
            if i not in self.lit:
                return False
        
        # 全数字マスの制約が満たされているか
        for b in self.board.num_indices:
            placed = sum(1 for i in self.board.adj[b] 
                        if i in self.status and self.status[i] == BULB)
            if placed != self.board.num_value[b]:
                return False
        
        return self._check_no_bulb_conflict()
    
    def is_terminal(self) -> bool:
        """終端状態かチェック (成功 or 失敗)"""
        if not self.valid:
            return True
        if self.is_solved():
            return True
        # 合法行動がない場合も終端
        return len(self.get_legal_actions()) == 0
    
    def encode(self) -> Tuple[int, ...]:
        """
        状態をハッシュ可能なタプルにエンコード
        
        各白マスの状態を 0/1/2 で表現し、タプル化
        """
        return tuple(self.status[i] for i in sorted(self.status.keys()))


class AkariEnv:
    """
    Akari パズルの Gym 互換環境
    
    Usage:
        env = AkariEnv(puzzle_json)
        obs = env.reset()
        obs, reward, done, info = env.step(action)
    
    Reward:
        - 成功: +100
        - 失敗: -50
        - 各ステップ: -1
        - シェーピング: +1 per newly satisfied number cell
    """
    
    # 報酬定数
    REWARD_SUCCESS = 100.0
    REWARD_FAILURE = -50.0
    REWARD_STEP = -0.5  # ステップペナルティを軽減
    REWARD_SHAPING = 2.0  # 数字マス満足報酬を増加
    REWARD_LIT = 0.5  # 照明報酬を追加
    
    def __init__(self, puzzle_json: Optional[dict] = None):
        """
        Args:
            puzzle_json: パズル JSON (width, height, grid)
        """
        self.board: Optional[Board] = None
        self.state: Optional[AkariState] = None
        self.steps = 0
        
        if puzzle_json is not None:
            self.load_puzzle(puzzle_json)
    
    def load_puzzle(self, puzzle_json: dict) -> None:
        """JSON からパズルを読み込み"""
        width = int(puzzle_json["width"])
        height = int(puzzle_json["height"])
        grid_tokens = flatten_grid(puzzle_json["grid"], width, height)
        self.board = Board(width, height, grid_tokens)
    
    @classmethod
    def from_file(cls, json_path: str) -> "AkariEnv":
        """JSON ファイルから環境を作成"""
        with open(json_path, "r", encoding="utf-8") as f:
            puzzle_json = json.load(f)
        return cls(puzzle_json)
    
    def reset(self, puzzle_json: Optional[dict] = None) -> Tuple[int, ...]:
        """
        環境をリセット
        
        Args:
            puzzle_json: 新しいパズル (None なら現在のパズルを再使用)
        
        Returns:
            初期観測 (状態のタプルエンコード)
        """
        if puzzle_json is not None:
            self.load_puzzle(puzzle_json)
        
        if self.board is None:
            raise ValueError("No puzzle loaded")
        
        self.state = AkariState(self.board)
        self.steps = 0
        
        return self.state.encode()
    
    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool, dict]:
        """
        行動を実行
        
        Args:
            action: cell_index * 2 + action_type
        
        Returns:
            (observation, reward, done, info)
        """
        if self.state is None:
            raise ValueError("Environment not reset")
        
        self.steps += 1
        
        # 行動を適用
        success, shaping, newly_lit = self.state.apply_action(action)
        
        # 報酬計算
        reward = self.REWARD_STEP
        reward += shaping * self.REWARD_SHAPING
        reward += newly_lit * self.REWARD_LIT  # 照明報酬
        
        done = False
        info = {"steps": self.steps, "success": False}
        
        if not success or not self.state.valid:
            # 矛盾発生
            reward += self.REWARD_FAILURE
            done = True
            info["reason"] = "contradiction"
        elif self.state.is_solved():
            # パズル完了
            reward += self.REWARD_SUCCESS
            done = True
            info["success"] = True
            info["reason"] = "solved"
        elif len(self.state.get_legal_actions()) == 0:
            # 手詰まり
            reward += self.REWARD_FAILURE
            done = True
            info["reason"] = "stuck"
        
        obs = self.state.encode()
        return obs, reward, done, info
    
    def get_legal_actions(self) -> List[int]:
        """現在の合法行動リスト"""
        if self.state is None:
            return []
        return self.state.get_legal_actions()
    
    def get_state_key(self) -> Tuple[int, ...]:
        """現在の状態キー (Q テーブル用)"""
        if self.state is None:
            raise ValueError("Environment not reset")
        return self.state.encode()
    
    def render(self) -> str:
        """盤面を文字列で描画"""
        if self.board is None or self.state is None:
            return "No puzzle loaded"
        
        lines = []
        for r in range(self.board.H):
            row = []
            for c in range(self.board.W):
                idx = r * self.board.W + c
                ct = self.board.cell_type[idx]
                
                if ct == WHITE:
                    st = self.state.status.get(idx, UNKNOWN)
                    if st == BULB:
                        ch = "O"
                    elif idx in self.state.lit:
                        ch = "*"
                    elif st == BLOCK:
                        ch = "x"
                    else:
                        ch = "."
                elif ct == BLACK:
                    ch = "#"
                else:  # NUM
                    ch = f"#{self.board.num_value[idx]}"
                
                row.append(ch)
            lines.append(" ".join(row))
        
        return "\n".join(lines)


# デモ用コード
if __name__ == "__main__":
    # サンプルパズル
    sample_puzzle = {
        "width": 5,
        "height": 5,
        "grid": [
            ".", ".", "#", ".", ".",
            ".", ".", ".", "#", ".",
            "#", ".", ".", "#2", ".",
            ".", ".", "#2", ".", ".",
            ".", ".", ".", ".", "#1"
        ]
    }
    
    env = AkariEnv(sample_puzzle)
    obs = env.reset()
    
    print("Initial state:")
    print(env.render())
    print(f"\nState encoding (first 10): {obs[:10]}...")
    print(f"Legal actions: {len(env.get_legal_actions())} available")
    
    # 数ステップ実行
    print("\n--- Taking some random actions ---")
    for i in range(3):
        actions = env.get_legal_actions()
        if not actions:
            print("No more actions available")
            break
        
        # ランダムに行動選択
        import random
        action = random.choice(actions)
        cell_idx = action // 2
        action_type = "BULB" if action % 2 == 1 else "BLOCK"
        
        print(f"\nStep {i+1}: Action {action} (cell {cell_idx}, {action_type})")
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward}, Done: {done}, Info: {info}")
        print(env.render())
        
        if done:
            break
