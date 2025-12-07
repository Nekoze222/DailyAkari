"""
Akari (Light Up) パズル環境 - OpenAI Gym 互換インターフェース

================================================================================
JSON フォーマット仕様
================================================================================
パズルは以下の JSON 形式で与えられる:
{
  "width": int,    # 盤面の幅（列数）
  "height": int,   # 盤面の高さ（行数）
  "grid": [str]    # 左上から右下へ、行優先で並んだマス情報
}

grid の各要素:
  "."   : 白マス（電球を置ける／照らされる必要がある）
  "#"   : 数字なし黒マス（電球を置けない、光を遮断）
  "#0"  : 数字0の黒マス（隣接4方向に電球0個）
  "#1"  : 数字1の黒マス（隣接4方向に電球1個）
  "#2"  : 数字2の黒マス（隣接4方向に電球2個）
  "#3"  : 数字3の黒マス（隣接4方向に電球3個）
  "#4"  : 数字4の黒マス（隣接4方向に電球4個）

座標系: grid[r * width + c] で (行r, 列c) のマスを取得（0-indexed）
================================================================================

行動空間:
  action = (cell_index, action_type)
    - cell_index: 白マスのインデックス（0 から num_white_cells - 1）
    - action_type: 0 = BULB（電球を置く）, 1 = BLOCK（電球禁止マークを置く）

報酬設計:
  - 解決時: +100
  - 矛盾検出時: -50
  - 各ステップ: -1
  - シェーピング報酬（オプション）:
    - 数字マス制約が満たされた: +1
    - 新しいマスを照らした: +0.1 * (照らしたマス数)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Dict, Tuple, Optional, Set, Any
import json
import copy


class CellState(IntEnum):
    UNKNOWN = 0   # 未確定（何も置かれていない）
    BULB = 1      # 電球を置いた
    BLOCK = 2     # 電球禁止（×マーク）


class CellType(IntEnum):
    WHITE = 0      # 白マス
    BLACK = 1      # 数字なし黒マス
    BLACK_0 = 10   # 数字0の黒マス
    BLACK_1 = 11   # 数字1の黒マス
    BLACK_2 = 12   # 数字2の黒マス
    BLACK_3 = 13   # 数字3の黒マス
    BLACK_4 = 14   # 数字4の黒マス

    @staticmethod
    def from_token(token: str) -> 'CellType':
        """JSON トークンから CellType を生成"""
        if token == ".":
            return CellType.WHITE
        elif token == "#":
            return CellType.BLACK
        elif token.startswith("#") and len(token) > 1 and token[1].isdigit():
            return CellType(10 + int(token[1]))
        else:
            raise ValueError(f"Unknown token: {token}")

    def is_black(self) -> bool:
        """黒マスかどうか"""
        return self != CellType.WHITE

    def is_numbered(self) -> bool:
        """数字付き黒マスかどうか"""
        return self.value >= 10

    def get_number(self) -> int:
        """数字を取得（数字付き黒マスのみ有効）"""
        if not self.is_numbered():
            raise ValueError("Not a numbered black cell")
        return self.value - 10


@dataclass
class LocalPatch:
    """
    ローカルパッチ：中心マスと近傍の状態を表現
    
    Learning Classifier System で条件マッチングに使用する
    ローカルな盤面情報の抽出結果
    """
    center_type: CellType              # 中心マスの種類
    center_state: Optional[CellState]  # 中心が白マスの場合の状態
    center_lit: bool                   # 中心が照らされているか
    neighbors_4: List[Tuple[CellType, Optional[CellState], bool]]  # 上下左右4方向
    neighbors_8: List[Tuple[CellType, Optional[CellState], bool]]  # 8方向（斜め含む）
    
    # 数字マスの場合の追加情報
    digit_required: int = 0            # 要求される電球数
    digit_placed: int = 0              # 既に置かれた電球数
    digit_possible: int = 0            # まだ置ける可能性のあるマス数
    
    # 未照明マスの場合の追加情報
    light_candidates: int = 0          # このマスを照らせる候補マス数

    def to_tuple(self) -> Tuple:
        """ハッシュ可能な形式に変換（条件マッチング用）"""
        return (
            self.center_type,
            self.center_state,
            self.center_lit,
            tuple(self.neighbors_4),
            self.digit_required,
            self.digit_placed,
            self.digit_possible,
            self.light_candidates
        )


@dataclass 
class ActionResult:
    """step() の結果を保持"""
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class AkariEnv:
    """
    Akari (Light Up) パズル環境
    
    OpenAI Gym 互換のインターフェースを提供し、
    Learning Classifier System と連携して動作する
    
    Attributes:
        width: 盤面の幅
        height: 盤面の高さ
        cell_types: 各マスの種類（不変）
        cell_states: 白マスの状態（UNKNOWN/BULB/BLOCK）
        lit: 各マスが照らされているかどうか
    """
    
    # 報酬定数
    REWARD_SOLVED = 100.0
    REWARD_CONTRADICTION = -50.0
    REWARD_STEP = -1.0
    REWARD_DIGIT_SATISFIED = 1.0
    REWARD_NEW_LIT = 0.1
    
    def __init__(self, json_puzzle: Optional[Dict] = None):
        """
        環境を初期化
        
        Args:
            json_puzzle: パズルを表すJSON辞書。Noneの場合は後でloadで読み込む
        """
        self.width = 0
        self.height = 0
        self.cell_types: List[List[CellType]] = []
        self.cell_states: List[List[CellState]] = []
        self.lit: List[List[bool]] = []
        
        # 数字マスの位置と要求数
        self.digit_cells: Dict[Tuple[int, int], int] = {}
        
        # 白マスのリスト（行動空間用）
        self.white_cells: List[Tuple[int, int]] = []
        
        # 各数字マスの満足状態を追跡（シェーピング報酬用）
        self._prev_satisfied_digits: Set[Tuple[int, int]] = set()
        
        # 前回の照明マス数（シェーピング報酬用）
        self._prev_lit_count = 0
        
        # Hard rule適用を無効にするフラグ（探索用）
        self.disable_hard_rules = False
        
        if json_puzzle is not None:
            self.load(json_puzzle)
    
    def load(self, json_puzzle: Dict) -> None:
        """
        JSON からパズルを読み込んで初期化
        
        Args:
            json_puzzle: パズルを表すJSON辞書
        """
        self.width = json_puzzle["width"]
        self.height = json_puzzle["height"]
        flat_grid = json_puzzle["grid"]
        
        if len(flat_grid) != self.width * self.height:
            raise ValueError(
                f"Grid size mismatch: expected {self.width * self.height}, "
                f"got {len(flat_grid)}"
            )
        
        # 盤面を2次元配列に変換
        self.cell_types = []
        self.cell_states = []
        self.lit = []
        self.digit_cells = {}
        self.white_cells = []
        
        for r in range(self.height):
            type_row = []
            state_row = []
            lit_row = []
            for c in range(self.width):
                token = flat_grid[r * self.width + c]
                cell_type = CellType.from_token(token)
                type_row.append(cell_type)
                
                if cell_type == CellType.WHITE:
                    state_row.append(CellState.UNKNOWN)
                    self.white_cells.append((r, c))
                else:
                    state_row.append(CellState.UNKNOWN)  # 黒マスは状態なし（ダミー）
                
                lit_row.append(False)
                
                if cell_type.is_numbered():
                    self.digit_cells[(r, c)] = cell_type.get_number()
            
            self.cell_types.append(type_row)
            self.cell_states.append(state_row)
            self.lit.append(lit_row)
        
        # 内部状態の初期化
        self._prev_satisfied_digits = set()
        self._prev_lit_count = 0
    
    @classmethod
    def from_file(cls, filepath: str) -> 'AkariEnv':
        """JSONファイルからパズルを読み込んで環境を生成"""
        with open(filepath, 'r') as f:
            puzzle = json.load(f)
        return cls(puzzle)
    
    def reset(self, json_puzzle: Optional[Dict] = None) -> Dict[str, Any]:
        """
        環境をリセット
        
        Args:
            json_puzzle: 新しいパズル（省略時は同じパズルをリセット）
        
        Returns:
            observation: 初期観測
        """
        if json_puzzle is not None:
            self.load(json_puzzle)
        else:
            # 同じパズルで状態だけリセット
            for r in range(self.height):
                for c in range(self.width):
                    if self.cell_types[r][c] == CellType.WHITE:
                        self.cell_states[r][c] = CellState.UNKNOWN
                    self.lit[r][c] = False
            self._prev_satisfied_digits = set()
            self._prev_lit_count = 0
        
        # ハードルールを適用して収束させる
        self._apply_hard_rules_until_fixpoint()
        
        return self._get_observation()
    
    def step(self, action: Tuple[int, int]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        行動を実行
        
        Args:
            action: (cell_index, action_type)
                - cell_index: white_cells リストのインデックス
                - action_type: 0=BULB, 1=BLOCK
        
        Returns:
            observation, reward, done, info
        """
        cell_idx, action_type = action
        
        if cell_idx < 0 or cell_idx >= len(self.white_cells):
            # 無効な行動
            return self._get_observation(), self.REWARD_CONTRADICTION, True, {"error": "invalid_cell_index"}
        
        r, c = self.white_cells[cell_idx]
        
        if self.cell_states[r][c] != CellState.UNKNOWN:
            # 既に確定済みのマス
            return self._get_observation(), self.REWARD_CONTRADICTION, True, {"error": "cell_already_set"}
        
        # 行動適用前の状態を保存（シェーピング報酬計算用）
        prev_lit_count = self._count_lit_cells()
        prev_satisfied = self._get_satisfied_digits()
        
        # 行動を適用
        if action_type == 0:  # BULB
            # 電球を置く前に合法性チェック
            if not self._is_legal_bulb_placement(r, c):
                return self._get_observation(), self.REWARD_CONTRADICTION, True, {"error": "illegal_bulb"}
            self.cell_states[r][c] = CellState.BULB
            self._light_from(r, c)
        else:  # BLOCK
            self.cell_states[r][c] = CellState.BLOCK
        
        # ハードルールを適用して収束させる（無効フラグがない場合のみ）
        contradiction = False
        if not self.disable_hard_rules:
            contradiction = self._apply_hard_rules_until_fixpoint()
        
        # 報酬計算
        reward = self.REWARD_STEP
        done = False
        info = {"action": (r, c, "BULB" if action_type == 0 else "BLOCK")}
        
        if contradiction:
            reward = self.REWARD_CONTRADICTION
            done = True
            info["status"] = "contradiction"
        elif self.is_solved():
            reward = self.REWARD_SOLVED
            done = True
            info["status"] = "solved"
        elif self._is_dead_end():
            reward = self.REWARD_CONTRADICTION
            done = True
            info["status"] = "dead_end"
        else:
            # シェーピング報酬
            new_lit_count = self._count_lit_cells()
            new_satisfied = self._get_satisfied_digits()
            
            # 新しく照らされたマスへの報酬
            lit_diff = new_lit_count - prev_lit_count
            if lit_diff > 0:
                reward += self.REWARD_NEW_LIT * lit_diff
            
            # 新しく満たされた数字マスへの報酬
            for digit_pos in new_satisfied - prev_satisfied:
                reward += self.REWARD_DIGIT_SATISFIED
            
            info["status"] = "ongoing"
        
        return self._get_observation(), reward, done, info
    
    # =========================================================================
    # ハードルール実装（A1/B1: 論理的に常に正しい推論）
    # =========================================================================
    
    def _apply_hard_rules_until_fixpoint(self) -> bool:
        """
        ハードルール（A1/B1系）を収束するまで繰り返し適用
        
        ハードルールは論理的に常に正しい推論で、step() 後に自動適用される。
        
        Returns:
            contradiction: 矛盾が検出された場合 True
        """
        changed = True
        while changed:
            changed = False
            
            # A1-0: ゼロ定石 - 数字0の周囲は全てBLOCK
            result = self._apply_rule_a1_0()
            if result == "contradiction":
                return True
            changed = changed or (result == "changed")
            
            # A1-1: ぴったり置き切り - 残り必要数 == 残りUNKNOWN数
            result = self._apply_rule_a1_1()
            if result == "contradiction":
                return True
            changed = changed or (result == "changed")
            
            # A1-2: 残り0の×確定 - 必要数が満たされたら残りはBLOCK
            result = self._apply_rule_a1_2()
            if result == "contradiction":
                return True
            changed = changed or (result == "changed")
            
            # B1: 一意照明 - 未照明マスを照らせる候補が1つだけならBULB
            result = self._apply_rule_b1()
            if result == "contradiction":
                return True
            changed = changed or (result == "changed")
            
            # 照明の再計算
            self._recompute_lighting()
        
        return False
    
    def _apply_rule_a1_0(self) -> str:
        """
        A1-0: ゼロ定石
        数字マス 0 の周囲4方向の白マスは全て BLOCK にする
        
        これはハードルール相当（論理的に常に正しい）
        """
        changed = False
        for (r, c), n in self.digit_cells.items():
            if n != 0:
                continue
            for nr, nc in self._neighbors4(r, c):
                if (self.cell_types[nr][nc] == CellType.WHITE and 
                    self.cell_states[nr][nc] == CellState.UNKNOWN):
                    self.cell_states[nr][nc] = CellState.BLOCK
                    changed = True
        return "changed" if changed else "unchanged"
    
    def _apply_rule_a1_1(self) -> str:
        """
        A1-1: ぴったり置き切り
        数字マスの「残り必要電球数」と「残りUNKNOWNマス数」が等しければ、
        全てのUNKNOWNマスにBULBを置く
        
        これはハードルール相当（論理的に常に正しい）
        """
        changed = False
        for (r, c), n in self.digit_cells.items():
            placed, unknown_cells = self._digit_stats(r, c)
            remaining = n - placed
            
            if remaining < 0:
                return "contradiction"  # 既に置きすぎ
            
            if remaining == len(unknown_cells) and remaining > 0:
                for nr, nc in unknown_cells:
                    if not self._is_legal_bulb_placement(nr, nc):
                        return "contradiction"
                    self.cell_states[nr][nc] = CellState.BULB
                    self._light_from(nr, nc)
                    changed = True
        
        return "changed" if changed else "unchanged"
    
    def _apply_rule_a1_2(self) -> str:
        """
        A1-2: 残り0の×確定
        数字マスの電球が既に必要数置かれていれば、残りのUNKNOWNはBLOCK
        
        これはハードルール相当（論理的に常に正しい）
        """
        changed = False
        for (r, c), n in self.digit_cells.items():
            placed, unknown_cells = self._digit_stats(r, c)
            remaining = n - placed
            
            if remaining == 0 and unknown_cells:
                for nr, nc in unknown_cells:
                    self.cell_states[nr][nc] = CellState.BLOCK
                    changed = True
        
        return "changed" if changed else "unchanged"
    
    def _apply_rule_b1(self) -> str:
        """
        B1: 一意照明ルール
        未照明の白マスを照らせる候補（UNKNOWN白マス）が1つだけなら、
        そのマスにBULBを置く
        
        これはハードルール相当（論理的に常に正しい）
        """
        changed = False
        
        for r in range(self.height):
            for c in range(self.width):
                if (self.cell_types[r][c] != CellType.WHITE or 
                    self.lit[r][c] or 
                    self.cell_states[r][c] == CellState.BULB):
                    continue
                
                # このマスを照らせる候補を探す
                candidates = self._get_light_candidates(r, c)
                
                if len(candidates) == 0:
                    # 照らす手段がない → 矛盾（BLOCKが置かれていない限り）
                    if self.cell_states[r][c] != CellState.BLOCK:
                        return "contradiction"
                elif len(candidates) == 1:
                    nr, nc = candidates[0]
                    if not self._is_legal_bulb_placement(nr, nc):
                        return "contradiction"
                    self.cell_states[nr][nc] = CellState.BULB
                    self._light_from(nr, nc)
                    changed = True
        
        return "changed" if changed else "unchanged"
    
    # =========================================================================
    # ヘルパーメソッド
    # =========================================================================
    
    def _neighbors4(self, r: int, c: int) -> List[Tuple[int, int]]:
        """上下左右4方向の隣接マス座標を返す"""
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                result.append((nr, nc))
        return result
    
    def _neighbors8(self, r: int, c: int) -> List[Tuple[int, int]]:
        """8方向の隣接マス座標を返す"""
        result = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    result.append((nr, nc))
        return result
    
    def _visible_cells(self, r: int, c: int) -> List[Tuple[int, int]]:
        """
        (r, c) から直線上に見える白マスのリスト（自分含む）
        黒マスで遮断される
        """
        if self.cell_types[r][c] != CellType.WHITE:
            return []
        
        result = [(r, c)]
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            while 0 <= nr < self.height and 0 <= nc < self.width:
                if self.cell_types[nr][nc].is_black():
                    break
                result.append((nr, nc))
                nr += dr
                nc += dc
        
        return result
    
    def _digit_stats(self, r: int, c: int) -> Tuple[int, List[Tuple[int, int]]]:
        """
        数字マス (r, c) の統計情報
        
        Returns:
            (placed_bulbs, unknown_cells): 置かれた電球数と未確定の隣接白マスリスト
        """
        placed = 0
        unknown = []
        
        for nr, nc in self._neighbors4(r, c):
            if self.cell_types[nr][nc] != CellType.WHITE:
                continue
            if self.cell_states[nr][nc] == CellState.BULB:
                placed += 1
            elif self.cell_states[nr][nc] == CellState.UNKNOWN:
                unknown.append((nr, nc))
        
        return placed, unknown
    
    def _is_legal_bulb_placement(self, r: int, c: int) -> bool:
        """
        (r, c) に電球を置くことが合法かどうか
        
        合法条件:
        - 白マスである
        - 状態がUNKNOWN
        - 同一直線上に既存の電球がない
        - 隣接数字マスの要求を超過しない
        """
        if self.cell_types[r][c] != CellType.WHITE:
            return False
        if self.cell_states[r][c] != CellState.UNKNOWN:
            return False
        
        # 視認性チェック：同一直線上に電球があってはならない
        for vr, vc in self._visible_cells(r, c):
            if (vr, vc) != (r, c) and self.cell_states[vr][vc] == CellState.BULB:
                return False
        
        # 数字制約チェック：これを置くと超過しないか
        for nr, nc in self._neighbors4(r, c):
            if (nr, nc) in self.digit_cells:
                required = self.digit_cells[(nr, nc)]
                placed, _ = self._digit_stats(nr, nc)
                if placed + 1 > required:
                    return False
        
        return True
    
    def _light_from(self, r: int, c: int) -> None:
        """(r, c) に電球を置いた時の照明効果を適用"""
        for vr, vc in self._visible_cells(r, c):
            self.lit[vr][vc] = True
    
    def _recompute_lighting(self) -> None:
        """照明状態を全て再計算"""
        for r in range(self.height):
            for c in range(self.width):
                self.lit[r][c] = False
        
        for r in range(self.height):
            for c in range(self.width):
                if self.cell_states[r][c] == CellState.BULB:
                    self._light_from(r, c)
    
    def _get_light_candidates(self, r: int, c: int) -> List[Tuple[int, int]]:
        """
        未照明マス (r, c) を照らせる候補（UNKNOWN状態の白マス）を返す
        """
        candidates = []
        for vr, vc in self._visible_cells(r, c):
            if (self.cell_states[vr][vc] == CellState.UNKNOWN and 
                self._is_legal_bulb_placement(vr, vc)):
                candidates.append((vr, vc))
        return candidates
    
    def _count_lit_cells(self) -> int:
        """照らされている白マスの数を返す"""
        count = 0
        for r in range(self.height):
            for c in range(self.width):
                if self.cell_types[r][c] == CellType.WHITE and self.lit[r][c]:
                    count += 1
        return count
    
    def _get_satisfied_digits(self) -> Set[Tuple[int, int]]:
        """要求数がちょうど満たされた数字マスの集合を返す"""
        satisfied = set()
        for (r, c), n in self.digit_cells.items():
            placed, _ = self._digit_stats(r, c)
            if placed == n:
                satisfied.add((r, c))
        return satisfied
    
    def is_solved(self) -> bool:
        """パズルが解けているかどうか"""
        # 全白マスが照らされている
        for r in range(self.height):
            for c in range(self.width):
                if self.cell_types[r][c] == CellType.WHITE and not self.lit[r][c]:
                    return False
        
        # 全数字マスの条件が満たされている
        for (r, c), n in self.digit_cells.items():
            placed, _ = self._digit_stats(r, c)
            if placed != n:
                return False
        
        return True
    
    def _is_dead_end(self) -> bool:
        """
        これ以上進められない行き詰まり状態かどうか
        """
        if self.is_solved():
            return False
        
        # 未照明で照らす手段がないマスがあれば詰み
        for r in range(self.height):
            for c in range(self.width):
                if (self.cell_types[r][c] == CellType.WHITE and 
                    not self.lit[r][c] and 
                    self.cell_states[r][c] != CellState.BULB):
                    candidates = self._get_light_candidates(r, c)
                    if len(candidates) == 0:
                        return True
        
        # 数字マスが満たせない状態ならば詰み
        for (r, c), n in self.digit_cells.items():
            placed, unknown = self._digit_stats(r, c)
            # 置いた数が超過
            if placed > n:
                return True
            # 残りのUNKNOWNを全部使っても足りない
            if placed + len(unknown) < n:
                return True
        
        return False
    
    def _get_observation(self) -> Dict[str, Any]:
        """現在の観測を生成"""
        return {
            "width": self.width,
            "height": self.height,
            "cell_types": [[ct.value for ct in row] for row in self.cell_types],
            "cell_states": [[cs.value for cs in row] for row in self.cell_states],
            "lit": [[l for l in row] for row in self.lit],
            "white_cells": self.white_cells.copy(),
            "digit_cells": dict(self.digit_cells),
        }
    
    def get_legal_actions(self) -> List[Tuple[int, int]]:
        """
        現在の状態で合法な行動のリスト
        
        Returns:
            List of (cell_index, action_type)
        """
        actions = []
        for idx, (r, c) in enumerate(self.white_cells):
            if self.cell_states[r][c] == CellState.UNKNOWN:
                # BULB が合法なら追加
                if self._is_legal_bulb_placement(r, c):
                    actions.append((idx, 0))
                # BLOCK は常に追加可能
                actions.append((idx, 1))
        return actions
    
    def get_local_patch(self, r: int, c: int) -> LocalPatch:
        """
        指定位置のローカルパッチを取得
        
        Learning Classifier System の条件マッチングに使用
        """
        center_type = self.cell_types[r][c]
        center_state = self.cell_states[r][c] if center_type == CellType.WHITE else None
        center_lit = self.lit[r][c]
        
        # 4方向の近傍
        neighbors_4 = []
        for nr, nc in self._neighbors4(r, c):
            n_type = self.cell_types[nr][nc]
            n_state = self.cell_states[nr][nc] if n_type == CellType.WHITE else None
            n_lit = self.lit[nr][nc]
            neighbors_4.append((n_type, n_state, n_lit))
        
        # 足りない方向は境界外として None を入れる
        while len(neighbors_4) < 4:
            neighbors_4.append((None, None, False))
        
        # 8方向の近傍
        neighbors_8 = []
        for nr, nc in self._neighbors8(r, c):
            n_type = self.cell_types[nr][nc]
            n_state = self.cell_states[nr][nc] if n_type == CellType.WHITE else None
            n_lit = self.lit[nr][nc]
            neighbors_8.append((n_type, n_state, n_lit))
        
        while len(neighbors_8) < 8:
            neighbors_8.append((None, None, False))
        
        # 数字マスの場合の追加情報
        digit_required = 0
        digit_placed = 0
        digit_possible = 0
        
        if center_type.is_numbered():
            digit_required = center_type.get_number()
            digit_placed, unknown_cells = self._digit_stats(r, c)
            digit_possible = len(unknown_cells)
        
        # 未照明白マスの場合の追加情報
        light_candidates = 0
        if center_type == CellType.WHITE and not center_lit:
            light_candidates = len(self._get_light_candidates(r, c))
        
        return LocalPatch(
            center_type=center_type,
            center_state=center_state,
            center_lit=center_lit,
            neighbors_4=neighbors_4,
            neighbors_8=neighbors_8,
            digit_required=digit_required,
            digit_placed=digit_placed,
            digit_possible=digit_possible,
            light_candidates=light_candidates
        )
    
    def clone(self) -> 'AkariEnv':
        """環境の深いコピーを作成"""
        new_env = AkariEnv()
        new_env.width = self.width
        new_env.height = self.height
        new_env.cell_types = [row.copy() for row in self.cell_types]
        new_env.cell_states = [row.copy() for row in self.cell_states]
        new_env.lit = [row.copy() for row in self.lit]
        new_env.digit_cells = dict(self.digit_cells)
        new_env.white_cells = self.white_cells.copy()
        new_env._prev_satisfied_digits = self._prev_satisfied_digits.copy()
        new_env._prev_lit_count = self._prev_lit_count
        new_env.disable_hard_rules = self.disable_hard_rules
        return new_env
    
    def render(self) -> str:
        """盤面を文字列で表現"""
        lines = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                ct = self.cell_types[r][c]
                if ct.is_black():
                    if ct.is_numbered():
                        row.append(str(ct.get_number()))
                    else:
                        row.append("#")
                else:
                    cs = self.cell_states[r][c]
                    if cs == CellState.BULB:
                        row.append("L")
                    elif cs == CellState.BLOCK:
                        row.append("x")
                    elif self.lit[r][c]:
                        row.append("*")
                    else:
                        row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)
    
    def get_cell_index(self, r: int, c: int) -> int:
        """座標から白マスインデックスを取得（-1は見つからない場合）"""
        try:
            return self.white_cells.index((r, c))
        except ValueError:
            return -1


if __name__ == "__main__":
    # 簡単なテスト
    puzzle = {
        "width": 10,
        "height": 10,
        "grid": [
            ".", "#0", "#", "#0", ".", "#", ".", ".", ".", ".",
            ".", "#0", ".", ".", ".", ".", ".", ".", ".", "#2",
            ".", ".", ".", ".", ".", ".", ".", ".", ".", ".",
            ".", "#0", ".", ".", "#3", ".", ".", "#", ".", ".",
            ".", "#0", ".", ".", ".", ".", ".", ".", ".", ".",
            ".", ".", ".", ".", ".", ".", ".", ".", ".", ".",
            "#", ".", ".", ".", ".", ".", ".", ".", ".", ".",
            ".", ".", "#1", "#0", ".", ".", ".", "#", ".", ".",
            ".", ".", ".", ".", ".", ".", ".", ".", ".", ".",
            "#1", ".", ".", ".", "#", "#1", ".", ".", ".", "#1"
        ]
    }
    
    env = AkariEnv(puzzle)
    obs = env.reset()
    print("Initial state:")
    print(env.render())
    print(f"\nWhite cells: {len(env.white_cells)}")
    print(f"Legal actions: {len(env.get_legal_actions())}")
