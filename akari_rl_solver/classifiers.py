"""
Learning Classifier System (LCS) for Akari Solver - Improved Version

================================================================================
バケツリレー型強化学習によるルール学習（改良版）
================================================================================

改良点:
1. ハードルールはenv.pyで自動適用されるため、classifierは探索ヒューリスティックに集中
2. 被覆度ベースの優先度付け
3. 孤立マス（照らす手段が少ない未照明マス）の優先
4. 学習効率の改善

設計書との対応:
- A1/B1系のハードルール: env.py の _apply_hard_rules_until_fixpoint() で自動適用
- C1/E1/D1/F1系: 必要に応じて探索ヒューリスティックとして組み込み
- 探索ルール: 本モジュールで classifier として実装・学習

================================================================================
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Callable, Any
from enum import Enum, auto
import random
import math

from env import AkariEnv, CellState, CellType, LocalPatch


class RuleType(Enum):
    """ルールの種類を表す列挙型"""
    # 探索ヒューリスティック系
    EXPLORE_ISOLATED = auto()          # 孤立した未照明マスを優先（照らす手段が少ない）
    EXPLORE_DIGIT_ADJACENT = auto()    # 数字マス隣接を優先
    EXPLORE_HIGH_COVERAGE = auto()     # 多くのマスを照らせる場所優先
    EXPLORE_UNLIT_PRIORITY = auto()    # 未照明マスを照らす優先
    EXPLORE_RANDOM = auto()            # ランダム探索（フォールバック）


@dataclass
class Classifier:
    """
    Learning Classifier System のルール（classifier）
    
    各ルールは探索ヒューリスティックを表し、学習によって強さが調整される。
    
    Attributes:
        id: ルールの一意識別子
        rule_type: ルールの種類
        strength: ルールの強さ（学習対象）
        specificity: ルールの特殊性（より具体的なルールほど高い）
        description: ルールの説明
    """
    id: int
    rule_type: RuleType
    strength: float = 1.0
    specificity: float = 1.0
    description: str = ""
    
    # 統計情報
    times_matched: int = 0
    times_selected: int = 0
    times_success: int = 0  # 成功エピソードで使われた回数
    times_failure: int = 0  # 失敗エピソードで使われた回数
    
    def compute_bid(self, score: float = 1.0, noise_factor: float = 0.1) -> float:
        """
        入札額を計算
        
        bid = strength * specificity * score * (1 + noise)
        
        Args:
            score: 行動のスコア（被覆度など）
            noise_factor: ノイズの大きさ（探索のため）
        
        Returns:
            bid: 入札額
        """
        noise = random.uniform(-noise_factor, noise_factor)
        bid = self.strength * self.specificity * score * (1 + noise)
        return max(0.01, bid)
    
    def __repr__(self) -> str:
        success_rate = self.times_success / max(1, self.times_success + self.times_failure)
        return (f"Classifier(type={self.rule_type.name}, "
                f"strength={self.strength:.3f}, "
                f"success_rate={success_rate:.2%})")


@dataclass
class ActionCandidate:
    """行動候補"""
    target_r: int
    target_c: int
    action_type: int  # 0: BULB, 1: BLOCK
    classifier: Optional[Classifier]
    score: float  # この行動のスコア（被覆度など）
    bid: float = 0.0
    
    def __repr__(self) -> str:
        action_name = "BULB" if self.action_type == 0 else "BLOCK"
        clf_name = self.classifier.rule_type.name if self.classifier else "NONE"
        return f"Action({action_name} at ({self.target_r},{self.target_c}), rule={clf_name}, bid={self.bid:.2f})"


class ClassifierPopulation:
    """
    ルール群（classifier population）の管理
    
    設計思想:
    - ハードルール（A1/B1）は env.py で自動適用される
    - このクラスは探索ヒューリスティックの学習に集中
    - バケツリレーアルゴリズムで強さを更新
    """
    
    def __init__(self, 
                 learning_rate: float = 0.15,
                 discount_factor: float = 0.95,
                 min_strength: float = 0.1,
                 max_strength: float = 5.0):
        """
        Args:
            learning_rate: 学習率（バケツリレーの分配率）
            discount_factor: 割引率（過去ルールへの減衰）
            min_strength: 強さの下限
            max_strength: 強さの上限
        """
        self.classifiers: Dict[RuleType, Classifier] = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.min_strength = min_strength
        self.max_strength = max_strength
        
        self._initialize_classifiers()
    
    def _initialize_classifiers(self) -> None:
        """
        探索ヒューリスティックルールを初期化
        
        これらは「人間知識の prior」として機能し、
        学習の過程で強さが調整される
        """
        
        # 孤立マス優先: 照らす手段が少ない未照明マスを優先
        # これは B1（一意照明）の緩和版で、候補が少ないマスを優先する
        self.classifiers[RuleType.EXPLORE_ISOLATED] = Classifier(
            id=0,
            rule_type=RuleType.EXPLORE_ISOLATED,
            strength=3.0,
            specificity=2.0,
            description="照らす手段が少ない未照明マスを優先（B1の緩和版）"
        )
        
        # 数字マス隣接: 未充足の数字マスに隣接する場所を優先
        # これは A1-1（ぴったり置き切り）につながりやすい
        self.classifiers[RuleType.EXPLORE_DIGIT_ADJACENT] = Classifier(
            id=1,
            rule_type=RuleType.EXPLORE_DIGIT_ADJACENT,
            strength=2.5,
            specificity=1.8,
            description="未充足の数字マス隣接を優先（A1につながる）"
        )
        
        # 高被覆: 多くのマスを照らせる場所を優先
        # 効率的に盤面を照らす
        self.classifiers[RuleType.EXPLORE_HIGH_COVERAGE] = Classifier(
            id=2,
            rule_type=RuleType.EXPLORE_HIGH_COVERAGE,
            strength=2.0,
            specificity=1.5,
            description="多くのマスを照らせる場所を優先"
        )
        
        # 未照明優先: まだ照らされていないマスを照らせる場所を優先
        self.classifiers[RuleType.EXPLORE_UNLIT_PRIORITY] = Classifier(
            id=3,
            rule_type=RuleType.EXPLORE_UNLIT_PRIORITY,
            strength=1.5,
            specificity=1.0,
            description="未照明マスを照らす場所を優先"
        )
        
        # ランダム: フォールバック
        self.classifiers[RuleType.EXPLORE_RANDOM] = Classifier(
            id=4,
            rule_type=RuleType.EXPLORE_RANDOM,
            strength=0.5,
            specificity=0.2,
            description="ランダムな空きマス（フォールバック）"
        )
    
    def _compute_coverage_score(self, env: AkariEnv, r: int, c: int) -> int:
        """
        (r, c) に電球を置いた時に新しく照らせるマス数を計算
        
        Args:
            env: 環境
            r, c: 対象マスの座標
        
        Returns:
            新しく照らせるマス数
        """
        if not env._is_legal_bulb_placement(r, c):
            return 0
        
        new_lit = 0
        for vr, vc in env._visible_cells(r, c):
            if not env.lit[vr][vc]:
                new_lit += 1
        return new_lit
    
    def _get_digit_adjacency_score(self, env: AkariEnv, r: int, c: int) -> Tuple[bool, float]:
        """
        (r, c) が未充足の数字マスに隣接しているかと、そのスコアを計算
        
        Returns:
            (隣接しているか, スコア)
        """
        total_score = 0.0
        has_adjacent = False
        
        for nr, nc in env._neighbors4(r, c):
            if (nr, nc) in env.digit_cells:
                required = env.digit_cells[(nr, nc)]
                placed, unknown = env._digit_stats(nr, nc)
                remaining = required - placed
                
                if remaining > 0:
                    has_adjacent = True
                    # 残り必要数が少ないほど高スコア（より確実）
                    # また、unknown が少ないほど高スコア（ぴったり置き切りに近い）
                    urgency = remaining / max(1, len(unknown))
                    total_score += urgency * 2
        
        return has_adjacent, total_score
    
    def _find_isolated_unlit(self, env: AkariEnv) -> List[Tuple[int, int, int, List[Tuple[int, int]]]]:
        """
        照らす手段が少ない未照明マスとその候補を見つける
        
        Returns:
            List of (unlit_r, unlit_c, num_candidates, candidate_positions)
        """
        isolated = []
        
        for r in range(env.height):
            for c in range(env.width):
                if (env.cell_types[r][c] == CellType.WHITE and
                    not env.lit[r][c] and
                    env.cell_states[r][c] != CellState.BULB):
                    
                    candidates = env._get_light_candidates(r, c)
                    
                    # 候補が1〜4個のマスを「孤立」とみなす
                    if 1 <= len(candidates) <= 4:
                        isolated.append((r, c, len(candidates), candidates))
        
        # 候補数が少ない順にソート（より孤立しているものを優先）
        isolated.sort(key=lambda x: x[2])
        return isolated
    
    def get_action_candidates(self, env: AkariEnv) -> List[ActionCandidate]:
        """
        現在の状態で可能な行動候補とそのスコアを取得
        
        各候補には適切なルールが割り当てられ、スコアが計算される。
        
        Returns:
            行動候補のリスト
        """
        candidates = []
        processed_targets = set()  # 重複を避ける
        
        # ===================================================================
        # Phase 1: 孤立した未照明マスを優先的に処理
        # 候補が少ないほど緊急性が高く、優先度を大幅に上げる
        # ===================================================================
        isolated = self._find_isolated_unlit(env)
        clf_isolated = self.classifiers[RuleType.EXPLORE_ISOLATED]
        
        for unlit_r, unlit_c, num_cands, light_cands in isolated:
            for tr, tc in light_cands:
                if (tr, tc) in processed_targets:
                    continue
                
                processed_targets.add((tr, tc))
                clf_isolated.times_matched += 1
                
                # 孤立マスの優先度を大幅に上げる
                # 候補が1つ: score=100, 2つ: score=50, 3つ: score=33, 4つ: score=25
                score = 100.0 / num_cands
                
                candidates.append(ActionCandidate(
                    target_r=tr, target_c=tc,
                    action_type=0,  # BULB
                    classifier=clf_isolated,
                    score=score
                ))
        
        # ===================================================================
        # Phase 2: 残りの合法手を処理
        # ===================================================================
        clf_digit = self.classifiers[RuleType.EXPLORE_DIGIT_ADJACENT]
        clf_coverage = self.classifiers[RuleType.EXPLORE_HIGH_COVERAGE]
        clf_unlit = self.classifiers[RuleType.EXPLORE_UNLIT_PRIORITY]
        clf_random = self.classifiers[RuleType.EXPLORE_RANDOM]
        
        for r, c in env.white_cells:
            if env.cell_states[r][c] != CellState.UNKNOWN:
                continue
            if not env._is_legal_bulb_placement(r, c):
                continue
            if (r, c) in processed_targets:
                continue
            
            coverage = self._compute_coverage_score(env, r, c)
            is_digit_adj, digit_score = self._get_digit_adjacency_score(env, r, c)
            
            # 数字マス隣接
            if is_digit_adj:
                clf_digit.times_matched += 1
                score = digit_score + coverage * 0.3
                candidates.append(ActionCandidate(
                    target_r=r, target_c=c,
                    action_type=0,
                    classifier=clf_digit,
                    score=score
                ))
                processed_targets.add((r, c))
            # 高被覆（閾値以上のマスを照らせる）
            elif coverage >= 4:
                clf_coverage.times_matched += 1
                score = coverage
                candidates.append(ActionCandidate(
                    target_r=r, target_c=c,
                    action_type=0,
                    classifier=clf_coverage,
                    score=score
                ))
                processed_targets.add((r, c))
            # 未照明優先（何かしら照らせる）
            elif coverage > 0:
                clf_unlit.times_matched += 1
                score = coverage
                candidates.append(ActionCandidate(
                    target_r=r, target_c=c,
                    action_type=0,
                    classifier=clf_unlit,
                    score=score
                ))
                processed_targets.add((r, c))
            # ランダム（何も照らせないが合法）
            else:
                clf_random.times_matched += 1
                score = 0.1
                candidates.append(ActionCandidate(
                    target_r=r, target_c=c,
                    action_type=0,
                    classifier=clf_random,
                    score=score
                ))
                processed_targets.add((r, c))
        
        # ===================================================================
        # Phase 3: 各候補の bid を計算
        # ===================================================================
        for cand in candidates:
            if cand.classifier:
                cand.bid = cand.classifier.compute_bid(cand.score)
            else:
                cand.bid = cand.score * 0.1
        
        return candidates
    
    def select_action(self, env: AkariEnv, 
                      epsilon: float = 0.1) -> Optional[ActionCandidate]:
        """
        マッチしたルールから行動を選択
        
        Args:
            env: 環境
            epsilon: ε-greedy の探索率
        
        Returns:
            選択された ActionCandidate or None
        """
        candidates = self.get_action_candidates(env)
        
        if not candidates:
            return None
        
        # ε-greedy: 一定確率でランダム選択
        if random.random() < epsilon:
            selected = random.choice(candidates)
            if selected.classifier:
                selected.classifier.times_selected += 1
            return selected
        
        # 入札ベースの選択（bid に比例した確率）
        total_bid = sum(c.bid for c in candidates)
        
        if total_bid <= 0:
            selected = random.choice(candidates)
        else:
            r = random.uniform(0, total_bid)
            cumulative = 0
            selected = candidates[-1]
            for cand in candidates:
                cumulative += cand.bid
                if cumulative >= r:
                    selected = cand
                    break
        
        if selected.classifier:
            selected.classifier.times_selected += 1
        
        return selected
    
    def update_strengths_bucket_brigade(self, 
                                        fired_history: List[Classifier],
                                        success: bool) -> None:
        """
        バケツリレーによる強さ更新
        
        エピソードで発火したルールの履歴に対して、
        報酬を後ろから前へ伝播させる。
        
        Args:
            fired_history: エピソード中に選択されたルールの履歴
            success: エピソードが成功（パズル解決）したかどうか
        """
        if not fired_history:
            return
        
        # 成功/失敗に応じた報酬
        # 成功時は正の報酬、失敗時は負の報酬
        final_reward = 1.0 if success else -0.2
        credit = final_reward
        
        # 後ろから前へ報酬を伝播（バケツリレー）
        for clf in reversed(fired_history):
            if clf is None:
                credit *= self.discount_factor
                continue
            
            # 統計更新
            if success:
                clf.times_success += 1
            else:
                clf.times_failure += 1
            
            # 強さの更新: s <- s + α * credit
            delta = self.learning_rate * credit
            new_strength = clf.strength + delta
            clf.strength = max(self.min_strength, 
                              min(self.max_strength, new_strength))
            
            # 信用の減衰
            credit *= self.discount_factor
    
    def get_stats(self) -> Dict[str, Any]:
        """ルール統計情報を取得"""
        stats = {}
        for rule_type, clf in self.classifiers.items():
            total = clf.times_success + clf.times_failure
            success_rate = clf.times_success / max(1, total)
            stats[rule_type.name] = {
                "strength": clf.strength,
                "matched": clf.times_matched,
                "selected": clf.times_selected,
                "success": clf.times_success,
                "failure": clf.times_failure,
                "success_rate": success_rate
            }
        return stats
    
    def __repr__(self) -> str:
        lines = ["ClassifierPopulation:"]
        for clf in self.classifiers.values():
            lines.append(f"  {clf}")
        return "\n".join(lines)


class BucketBrigadeAgent:
    """
    バケツリレー型強化学習エージェント
    
    Learning Classifier System を使用して Akari パズルを解く。
    
    動作の流れ:
    1. 環境から状態を取得
    2. ClassifierPopulation から行動候補を取得
    3. 入札ベースで行動を選択
    4. 行動を実行し、報酬を受け取る
    5. エピソード終了時にバケツリレーで学習
    """
    
    def __init__(self, 
                 learning_rate: float = 0.15,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.3,
                 epsilon_decay: float = 0.998,
                 min_epsilon: float = 0.05,
                 seed: Optional[int] = None):
        """
        Args:
            learning_rate: 学習率
            discount_factor: 割引率
            epsilon: 探索率（ε-greedy）
            epsilon_decay: 探索率の減衰率
            min_epsilon: 探索率の下限
            seed: 乱数シード
        """
        if seed is not None:
            random.seed(seed)
        
        self.population = ClassifierPopulation(
            learning_rate=learning_rate,
            discount_factor=discount_factor
        )
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # 統計情報
        self.episodes = 0
        self.solved_count = 0
        self.total_steps = 0
    
    def run_episode(self, env: AkariEnv, 
                    max_steps: int = 200,
                    train: bool = True) -> Tuple[bool, int, float]:
        """
        1エピソードを実行
        
        Args:
            env: Akari 環境
            max_steps: 最大ステップ数
            train: 学習するかどうか
        
        Returns:
            (solved, steps, total_reward)
        """
        obs = env.reset()
        fired_history: List[Classifier] = []
        total_reward = 0.0
        steps = 0
        
        for step in range(max_steps):
            if env.is_solved():
                break
            
            # 行動選択
            action_cand = self.population.select_action(
                env, 
                epsilon=self.epsilon if train else 0.0
            )
            
            if action_cand is None:
                # 合法手がない → 行き詰まり
                break
            
            # 行動を環境のインデックスに変換
            cell_idx = env.get_cell_index(action_cand.target_r, action_cand.target_c)
            if cell_idx < 0:
                break
            
            # 行動実行
            obs, reward, done, info = env.step((cell_idx, action_cand.action_type))
            total_reward += reward
            steps += 1
            
            # 履歴に追加
            fired_history.append(action_cand.classifier)
            
            if done:
                break
        
        solved = env.is_solved()
        
        # 学習
        if train:
            self.population.update_strengths_bucket_brigade(
                fired_history, solved
            )
            
            # 探索率を減衰
            self.epsilon = max(
                self.min_epsilon,
                self.epsilon * self.epsilon_decay
            )
        
        # 統計更新
        self.episodes += 1
        if solved:
            self.solved_count += 1
        self.total_steps += steps
        
        return solved, steps, total_reward
    
    def train(self, puzzles: List[Dict], 
              episodes_per_puzzle: int = 100,
              max_steps: int = 200,
              verbose: bool = True) -> Dict[str, Any]:
        """
        複数パズルで学習
        
        Args:
            puzzles: パズル JSON のリスト
            episodes_per_puzzle: パズルあたりのエピソード数
            max_steps: 1エピソードの最大ステップ数
            verbose: 進捗表示
        
        Returns:
            学習結果の統計情報
        """
        results = {
            "total_episodes": 0,
            "total_solved": 0,
            "puzzle_results": []
        }
        
        for puzzle_idx, puzzle in enumerate(puzzles):
            env = AkariEnv(puzzle)
            puzzle_solved = 0
            
            for ep in range(episodes_per_puzzle):
                solved, steps, reward = self.run_episode(
                    env, max_steps=max_steps, train=True
                )
                
                if solved:
                    puzzle_solved += 1
                
                results["total_episodes"] += 1
                if solved:
                    results["total_solved"] += 1
                
                if verbose and (ep + 1) % max(1, episodes_per_puzzle // 10) == 0:
                    solve_rate = puzzle_solved / (ep + 1)
                    print(f"Puzzle {puzzle_idx + 1}, Episode {ep + 1}/{episodes_per_puzzle}: "
                          f"solve_rate={solve_rate:.2%}, epsilon={self.epsilon:.3f}")
            
            results["puzzle_results"].append({
                "puzzle_idx": puzzle_idx,
                "episodes": episodes_per_puzzle,
                "solved": puzzle_solved,
                "solve_rate": puzzle_solved / episodes_per_puzzle
            })
        
        return results
    
    def solve(self, env: AkariEnv, max_steps: int = 200) -> Tuple[bool, AkariEnv]:
        """
        学習済みモデルでパズルを解く（greedy）
        
        Args:
            env: 解くパズルの環境
            max_steps: 最大ステップ数
        
        Returns:
            (solved, final_env)
        """
        env_copy = env.clone()
        obs = env_copy.reset()
        
        for step in range(max_steps):
            if env_copy.is_solved():
                break
            
            action_cand = self.population.select_action(env_copy, epsilon=0.0)
            
            if action_cand is None:
                break
            
            cell_idx = env_copy.get_cell_index(action_cand.target_r, action_cand.target_c)
            
            if cell_idx < 0:
                break
            
            obs, reward, done, info = env_copy.step((cell_idx, action_cand.action_type))
            
            if done:
                break
        
        return env_copy.is_solved(), env_copy
    
    def _get_forced_moves(self, env: AkariEnv) -> List[Tuple[int, int]]:
        """
        制約伝播: 数字マスで、残りのスペースがちょうど必要な電球数と一致する場合、
        そこに電球を置くしかない。
        
        Returns:
            強制的に電球を置くべきマスのリスト
        """
        forced = []
        
        for (r, c), digit in env.digit_cells.items():
            count_bulbs = 0
            empty_neighbors = []
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    state = env.cell_states[nr][nc]
                    if state == CellState.BULB:
                        count_bulbs += 1
                    elif state == CellState.UNKNOWN:
                        if env._is_legal_bulb_placement(nr, nc):
                            empty_neighbors.append((nr, nc))
            
            needed = digit - count_bulbs
            if needed == len(empty_neighbors) and needed > 0:
                # 残りのスペース全てに電球を置くしかない
                forced.extend(empty_neighbors)
        
        return forced
    
    def _apply_forced_moves(self, env: AkariEnv) -> bool:
        """
        強制手を全て適用
        
        Returns:
            矛盾が発生した場合 False
        """
        max_iterations = 100  # 無限ループ防止
        for _ in range(max_iterations):
            forced = self._get_forced_moves(env)
            if not forced:
                break
            
            # 1つだけ適用して再チェック（照明状態が変わるため）
            applied = False
            for r, c in forced:
                if env.cell_states[r][c] != CellState.UNKNOWN:
                    continue
                if not env._is_legal_bulb_placement(r, c):
                    # この位置は既に照らされている（他の強制手で照らされた）
                    # 数字制約違反でなければ問題ない
                    continue
                
                cell_idx = env.get_cell_index(r, c)
                if cell_idx < 0:
                    continue
                
                _, _, done, info = env.step((cell_idx, 0))
                if info.get("status") == "contradiction":
                    return False
                
                applied = True
                break  # 1つ適用したら再チェック
            
            if not applied:
                # 全ての強制手が既に処理済みか違法
                break
        
        # 数字制約が満たせなくなっていないかチェック
        for (r, c), digit in env.digit_cells.items():
            count_bulbs = 0
            empty_legal = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    state = env.cell_states[nr][nc]
                    if state == CellState.BULB:
                        count_bulbs += 1
                    elif state == CellState.UNKNOWN:
                        if env._is_legal_bulb_placement(nr, nc):
                            empty_legal += 1
            
            needed = digit - count_bulbs
            if needed > empty_legal:
                # 必要な電球数より配置可能なスペースが少ない
                return False
            if needed < 0:
                # 電球が多すぎる
                return False
        
        return True
    
    def solve_with_backtrack(self, env: AkariEnv, 
                              max_depth: int = 100,
                              max_nodes: int = 50000) -> Tuple[bool, AkariEnv]:
        """
        バックトラッキング付きでパズルを解く（制約伝播強化版）
        
        強化学習で学習した優先度を使いつつ、行き詰まったら
        バックトラックして別の手を試す。
        数字マス制約から強制的な手を導出し、探索を効率化。
        
        Args:
            env: 解くパズルの環境
            max_depth: 最大探索深さ
            max_nodes: 最大探索ノード数
        
        Returns:
            (solved, final_env)
        """
        initial_env = env.clone()
        initial_env.reset()
        
        # 探索中はhard ruleを無効にする（自分で制約伝播を管理）
        initial_env.disable_hard_rules = True
        
        # 初期の強制手を適用
        if not self._apply_forced_moves(initial_env):
            return False, initial_env
        
        if initial_env.is_solved():
            return True, initial_env
        
        # 状態をシリアライズ/デシリアライズするためのヘルパー
        def serialize_state(e: AkariEnv) -> bytes:
            import pickle
            return pickle.dumps((
                [[s for s in row] for row in e.cell_states],
                [[l for l in row] for row in e.lit]
            ))
        
        def deserialize_state(e: AkariEnv, state: bytes) -> None:
            import pickle
            cell_states, lit = pickle.loads(state)
            for r in range(e.height):
                for c in range(e.width):
                    e.cell_states[r][c] = cell_states[r][c]
                    e.lit[r][c] = lit[r][c]
        
        # スタック: (serialized_state, candidate_index, candidates_list)
        stack = []
        
        # 初期状態の候補を取得
        initial_state = serialize_state(initial_env)
        initial_candidates = self.population.get_action_candidates(initial_env)
        initial_candidates.sort(key=lambda x: -x.bid)
        stack.append((initial_state, 0, [(c.target_r, c.target_c, c.action_type, c.bid) for c in initial_candidates]))
        
        nodes_explored = 0
        best_env = initial_env.clone()
        best_lit_count = initial_env._count_lit_cells()
        
        # 作業用環境（クローンを避ける）
        work_env = initial_env.clone()
        
        while stack and nodes_explored < max_nodes:
            current_state, cand_idx, candidates = stack[-1]
            
            # 全候補を試した場合はバックトラック
            if cand_idx >= len(candidates) or cand_idx >= 12:  # 各レベルで最大12候補
                stack.pop()
                continue
            
            # 深さ制限チェック
            if len(stack) > max_depth:
                stack.pop()
                continue
            
            # 次の候補を取得
            tr, tc, action_type, bid = candidates[cand_idx]
            stack[-1] = (current_state, cand_idx + 1, candidates)
            
            nodes_explored += 1
            
            # 状態を復元
            deserialize_state(work_env, current_state)
            
            # 既に配置済みならスキップ
            if work_env.cell_states[tr][tc] != CellState.UNKNOWN:
                continue
            
            # 行動を適用
            cell_idx = work_env.get_cell_index(tr, tc)
            if cell_idx < 0:
                continue
            
            if not work_env._is_legal_bulb_placement(tr, tc):
                continue
            
            _, reward, done, info = work_env.step((cell_idx, action_type))
            
            if info.get("status") == "contradiction":
                continue
            
            # 強制手を適用
            if not self._apply_forced_moves(work_env):
                continue  # 矛盾
            
            # 進捗を追跡
            lit_count = work_env._count_lit_cells()
            if lit_count > best_lit_count:
                best_lit_count = lit_count
                best_env = work_env.clone()
            
            if work_env.is_solved():
                return True, work_env.clone()
            
            # 次の候補を取得してスタックにプッシュ
            next_state = serialize_state(work_env)
            next_candidates = self.population.get_action_candidates(work_env)
            if next_candidates:
                next_candidates.sort(key=lambda x: -x.bid)
                stack.append((next_state, 0, [(c.target_r, c.target_c, c.action_type, c.bid) for c in next_candidates]))
        
        return False, best_env
    
    def get_stats(self) -> Dict[str, Any]:
        """エージェントの統計情報"""
        return {
            "episodes": self.episodes,
            "solved_count": self.solved_count,
            "solve_rate": self.solved_count / max(1, self.episodes),
            "total_steps": self.total_steps,
            "avg_steps": self.total_steps / max(1, self.episodes),
            "epsilon": self.epsilon,
            "classifier_stats": self.population.get_stats()
        }


if __name__ == "__main__":
    # 簡単なテスト
    from env import AkariEnv
    
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
    
    print("Testing Improved Classifier Population...")
    env = AkariEnv(puzzle)
    env.reset()
    
    population = ClassifierPopulation()
    print(f"Initialized {len(population.classifiers)} classifiers")
    
    candidates = population.get_action_candidates(env)
    print(f"Found {len(candidates)} action candidates")
    
    print("\nTop 5 candidates by bid:")
    for cand in sorted(candidates, key=lambda x: -x.bid)[:5]:
        print(f"  {cand}")
    
    print("\nClassifier initial states:")
    print(population)
