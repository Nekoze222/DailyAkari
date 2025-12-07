#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonteCarloAgent: モンテカルロ制御 (ε-greedy 方策改善) によるエージェント

設計方針:
---------
1. Q テーブル: Q[state_key][action] で行動価値を保持
2. 訪問回数: N[state_key][action] で更新回数を追跡
3. ε-greedy 方策: 確率 ε でランダム、1-ε で Q 最大の行動を選択
4. Every-visit MC: エピソード中の全 (s, a) ペアで Q を更新
5. 更新式: Q(s,a) ← Q(s,a) + α * (G - Q(s,a))
   - α は 1/N(s,a) (サンプル平均) または固定学習率

最適化:
-------
- UCB探索ボーナス: 訪問回数が少ない行動に探索ボーナス
- ヒューリスティック: BULB行動を優先、照明効率を考慮
- 楽観的初期化: 未訪問行動に高い初期値

割引率 γ について:
- Akari は有限エピソードなので γ=1.0 を使用
- これにより「最終結果に忠実な評価」となる
"""

from __future__ import annotations
import math
import random
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple, Optional

from env import AkariEnv


class MonteCarloAgent:
    """
    モンテカルロ強化学習エージェント (最適化版)
    
    Attributes:
        epsilon: 探索率 (0.0 ~ 1.0)
        gamma: 割引率 (通常 1.0)
        alpha: 学習率 (None の場合はサンプル平均 1/N を使用)
        Q: Q テーブル Q[state_key][action] -> float
        N: 訪問回数 N[state_key][action] -> int
        ucb_c: UCB探索係数
        optimistic_init: 楽観的初期値
    """
    
    def __init__(
        self,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        alpha: Optional[float] = None,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.01,
        ucb_c: float = 2.0,
        optimistic_init: float = 50.0
    ):
        """
        Args:
            epsilon: 初期探索率
            gamma: 割引率 (有限エピソードなので 1.0 推奨)
            alpha: 固定学習率 (None ならサンプル平均)
            epsilon_decay: エピソードごとの ε 減衰率
            epsilon_min: ε の下限
            ucb_c: UCB探索係数 (大きいほど探索重視)
            optimistic_init: 未訪問行動の楽観的初期値
        """
        self.epsilon = epsilon
        self.epsilon_initial = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.alpha = alpha
        self.ucb_c = ucb_c
        self.optimistic_init = optimistic_init
        
        # Q テーブル: Q[state_key][action] -> value
        # 楽観的初期化: 未訪問は高い値を返す
        self.Q: DefaultDict[Tuple, DefaultDict[int, float]] = defaultdict(
            lambda: defaultdict(lambda: self.optimistic_init)
        )
        
        # 訪問回数: N[state_key][action] -> count
        self.N: DefaultDict[Tuple, DefaultDict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        
        # 状態ごとの総訪問回数
        self.N_state: DefaultDict[Tuple, int] = defaultdict(int)
        
        # 統計情報
        self.episode_count = 0
        self.total_updates = 0
    
    def select_action(
        self,
        state_key: Tuple,
        legal_actions: List[int],
        greedy: bool = False,
        env: Optional[AkariEnv] = None
    ) -> int:
        """
        UCB + ε-greedy 方策に従って行動を選択
        
        Args:
            state_key: 状態のタプル表現
            legal_actions: 合法行動のリスト
            greedy: True なら純粋に greedy (評価用)
            env: 環境 (ヒューリスティック計算用、オプション)
        
        Returns:
            選択された行動
        """
        if not legal_actions:
            raise ValueError("No legal actions available")
        
        # BULB行動のみをフィルタ (BLOCKは補助的)
        bulb_actions = [a for a in legal_actions if a % 2 == 1]
        if bulb_actions:
            primary_actions = bulb_actions
        else:
            primary_actions = legal_actions
        
        # 探索: 確率 ε でランダム
        if not greedy and random.random() < self.epsilon:
            return random.choice(primary_actions)
        
        # UCB + Q値で行動選択
        q_values = self.Q[state_key]
        n_state = self.N_state[state_key] + 1  # 0除算防止
        
        best_action = None
        best_score = float("-inf")
        
        for action in primary_actions:
            q = q_values[action]
            n_action = self.N[state_key][action]
            
            if greedy:
                # 評価時はQ値のみ
                score = q
            else:
                # UCBボーナス
                if n_action == 0:
                    ucb_bonus = self.optimistic_init
                else:
                    ucb_bonus = self.ucb_c * math.sqrt(math.log(n_state) / n_action)
                score = q + ucb_bonus
            
            if score > best_score:
                best_score = score
                best_action = action
            elif score == best_score and best_action is not None:
                # タイブレーク: 訪問回数が少ない方
                if self.N[state_key][action] < self.N[state_key][best_action]:
                    best_action = action
        
        if best_action is None:
            return random.choice(primary_actions)
        
        return best_action
    
    def generate_episode(
        self,
        env: AkariEnv,
        max_steps: int = 1000
    ) -> List[Tuple[Tuple, int, float]]:
        """
        環境と対話してエピソードを生成
        
        Args:
            env: 環境
            max_steps: 最大ステップ数
        
        Returns:
            [(state_key, action, reward), ...] のリスト
        """
        episode: List[Tuple[Tuple, int, float]] = []
        
        obs = env.reset()
        state_key = env.get_state_key()
        
        for _ in range(max_steps):
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break
            
            action = self.select_action(state_key, legal_actions)
            obs, reward, done, info = env.step(action)
            
            episode.append((state_key, action, reward))
            
            if done:
                break
            
            state_key = env.get_state_key()
        
        return episode
    
    def update_from_episode(
        self,
        episode: List[Tuple[Tuple, int, float]]
    ) -> Dict[str, float]:
        """
        エピソードから Q テーブルを更新 (every-visit MC)
        
        Args:
            episode: [(state_key, action, reward), ...]
        
        Returns:
            統計情報 (total_return, num_updates)
        """
        if not episode:
            return {"total_return": 0.0, "num_updates": 0}
        
        # 累積報酬 G を後ろから計算
        G = 0.0
        updates = 0
        
        for t in range(len(episode) - 1, -1, -1):
            state_key, action, reward = episode[t]
            G = self.gamma * G + reward
            
            # Every-visit: 全ての訪問で更新
            self.N[state_key][action] += 1
            self.N_state[state_key] += 1
            n = self.N[state_key][action]
            
            # 学習率の決定 (固定学習率を推奨)
            if self.alpha is not None:
                lr = self.alpha
            else:
                # 減衰学習率: より速く収束
                lr = max(0.1, 1.0 / (1 + 0.1 * n))
            
            # Q 値の更新: Q(s,a) ← Q(s,a) + α * (G - Q(s,a))
            old_q = self.Q[state_key][action]
            self.Q[state_key][action] = old_q + lr * (G - old_q)
            
            updates += 1
            self.total_updates += 1
        
        self.episode_count += 1
        
        return {
            "total_return": G,
            "num_updates": updates,
            "episode_length": len(episode)
        }
    
    def decay_epsilon(self) -> None:
        """ε を減衰させる"""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )
    
    def evaluate(
        self,
        env: AkariEnv,
        num_episodes: int = 10,
        max_steps: int = 1000
    ) -> Dict[str, float]:
        """
        Greedy 方策での評価
        
        Args:
            env: 環境
            num_episodes: 評価エピソード数
            max_steps: 各エピソードの最大ステップ数
        
        Returns:
            統計情報 (success_rate, avg_return, avg_steps)
        """
        successes = 0
        total_return = 0.0
        total_steps = 0
        
        for _ in range(num_episodes):
            obs = env.reset()
            state_key = env.get_state_key()
            episode_return = 0.0
            steps = 0
            
            for step in range(max_steps):
                legal_actions = env.get_legal_actions()
                if not legal_actions:
                    break
                
                # Greedy 選択
                action = self.select_action(state_key, legal_actions, greedy=True)
                obs, reward, done, info = env.step(action)
                
                episode_return += reward
                steps += 1
                
                if done:
                    if info.get("success", False):
                        successes += 1
                    break
                
                state_key = env.get_state_key()
            
            total_return += episode_return
            total_steps += steps
        
        return {
            "success_rate": successes / num_episodes,
            "avg_return": total_return / num_episodes,
            "avg_steps": total_steps / num_episodes
        }
    
    def get_stats(self) -> Dict[str, any]:
        """現在の統計情報を取得"""
        num_states = len(self.Q)
        total_actions = sum(len(actions) for actions in self.Q.values())
        
        return {
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "total_updates": self.total_updates,
            "num_states": num_states,
            "num_state_action_pairs": total_actions
        }
    
    def reset_stats(self) -> None:
        """統計情報をリセット (Q テーブルは保持)"""
        self.epsilon = self.epsilon_initial
        self.episode_count = 0
        self.total_updates = 0
    
    def clear(self) -> None:
        """Q テーブルと統計情報を完全にリセット"""
        self.Q.clear()
        self.N.clear()
        self.N_state.clear()
        self.reset_stats()


class MonteCarloAgentWithBaseline(MonteCarloAgent):
    """
    ベースライン付きモンテカルロエージェント
    
    分散低減のため、状態価値 V(s) をベースラインとして使用
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 状態価値: V[state_key] -> value
        self.V: DefaultDict[Tuple, float] = defaultdict(float)
        self.V_count: DefaultDict[Tuple, int] = defaultdict(int)
    
    def update_from_episode(
        self,
        episode: List[Tuple[Tuple, int, float]]
    ) -> Dict[str, float]:
        """ベースライン付き更新"""
        if not episode:
            return {"total_return": 0.0, "num_updates": 0}
        
        G = 0.0
        updates = 0
        
        for t in range(len(episode) - 1, -1, -1):
            state_key, action, reward = episode[t]
            G = self.gamma * G + reward
            
            # 状態価値の更新
            self.V_count[state_key] += 1
            v_n = self.V_count[state_key]
            v_lr = 1.0 / v_n if self.alpha is None else self.alpha
            self.V[state_key] += v_lr * (G - self.V[state_key])
            
            # Q 値の更新 (ベースラインを引く)
            self.N[state_key][action] += 1
            n = self.N[state_key][action]
            lr = 1.0 / n if self.alpha is None else self.alpha
            
            # δ = G - V(s) をターゲットに使用
            baseline = self.V[state_key]
            advantage = G - baseline
            old_q = self.Q[state_key][action]
            self.Q[state_key][action] = old_q + lr * (G - old_q)
            
            updates += 1
            self.total_updates += 1
        
        self.episode_count += 1
        
        return {
            "total_return": G,
            "num_updates": updates,
            "episode_length": len(episode)
        }


# デモ用コード
if __name__ == "__main__":
    from env import AkariEnv
    
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
    agent = MonteCarloAgent(epsilon=0.3, gamma=1.0, alpha=0.1)
    
    print("=== Monte Carlo Agent Demo ===\n")
    
    # いくつかのエピソードを実行
    for i in range(5):
        episode = agent.generate_episode(env)
        stats = agent.update_from_episode(episode)
        print(f"Episode {i+1}:")
        print(f"  Length: {stats['episode_length']}")
        print(f"  Total return: {stats['total_return']:.2f}")
    
    print(f"\n--- Agent Stats ---")
    for k, v in agent.get_stats().items():
        print(f"  {k}: {v}")
    
    # 評価
    print("\n--- Evaluation (greedy policy) ---")
    eval_stats = agent.evaluate(env, num_episodes=10)
    for k, v in eval_stats.items():
        print(f"  {k}: {v:.3f}")
