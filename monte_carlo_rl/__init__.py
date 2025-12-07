"""
Monte Carlo RL Solver for Akari/Light Up Puzzle

このパッケージは、モンテカルロ強化学習を用いた Akari パズルソルバを提供します。

モジュール:
- env.py: AkariEnv (Gym互換の環境)
- mc_agent.py: MonteCarloAgent (Q学習エージェント)
- train.py: 学習スクリプト

使用例:
-------
from monte_carlo_rl import AkariEnv, MonteCarloAgent

# 環境作成
puzzle = {"width": 5, "height": 5, "grid": [...]}
env = AkariEnv(puzzle)

# エージェント作成
agent = MonteCarloAgent(epsilon=0.3)

# 学習
for _ in range(1000):
    episode = agent.generate_episode(env)
    agent.update_from_episode(episode)
    agent.decay_epsilon()
"""

from .env import AkariEnv, AkariState, Board
from .mc_agent import MonteCarloAgent, MonteCarloAgentWithBaseline

__all__ = [
    "AkariEnv",
    "AkariState",
    "Board",
    "MonteCarloAgent",
    "MonteCarloAgentWithBaseline",
]
