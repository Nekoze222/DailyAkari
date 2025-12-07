#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py: モンテカルロ強化学習によるAkariソルバの学習スクリプト

使用例:
-------
# 単一パズルで学習
python train.py puzzle.json --episodes 5000

# 複数パズルで学習
python train.py puzzle.json puzzle2.json puzzle3.json --episodes 10000

# パラメータ指定
python train.py puzzle.json --episodes 5000 --epsilon 0.2 --alpha 0.1
"""

from __future__ import annotations
import argparse
import json
import os
import random
import time
from typing import List, Dict, Any

from env import AkariEnv
from mc_agent import MonteCarloAgent


def load_puzzles(json_paths: List[str]) -> List[dict]:
    """複数のパズル JSON ファイルを読み込む"""
    puzzles = []
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            puzzle = json.load(f)
            puzzle["_path"] = path  # デバッグ用にパスを保存
            puzzles.append(puzzle)
    return puzzles


def train(
    agent: MonteCarloAgent,
    puzzles: List[dict],
    num_episodes: int = 5000,
    eval_interval: int = 100,
    eval_episodes: int = 10,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    モンテカルロ制御による学習ループ
    
    Args:
        agent: モンテカルロエージェント
        puzzles: 学習用パズルのリスト
        num_episodes: 総エピソード数
        eval_interval: 評価間隔
        eval_episodes: 評価時のエピソード数
        verbose: 詳細出力
    
    Returns:
        学習履歴 (returns, success_rates, etc.)
    """
    history = {
        "episode": [],
        "return": [],
        "success_rate": [],
        "epsilon": [],
        "episode_length": []
    }
    
    env = AkariEnv()
    start_time = time.time()
    
    # 各パズル用の環境をキャッシュ
    envs = [AkariEnv(p) for p in puzzles]
    
    for episode in range(1, num_episodes + 1):
        # パズルをランダムに選択
        puzzle_idx = random.randint(0, len(puzzles) - 1)
        current_env = envs[puzzle_idx]
        
        # エピソード生成と Q 更新
        ep_data = agent.generate_episode(current_env)
        stats = agent.update_from_episode(ep_data)
        
        # ε 減衰
        agent.decay_epsilon()
        
        # 定期評価
        if episode % eval_interval == 0:
            # 全パズルで評価
            total_success = 0
            total_return = 0.0
            total_steps = 0
            
            for eval_env in envs:
                eval_stats = agent.evaluate(
                    eval_env,
                    num_episodes=eval_episodes,
                    max_steps=500
                )
                total_success += eval_stats["success_rate"] * eval_episodes
                total_return += eval_stats["avg_return"] * eval_episodes
                total_steps += eval_stats["avg_steps"] * eval_episodes
            
            n_total = len(envs) * eval_episodes
            avg_success = total_success / n_total
            avg_return = total_return / n_total
            avg_steps = total_steps / n_total
            
            history["episode"].append(episode)
            history["return"].append(avg_return)
            history["success_rate"].append(avg_success)
            history["epsilon"].append(agent.epsilon)
            history["episode_length"].append(stats["episode_length"])
            
            if verbose:
                elapsed = time.time() - start_time
                print(
                    f"Episode {episode:5d} | "
                    f"Success: {avg_success:.2%} | "
                    f"Return: {avg_return:8.2f} | "
                    f"ε: {agent.epsilon:.4f} | "
                    f"States: {len(agent.Q):6d} | "
                    f"Time: {elapsed:.1f}s"
                )
    
    return history


def demo_solution(agent: MonteCarloAgent, puzzle: dict, verbose: bool = True) -> bool:
    """
    学習済みエージェントでパズルを解く（デモ表示）
    
    Returns:
        成功したかどうか
    """
    env = AkariEnv(puzzle)
    obs = env.reset()
    
    if verbose:
        print("\n" + "=" * 50)
        print("Initial state:")
        print(env.render())
        print()
    
    state_key = env.get_state_key()
    step = 0
    
    while True:
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            break
        
        # Greedy 選択
        action = agent.select_action(state_key, legal_actions, greedy=True)
        cell_idx = action // 2
        action_type = "BULB" if action % 2 == 1 else "BLOCK"
        
        obs, reward, done, info = env.step(action)
        step += 1
        
        if verbose:
            r, c = divmod(cell_idx, env.board.W)
            print(f"Step {step}: ({r},{c}) = {action_type}, reward = {reward:.1f}")
        
        if done:
            if verbose:
                print("\n" + "-" * 50)
                print("Final state:")
                print(env.render())
                print()
                if info.get("success", False):
                    print("✅ SOLVED!")
                else:
                    print(f"❌ Failed: {info.get('reason', 'unknown')}")
            return info.get("success", False)
        
        state_key = env.get_state_key()
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Train Monte Carlo RL agent for Akari puzzle"
    )
    parser.add_argument(
        "json_paths",
        nargs="+",
        help="Path(s) to puzzle JSON file(s)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="Number of training episodes (default: 5000)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.3,
        help="Initial exploration rate (default: 0.3)"
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.9995,
        help="Epsilon decay rate per episode (default: 0.9995)"
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.01,
        help="Minimum epsilon (default: 0.01)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Learning rate (default: None = sample average)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Discount factor (default: 1.0)"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Evaluation interval (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Show demo solution after training"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress training progress output"
    )
    
    args = parser.parse_args()
    
    # 乱数シード設定
    random.seed(args.seed)
    
    # パズル読み込み
    print(f"Loading {len(args.json_paths)} puzzle(s)...")
    puzzles = load_puzzles(args.json_paths)
    
    for p in puzzles:
        print(f"  - {p.get('_path', 'unknown')}: {p['width']}x{p['height']}")
    
    # エージェント作成
    agent = MonteCarloAgent(
        epsilon=args.epsilon,
        gamma=args.gamma,
        alpha=args.alpha,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min
    )
    
    print(f"\n{'='*50}")
    print("Training Configuration:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Epsilon: {args.epsilon} (decay: {args.epsilon_decay}, min: {args.epsilon_min})")
    print(f"  Gamma: {args.gamma}")
    print(f"  Alpha: {args.alpha if args.alpha else 'sample average'}")
    print(f"  Seed: {args.seed}")
    print(f"{'='*50}\n")
    
    # 学習
    history = train(
        agent=agent,
        puzzles=puzzles,
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        verbose=not args.quiet
    )
    
    # 最終統計
    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"{'='*50}")
    agent_stats = agent.get_stats()
    print(f"Final epsilon: {agent_stats['epsilon']:.6f}")
    print(f"Total episodes: {agent_stats['episode_count']}")
    print(f"Total updates: {agent_stats['total_updates']}")
    print(f"Unique states visited: {agent_stats['num_states']}")
    print(f"State-action pairs: {agent_stats['num_state_action_pairs']}")
    
    if history["success_rate"]:
        print(f"\nFinal success rate: {history['success_rate'][-1]:.2%}")
        print(f"Final avg return: {history['return'][-1]:.2f}")
    
    # デモ
    if args.demo:
        print(f"\n{'='*50}")
        print("Demo: Solving puzzles with learned policy")
        print(f"{'='*50}")
        
        for i, puzzle in enumerate(puzzles):
            print(f"\n--- Puzzle {i+1}: {puzzle.get('_path', 'unknown')} ---")
            success = demo_solution(agent, puzzle, verbose=True)
    
    return agent, history


if __name__ == "__main__":
    main()
