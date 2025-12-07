"""
Akari Bucket Brigade Solver - Training and Evaluation Script

================================================================================
学習スクリプト
================================================================================

このスクリプトは、バケツリレー型強化学習エージェントを使用して
Akari パズルを解くための学習と評価を行う。

使用方法:
    python train.py                     # デフォルト設定で学習
    python train.py --puzzle puzzle.json  # 指定パズルで学習
    python train.py --episodes 5000     # エピソード数を指定

================================================================================
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any

# 同じディレクトリからインポート
from env import AkariEnv
from classifiers import BucketBrigadeAgent, ClassifierPopulation


def load_puzzle(filepath: str) -> Dict:
    """JSONファイルからパズルを読み込む"""
    with open(filepath, 'r') as f:
        return json.load(f)


def run_training(puzzles: List[Dict],
                 episodes_per_puzzle: int = 1000,
                 max_steps: int = 200,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 epsilon: float = 0.2,
                 epsilon_decay: float = 0.999,
                 seed: int = 42,
                 verbose: bool = True) -> BucketBrigadeAgent:
    """
    学習を実行
    
    Args:
        puzzles: パズル JSON のリスト
        episodes_per_puzzle: パズルあたりのエピソード数
        max_steps: 1エピソードの最大ステップ数
        learning_rate: 学習率
        discount_factor: 割引率
        epsilon: 初期探索率
        epsilon_decay: 探索率の減衰率
        seed: 乱数シード
        verbose: 進捗表示
    
    Returns:
        学習済みエージェント
    """
    agent = BucketBrigadeAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        seed=seed
    )
    
    if verbose:
        print("=" * 60)
        print("Akari Bucket Brigade Solver - Training")
        print("=" * 60)
        print(f"Number of puzzles: {len(puzzles)}")
        print(f"Episodes per puzzle: {episodes_per_puzzle}")
        print(f"Max steps per episode: {max_steps}")
        print(f"Learning rate: {learning_rate}")
        print(f"Discount factor: {discount_factor}")
        print(f"Initial epsilon: {epsilon}")
        print(f"Epsilon decay: {epsilon_decay}")
        print(f"Random seed: {seed}")
        print("=" * 60)
        print()
    
    start_time = time.time()
    
    results = agent.train(
        puzzles=puzzles,
        episodes_per_puzzle=episodes_per_puzzle,
        max_steps=max_steps,
        verbose=verbose
    )
    
    elapsed = time.time() - start_time
    
    if verbose:
        print()
        print("=" * 60)
        print("Training Complete")
        print("=" * 60)
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Total episodes: {results['total_episodes']}")
        print(f"Total solved: {results['total_solved']}")
        print(f"Overall solve rate: {results['total_solved'] / max(1, results['total_episodes']):.2%}")
        print()
        
        # パズルごとの結果
        print("Per-puzzle results:")
        for pr in results["puzzle_results"]:
            print(f"  Puzzle {pr['puzzle_idx'] + 1}: "
                  f"{pr['solved']}/{pr['episodes']} ({pr['solve_rate']:.2%})")
        print()
        
        # ルール統計
        print("Classifier strengths after training:")
        stats = agent.get_stats()["classifier_stats"]
        for name, s in sorted(stats.items(), key=lambda x: -x[1]["strength"]):
            success = s.get('success', s.get('correct', 0))
            failure = s.get('failure', 0)
            success_rate = s.get('success_rate', success / max(1, success + failure))
            print(f"  {name:30s}: strength={s['strength']:.3f}, "
                  f"selected={s['selected']}, success_rate={success_rate:.2%}")
    
    return agent


def evaluate_agent(agent: BucketBrigadeAgent,
                   puzzles: List[Dict],
                   max_steps: int = 200,
                   use_backtrack: bool = True,
                   verbose: bool = True) -> Dict[str, Any]:
    """
    学習済みエージェントを評価
    
    Args:
        agent: 学習済みエージェント
        puzzles: 評価用パズルのリスト
        max_steps: 最大ステップ数
        use_backtrack: バックトラッキングを使用するか
        verbose: 詳細表示
    
    Returns:
        評価結果
    """
    results = {
        "total": len(puzzles),
        "solved_greedy": 0,
        "solved_backtrack": 0,
        "details": []
    }
    
    for idx, puzzle in enumerate(puzzles):
        env = AkariEnv(puzzle)
        
        # Greedy評価
        solved_greedy, final_env_greedy = agent.solve(env, max_steps=max_steps)
        
        # バックトラック評価
        solved_backtrack = False
        final_env_backtrack = None
        if use_backtrack:
            solved_backtrack, final_env_backtrack = agent.solve_with_backtrack(
                env, max_depth=max_steps, max_backtracks=5000
            )
        
        results["details"].append({
            "puzzle_idx": idx,
            "solved_greedy": solved_greedy,
            "solved_backtrack": solved_backtrack
        })
        
        if solved_greedy:
            results["solved_greedy"] += 1
        if solved_backtrack:
            results["solved_backtrack"] += 1
        
        if verbose:
            greedy_status = "✓" if solved_greedy else "✗"
            backtrack_status = "✓" if solved_backtrack else "✗"
            print(f"Puzzle {idx + 1}: Greedy={greedy_status}, Backtrack={backtrack_status}")
            if solved_backtrack and final_env_backtrack:
                print(final_env_backtrack.render())
                print()
            elif solved_greedy:
                print(final_env_greedy.render())
                print()
    
    results["solve_rate_greedy"] = results["solved_greedy"] / max(1, results["total"])
    results["solve_rate_backtrack"] = results["solved_backtrack"] / max(1, results["total"])
    
    if verbose:
        print()
        print(f"Greedy: {results['solved_greedy']}/{results['total']} "
              f"({results['solve_rate_greedy']:.2%})")
        if use_backtrack:
            print(f"Backtrack: {results['solved_backtrack']}/{results['total']} "
                  f"({results['solve_rate_backtrack']:.2%})")
    
    return results


def demo_step_by_step(puzzle: Dict, agent: BucketBrigadeAgent, max_steps: int = 50):
    """
    エージェントの動作をステップごとに表示するデモ
    
    Args:
        puzzle: パズル JSON
        agent: 学習済みエージェント
        max_steps: 最大ステップ数
    """
    env = AkariEnv(puzzle)
    obs = env.reset()
    
    print("=" * 60)
    print("Step-by-Step Demo")
    print("=" * 60)
    print("\nInitial state:")
    print(env.render())
    print()
    
    for step in range(max_steps):
        if env.is_solved():
            print("=" * 60)
            print("SOLVED!")
            print("=" * 60)
            break
        
        action_cand = agent.population.select_action(env, epsilon=0.0)
        
        if action_cand is None:
            print("No legal action available")
            break
        
        target_r, target_c = action_cand.target_r, action_cand.target_c
        action_type = action_cand.action_type
        clf = action_cand.classifier
        
        cell_idx = env.get_cell_index(target_r, target_c)
        
        if cell_idx < 0:
            print("Invalid target cell")
            break
        
        action_name = "BULB" if action_type == 0 else "BLOCK"
        rule_name = clf.rule_type.name if clf else "RANDOM"
        
        print(f"Step {step + 1}: Apply {action_name} at ({target_r}, {target_c})")
        print(f"  Rule: {rule_name}")
        if clf:
            print(f"  Strength: {clf.strength:.3f}")
        
        obs, reward, done, info = env.step((cell_idx, action_type))
        
        print(f"  Reward: {reward:.2f}")
        print()
        print(env.render())
        print()
        
        if done:
            if env.is_solved():
                print("=" * 60)
                print("SOLVED!")
                print("=" * 60)
            else:
                print(f"Episode ended: {info.get('status', 'unknown')}")
            break
    
    return env.is_solved()


def run_tests():
    """
    テストコード: reset(), step() の動作確認
    """
    print("=" * 60)
    print("Running Tests")
    print("=" * 60)
    print()
    
    # サンプルパズル
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
    
    # Test 1: 環境の初期化と reset
    print("Test 1: Environment initialization and reset")
    env = AkariEnv(puzzle)
    obs = env.reset()
    print(f"  Width: {env.width}, Height: {env.height}")
    print(f"  White cells: {len(env.white_cells)}")
    print(f"  Digit cells: {len(env.digit_cells)}")
    print(f"  Initial state:")
    print(env.render())
    print()
    
    # Test 2: 合法行動の取得
    print("Test 2: Get legal actions")
    actions = env.get_legal_actions()
    print(f"  Legal actions: {len(actions)}")
    print(f"  First 5 actions: {actions[:5]}")
    print()
    
    # Test 3: step() の実行
    print("Test 3: Execute step()")
    if actions:
        action = actions[0]
        cell_idx, action_type = action
        r, c = env.white_cells[cell_idx]
        action_name = "BULB" if action_type == 0 else "BLOCK"
        print(f"  Executing: {action_name} at cell {cell_idx} ({r}, {c})")
        
        obs, reward, done, info = env.step(action)
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        print(f"  Info: {info}")
        print(f"  State after step:")
        print(env.render())
    print()
    
    # Test 4: ClassifierPopulation の動作
    print("Test 4: ClassifierPopulation action candidates")
    env.reset()  # リセット
    population = ClassifierPopulation()
    print(f"  Total classifiers: {len(population.classifiers)}")
    
    candidates = population.get_action_candidates(env)
    print(f"  Action candidates: {len(candidates)}")
    
    if candidates:
        print(f"  Top 3 candidates by bid:")
        candidates.sort(key=lambda x: -x.bid)
        for cand in candidates[:3]:
            print(f"    ({cand.target_r},{cand.target_c}) bid={cand.bid:.2f} rule={cand.classifier.rule_type.name}")
    print()
    
    # Test 5: エージェントによる1エピソード
    print("Test 5: Run one episode with agent")
    env.reset()
    agent = BucketBrigadeAgent(seed=42)
    solved, steps, reward = agent.run_episode(env, max_steps=100, train=False)
    print(f"  Solved: {solved}")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {reward:.2f}")
    print(f"  Final state:")
    print(env.render())
    print()
    
    # Test 6: バックトラック付き解法
    print("Test 6: Solve with backtracking")
    solved, final_env = agent.solve_with_backtrack(env, max_nodes=100000)
    print(f"  Solved: {solved}")
    print(f"  Final state:")
    print(final_env.render())
    print()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Akari Bucket Brigade Solver - Training and Evaluation"
    )
    parser.add_argument(
        "--puzzle", "-p",
        type=str,
        default=None,
        help="Path to puzzle JSON file"
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=2000,
        help="Number of episodes per puzzle (default: 2000)"
    )
    parser.add_argument(
        "--max-steps", "-s",
        type=int,
        default=200,
        help="Maximum steps per episode (default: 200)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="Discount factor (default: 0.9)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="Initial exploration rate (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run tests instead of training"
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run step-by-step demo"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
        return
    
    # パズルの読み込み
    if args.puzzle:
        puzzles = [load_puzzle(args.puzzle)]
    else:
        # デフォルトのサンプルパズル
        puzzles = [{
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
        }]
    
    # 学習
    agent = run_training(
        puzzles=puzzles,
        episodes_per_puzzle=args.episodes,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        discount_factor=args.gamma,
        epsilon=args.epsilon,
        seed=args.seed,
        verbose=not args.quiet
    )
    
    # デモモード
    if args.demo:
        print()
        demo_step_by_step(puzzles[0], agent)
    
    # 最終評価
    print()
    print("Final evaluation (greedy):")
    evaluate_agent(agent, puzzles, max_steps=args.max_steps, verbose=True)


if __name__ == "__main__":
    main()
