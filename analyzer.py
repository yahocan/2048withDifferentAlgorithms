import copy
import random
from expectimax2048 import Game2048 as ExpectimaxGame
from hillclimbing2048 import Game2048 as HillClimbingGame


def analyze_expectimax_rollouts():
    """Monte Carlo Rollout sayısının Expectimax performansına etkisini analiz eder."""
    print("Expectimax Rollout Analizi")
    print("--------------------------")

    # Farklı rollout değerleri ile test et
    rollout_values = [5, 10, 15, 20]

    for rollout in rollout_values:
        # Test süresi kısa tutmak için daha az hamle simüle et
        game = ExpectimaxGame(run_without_gui=True)

        # Monte Carlo rollout değerini geçersiz kıl
        original_monte_carlo = game.monte_carlo_expectimax

        def test_monte_carlo(self, depth, agent_type, test_grid, rollouts=rollout):
            return original_monte_carlo(depth, agent_type, test_grid, rollouts)

        # Test fonksiyonunu geçici olarak yerleştir
        game.monte_carlo_expectimax = test_monte_carlo.__get__(game, ExpectimaxGame)

        # 20 hamle yap ve heuristikleri hesapla
        total_empty_count = 0
        total_mono_score = 0
        total_smooth_score = 0
        total_corner_score = 0
        total_clustering = 0
        total_ratio = 0
        total_merge = 0

        moves = 0
        while moves < 20 and game.can_move(game.grid):
            best_move = game.get_best_move(depth=2)
            if best_move:
                best_move()
                game.add_new_tile()

                # Skorları hesapla
                empty_count = sum(row.count(0) for row in game.grid) * 10000
                mono_score = game._calculate_monotonicity()
                smooth_score = game._calculate_smoothness()
                corner_score = game._calculate_corner_score()
                clustering = game._calculate_tile_clustering() * 50
                ratio = game._calculate_empty_tile_ratio() * 2000
                merge = game._calculate_merge_potential() * 1000

                total_empty_count += empty_count
                total_mono_score += mono_score
                total_smooth_score += smooth_score
                total_corner_score += corner_score
                total_clustering += clustering
                total_ratio += ratio
                total_merge += merge

                moves += 1

        # Ortalama skorları yazdır
        print(f"Rollout: {rollout}")
        print(f"  Ortalama Empty Count: {total_empty_count/moves:.2f}")
        print(f"  Ortalama Monotonicity: {total_mono_score/moves:.2f}")
        print(f"  Ortalama Smoothness: {total_smooth_score/moves:.2f}")
        print(f"  Ortalama Corner Score: {total_corner_score/moves:.2f}")
        print(f"  Ortalama Clustering: {total_clustering/moves:.2f}")
        print(f"  Ortalama Empty Tile Ratio: {total_ratio/moves:.2f}")
        print(f"  Ortalama Merge Potential: {total_merge/moves:.2f}")
        print()


def analyze_hill_climbing_heuristics():
    """Hill Climbing heuristiklerinin etkisini analiz eder."""
    print("Hill Climbing Heuristic Analizi")
    print("-------------------------------")

    # Farklı ağırlıklar test et
    weights = [
        {"empty": 100, "mono": 1 / 10, "smooth": 5, "max_place": 10},  # Mevcut
        {"empty": 200, "mono": 1 / 5, "smooth": 2, "max_place": 5},  # Alternatif 1
        {"empty": 50, "mono": 1 / 20, "smooth": 10, "max_place": 20},  # Alternatif 2
    ]

    for i, weight in enumerate(weights):
        game = HillClimbingGame(run_without_gui=True)

        # Evaluate fonksiyonunu geçersiz kıl
        original_evaluate = game.evaluate

        def test_evaluate(self):
            empty_cell_score = sum(row.count(0) for row in self.grid) * weight["empty"]
            monotonicity_score = self._calculate_monotonicity() * weight["mono"]
            smoothness_score = self._calculate_smoothness() * weight["smooth"]
            max_tile_placement_score = (
                self._evaluate_max_tile_placement() * weight["max_place"]
            )

            return (
                empty_cell_score
                + monotonicity_score
                + smoothness_score
                + max_tile_placement_score
            )

        # Test fonksiyonunu geçici olarak yerleştir
        game.evaluate = test_evaluate.__get__(game, HillClimbingGame)

        # 20 hamle yap ve heuristikleri hesapla
        moves_up = 0
        moves_down = 0
        moves_left = 0
        moves_right = 0

        moves = 0
        while moves < 20 and game.can_move():
            best_move = game.get_best_move()
            if best_move:
                if best_move == game.move_up:
                    moves_up += 1
                elif best_move == game.move_down:
                    moves_down += 1
                elif best_move == game.move_left:
                    moves_left += 1
                elif best_move == game.move_right:
                    moves_right += 1

                best_move()
                game.add_new_tile()
                moves += 1

        # Hamle dağılımını yazdır
        print(f"Ağırlık Seti {i+1}: {weight}")
        print(f"  Yukarı: {moves_up}")
        print(f"  Aşağı: {moves_down}")
        print(f"  Sol: {moves_left}")
        print(f"  Sağ: {moves_right}")
        print()


if __name__ == "__main__":
    analyze_expectimax_rollouts()
    analyze_hill_climbing_heuristics()
