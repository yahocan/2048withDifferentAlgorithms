import time
import csv
from astar2048 import Game2048 as AStar2048
from expectimax2048 import Game2048 as Expectimax2048
from greedy2048 import Game2048 as Greedy2048
from hillclimbing2048 import Game2048 as HillClimbing2048
from minimax2048 import Game2048 as Minimax2048

ALGORITHMS = {
    "A*": AStar2048,
    "Expectimax": Expectimax2048,
    "Greedy": Greedy2048,
    "Hill Climbing": HillClimbing2048,
    "Minimax": Minimax2048,
}

NUM_GAMES = 100
RESULTS_FILE = "test_results.csv"


def run_test(algorithm_name, algorithm_class):
    scores = []
    max_tiles = []
    times = []

    for i in range(NUM_GAMES):
        start_time = time.time()
        game = algorithm_class(
            run_without_gui=True
        )  # Use the run_without_gui parameter
        final_score, max_tile = game.run()  # Call the run method
        end_time = time.time()

        scores.append(final_score)
        max_tiles.append(max_tile)
        times.append(end_time - start_time)

        if i % 10 == 0:  # Progress update every 10 games
            print(f"  Progress: {i}/{NUM_GAMES} games completed")

    return scores, max_tiles, times


def main():
    with open(RESULTS_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Algorithm", "Avg Score", "Max Score", "Avg Max Tile", "Avg Time"]
        )

        for name, algo in ALGORITHMS.items():
            print(f"Running {name}...")
            scores, max_tiles, times = run_test(name, algo)
            writer.writerow(
                [
                    name,
                    sum(scores) / NUM_GAMES,
                    max(scores),
                    sum(max_tiles) / NUM_GAMES,
                    sum(times) / NUM_GAMES,
                ]
            )
            print(f"Finished {name}.")


if __name__ == "__main__":
    main()
