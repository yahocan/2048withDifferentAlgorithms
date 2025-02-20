# batch_test.py
import time
import csv
from astar2048 import Game2048 as AStarGame
from expectimax2048 import Game2048 as ExpectimaxGame
from greedy2048 import Game2048 as GreedyGame
from hillclimbing2048 import Game2048 as HillClimbingGame
from minimax2048 import Game2048 as MinimaxGame


def run_tests(game_class, num_runs=50):
    """Algoritma testlerini çalıştırır ve performans metriklerini döndürür."""
    scores = []
    max_tiles = []
    durations = []

    for _ in range(num_runs):
        try:
            game = game_class()
            start_time = time.time()

            # Oyunu otomatik oynat (GUI'siz mod için)
            while True:
                if not game.autoplay():
                    break

            end_time = time.time()

            # Metrikleri kaydet
            scores.append(game.score)
            max_tile = max(max(row) for row in game.grid)
            max_tiles.append(max_tile)
            durations.append(end_time - start_time)

        except Exception as e:
            print(f"Hata: {str(e)}")
            continue

    return {
        "Ortalama Skor": sum(scores) / len(scores),
        "Maks Skor": max(scores),
        "Ortalama Tile": sum(max_tiles) / len(max_tiles),
        "Maks Tile": max(max_tiles),
        "Ortalama Süre": sum(durations) / len(durations),
    }


def main():
    # Test edilecek algoritmalar
    algorithms = {
        "A*": AStarGame,
        "Expectimax": ExpectimaxGame,
        "Greedy": GreedyGame,
        "Hill Climbing": HillClimbingGame,
        "Minimax": MinimaxGame,
    }

    # Sonuçları saklamak için sözlük
    results = {}

    # Her algoritmayı test et
    for alg_name, alg_class in algorithms.items():
        print(f"{alg_name} testi başladı...")
        results[alg_name] = run_tests(alg_class)
        print(f"{alg_name} testi tamamlandı.\n")

    # Sonuçları göster
    print("\n\n=== Performans Karşılaştırması ===")
    print(
        f"{'Algoritma':<15} | {'Ort. Skor':<10} | {'Maks Skor':<10} | {'Ort. Tile':<10} | {'Maks Tile':<10} | {'Ort. Süre (sn)':<15}"
    )
    print("-" * 85)
    for alg, data in results.items():
        print(
            f"{alg:<15} | {data['Ortalama Skor']:<10.1f} | {data['Maks Skor']:<10} | {data['Ortalama Tile']:<10.1f} | {data['Maks Tile']:<10} | {data['Ortalama Süre']:<15.2f}"
        )

    # CSV'ye kaydet
    with open("sonuclar.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Algoritma",
                "Ortalama Skor",
                "Maks Skor",
                "Ortalama Tile",
                "Maks Tile",
                "Ortalama Süre (sn)",
            ]
        )
        for alg, data in results.items():
            writer.writerow(
                [
                    alg,
                    data["Ortalama Skor"],
                    data["Maks Skor"],
                    data["Ortalama Tile"],
                    data["Maks Tile"],
                    data["Ortalama Süre"],
                ]
            )


if __name__ == "__main__":
    main()
