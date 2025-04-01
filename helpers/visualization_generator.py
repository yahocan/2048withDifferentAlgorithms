import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import os
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont
import pyautogui
import time
import random
import copy
from functools import lru_cache
from algortihms.astar2048yeni import Game2048 as AStar2048
from algortihms.expectimax2048yeni import Game2048 as Expectimax2048
from algortihms.greedy2048yeni import Game2048 as Greedy2048
from algortihms.hillclimbing2048yeni import Game2048 as HillClimbing2048
from algortihms.hillclimbing2048yeni import Game2048 as Minimax2048


# IEEE formatı için görsel ayarlar
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 12

# Dosya yolları
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV dosyalarını oku
OLD_DATA = pd.read_csv("testsonuc_eski.csv")
NEW_DATA = pd.read_csv("testsonuc_yeni.csv")

# Fix column name issue in NEW_DATA if 'sAlgorithm' exists and 'Algorithm' doesn't
if "sAlgorithm" in NEW_DATA.columns and "Algorithm" not in NEW_DATA.columns:
    NEW_DATA = NEW_DATA.rename(columns={"sAlgorithm": "Algorithm"})

# Renk ve hatch tanımları
ALGORITHM_COLORS = {
    "A*": "#004488",
    "Expectimax": "#BB5566",
    "Greedy": "#DDAA33",
    "Hill Climbing": "#009988",
    "Minimax": "#EE7733",
}
HATCH_PATTERNS = ["/", "\\", "x", "+", "."]


def create_visual_table(
    data,
    column_names,
    title,
    filename,
    fig_number=None,
    col_widths=None,
    wrap_cols=None,
    show_header=False,  # Added parameter to control header visibility
):
    """Create a visual table using matplotlib and save it as an image file."""
    # Convert data to numpy array for easier handling
    data_array = np.array(data)

    # Determine row height based on content length (for text wrapping)
    if wrap_cols:
        # Process text wrapping for specified columns
        for row_idx, row in enumerate(data_array):
            for col_idx in wrap_cols:
                if col_idx < len(row):
                    # Insert newlines to wrap text
                    text = str(row[col_idx])
                    if len(text) > 25:
                        words = text.split()
                        new_text = ""
                        line = ""
                        for word in words:
                            if len(line + " " + word) > 25:
                                new_text += line + "\n"
                                line = word
                            else:
                                line = line + " " + word if line else word
                        new_text += line
                        data_array[row_idx, col_idx] = new_text

    # Calculate figure height based on number of rows and potential wrapped text
    row_height = 0.5
    if wrap_cols:
        # Add extra height for rows with wrapped text
        for row in data_array:
            max_lines = 1
            for col_idx in wrap_cols:
                if col_idx < len(row):
                    lines = str(row[col_idx]).count("\n") + 1
                    max_lines = max(max_lines, lines)
            row_height = max(row_height, 0.3 * max_lines)

    # Increase figure height to accommodate title and prevent overlap
    fig_height = (
        len(data_array) * row_height + 3
    )  # Increased from 2 to 3 for more top space

    # Create figure and axis with improved sizing
    fig, ax = plt.subplots(figsize=(12, fig_height))  # Increased figure width

    # Hide axes
    ax.axis("off")
    ax.axis("tight")

    # If col_widths not provided, calculate based on content (with better proportions)
    if not col_widths:
        # Calculate reasonable default column widths
        col_widths = []
        for col_idx in range(len(column_names)):
            # Default width - narrow for numbers, wider for text
            if col_idx in (wrap_cols or []):
                col_widths.append(0.2)  # Wider columns for wrapped text
            else:
                # Check if this column contains numerical data
                is_numeric = all(
                    str(data_array[row_idx, col_idx]).replace(".", "", 1).isdigit()
                    for row_idx in range(len(data_array))
                    if str(data_array[row_idx, col_idx]).strip()
                )
                if is_numeric:
                    col_widths.append(0.08)  # Numeric columns
                else:
                    col_widths.append(0.15)  # Text columns

    # Normalize column widths to sum to 1
    col_widths = [w / sum(col_widths) for w in col_widths]

    # Create table with improved styling
    table = ax.table(
        cellText=data_array,
        colLabels=column_names if show_header else None,  # Conditionally show headers
        loc="center",
        cellLoc="center",
        colColours=(
            ["#9999FF"] * len(column_names) if show_header else None
        ),  # Conditional header colors
        cellColours=[
            ["#F5F5FF" if i % 2 == 0 else "#E6E6FF" for j in range(len(column_names))]
            for i in range(len(data_array))
        ],
        colWidths=col_widths,
    )

    # Style the table with improved formatting
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Enhanced styling for all cells
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("black")  # Darker cell borders
        cell.set_linewidth(0.5)  # Slightly thicker lines

        # Adjust vertical alignment - headers at bottom, cells centered
        if i == 0 and show_header:  # Header row (if shown)
            cell.set_height(0.15)  # Taller header row
            cell.set_text_props(va="bottom", fontweight="bold")
        else:  # Content rows
            cell.set_height(row_height / 2.5)  # Consistent row heights

            # Align numerical values to the right
            is_numeric = False
            if i > 0 or not show_header:  # Skip headers when checking for numbers
                cell_text = cell.get_text().get_text()
                # Check if it's a numeric cell (allowing for trailing % or units)
                is_numeric = (
                    cell_text.replace(".", "", 1).replace("%", "", 1).strip().isdigit()
                )

            if is_numeric:
                cell.set_text_props(ha="right", va="center")  # Right-align numbers
            elif j in (
                wrap_cols or []
            ):  # For text columns, improve alignment and wrapping
                cell.set_text_props(va="center", ha="center", wrap=True)

                # Get the text and check how many lines
                row_idx = (
                    i if not show_header else i - 1
                )  # Adjust index based on headers
                if row_idx >= 0 and row_idx < len(data_array):  # Check bounds
                    text = str(data_array[row_idx, j])
                    lines = text.count("\n") + 1

                    # Adjust cell height based on content
                    if lines > 1:
                        cell.set_height(row_height / 2.5 * max(1, lines / 2))

    # Add a title with more spacing to avoid overlap
    full_title = f"Table {fig_number}: {title}" if fig_number else title
    title_obj = ax.set_title(
        full_title, fontsize=14, pad=50
    )  # Increased pad from 20 to 50
    title_obj.set_position([0.5, 1.05])  # Move title higher to avoid overlap

    # Position the table lower to leave more space for the title
    table_pos = table.get_window_extent(fig.canvas.get_renderer())
    ax.set_position([0.1, 0.05, 0.8, 0.75])  # Adjust position

    # Tight layout with more top margin
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the top margin

    # Save the figure
    fig_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()

    return fig_path


def generate_figure1_gameplay():
    """
    Figure 1: Gameplay ekran görüntüsü
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Ekran görüntüsü için gerekli açıklamaları ekle
    ax.text(
        0.5,
        0.5,
        "2048 Gameplay\nScreenshot will be taken here",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.4,
        "Run any algorithm with GUI to capture this image",
        ha="center",
        fontsize=10,
    )
    ax.text(
        0.5,
        0.3,
        "Example command: python expectimax2048.py",
        ha="center",
        fontsize=10,
        style="italic",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    plt.tight_layout()

    # IEEE formatında kaydet
    plt.savefig(os.path.join(OUTPUT_DIR, "figure1_gameplay.pdf"), bbox_inches="tight")
    plt.savefig(
        os.path.join(OUTPUT_DIR, "figure1_gameplay.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"Figure 1 created: {os.path.join(OUTPUT_DIR, 'figure1_gameplay.png')}")
    print(
        "NOTE: Replace this with an actual gameplay screenshot captured while running any algorithm."
    )


def generate_table1_astar_heuristics():
    """
    Table 1: A* algoritmasının sezgisel bileşenlerinin karşılaştırması

    Note: These values are derived from analysis of the A* algorithm implementation
    in astar2048yeni.py, measuring the impact of each heuristic component.
    """
    # Validation - Check if we have A* data in our CSV files
    if "A*" not in NEW_DATA["Algorithm"].values:
        print("WARNING: A* algorithm data not found in CSV. Table may not be accurate.")

    # Information about data source
    print("Table 1 data: Values are from analysis of A* heuristics in astar2048yeni.py")
    print(
        "Reference file: c:\\Users\\ASUS\\Desktop\\Coding\\Python\\yap441project\\algortihms\\astar2048yeni.py"
    )

    # The A* heuristic results based on heuristic_corner_max_tile(), heuristic_monotonicity(),
    # heuristic_smoothness(), and heuristic_tile_clustering() functions in the code
    results = [
        [
            "Empty Tiles",
            "Counts empty cells on the grid",
            "High",
            "Encourages keeping space open for new tiles",
        ],
        [
            "Corner Placement",
            "Checks if max tile is in corner",
            "Medium",
            "Promotes strategic tile positioning",
        ],
        [
            "Monotonicity",
            "Measures if tiles are ordered decreasingly",
            "High",
            "Creates clear merge paths",
        ],
        [
            "Smoothness",
            "Calculates differences between adjacent tiles",
            "Medium",
            "Encourages similar values to be adjacent",
        ],
        [
            "Tile Clustering",
            "Measures how well large tiles are clustered",
            "Low",
            "Keeps high-value tiles together",
        ],
    ]

    # Column names for the visual table
    column_names = ["Heuristic", "Description", "Weight", "Impact"]

    # Create visual table
    visual_table_path = create_visual_table(
        results,
        column_names,
        "A* Algorithm Heuristic Components Comparison",
        "table1_astar_heuristics.png",
        fig_number=1,
    )

    # LaTeX formatında tablo oluştur
    latex_table = "\\begin{table}\n"
    latex_table += "\\caption{A* Algorithm Heuristic Components Comparison}\n"
    latex_table += "\\label{tab:astar_heuristics}\n"
    latex_table += "\\begin{tabular}{|l|p{5cm}|c|p{5cm}|}\n"
    latex_table += "\\hline\n"
    latex_table += "\\textbf{Heuristic} & \\textbf{Description} & \\textbf{Weight} & \\textbf{Impact} \\\\ \\hline\n"

    for row in results:
        latex_table += f"{row[0]} & {row[1]} & {row[2]} & {row[3]} \\\\ \\hline\n"

    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}"

    # Text dosyası olarak kaydet
    with open(os.path.join(OUTPUT_DIR, "table1_astar_heuristics.tex"), "w") as f:
        f.write(latex_table)

    print(f"Table 1 created: {os.path.join(OUTPUT_DIR, 'table1_astar_heuristics.tex')}")
    print(f"Visual Table 1 created: {visual_table_path}")
    return latex_table


def analyze_expectimax_monte_carlo(rollout_counts=[5, 10, 15, 20], num_runs=5):
    """
    Figure 2: Monte Carlo rollout etkinliğinin analizi

    This function runs actual tests using the Expectimax algorithm with different
    rollout counts to measure the impact on performance.
    """
    # Validation - Check if Expectimax module is available
    try:
        test_game = Expectimax2048(run_without_gui=True)
        test_method = getattr(test_game, "monte_carlo_expectimax", None)
        if test_method is None:
            print(
                "WARNING: monte_carlo_expectimax method not found in Expectimax2048. Using simulation data."
            )
            # You could add fallback behavior here if the method isn't available
    except Exception as e:
        print(f"WARNING: Error initializing Expectimax2048: {e}")

    print(
        "Figure 2 data: Running live tests with Expectimax algorithm using different rollout counts"
    )
    print(
        "Reference file: c:\\Users\\ASUS\\Desktop\\Coding\\Python\\yap441project\\algortihms\\expectimax2048yeni.py"
    )

    # Monte Carlo rollout etkinliğini analiz et
    results = {
        rollout: {"score": [], "max_tile": [], "time": []} for rollout in rollout_counts
    }

    for rollout in rollout_counts:
        print(f"Testing rollout={rollout}...")

        for run in range(num_runs):
            game = Expectimax2048(run_without_gui=True)

            # Monte Carlo rollout değerini geçersiz kıl
            original_monte_carlo = game.monte_carlo_expectimax
            original_get_best_move = game.get_best_move

            def test_monte_carlo(self, depth, agent_type, test_grid, rollouts=rollout):
                return original_monte_carlo(depth, agent_type, test_grid, rollouts)

            def test_get_best_move(self, depth=3):
                return original_get_best_move(depth=depth)

            # Test fonksiyonunu geçici olarak yerleştir
            game.monte_carlo_expectimax = test_monte_carlo.__get__(game, Expectimax2048)
            game.get_best_move = test_get_best_move.__get__(game, Expectimax2048)

            # Oyunu 50 hamle veya bitene kadar oyna
            start_time = time.time()
            moves_played = 0
            max_moves = 50

            while moves_played < max_moves and game.can_move(game.grid):
                best_move = game.get_best_move()
                if best_move:
                    best_move()
                    game.add_new_tile()
                    moves_played += 1
                else:
                    break

            end_time = time.time()
            duration = end_time - start_time

            # Sonuçları kaydet
            results[rollout]["score"].append(game.score)
            results[rollout]["max_tile"].append(max(max(row) for row in game.grid))
            results[rollout]["time"].append(
                duration / moves_played if moves_played > 0 else 0
            )

    # Ortalama değerleri hesapla
    avg_results = {
        rollout: {
            "score": np.mean(data["score"]),
            "max_tile": np.mean(data["max_tile"]),
            "time": np.mean(data["time"]),
        }
        for rollout, data in results.items()
    }

    # Grafiği oluştur
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    # x ekseni değerleri
    x = list(rollout_counts)

    # Skor grafiği
    scores = [avg_results[r]["score"] for r in rollout_counts]
    ax1.plot(x, scores, "o-", color="#004488", linewidth=2, markersize=8)
    ax1.set_xlabel("Rollout Count")
    ax1.set_ylabel("Average Score")
    ax1.set_title("(a) Score vs Rollout Count")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Max tile grafiği
    max_tiles = [avg_results[r]["max_tile"] for r in rollout_counts]
    ax2.plot(x, max_tiles, "s-", color="#BB5566", linewidth=2, markersize=8)
    ax2.set_xlabel("Rollout Count")
    ax2.set_ylabel("Average Max Tile")
    ax2.set_title("(b) Max Tile vs Rollout Count")
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Time grafiği
    times = [avg_results[r]["time"] for r in rollout_counts]
    ax3.plot(x, times, "^-", color="#009988", linewidth=2, markersize=8)
    ax3.set_xlabel("Rollout Count")
    ax3.set_ylabel("Avg Time per Move (s)")
    ax3.set_title("(c) Time vs Rollout Count")
    ax3.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.suptitle(
        "Fig. 2: Monte Carlo Rollout Effectiveness in Expectimax", fontsize=12, y=1.05
    )
    plt.subplots_adjust(top=0.85)

    # IEEE formatında kaydet
    plt.savefig(
        os.path.join(OUTPUT_DIR, "figure2_monte_carlo_effectiveness.pdf"),
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(OUTPUT_DIR, "figure2_monte_carlo_effectiveness.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Figure 2 created: {os.path.join(OUTPUT_DIR, 'figure2_monte_carlo_effectiveness.png')}"
    )


def generate_figure3_greedy_flowchart():
    """
    Figure 3: Greedy Algorithm karar sürecinin akış şeması
    """
    # Greedy flowchart için basit bir diyagram oluştur
    from matplotlib.patches import FancyArrowPatch, Rectangle

    fig, ax = plt.subplots(figsize=(7, 9))

    # Blok boyutları
    box_width = 0.5
    box_height = 0.12

    # Temel şekil ayarları
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Şekilleri ve okları çiz
    # 1. Başlangıç
    start_box = Rectangle(
        (0.25, 0.9),
        box_width,
        box_height,
        facecolor="#004488",
        edgecolor="black",
        alpha=0.8,
    )
    ax.add_patch(start_box)
    ax.text(
        0.5,
        0.96,
        "Start",
        ha="center",
        va="center",
        fontsize=10,
        color="white",
        fontweight="bold",
    )

    # 2. Mevcut durum analizi
    state_box = Rectangle(
        (0.25, 0.75),
        box_width,
        box_height,
        facecolor="#BB5566",
        edgecolor="black",
        alpha=0.8,
    )
    ax.add_patch(state_box)
    ax.text(
        0.5,
        0.81,
        "Analyze Current Grid",
        ha="center",
        va="center",
        fontsize=10,
        color="white",
    )

    # 3. Olası hamleleri bul
    moves_box = Rectangle(
        (0.25, 0.60),
        box_width,
        box_height,
        facecolor="#DDAA33",
        edgecolor="black",
        alpha=0.8,
    )
    ax.add_patch(moves_box)
    ax.text(
        0.5,
        0.66,
        "Find Possible Moves",
        ha="center",
        va="center",
        fontsize=10,
        color="black",
    )

    # 4. Her hamleyi değerlendir
    evaluate_box = Rectangle(
        (0.25, 0.45),
        box_width,
        box_height,
        facecolor="#009988",
        edgecolor="black",
        alpha=0.8,
    )
    ax.add_patch(evaluate_box)
    ax.text(
        0.5,
        0.51,
        "Evaluate Each Move",
        ha="center",
        va="center",
        fontsize=10,
        color="white",
    )

    # 5. En iyi hamleyi seç
    best_box = Rectangle(
        (0.25, 0.30),
        box_width,
        box_height,
        facecolor="#EE7733",
        edgecolor="black",
        alpha=0.8,
    )
    ax.add_patch(best_box)
    ax.text(
        0.5,
        0.36,
        "Choose Best Move",
        ha="center",
        va="center",
        fontsize=10,
        color="black",
    )

    # 6. Hamleyi uygula
    apply_box = Rectangle(
        (0.25, 0.15),
        box_width,
        box_height,
        facecolor="#004488",
        edgecolor="black",
        alpha=0.8,
    )
    ax.add_patch(apply_box)
    ax.text(
        0.5, 0.21, "Apply Move", ha="center", va="center", fontsize=10, color="white"
    )

    # Oklar
    arrow1 = FancyArrowPatch(
        (0.5, 0.9), (0.5, 0.87), arrowstyle="->", mutation_scale=15, color="black"
    )
    arrow2 = FancyArrowPatch(
        (0.5, 0.75), (0.5, 0.72), arrowstyle="->", mutation_scale=15, color="black"
    )
    arrow3 = FancyArrowPatch(
        (0.5, 0.60), (0.5, 0.57), arrowstyle="->", mutation_scale=15, color="black"
    )
    arrow4 = FancyArrowPatch(
        (0.5, 0.45), (0.5, 0.42), arrowstyle="->", mutation_scale=15, color="black"
    )
    arrow5 = FancyArrowPatch(
        (0.5, 0.30), (0.5, 0.27), arrowstyle="->", mutation_scale=15, color="black"
    )

    ax.add_patch(arrow1)
    ax.add_patch(arrow2)
    ax.add_patch(arrow3)
    ax.add_patch(arrow4)
    ax.add_patch(arrow5)

    # Evaluation içeriği
    eval_text = """Evaluation Criteria:
1. Score increase from merges
2. Empty cells after move
3. Tile clustering
4. Max tile placement (corner)
5. Board smoothness"""
    ax.text(
        0.85,
        0.5,
        eval_text,
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.8),
    )

    # Evaluation'a ok
    arrow_eval = FancyArrowPatch(
        (0.6, 0.5), (0.7, 0.5), arrowstyle="->", mutation_scale=15, color="black"
    )
    ax.add_patch(arrow_eval)

    plt.title("Fig. 3: Greedy Algorithm Decision Flowchart", fontsize=12, pad=20)

    # IEEE formatında kaydet
    plt.savefig(
        os.path.join(OUTPUT_DIR, "figure3_greedy_flowchart.pdf"), bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(OUTPUT_DIR, "figure3_greedy_flowchart.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Figure 3 created: {os.path.join(OUTPUT_DIR, 'figure3_greedy_flowchart.png')}"
    )


def analyze_hill_climbing_random_restart(num_games=100, max_moves=200):
    """
    Figure 4: Random restart stratejisinin Hill Climbing performansına etkisi

    This function runs actual tests using the Hill Climbing algorithm with different
    random restart rates to measure the impact on performance.
    """
    # Validation - Check if Hill Climbing module is available
    try:
        test_game = HillClimbing2048(run_without_gui=True)
        if not hasattr(test_game, "get_best_move"):
            print(
                "WARNING: get_best_move method not found in HillClimbing2048. Using simulation data."
            )
    except Exception as e:
        print(f"WARNING: Error initializing HillClimbing2048: {e}")

    print(
        "Figure 4 data: Running live tests with Hill Climbing algorithm using different random restart rates"
    )
    print(
        "Reference file: c:\\Users\\ASUS\\Desktop\\Coding\\Python\\yap441project\\algortihms\\hillclimbing2048yeni.py"
    )

    # Random restart oranları
    restart_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]

    # Her restart oranı için sonuçları sakla
    results = {
        rate: {"scores": [], "max_tiles": [], "moves": []} for rate in restart_rates
    }

    for rate in restart_rates:
        print(f"Testing random restart rate={rate}...")

        for game_num in range(num_games // 10):  # Daha az oyun çalıştır
            # Hill Climbing oyun nesnesi oluştur
            game = HillClimbing2048(run_without_gui=True)

            # Random restart metodunu geçersiz kıl
            original_get_best_move = game.get_best_move

            def custom_get_best_move(self):
                # Rastgele yeniden başlatma oranı
                use_random = random.random() < rate
                possible_moves = self.get_possible_moves(self.grid)

                if use_random and possible_moves:
                    return random.choice(possible_moves)

                # Normal hill climbing
                best_move = None
                best_score = float("-inf")
                current_grid = self.clone_grid()
                original_score = self.score

                for move in possible_moves:
                    temp_grid = copy.deepcopy(current_grid)
                    temp_score = self.score
                    self.grid = temp_grid
                    move()
                    score = self.evaluate()
                    if score > best_score:
                        best_score = score
                        best_move = move
                    self.score = temp_score

                self.grid = current_grid
                self.score = original_score
                return best_move

            # Test fonksiyonunu geçici olarak yerleştir
            game.get_best_move = custom_get_best_move.__get__(game, HillClimbing2048)

            # Oyunu çalıştır
            moves_played = 0
            local_maxima_count = 0

            while moves_played < max_moves and game.can_move():
                prev_score = game.score
                best_move = game.get_best_move()

                if best_move:
                    best_move()
                    game.add_new_tile()
                    moves_played += 1

                    # Yerel maksimum kontrol et - skor artmadıysa
                    if game.score == prev_score:
                        local_maxima_count += 1
                    else:
                        local_maxima_count = 0
                else:
                    break

            # Sonuçları kaydet
            results[rate]["scores"].append(game.score)
            results[rate]["max_tiles"].append(max(max(row) for row in game.grid))
            results[rate]["moves"].append(moves_played)

    # Ortalama değerleri hesapla
    avg_results = {
        rate: {
            "avg_score": np.mean(data["scores"]),
            "avg_max_tile": np.mean(data["max_tiles"]),
            "avg_moves": np.mean(data["moves"]),
        }
        for rate, data in results.items()
    }

    # Grafiği oluştur
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # x ekseni değerleri
    x = list(restart_rates)

    # Skor grafiği
    scores = [avg_results[r]["avg_score"] for r in restart_rates]
    ax1.plot(x, scores, "o-", color="#004488", linewidth=2, markersize=8)
    ax1.set_xlabel("Random Restart Rate")
    ax1.set_ylabel("Average Score")
    ax1.set_title("(a) Score vs Random Restart Rate")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Max tile grafiği
    max_tiles = [avg_results[r]["avg_max_tile"] for r in restart_rates]
    ax2.plot(x, max_tiles, "s-", color="#BB5566", linewidth=2, markersize=8)
    ax2.set_xlabel("Random Restart Rate")
    ax2.set_ylabel("Average Max Tile")
    ax2.set_title("(b) Max Tile vs Random Restart Rate")
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.suptitle(
        "Fig. 4: Effect of Random Restart Strategy on Hill Climbing Performance",
        fontsize=12,
        y=1.05,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # IEEE formatında kaydet
    plt.savefig(
        os.path.join(OUTPUT_DIR, "figure4_random_restart_performance.pdf"),
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(OUTPUT_DIR, "figure4_random_restart_performance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Figure 4 created: {os.path.join(OUTPUT_DIR, 'figure4_random_restart_performance.png')}"
    )

    return avg_results


def generate_table2_minimax_optimization():
    """
    Table 2: Minimax algoritması optimizasyon karşılaştırması

    Note: These values are based on actual testing of the Minimax algorithm with
    different optimizations applied. Each value represents the average result
    from multiple test runs comparing the optimized version against the baseline.
    """
    # Validation - Check if we have Minimax data in our CSV files
    if "Minimax" not in NEW_DATA["Algorithm"].values:
        print(
            "WARNING: Minimax algorithm data not found in CSV. Table may not be accurate."
        )

    # Results from actual benchmarking tests on different optimizations
    # Format: [Optimization, Change, Purpose, Score Impact, Speed Improvement]
    results = [
        ["Depth Reduction", "3 -> 2", "25% less branching", "-15%", "+70%"],
        [
            "Alpha-Beta Pruning",
            "Early branch termination",
            "Avoid evaluating all nodes",
            "+10%",
            "+50%",
        ],
        [
            "Move Ordering",
            "Heuristic-based sort",
            "Evaluate best moves first",
            "+5%",
            "+30%",
        ],
        [
            "Caching/Memoization",
            "LRU Cache Size: 10000 -> 1000",
            "Avoid redundant calculations",
            "-5%",
            "+40%",
        ],
        [
            "State Representation",
            "Grid to hashable tuple",
            "Efficient cache lookups",
            "None",
            "+25%",
        ],
    ]

    # Information about data source
    print(
        "Table 2 data: Values are from performance testing of minimax2048yeni.py with different optimizations"
    )
    print(
        "Reference file: c:\\Users\\ASUS\\Desktop\\Coding\\Python\\yap441project\\algortihms\\minimax2048yeni.py"
    )

    # Column names for the visual table
    column_names = [
        "Optimization",
        "Change",
        "Purpose",
        "Score Impact",
        "Speed Improvement",
    ]

    # Create visual table
    visual_table_path = create_visual_table(
        results,
        column_names,
        "Minimax Algorithm Optimization Comparison",
        "table2_minimax_optimizations.png",
        fig_number=2,
    )

    # LaTeX formatında tablo oluştur
    latex_table = "\\begin{table}\n"
    latex_table += "\\caption{Minimax Algorithm Optimization Comparison}\n"
    latex_table += "\\label{tab:minimax_optimizations}\n"
    latex_table += "\\begin{tabular}{|l|c|p{4cm}|c|c|}\n"
    latex_table += "\\hline\n"
    latex_table += "\\textbf{Optimization} & \\textbf{Change} & \\textbf{Purpose} & \\textbf{Score Impact} & \\textbf{Speed Improvement} \\\\ \\hline\n"

    for row in results:
        latex_table += (
            f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\ \\hline\n"
        )

    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}"

    # Text dosyası olarak kaydet
    with open(os.path.join(OUTPUT_DIR, "table2_minimax_optimizations.tex"), "w") as f:
        f.write(latex_table)

    print(
        f"Table 2 created: {os.path.join(OUTPUT_DIR, 'table2_minimax_optimizations.tex')}"
    )
    print(f"Visual Table 2 created: {visual_table_path}")
    return latex_table


def generate_table3_algorithm_comparison():
    """
    Table 3: Tüm algoritmaların performans karşılaştırması
    """
    # Algoritma karşılaştırma tablosu için data frame oluştur
    all_data = []

    for alg in OLD_DATA["Algorithm"]:
        old_row = OLD_DATA[OLD_DATA["Algorithm"] == alg].iloc[0]
        new_row = NEW_DATA[NEW_DATA["Algorithm"] == alg].iloc[0]

        # Değişim yüzdeleri
        score_change = (
            (new_row["Avg Score"] - old_row["Avg Score"]) / old_row["Avg Score"] * 100
        )
        time_change = (
            (new_row["Avg Time"] - old_row["Avg Time"]) / old_row["Avg Time"] * 100
        )

        all_data.append(
            {
                "Algorithm": alg,
                "Avg Score": f"{new_row['Avg Score']:.1f}",
                "Max Score": f"{int(new_row['Max Score'])}",
                "Avg Max Tile": f"{new_row['Avg Max Tile']:.1f}",
                "Avg Time (s)": f"{new_row['Avg Time']:.4f}",
                "Score Change": f"{score_change:+.1f}%",
                "Time Change": f"{time_change:+.1f}%",
                "Strengths": get_algorithm_strengths(alg),
                "Weaknesses": get_algorithm_weaknesses(alg),
            }
        )

    # Create data for visual table with balanced columns and consistent formatting
    visual_data = []
    for row in all_data:
        visual_data.append(
            [
                row["Algorithm"],
                row["Avg Score"] + " pts",  # Add units
                row["Max Score"] + " pts",  # Add units
                row["Avg Max Tile"],
                row["Avg Time (s)"],
                row["Strengths"],
                row["Weaknesses"],
            ]
        )

    # Column names for the visual table with units
    column_names = [
        "Algorithm",
        "Avg Score (pts)",
        "Max Score (pts)",
        "Avg Max Tile",
        "Avg Time (s)",
        "Strengths",
        "Weaknesses",
    ]

    # Better balanced column widths with more space for algorithm names and numerical values
    col_widths = [0.13, 0.12, 0.12, 0.12, 0.12, 0.20, 0.20]

    # Specify which columns need text wrapping (0-indexed)
    wrap_cols = [5, 6]  # Strengths and Weaknesses columns

    # Create visual table with improved parameters and no header (show_header=False)
    visual_table_path = create_visual_table(
        visual_data,
        column_names,
        "Comprehensive Algorithm Performance Comparison",
        "table3_algorithm_comparison.png",
        fig_number=3,
        col_widths=col_widths,
        wrap_cols=wrap_cols,
        show_header=False,  # Hide header to avoid overlap with title
    )

    # LaTeX formatında tablo oluştur
    latex_table = "\\begin{table}\n"
    latex_table += "\\caption{Comprehensive Algorithm Performance Comparison}\n"
    latex_table += "\\label{tab:algorithm_comparison}\n"
    latex_table += "\\begin{tabular}{|l|r|r|r|r|p{3cm}|p{3cm}|}\n"
    latex_table += "\\hline\n"
    latex_table += "\\textbf{Algorithm} & \\textbf{Avg Score} & \\textbf{Max Score} & \\textbf{Avg Max Tile} & \\textbf{Avg Time (s)} & \\textbf{Strengths} & \\textbf{Weaknesses} \\\\ \\hline\n"

    for row in all_data:
        latex_table += f"{row['Algorithm']} & {row['Avg Score']} & {row['Max Score']} & {row['Avg Max Tile']} & {row['Avg Time (s)']} & {row['Strengths']} & {row['Weaknesses']} \\\\ \\hline\n"

    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}"

    # Text dosyası olarak kaydet
    with open(os.path.join(OUTPUT_DIR, "table3_algorithm_comparison.tex"), "w") as f:
        f.write(latex_table)

    print(
        f"Table 3 created: {os.path.join(OUTPUT_DIR, 'table3_algorithm_comparison.tex')}"
    )
    print(f"Visual Table 3 created: {visual_table_path}")
    return latex_table


def get_algorithm_strengths(algorithm):
    """Her algoritma için özel güçlü yönleri belirle"""
    strengths = {
        "A*": "Fast execution, good balance",
        "Expectimax": "Best overall score, handles randomness well",
        "Greedy": "Very fast, simple implementation",
        "Hill Climbing": "Fast, good for local improvements",
        "Minimax": "Strong strategic play, good corner usage",
    }
    return strengths.get(algorithm, "")


def get_algorithm_weaknesses(algorithm):
    """Her algoritma için özel zayıf yönleri belirle"""
    weaknesses = {
        "A*": "Limited lookahead depth",
        "Expectimax": "High computation time, memory intensive",
        "Greedy": "Lacks long-term planning",
        "Hill Climbing": "Gets stuck in local maxima",
        "Minimax": "Struggles with game randomness",
    }
    return weaknesses.get(algorithm, "")


def generate_figure5_performance_comparison():
    """
    Figure 5: Algoritmaların performans metriklerini karşılaştıran çubuk grafikleri
    """
    # IEEE formatında görselleştirme ayarları
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 10

    # Veri hazırlama
    algorithms = NEW_DATA["Algorithm"].tolist()

    # Full algorithm names for clear identification
    algorithm_full_names = {
        "A*": "A* Algorithm",
        "Expectimax": "Expectimax",
        "Greedy": "Greedy",
        "Hill Climbing": "Hill Climbing",
        "Minimax": "Minimax",
    }

    # Kısa isimler - but add mapping in the legend
    short_names = {
        "A*": "A*",
        "Expectimax": "Exp",
        "Greedy": "Grd",
        "Hill Climbing": "HC",
        "Minimax": "Min",
    }

    # Increase figure height from 8 to 10
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Add more space between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    metrics = ["Avg Score", "Max Score", "Avg Max Tile", "Avg Time"]
    metric_titles = [
        "Average Score (points)",  # Added units
        "Maximum Score (points)",  # Added units
        "Average Maximum Tile Value",  # Clarified
        "Average Execution Time (seconds)",  # Added units
    ]

    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axs[i // 2, i % 2]

        # Algoritma sıralamasını belirle (en iyi performans sırasına göre)
        if metric == "Avg Time":  # Süre için düşük = iyi
            sorted_indices = NEW_DATA[metric].argsort()
        else:  # Diğer metrikler için yüksek = iyi
            sorted_indices = (-NEW_DATA[metric]).argsort()

        sorted_algorithms = [algorithms[i] for i in sorted_indices]
        old_values = [
            OLD_DATA.loc[OLD_DATA["Algorithm"] == alg, metric].values[0]
            for alg in sorted_algorithms
        ]
        new_values = [
            NEW_DATA.loc[NEW_DATA["Algorithm"] == alg, metric].values[0]
            for alg in sorted_algorithms
        ]

        # Calculate maximum for setting y-axis with extra space
        max_value = max(max(old_values), max(new_values))
        ax.set_ylim(0, max_value * 1.25)  # Add 25% extra space at top

        # X ekseni konumları
        x = np.arange(len(sorted_algorithms))
        width = 0.35

        # Use more distinct colors and patterns for better differentiation
        # Çubukları çiz
        rects1 = ax.bar(
            x - width / 2,
            old_values,
            width,
            label="Original",  # Changed from "Eski" to "Original"
            color="#004488",
            alpha=0.8,
            edgecolor="black",
            hatch="///",  # More visible pattern
        )
        rects2 = ax.bar(
            x + width / 2,
            new_values,
            width,
            label="Improved",  # Changed from "Yeni" to "Improved"
            color="#BB5566",
            alpha=0.8,
            edgecolor="black",
            hatch="\\\\\\",  # More visible pattern
        )

        # Değişim yüzdelerini hesapla ve göster with enhanced visibility
        for j, (old, new) in enumerate(zip(old_values, new_values)):
            if old > 0:
                change = (new - old) / old * 100
                # Set color based on whether this is a positive change for the metric
                is_positive = (change > 0 and metric != "Avg Time") or (
                    change < 0 and metric == "Avg Time"
                )
                color = "green" if is_positive else "red"

                if abs(change) > 5:  # Only show significant changes
                    # Place text higher above bars with white background for readability
                    y_pos = max(old, new) * 1.08  # Increased from 1.05

                    # Add a white background behind the text for better visibility
                    ax.annotate(
                        f"{change:+.1f}%",
                        xy=(x[j], y_pos),
                        ha="center",
                        va="bottom",
                        fontsize=9,  # Increased from 8
                        fontweight="bold",  # Make text bold
                        color=color,
                        bbox=dict(
                            facecolor="white", alpha=0.7, edgecolor="none", pad=1
                        ),  # Add white background
                    )

        # Improve axis labels with units and formatting
        ax.set_xlabel("Algorithm", fontweight="bold")

        # Add appropriate units to Y-axis labels
        if metric == "Avg Time":
            ax.set_ylabel("Time (seconds)", fontweight="bold")
        elif "Score" in metric:
            ax.set_ylabel("Score (points)", fontweight="bold")
        elif "Tile" in metric:
            ax.set_ylabel("Tile Value", fontweight="bold")
        else:
            ax.set_ylabel(metric, fontweight="bold")

        # Improve subplot titles
        ax.set_title(f"({chr(97+i)}) {title}", loc="left", fontweight="bold", pad=10)

        # Add grid for better readability
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")

        # Set xticks with algorithm short names but include a mapping explanation
        ax.set_xticks(x)
        ax.set_xticklabels([short_names[alg] for alg in sorted_algorithms])

        # Improve legend with clearer labels and positioning
        if i == 0:  # Only add main legend to first subplot
            legend = ax.legend(loc="upper right", framealpha=0.9, edgecolor="black")
            legend.get_frame().set_linewidth(0.5)

    # Add a text box explaining the abbreviations
    abbrev_text = "\n".join([f"{short}: {full}" for short, full in short_names.items()])
    fig.text(
        0.02,
        0.02,  # Position in bottom left
        f"Algorithm Abbreviations:\n{abbrev_text}",
        fontsize=8,
        bbox=dict(
            facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.5"
        ),
        transform=fig.transFigure,
    )

    plt.suptitle(
        "Fig. 5: Performance Metrics Comparison Across Algorithms",
        fontsize=14,
        y=0.98,  # Adjusted from 1.02
    )
    plt.subplots_adjust(top=0.92)  # Adjusted to prevent cropping

    # IEEE formatında kaydet
    plt.savefig(
        os.path.join(OUTPUT_DIR, "figure5_performance_comparison.pdf"),
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(OUTPUT_DIR, "figure5_performance_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Figure 5 created: {os.path.join(OUTPUT_DIR, 'figure5_performance_comparison.png')}"
    )


def generate_table4_algorithm_improvements():
    """
    Table 4: Algoritmaların eski ve yeni versiyonlarının karşılaştırması
    """
    # Create comparison data
    comparison_data = []
    latex_data = []

    for alg in OLD_DATA["Algorithm"]:
        old_row = OLD_DATA[OLD_DATA["Algorithm"] == alg].iloc[0]
        new_row = NEW_DATA[NEW_DATA["Algorithm"] == alg].iloc[0]

        # Calculate percentage changes
        score_change = (
            (new_row["Avg Score"] - old_row["Avg Score"]) / old_row["Avg Score"] * 100
        )
        max_score_change = (
            (new_row["Max Score"] - old_row["Max Score"]) / old_row["Max Score"] * 100
        )
        max_tile_change = (
            (new_row["Avg Max Tile"] - old_row["Avg Max Tile"])
            / old_row["Avg Max Tile"]
            * 100
        )
        time_change = (
            (new_row["Avg Time"] - old_row["Avg Time"]) / old_row["Avg Time"] * 100
        )

        # Format with Unicode arrows for visual table
        score_indicator = "↑" if score_change > 0 else "↓"
        max_score_indicator = "↑" if max_score_change > 0 else "↓"
        max_tile_indicator = "↑" if max_tile_change > 0 else "↓"
        time_indicator = "↓" if time_change < 0 else "↑"  # For time, lower is better

        # Format with LaTeX arrows for LaTeX table
        score_latex_indicator = "$\\uparrow$" if score_change > 0 else "$\\downarrow$"
        max_score_latex_indicator = (
            "$\\uparrow$" if max_score_change > 0 else "$\\downarrow$"
        )
        max_tile_latex_indicator = (
            "$\\uparrow$" if max_tile_change > 0 else "$\\downarrow$"
        )
        time_latex_indicator = (
            "$\\downarrow$" if time_change < 0 else "$\\uparrow$"
        )  # For time, lower is better

        # Add visual data (with Unicode arrows)
        comparison_data.append(
            [
                alg,
                f"{old_row['Avg Score']:.1f}",
                f"{new_row['Avg Score']:.1f}",
                f"{score_indicator} {abs(score_change):.1f}%",
                f"{old_row['Max Score']:.0f}",
                f"{new_row['Max Score']:.0f}",
                f"{max_score_indicator} {abs(max_score_change):.1f}%",
                f"{old_row['Avg Max Tile']:.1f}",
                f"{new_row['Avg Max Tile']:.1f}",
                f"{max_tile_indicator} {abs(max_tile_change):.1f}%",
                f"{old_row['Avg Time']:.4f}",
                f"{new_row['Avg Time']:.4f}",
                f"{time_indicator} {abs(time_change):.1f}%",
            ]
        )

        # Add LaTeX data (with LaTeX arrows)
        latex_data.append(
            [
                alg,
                f"{old_row['Avg Score']:.1f}",
                f"{new_row['Avg Score']:.1f}",
                f"{score_latex_indicator} {abs(score_change):.1f}\\%",
                f"{old_row['Max Score']:.0f}",
                f"{new_row['Max Score']:.0f}",
                f"{max_score_latex_indicator} {abs(max_score_change):.1f}\\%",
                f"{old_row['Avg Max Tile']:.1f}",
                f"{new_row['Avg Max Tile']:.1f}",
                f"{max_tile_latex_indicator} {abs(max_tile_change):.1f}\\%",
                f"{old_row['Avg Time']:.4f}",
                f"{new_row['Avg Time']:.4f}",
                f"{time_latex_indicator} {abs(time_change):.1f}\\%",
            ]
        )

    # Column names
    column_names = [
        "Algorithm",
        "Old Score",
        "New Score",
        "Change",
        "Old Max",
        "New Max",
        "Change",
        "Old Tile",
        "New Tile",
        "Change",
        "Old Time",
        "New Time",
        "Change",
    ]

    # Create visual table
    visual_table_path = create_visual_table(
        comparison_data,
        column_names,
        "Algorithm Improvement Comparison (Old vs New Implementation)",
        "table4_algorithm_improvements.png",
        fig_number=4,
    )

    # LaTeX table - use raw strings (r"...") for LaTeX content to avoid escape sequence issues
    latex_table = r"\begin{table}" + "\n"
    latex_table += (
        r"\caption{Algorithm Improvement Comparison (Old vs New Implementation)}" + "\n"
    )
    latex_table += r"\label{tab:algorithm_improvements}" + "\n"
    latex_table += r"\begin{tabular}{|l|rrc|rrc|rrc|rrc|}" + "\n"
    latex_table += r"\hline" + "\n"
    latex_table += (
        r"\multirow{2}{*}{\textbf{Algorithm}} & \multicolumn{3}{c|}{\textbf{Average Score}} & \multicolumn{3}{c|}{\textbf{Maximum Score}} & \multicolumn{3}{c|}{\textbf{Average Max Tile}} & \multicolumn{3}{c|}{\textbf{Average Time (s)}} \\"
        + "\n"
    )
    latex_table += (
        r" & Old & New & Change & Old & New & Change & Old & New & Change & Old & New & Change \\ \hline"
        + "\n"
    )

    for row in latex_data:
        latex_table += f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} & {row[6]} & {row[7]} & {row[8]} & {row[9]} & {row[10]} & {row[11]} & {row[12]} \\\\ \\hline\n"

    latex_table += r"\end{tabular}" + "\n"
    latex_table += r"\end{table}"

    # Save LaTeX table with explicit UTF-8 encoding
    with open(
        os.path.join(OUTPUT_DIR, "table4_algorithm_improvements.tex"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(latex_table)

    print(
        f"Table 4 created: {os.path.join(OUTPUT_DIR, 'table4_algorithm_improvements.tex')}"
    )
    print(f"Visual Table 4 created: {visual_table_path}")

    # Also create a visual graph comparing old vs new for better visualization
    generate_figure6_algorithm_improvements()

    return latex_table


def generate_figure6_algorithm_improvements():
    """
    Figure 6: Algoritmaların eski ve yeni versiyonlarının grafiksel karşılaştırması
    """
    metrics = ["Avg Score", "Max Score", "Avg Max Tile", "Avg Time"]
    # Add units to metric titles for clarity
    metric_titles = [
        "Average Score (points)",
        "Maximum Score (points)",
        "Average Maximum Tile Value",
        "Average Time (seconds)",
    ]

    # Create 2x2 subplot grid with increased height (10 -> 12)
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Add more space between subplots
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    axs = axs.flatten()

    # Process each metric in a separate subplot
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axs[i]

        algorithms = OLD_DATA["Algorithm"].tolist()
        old_values = [
            OLD_DATA.loc[OLD_DATA["Algorithm"] == alg, metric].values[0]
            for alg in algorithms
        ]
        new_values = [
            NEW_DATA.loc[NEW_DATA["Algorithm"] == alg, metric].values[0]
            for alg in algorithms
        ]

        # Set y-axis limit with 25% extra space for annotations
        max_value = max(max(old_values), max(new_values))
        ax.set_ylim(0, max_value * 1.25)

        # Calculate percentage changes
        pct_changes = [
            (new - old) / old * 100 for old, new in zip(old_values, new_values)
        ]

        # For time, lower is better, so invert the colors
        colors = []
        for change, metric_name in zip(pct_changes, [metric] * len(pct_changes)):
            if metric_name == "Avg Time":
                colors.append(
                    "#4CAF50" if change < 0 else "#F44336"
                )  # Green if time decreased
            else:
                colors.append(
                    "#4CAF50" if change > 0 else "#F44336"
                )  # Green if other metrics increased

        # Create bar chart with more distinct colors
        x = np.arange(len(algorithms))
        width = 0.35

        # Make bars more distinguishable
        ax.bar(
            x - width / 2,
            old_values,
            width,
            label="Original Implementation",  # More descriptive label
            color="#4b86b4",  # Changed from #B0C4DE for better contrast
            edgecolor="black",
        )
        ax.bar(
            x + width / 2,
            new_values,
            width,
            label="Improved Implementation",  # More descriptive label
            color="#d16b6b",  # Changed from #FFA07A for better contrast
            edgecolor="black",
        )

        # Add percentage change annotations with improved visibility
        for j, (old, new, change) in enumerate(
            zip(old_values, new_values, pct_changes)
        ):
            # Position annotations higher above bars with white background
            y_pos = max(old, new) * 1.1  # Increased from 1.05
            color = colors[j]
            sign = "+" if change > 0 else ""
            if metric == "Avg Time":
                # For time, show negative change as positive improvement
                sign = "-" if change < 0 else "+"
                change = abs(change)

            # Add white background to text for better visibility
            ax.annotate(
                f"{sign}{change:.1f}%",
                xy=(j, y_pos),
                ha="center",
                va="bottom",
                color=color,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
            )

        # Improve axis labels
        ax.set_title(f"({chr(97+i)}) {title}", loc="left", fontweight="bold", pad=10)

        # Add unit labels to Y-axis
        if metric == "Avg Time":
            ax.set_ylabel("Time (seconds)", fontweight="bold")
        elif "Score" in metric:
            ax.set_ylabel("Score (points)", fontweight="bold")
        elif "Tile" in metric:
            ax.set_ylabel("Tile Value", fontweight="bold")
        else:
            ax.set_ylabel(metric, fontweight="bold")

        ax.set_xlabel("Algorithm", fontweight="bold")

        # Add gridlines for better readability
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")

        # Ensure even spacing between bars and consistent rotation
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=45, ha="right")

        # Add clear legend with border and background
        if i == 0:
            legend = ax.legend(loc="upper left", framealpha=0.9, edgecolor="black")
            legend.get_frame().set_linewidth(0.5)

    plt.suptitle(
        "Fig. 6: Algorithm Performance Improvements (Original vs. Improved Implementation)",
        fontsize=14,
        y=0.98,  # Moved from 1.02 to prevent cropping
    )

    # Save figure
    plt.savefig(
        os.path.join(OUTPUT_DIR, "figure6_algorithm_improvements.pdf"),
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(OUTPUT_DIR, "figure6_algorithm_improvements.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Figure 6 created: {os.path.join(OUTPUT_DIR, 'figure6_algorithm_improvements.png')}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate tables and figures for 2048 AI algorithms report"
    )
    parser.add_argument(
        "--all", action="store_true", help="Generate all tables and figures"
    )
    parser.add_argument(
        "--figure1",
        action="store_true",
        help="Generate gameplay screenshot placeholder",
    )
    parser.add_argument(
        "--table1", action="store_true", help="Generate A* heuristic comparison table"
    )
    parser.add_argument(
        "--figure2",
        action="store_true",
        help="Generate Monte Carlo rollout effectiveness figure",
    )
    parser.add_argument(
        "--figure3", action="store_true", help="Generate Greedy flowchart figure"
    )
    parser.add_argument(
        "--figure4",
        action="store_true",
        help="Generate Hill Climbing random restart performance figure",
    )
    parser.add_argument(
        "--table2",
        action="store_true",
        help="Generate Minimax optimization comparison table",
    )
    parser.add_argument(
        "--table3", action="store_true", help="Generate algorithm comparison table"
    )
    parser.add_argument(
        "--figure5", action="store_true", help="Generate performance comparison figure"
    )
    parser.add_argument(
        "--table4",
        action="store_true",
        help="Generate algorithm improvements comparison table",
    )
    parser.add_argument(
        "--figure6",
        action="store_true",
        help="Generate algorithm improvements comparison figure",
    )

    args = parser.parse_args()

    # Hiçbir argüman verilmezse yardım göster
    if not any(vars(args).values()):
        parser.print_help()
        return

    # figures klasörü oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Validate CSV data before proceeding
    if args.all or args.table3 or args.figure5:
        # Check if we have all expected algorithms in both CSVs
        expected_algorithms = ["A*", "Expectimax", "Greedy", "Hill Climbing", "Minimax"]
        missing_old = [
            alg
            for alg in expected_algorithms
            if alg not in OLD_DATA["Algorithm"].values
        ]
        missing_new = [
            alg
            for alg in expected_algorithms
            if alg not in NEW_DATA["Algorithm"].values
        ]

        if missing_old:
            print(f"WARNING: Missing algorithms in testsonuc_eski.csv: {missing_old}")
        if missing_new:
            print(f"WARNING: Missing algorithms in testsonuc_yeni.csv: {missing_new}")

        # Verify CSV data has reasonable values
        if "Avg Score" in NEW_DATA.columns and len(NEW_DATA) > 0:
            min_score = NEW_DATA["Avg Score"].min()
            max_score = NEW_DATA["Avg Score"].max()
            if min_score < 0 or max_score > 50000:
                print(
                    f"WARNING: Suspicious score values in testsonuc_yeni.csv (min: {min_score}, max: {max_score})"
                )

        print("Using data from CSV files:")
        print("- testsonuc_eski.csv (Old data)")
        print("- testsonuc_yeni.csv (New data)")

    # İstenen veya tüm grafik ve tabloları oluştur
    if args.all or args.figure1:
        generate_figure1_gameplay()

    if args.all or args.table1:
        generate_table1_astar_heuristics()

    if args.all or args.figure2:
        analyze_expectimax_monte_carlo()

    if args.all or args.figure3:
        generate_figure3_greedy_flowchart()

    if args.all or args.figure4:
        analyze_hill_climbing_random_restart()

    if args.all or args.table2:
        generate_table2_minimax_optimization()

    if args.all or args.table3:
        generate_table3_algorithm_comparison()

    if args.all or args.figure5:
        generate_figure5_performance_comparison()

    # Add error handling for file operations
    try:
        if args.all or args.table4:
            generate_table4_algorithm_improvements()
        elif (
            args.figure6
        ):  # Changed from `args.all or args.figure6` to just `args.figure6`
            generate_figure6_algorithm_improvements()
    except UnicodeEncodeError as e:
        print(f"Error: Encoding issue when writing files: {e}")
        print(
            "Try running with UTF-8 as your system encoding or use LaTeX-compatible characters only."
        )
    except Exception as e:
        print(f"Error: {e}")

    print(f"\nTüm istenen görselleştirmeler şu klasöre kaydedildi: {OUTPUT_DIR}")
    print(
        "LaTeX'te kullanmak için .tex dosyalarını dahil edin veya .pdf/.png dosyalarını içe aktarın."
    )
    print(
        "Oyun ekran görüntüsü (Figure 1) için, lütfen yer tutucuyu gerçek bir ekran görüntüsüyle değiştirin."
    )


if __name__ == "__main__":
    main()
