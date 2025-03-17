import tkinter as tk
import random
import copy
import os
import datetime
import time  # Add this import for time-related functions
from typing import Callable, List, Optional, Tuple
import pyautogui  # Change from ImageGrab to pyautogui
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
)  # Add ImageDraw and ImageFont for text on screenshots
from functools import lru_cache

GRID_SIZE = 4
NEW_TILE_VALUES = [2, 4]
BACKGROUND_COLOR = "#bbada0"
TILE_COLORS = {
    0: "#cdc1b4",
    2: "#eee4da",
    4: "#ede0c8",
    8: "#f2b179",
    16: "#f59563",
    32: "#f67c5f",
    64: "#f65e3b",
    128: "#edcf72",
    256: "#edcc61",
    512: "#edc850",
    1024: "#edc53f",
    2048: "#edc22e",
}
FONT = ("Verdana", 24, "bold")


class Game2048:
    def __init__(self, run_without_gui=False):
        self.run_without_gui = run_without_gui
        self.grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.cells = []
        self.score = 0
        self.game_running = True
        self.move_count = 0  # Track number of moves for screenshot naming

        # Create screenshots directory if it doesn't exist
        self.screenshots_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "screenshots"
        )
        os.makedirs(self.screenshots_dir, exist_ok=True)

        if not run_without_gui:
            self.window = tk.Tk()
            self.window.title("2048")
            self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.init_gui()
            self.start_game()
            self.ai_task = None
            self.schedule_ai_move()
            self.window.mainloop()
        else:
            self.start_game()

    def run(self):
        """Run the game without GUI and return final score and max tile."""
        if not self.run_without_gui:
            return self.score, max(max(row) for row in self.grid)

        # Run the game until game over
        while self.can_move(self.grid):
            best_move = self.get_best_move()
            if best_move:
                best_move()
                self.add_new_tile()
                self.move_count += 1
            else:
                break

        # Find the max tile value
        max_tile = max(max(row) for row in self.grid)
        return self.score, max_tile

    def init_gui(self):
        """Initialize the game's graphical user interface."""
        if self.run_without_gui:
            return
        try:
            self.frame = tk.Frame(self.window, bg=BACKGROUND_COLOR)
            self.frame.grid()

            self.score_label = tk.Label(
                self.frame,
                text=f"Score: {self.score}",
                font=("Verdana", 20, "bold"),
                bg=BACKGROUND_COLOR,
                padx=5,
                pady=5,
            )
            self.score_label.grid(row=0, column=0, columnspan=GRID_SIZE, sticky="nsew")

            for i in range(GRID_SIZE):
                row_cells = []
                for j in range(GRID_SIZE):
                    cell_frame = tk.Frame(
                        self.frame, bg=TILE_COLORS[0], width=100, height=100
                    )
                    cell_frame.grid(row=i + 1, column=j, padx=5, pady=5)
                    cell = tk.Label(
                        self.frame,
                        text="",
                        width=4,
                        height=2,
                        font=FONT,
                        bg=TILE_COLORS[0],
                    )
                    cell.grid(row=i + 1, column=j)
                    row_cells.append(cell)
                self.cells.append(row_cells)
        except tk.TclError as e:
            print(f"Error initializing GUI: {e}")
            self.game_running = False

    def evaluate(self) -> float:
        """Evaluate the current board position using multiple heuristics."""
        empty_count = sum(row.count(0) for row in self.grid) * 10000
        mono_score = self._calculate_monotonicity()
        smooth_score = self._calculate_smoothness()
        corner_score = self._calculate_corner_score()

        # New heuristics
        clustering_score = self._calculate_tile_clustering() * 50
        empty_tile_ratio = self._calculate_empty_tile_ratio() * 2000
        merge_potential = self._calculate_merge_potential() * 1000

        return (
            empty_count
            + mono_score
            + smooth_score
            + corner_score
            + clustering_score
            + empty_tile_ratio
            + merge_potential
        )

    def _calculate_monotonicity(self) -> float:
        """Calculate how monotonic (ordered) the grid is."""
        mono_score = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if self.grid[i][j] and self.grid[i][j + 1]:
                    if self.grid[i][j] >= self.grid[i][j + 1]:
                        mono_score += self.grid[i][j]
        for j in range(GRID_SIZE):
            for i in range(GRID_SIZE - 1):
                if self.grid[i][j] and self.grid[i + 1][j]:
                    if self.grid[i][j] >= self.grid[i + 1][j]:
                        mono_score += self.grid[i][j]
        return mono_score

    def _calculate_smoothness(self) -> float:
        """Calculate the smoothness of the grid (difference between adjacent tiles)."""
        smoothness = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] != 0:
                    if j < GRID_SIZE - 1 and self.grid[i][j + 1] != 0:
                        smoothness -= abs(self.grid[i][j] - self.grid[i][j + 1])
                    if i < GRID_SIZE - 1 and self.grid[i + 1][j] != 0:
                        smoothness -= abs(self.grid[i][j] - self.grid[i + 1][j])
        return smoothness

    def _calculate_corner_score(self) -> float:
        """Calculate score based on high values in corners."""
        corners = [
            self.grid[0][0],
            self.grid[0][GRID_SIZE - 1],
            self.grid[GRID_SIZE - 1][0],
            self.grid[GRID_SIZE - 1][GRID_SIZE - 1],
        ]
        return max(corners) * 2.0

    def _calculate_tile_clustering(self) -> float:
        """Calculate how well large tiles are clustered together."""
        clustering_score = 0

        # Find the mean position of tiles weighted by their values
        total_value = 0
        weighted_x_sum = 0
        weighted_y_sum = 0

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] > 0:
                    total_value += self.grid[i][j]
                    weighted_x_sum += j * self.grid[i][j]
                    weighted_y_sum += i * self.grid[i][j]

        if total_value > 0:
            mean_x = weighted_x_sum / total_value
            mean_y = weighted_y_sum / total_value

            # Calculate distance from each tile to the mean position
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    if self.grid[i][j] > 0:
                        # Higher values for larger tiles that are closer to the center of mass
                        distance = ((i - mean_y) ** 2 + (j - mean_x) ** 2) ** 0.5
                        clustering_score += self.grid[i][j] / (1 + distance)

        return clustering_score

    def _calculate_empty_tile_ratio(self) -> float:
        """Calculate the ratio of empty tiles to non-empty tiles."""
        empty_count = sum(row.count(0) for row in self.grid)
        non_empty_count = GRID_SIZE * GRID_SIZE - empty_count

        if non_empty_count == 0:  # Avoid division by zero
            return GRID_SIZE * GRID_SIZE

        return empty_count / (non_empty_count + 1)  # +1 to avoid division by zero

    def _calculate_merge_potential(self) -> float:
        """Calculate the potential for merging tiles on the board."""
        merge_score = 0

        # Check horizontal merge potential
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if self.grid[i][j] != 0 and self.grid[i][j] == self.grid[i][j + 1]:
                    merge_score += self.grid[i][j] * 2  # Value after merge

        # Check vertical merge potential
        for i in range(GRID_SIZE - 1):
            for j in range(GRID_SIZE):
                if self.grid[i][j] != 0 and self.grid[i][j] == self.grid[i + 1][j]:
                    merge_score += self.grid[i][j] * 2  # Value after merge

        return merge_score

    def expectimax(
        self, depth: int, agent_type: str, test_grid: List[List[int]]
    ) -> float:
        """Implement expectimax algorithm with three types of nodes."""
        if depth == 0:
            return self.evaluate()

        original_grid = self.grid
        original_score = self.score
        self.grid = test_grid

        if agent_type == "max":
            possible_moves = self.get_possible_moves(test_grid)
            if not possible_moves:
                self.grid = original_grid
                self.score = original_score
                return self.evaluate()

            max_value = float("-inf")
            for move in possible_moves:
                temp_grid = copy.deepcopy(test_grid)
                self.grid = temp_grid
                move()
                value = self.expectimax(depth - 1, "chance", self.grid)
                max_value = max(max_value, value)

            self.grid = original_grid
            self.score = original_score
            return max_value

        elif agent_type == "chance":
            empty_cells = [
                (i, j)
                for i in range(GRID_SIZE)
                for j in range(GRID_SIZE)
                if test_grid[i][j] == 0
            ]

            if not empty_cells:
                self.grid = original_grid
                self.score = original_score
                return self.evaluate()

            avg_value = 0
            num_possibilities = len(empty_cells) * len(NEW_TILE_VALUES)
            probabilities = {2: 0.9, 4: 0.1}

            for i, j in empty_cells:
                for new_value in NEW_TILE_VALUES:
                    temp_grid = copy.deepcopy(test_grid)
                    temp_grid[i][j] = new_value
                    value = self.expectimax(depth - 1, "max", temp_grid)
                    avg_value += value * probabilities[new_value] / len(empty_cells)

            self.grid = original_grid
            self.score = original_score
            return avg_value

    def get_best_move(self, depth: int = 3) -> Optional[Callable]:
        """Find the best move using expectimax algorithm."""
        try:
            best_move = None
            max_value = float("-inf")
            current_grid = self.clone_grid()
            original_score = self.score

            for move in self.get_possible_moves(current_grid):
                temp_grid = copy.deepcopy(current_grid)
                temp_score = self.score
                self.grid = temp_grid
                move()
                value = self.expectimax(depth - 1, "chance", self.grid)
                if value > max_value:
                    max_value = value
                    best_move = move
                self.score = temp_score

            self.grid = current_grid
            self.score = original_score
            return best_move
        except Exception as e:
            print(f"Error in get_best_move: {e}")
            return None

    def start_game(self):
        """Start the game by adding two initial tiles."""
        self.add_new_tile()
        self.add_new_tile()
        self.update_gui()

        # Take screenshot at the beginning of the game
        if not self.run_without_gui:
            self.window.after(500, lambda: self.take_screenshot("start"))

    def add_new_tile(self):
        """Add a new tile (2 or 4) to a random empty cell."""
        empty_cells = [
            (i, j)
            for i in range(GRID_SIZE)
            for j in range(GRID_SIZE)
            if self.grid[i][j] == 0
        ]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.grid[i][j] = random.choices(NEW_TILE_VALUES, weights=[0.9, 0.1])[0]

    def update_gui(self):
        """Update the GUI to reflect the current state of the grid."""
        if self.run_without_gui:
            return
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                value = self.grid[i][j]
                self.cells[i][j].configure(
                    text=str(value) if value != 0 else "", bg=TILE_COLORS[value]
                )
        self.score_label.configure(text=f"Score: {self.score}")
        self.window.update_idletasks()

    def clone_grid(self) -> List[List[int]]:
        """Return a deep copy of the current grid."""
        return copy.deepcopy(self.grid)

    def get_possible_moves(self, grid: List[List[int]]) -> List[Callable]:
        """Return a list of possible move functions."""
        moves = []
        if self.can_move_left(grid):
            moves.append(self.move_left)
        if self.can_move_right(grid):
            moves.append(self.move_right)
        if self.can_move_up(grid):
            moves.append(self.move_up)
        if self.can_move_down(grid):
            moves.append(self.move_down)
        return moves

    def take_screenshot(self, reason="move"):
        """Take a screenshot of the current game state."""
        if self.run_without_gui:
            return
        try:
            # Ensure window is updated before taking screenshot
            self.window.update_idletasks()
            self.window.update()

            # Get window position and size
            x, y, width, height = (
                self.window.winfo_rootx(),
                self.window.winfo_rooty(),
                self.window.winfo_width(),
                self.window.winfo_height(),
            )

            # Set filename based on reason
            if reason == "start":
                filename = "game_start_expectimax.png"
                text = "BAŞLANGIÇ"  # Changed from "Expectimax Algorithm" to "BAŞLANGIÇ"
            elif reason == "mid":
                filename = "game_mid_expectimax.png"
                text = "Expectimax Algorithm"
            elif reason == "gameover":
                filename = "game_over_expectimax.png"
                text = "Expectimax Algorithm"
            else:
                return  # Don't take screenshots for other reasons

            filepath = os.path.join(self.screenshots_dir, filename)

            # Capture screenshot using pyautogui like in the greedy version
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            screenshot = screenshot.convert("RGB")

            # Add algorithm name to the screenshot
            draw = ImageDraw.Draw(screenshot)
            try:
                font = ImageFont.truetype("arial.ttf", 36)
            except IOError:
                # Fallback to default font if arial.ttf is not available
                font = ImageFont.load_default()

            draw.text((10, 50), text, fill="black", font=font)

            # Save the screenshot
            screenshot.save(filepath)
            print(f"Screenshot saved: {filepath}")
        except Exception as e:
            print(f"Error taking screenshot: {e}")

    def schedule_ai_move(self):
        """Schedule the AI to make a move."""
        if self.run_without_gui:
            return
        if self.game_running:
            best_move = self.get_best_move()
            if best_move:
                best_move()
                self.add_new_tile()
                self.update_gui()
                self.move_count += 1

                # Take screenshot only at move 100
                if self.move_count == 200:
                    self.take_screenshot("mid")

                if self.game_over():
                    self.game_running = False
                    self.take_screenshot("gameover")  # Take screenshot at game over
                    print("Game Over!")
            self.window.after(50, self.schedule_ai_move)

    def ai_play(self):
        """Make the AI play a move."""
        best_move = self.get_best_move()
        if best_move:
            best_move()
            self.add_new_tile()
            self.update_gui()

    def move_left(self):
        """Move tiles left."""
        self.move(self.grid, self.compress, self.merge, self.compress)

    def move_right(self):
        """Move tiles right."""
        self.move(
            self.grid,
            self.reverse,
            self.compress,
            self.merge,
            self.compress,
            self.reverse,
        )

    def move_up(self):
        """Move tiles up."""
        self.move(
            self.grid,
            self.transpose,
            self.compress,
            self.merge,
            self.compress,
            self.transpose,
        )

    def move_down(self):
        """Move tiles down."""
        self.move(
            self.grid,
            self.transpose,
            self.reverse,
            self.compress,
            self.merge,
            self.compress,
            self.reverse,
            self.transpose,
        )

    def move(self, grid, *steps):
        """Apply a sequence of steps to the grid."""
        for step in steps:
            step(grid)

    def compress(self, grid):
        """Compress the grid by sliding tiles to the left."""
        for i in range(GRID_SIZE):
            new_row = [tile for tile in grid[i] if tile != 0]
            new_row += [0] * (GRID_SIZE - len(new_row))
            grid[i] = new_row

    def merge(self, grid):
        """Merge tiles in the grid."""
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if grid[i][j] == grid[i][j + 1] and grid[i][j] != 0:
                    grid[i][j] *= 2
                    grid[i][j + 1] = 0
                    self.score += grid[i][j]

    def reverse(self, grid):
        """Reverse the rows of the grid."""
        for i in range(GRID_SIZE):
            grid[i] = grid[i][::-1]

    def transpose(self, grid):
        """Transpose the grid (swap rows and columns)."""
        grid[:] = [list(row) for row in zip(*grid)]

    def can_move(self, grid) -> bool:
        """Check if any moves are possible."""
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if grid[i][j] == 0:
                    return True
                if j < GRID_SIZE - 1 and grid[i][j] == grid[i][j + 1]:
                    return True
                if i < GRID_SIZE - 1 and grid[i][j] == grid[i + 1][j]:
                    return True
        return False

    def can_move_left(self, grid) -> bool:
        """Check if a move left is possible."""
        for i in range(GRID_SIZE):
            for j in range(1, GRID_SIZE):
                if grid[i][j] != 0 and (
                    grid[i][j - 1] == 0 or grid[i][j - 1] == grid[i][j]
                ):
                    return True
        return False

    def can_move_right(self, grid) -> bool:
        """Check if a move right is possible."""
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if grid[i][j] != 0 and (
                    grid[i][j + 1] == 0 or grid[i][j + 1] == grid[i][j]
                ):
                    return True
        return False

    def can_move_up(self, grid) -> bool:
        """Check if a move up is possible."""
        for i in range(1, GRID_SIZE):
            for j in range(GRID_SIZE):
                if grid[i][j] != 0 and (
                    grid[i - 1][j] == 0 or grid[i - 1][j] == grid[i][j]
                ):
                    return True
        return False

    def can_move_down(self, grid) -> bool:
        """Check if a move down is possible."""
        for i in range(GRID_SIZE - 1):
            for j in range(GRID_SIZE):
                if grid[i][j] != 0 and (
                    grid[i + 1][j] == 0 or grid[i + 1][j] == grid[i][j]
                ):
                    return True
        return False

    def game_over(self) -> bool:
        """Check if the game is over."""
        if not self.can_move(self.grid):
            return True
        return False

    def on_closing(self):
        """Handle the window closing event."""
        self.game_running = False
        self.window.destroy()

    def is_terminal(self, grid: List[List[int]]) -> bool:
        """Check if the grid is in a terminal state (no moves possible)."""
        return not self.can_move(grid)

    def grid_to_tuple(self, grid: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
        """Convert grid to hashable tuple format for memoization."""
        return tuple(tuple(row) for row in grid)

    @lru_cache(maxsize=10000)
    def cached_expectimax(
        self,
        depth: int,
        agent_type: str,
        grid_tuple: Tuple[Tuple[int, ...], ...],
        score: int,
    ) -> float:
        """Memoized version of expectimax."""
        # Convert tuple back to list
        grid = [list(row) for row in grid_tuple]

        # Save original grid and score
        original_grid = self.grid
        original_score = self.score

        # Use test grid temporarily
        self.grid = grid
        self.score = score

        # Run expectimax
        result = self.monte_carlo_expectimax(depth, agent_type, self.grid)

        # Restore original grid and score
        self.grid = original_grid
        self.score = original_score

        return result

    # Monte Carlo sürümü - rollout sayısı optimize edildi
    def monte_carlo_expectimax(
        self,
        depth: int,
        agent_type: str,
        test_grid: List[List[int]],
        rollouts: int = 10,
    ) -> float:
        """Monte Carlo version of expectimax algorithm with optimized rollouts."""
        # Early termination
        if depth == 0 or self.is_terminal(test_grid):
            return self.evaluate()

        original_grid = self.grid
        original_score = self.score
        self.grid = test_grid

        if agent_type == "max":
            possible_moves = self.get_possible_moves(test_grid)
            if not possible_moves:
                self.grid = original_grid
                self.score = original_score
                return self.evaluate()

            max_value = float("-inf")
            for move in possible_moves:
                temp_grid = copy.deepcopy(test_grid)
                self.grid = temp_grid
                move()
                grid_tuple = self.grid_to_tuple(self.grid)
                value = self.cached_expectimax(
                    depth - 1, "chance", grid_tuple, self.score
                )
                max_value = max(max_value, value)

            self.grid = original_grid
            self.score = original_score
            return max_value

        elif agent_type == "chance":
            empty_cells = [
                (i, j)
                for i in range(GRID_SIZE)
                for j in range(GRID_SIZE)
                if test_grid[i][j] == 0
            ]

            if not empty_cells:
                self.grid = original_grid
                self.score = original_score
                return self.evaluate()

            # Use Monte Carlo sampling with weight-biased selection
            total_value = 0
            probabilities = {2: 0.9, 4: 0.1}

            # Rollout değişikliği: 15'ten 10'a düşürüldü ve hücreler önem derecesine göre seçiliyor
            # Köşeler daha önemli olduğu için köşelere yakın hücrelere öncelik ver
            weighted_cells = []
            for i, j in empty_cells:
                # Köşelere yakınlık hesapla (0,0 köşesine yakınsa yüksek puan)
                corner_distance = min(
                    [
                        (i**2 + j**2),  # Sol üst köşeye mesafe
                        (i**2 + (GRID_SIZE - 1 - j) ** 2),  # Sağ üst köşeye mesafe
                        ((GRID_SIZE - 1 - i) ** 2 + j**2),  # Sol alt köşeye mesafe
                        (
                            (GRID_SIZE - 1 - i) ** 2 + (GRID_SIZE - 1 - j) ** 2
                        ),  # Sağ alt köşeye mesafe
                    ]
                )
                # Daha düşük mesafe = daha yüksek ağırlık
                weight = 1.0 / (1.0 + corner_distance)
                weighted_cells.append((weight, (i, j)))

            # Ağırlıklara göre sırala ve en iyi hücreleri seç
            weighted_cells.sort(reverse=True)  # En yüksek ağırlıklı hücreler ilk sırada

            # Rollout sayısını ve boş hücre sayısını karşılaştır
            actual_rollouts = min(rollouts, len(empty_cells))

            # En iyi hücrelerden başlayarak rollout yap
            for _, (i, j) in weighted_cells[:actual_rollouts]:
                for new_value in NEW_TILE_VALUES:
                    temp_grid = copy.deepcopy(test_grid)
                    temp_grid[i][j] = new_value

                    grid_tuple = self.grid_to_tuple(temp_grid)
                    value = self.cached_expectimax(
                        depth - 1, "max", grid_tuple, self.score
                    )

                    # Weight the value by probability
                    total_value += value * probabilities[new_value]

            # Average the results
            avg_value = total_value / (actual_rollouts * len(NEW_TILE_VALUES))

            self.grid = original_grid
            self.score = original_score
            return avg_value

    def get_best_move(self, depth: int = 3) -> Optional[Callable]:
        """Find the best move using optimized Monte Carlo expectimax algorithm."""
        try:
            start_time = time.time()
            best_move = None
            max_value = float("-inf")
            current_grid = self.clone_grid()
            original_score = self.score

            # Rollout sayısını 10'a düşür
            rollouts = 10

            for move in self.get_possible_moves(current_grid):
                temp_grid = copy.deepcopy(current_grid)
                temp_score = self.score
                self.grid = temp_grid
                move()
                grid_tuple = self.grid_to_tuple(self.grid)
                value = self.cached_expectimax(
                    depth - 1, "chance", grid_tuple, self.score
                )
                if value > max_value:
                    max_value = value
                    best_move = move
                self.score = temp_score

            self.grid = current_grid
            self.score = original_score

            # Cache temizlemesini sadece işlem uzun sürdüğünde ve periyodik olarak yap
            if time.time() - start_time > 0.5 or self.move_count % 150 == 0:
                self.cached_expectimax.cache_clear()

            return best_move
        except Exception as e:
            print(f"Error in get_best_move: {e}")
            return None


if __name__ == "__main__":
    Game2048()
