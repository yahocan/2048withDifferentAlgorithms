import tkinter as tk
import random
import copy
from typing import Callable, List, Optional, Tuple
import pyautogui
from PIL import Image, ImageDraw, ImageFont
from functools import lru_cache
import time
import multiprocessing
from multiprocessing import Pool

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
        self.move_count = 0  # Add a counter for moves
        self.screenshot_taken = False  # Flag to check if screenshot is taken

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
        while self.can_move():
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

            # Move score label to row 0 (top) and make it visible
            self.score_label = tk.Label(
                self.frame,
                text=f"Score: {self.score}",
                font=("Verdana", 20, "bold"),
                bg=BACKGROUND_COLOR,
                padx=5,
                pady=5,
            )
            self.score_label.grid(row=0, column=0, columnspan=GRID_SIZE, sticky="nsew")

            # Move the game grid to start at row 1
            for i in range(GRID_SIZE):
                row_cells = []
                for j in range(GRID_SIZE):
                    cell_frame = tk.Frame(
                        self.frame, bg=TILE_COLORS[0], width=100, height=100
                    )
                    cell_frame.grid(
                        row=i + 1, column=j, padx=5, pady=5
                    )  # Note the i+1 here
                    cell = tk.Label(
                        self.frame,
                        text="",
                        width=4,
                        height=2,
                        font=FONT,
                        bg=TILE_COLORS[0],
                    )
                    cell.grid(row=i + 1, column=j)  # Note the i+1 here
                    row_cells.append(cell)
                self.cells.append(row_cells)
        except tk.TclError as e:
            print(f"Error initializing GUI: {e}")
            self.game_running = False

    def start_game(self):
        """Initialize the game state."""
        self.add_new_tile()
        self.add_new_tile()
        self.update_gui()

    def add_new_tile(self) -> bool:
        """Add a new tile to the grid. Returns True if successful, False if no empty cells."""
        empty_cells = [
            (i, j)
            for i in range(GRID_SIZE)
            for j in range(GRID_SIZE)
            if self.grid[i][j] == 0
        ]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.grid[i][j] = random.choice(NEW_TILE_VALUES)
            return True
        return False

    def update_gui(self):
        """Update the GUI to reflect the current game state."""
        if self.run_without_gui:
            return

        try:
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    value = self.grid[i][j]
                    self.cells[i][j].config(
                        text=str(value) if value else "",
                        bg=TILE_COLORS.get(value, "#edc22e"),
                    )
            self.score_label.config(text=f"Score: {self.score}")
        except tk.TclError:
            self.game_running = False

    def clone_grid(self) -> List[List[int]]:
        """Create a deep copy of the current grid."""
        return copy.deepcopy(self.grid)

    def get_possible_moves(self, test_grid: List[List[int]]) -> List[Callable]:
        """Get all possible moves for a given grid state."""
        moves = []
        original_grid = self.grid
        original_score = self.score
        self.grid = test_grid

        for move in [self.move_up, self.move_down, self.move_left, self.move_right]:
            temp_grid = copy.deepcopy(test_grid)
            self.grid = temp_grid
            if move():
                moves.append(move)

        self.score = original_score
        self.grid = original_grid
        return moves

    def evaluate(self) -> int:
        """Evaluate the current board position using multiple heuristics."""
        # Base heuristic: empty cell count
        empty_cell_score = sum(row.count(0) for row in self.grid) * 100

        # New heuristics
        monotonicity_score = self._calculate_monotonicity() / 10
        smoothness_score = self._calculate_smoothness() * 5
        clustering_score = self._calculate_clustering() / 50
        max_tile_placement_score = self._evaluate_max_tile_placement() * 15

        return (
            empty_cell_score
            + monotonicity_score
            + smoothness_score
            + clustering_score
            + max_tile_placement_score
        )

    def _calculate_monotonicity(self) -> float:
        """Calculate how monotonic (ordered) the grid is."""
        mono_score = 0

        # Check horizontal monotonicity (decreasing from left to right)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if self.grid[i][j] and self.grid[i][j + 1]:
                    if self.grid[i][j] >= self.grid[i][j + 1]:
                        mono_score += self.grid[i][j]

        # Check vertical monotonicity (decreasing from top to bottom)
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

    def _calculate_clustering(self) -> float:
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

    def _evaluate_max_tile_placement(self) -> int:
        """Evaluate the position of the maximum tile on the grid."""
        max_val = 0
        max_i, max_j = 0, 0

        # Find the maximum tile value and its position
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] > max_val:
                    max_val = self.grid[i][j]
                    max_i, max_j = i, j

        # Check if the max tile is in a corner (best position)
        corner_positions = [
            (0, 0),
            (0, GRID_SIZE - 1),
            (GRID_SIZE - 1, 0),
            (GRID_SIZE - 1, GRID_SIZE - 1),
        ]
        if (max_i, max_j) in corner_positions:
            return max_val  # Maximum score for corner placement

        # Check if it's on an edge
        if max_i == 0 or max_i == GRID_SIZE - 1 or max_j == 0 or max_j == GRID_SIZE - 1:
            return max_val // 2  # Half score for edge placement

        return 0  # No bonus for center placement

    def is_terminal(self, grid: List[List[int]]) -> bool:
        """Check if the grid is in a terminal state (no moves possible)."""
        # Check for empty cells
        if any(0 in row for row in grid):
            return False

        # Check for possible merges horizontally
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if grid[i][j] == grid[i][j + 1]:
                    return False

        # Check for possible merges vertically
        for i in range(GRID_SIZE - 1):
            for j in range(GRID_SIZE):
                if grid[i][j] == grid[i + 1][j]:
                    return False

        return True  # No moves possible

    def evaluate_move_for_sorting(
        self, move: Callable, test_grid: List[List[int]]
    ) -> float:
        """Evaluate a move quickly for move ordering purposes."""
        temp_grid = copy.deepcopy(test_grid)
        original_grid = self.grid
        self.grid = temp_grid

        # Perform the move
        move()

        # Quick heuristic: prefer more empty cells and higher scores
        empty_count = sum(row.count(0) for row in self.grid)
        max_value = max(max(row) for row in self.grid)
        corner_score = self._evaluate_max_tile_placement()

        # Restore original grid
        self.grid = original_grid

        return empty_count * 10 + max_value + corner_score

    def get_sorted_moves(self, test_grid: List[List[int]]) -> List[Callable]:
        """Get possible moves sorted by heuristic evaluation (best first)."""
        possible_moves = self.get_possible_moves(test_grid)

        # Sort moves by a quick evaluation (best first)
        return sorted(
            possible_moves,
            key=lambda move: self.evaluate_move_for_sorting(move, test_grid),
            reverse=True,
        )

    def grid_to_tuple(self, grid: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
        """Convert grid to hashable tuple format for memoization."""
        return tuple(tuple(row) for row in grid)

    # Cached version of minimax with lru_cache for memoization
    @lru_cache(maxsize=1000)  # 10000'den 1000'e düşürüldü
    def cached_minimax(
        self,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool,
        grid_tuple: Tuple[Tuple[int, ...], ...],
        score: int,
    ) -> float:
        """Memoized version of minimax."""
        # Convert tuple back to list
        grid = [list(row) for row in grid_tuple]

        # Save original grid and score
        original_grid = self.grid
        original_score = self.score

        # Use test grid temporarily
        self.grid = grid
        self.score = score

        # Run minimax
        result = self.minimax(depth, alpha, beta, maximizing_player, self.grid)

        # Restore original grid and score
        self.grid = original_grid
        self.score = original_score

        return result

    def minimax(
        self,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool,
        test_grid: List[List[int]],
    ) -> float:
        """Optimized minimax algorithm with alpha-beta pruning and early termination."""
        # Early termination check
        if depth == 0 or self.is_terminal(test_grid):
            return self.evaluate()

        original_grid = self.grid
        original_score = self.score
        self.grid = test_grid

        if maximizing_player:
            max_eval = float("-inf")
            # Use sorted moves for better alpha-beta pruning
            for move in self.get_sorted_moves(test_grid):
                temp_grid = copy.deepcopy(test_grid)
                self.grid = temp_grid
                move()

                # Use memoization for recursive calls
                grid_tuple = self.grid_to_tuple(self.grid)
                eval = self.cached_minimax(
                    depth - 1, alpha, beta, False, grid_tuple, self.score
                )

                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)

                if beta <= alpha:
                    break  # Alpha-beta pruning

            self.score = original_score
            self.grid = original_grid
            return max_eval
        else:
            min_eval = float("inf")
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

            # Probabilistic weighting for 2 and 4 tiles
            probabilities = {2: 0.9, 4: 0.1}

            for i, j in empty_cells:
                for new_value in NEW_TILE_VALUES:
                    temp_grid = copy.deepcopy(test_grid)
                    temp_grid[i][j] = new_value

                    # Weight by probability of tile appearing
                    grid_tuple = self.grid_to_tuple(temp_grid)
                    eval = self.cached_minimax(
                        depth - 1, alpha, beta, True, grid_tuple, self.score
                    )
                    weighted_eval = eval * probabilities[new_value]

                    min_eval = min(min_eval, weighted_eval)
                    beta = min(beta, weighted_eval)

                    if beta <= alpha:
                        break  # Alpha-beta pruning

            self.grid = original_grid
            self.score = original_score
            return min_eval

    def parallel_process_move(self, move, grid, depth):
        """Modified helper function for parallel move evaluation without pickling issues."""
        # Process a single move evaluation without passing self to multiprocessing
        temp_grid = copy.deepcopy(grid)

        # Create a new instance of Game2048 just for this evaluation
        # to avoid pickling issues with the tkinter window
        temp_game = Game2048(run_without_gui=True)
        temp_game.grid = temp_grid
        temp_game.score = self.score

        # Make the move
        move_func = getattr(temp_game, move.__name__)
        move_func()

        # Evaluate using minimax directly on the temporary game
        result = temp_game.evaluate()

        return (move, result)

    def get_best_move(
        self, depth: int = 2
    ) -> Optional[Callable]:  # derinlik 3'ten 2'ye düşürüldü
        """Get best move using optimized minimax without parallel processing."""
        try:
            best_move = None
            max_eval = float("-inf")
            current_grid = self.clone_grid()
            original_score = self.score
            possible_moves = self.get_sorted_moves(current_grid)

            # For now, always use sequential processing to avoid pickling errors
            # Sequential processing
            for move in possible_moves:
                temp_grid = copy.deepcopy(current_grid)
                temp_score = self.score
                self.grid = temp_grid
                move()

                # Use memoization
                grid_tuple = self.grid_to_tuple(self.grid)
                eval = self.cached_minimax(
                    depth - 1,
                    float("-inf"),
                    float("inf"),
                    False,
                    grid_tuple,
                    self.score,
                )

                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                self.score = original_score

            self.grid = current_grid
            self.score = original_score

            # Periyodik olarak cache'i temizle (her 100 hamlede bir)
            if self.move_count % 100 == 0:
                self.cached_minimax.cache_clear()

            return best_move

        except Exception as e:
            print(f"Error in get_best_move: {e}")
            return None

    def schedule_ai_move(self):
        """Schedule the next AI move if the game is still running."""
        if self.run_without_gui:
            return

        if self.game_running:
            self.ai_task = self.window.after(50, self.ai_play)

    def take_screenshot(self, filename: str, text: str):
        """Take a screenshot of the game window and add text."""
        if self.run_without_gui:
            return

        x, y, width, height = (
            self.window.winfo_rootx(),
            self.window.winfo_rooty(),
            self.window.winfo_width(),
            self.window.winfo_height(),
        )
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        screenshot = screenshot.convert("RGB")
        draw = ImageDraw.Draw(screenshot)
        font = ImageFont.truetype("arial.ttf", 36)
        draw.text((10, 50), text, fill="black", font=font)  # Adjusted position
        screenshot.save(filename)

    def ai_play(self):
        """Execute AI move and schedule the next one if the game is still running."""
        if self.run_without_gui:
            return

        if self.game_running and self.can_move():
            try:
                best_move = self.get_best_move()
                if best_move:
                    best_move()
                    self.add_new_tile()
                    self.update_gui()

                    self.move_count += 1  # Increment move counter

                    # Take a screenshot after a certain number of moves
                    if self.move_count == 70 and not self.screenshot_taken:
                        self.take_screenshot(
                            "game_mid_minimax.png", "Minimax Algorithm"
                        )
                        self.screenshot_taken = True

                    if not self.can_move():
                        self.game_over()
                    else:
                        self.schedule_ai_move()
            except Exception as e:
                print(f"Error in AI play: {e}")
                self.game_over()
        elif not self.can_move():
            self.game_over()

    def move_left(self) -> bool:
        """Move tiles left and merge if possible."""
        return self.move(lambda row: self.compress(row), lambda row: self.merge(row))

    def move_right(self) -> bool:
        """Move tiles right and merge if possible."""
        return self.move(
            lambda row: self.compress(row[::-1])[::-1],
            lambda row: self.merge(row[::-1])[::-1],
        )

    def move_up(self) -> bool:
        """Move tiles up and merge if possible."""
        return self.move_columns(
            lambda col: self.compress(col), lambda col: self.merge(col)
        )

    def move_down(self) -> bool:
        """Move tiles down and merge if possible."""
        return self.move_columns(
            lambda col: self.compress(col[::-1])[::-1],
            lambda col: self.merge(col[::-1])[::-1],
        )

    def move(self, compress_fn: Callable, merge_fn: Callable) -> bool:
        """Generic move function for horizontal moves."""
        moved = False
        for i in range(GRID_SIZE):
            original = self.grid[i][:]
            compressed = compress_fn(original)
            merged = merge_fn(compressed)
            new_row = compress_fn(merged)
            if new_row != original:
                self.grid[i] = new_row
                moved = True
        return moved

    def move_columns(self, compress_fn: Callable, merge_fn: Callable) -> bool:
        """Generic move function for vertical moves."""
        moved = False
        for j in range(GRID_SIZE):
            original = [self.grid[i][j] for i in range(GRID_SIZE)]
            compressed = compress_fn(original)
            merged = merge_fn(compressed)
            new_col = compress_fn(merged)
            if new_col != original:
                for i in range(GRID_SIZE):
                    self.grid[i][j] = new_col[i]
                moved = True
        return moved

    def compress(self, row: List[int]) -> List[int]:
        """Remove zeros and shift numbers to one side."""
        return [num for num in row if num != 0] + [0] * row.count(0)

    def merge(self, row: List[int]) -> List[int]:
        """Merge adjacent equal numbers."""
        for i in range(len(row) - 1):
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]  # Update score
        return row

    def can_move(self) -> bool:
        """Check if any moves are possible."""
        # Check for empty cells
        if any(0 in row for row in self.grid):
            return True

        # Check for possible merges horizontally
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if self.grid[i][j] == self.grid[i][j + 1]:
                    return True

        # Check for possible merges vertically
        for i in range(GRID_SIZE - 1):
            for j in range(GRID_SIZE):
                if self.grid[i][j] == self.grid[i + 1][j]:
                    return True

        return False

    def game_over(self):
        if self.run_without_gui:
            return

        self.game_running = False
        try:
            game_over_label = tk.Label(
                self.frame,
                text=f"Game Over! Your score: {self.score}",
                font=("Verdana", 30, "bold"),
                bg=BACKGROUND_COLOR,
            )
            game_over_label.grid(
                row=GRID_SIZE + 1, column=0, columnspan=GRID_SIZE
            )  # Place below the grid
            self.take_screenshot(
                "game_over_minimax.png", "Minimax Algorithm"
            )  # Take screenshot on game over
        except tk.TclError as e:
            print(f"Error displaying game over: {e}")

    def on_closing(self):
        """Handle window closing event."""
        self.game_running = False
        if self.ai_task:
            self.window.after_cancel(self.ai_task)
        self.window.destroy()


if __name__ == "__main__":
    Game2048()
