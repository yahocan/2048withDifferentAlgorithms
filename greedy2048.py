import tkinter as tk
import random
import copy
from typing import Callable, List, Optional, Tuple
import pyautogui
from PIL import Image, ImageDraw, ImageFont

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

    def evaluate_move(self, move: Callable) -> int:
        original_grid = self.clone_grid()
        original_score = self.score
        temp_grid = self.clone_grid()
        self.grid = temp_grid
        move_happened = move()  # Check if the move actually changes the grid

        if move_happened:
            # Calculate basic score difference
            score_diff = self.score - original_score

            # Add new heuristics
            empty_cells_score = self._count_empty_cells() * 10
            clustering_score = self._calculate_clustering() / 100
            max_tile_placement_score = self._evaluate_max_tile_placement() * 10

            total_score = (
                score_diff
                + empty_cells_score
                + clustering_score
                + max_tile_placement_score
            )
        else:
            total_score = -1  # Return -1 for invalid moves

        self.grid = original_grid
        self.score = original_score
        return total_score

    def _count_empty_cells(self) -> int:
        """Count the number of empty cells on the grid."""
        return sum(row.count(0) for row in self.grid)

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

    def get_best_move(self) -> Optional[Callable]:
        best_move = None
        max_score = -1
        current_grid = self.clone_grid()

        # Only consider moves that are possible (change the grid)
        possible_moves = self.get_possible_moves(current_grid)

        for move in possible_moves:
            score_diff = self.evaluate_move(move)
            if score_diff > max_score:
                max_score = score_diff
                best_move = move

        return best_move

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
                        self.take_screenshot("game_mid_greedy.png", "Greedy Algorithm")
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
                "game_over_greedy.png", "Greedy Algorithm"
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
