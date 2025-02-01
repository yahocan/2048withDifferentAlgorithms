import tkinter as tk
import random
import copy
from typing import Callable, List, Optional, Tuple

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
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("2048")
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.cells = []
        self.score = 0
        self.game_running = True
        self.init_gui()
        self.start_game()
        self.ai_task = None
        self.schedule_ai_move()
        self.window.mainloop()

    def init_gui(self):
        """Initialize the game's graphical user interface."""
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
        return empty_count + mono_score + smooth_score + corner_score

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

    def schedule_ai_move(self):
        """Schedule the AI to make a move."""
        if self.game_running:
            best_move = self.get_best_move()
            if best_move:
                best_move()
                self.add_new_tile()
                self.update_gui()
                if self.game_over():
                    self.game_running = False
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


if __name__ == "__main__":
    Game2048()
