# heuristic: the biggest number in the grid is on the left corner

import tkinter as tk
import random
import copy
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
        self.move_count = 0  # Add a counter for moves
        self.screenshot_taken = False  # Flag to check if screenshot is taken

        if not run_without_gui:
            self.window = tk.Tk()
            self.window.title("2048")
            self.init_gui()
            self.start_game()
            self.autoplay()
            self.window.mainloop()
        else:
            self.start_game()

    def run(self):
        """Run the game without GUI and return final score and max tile."""
        if not self.run_without_gui:
            return self.score, max(max(row) for row in self.grid)

        # Run the game until game over
        while self.can_move():
            best_move = self.a_star_best_move()
            if best_move:
                key_map = {
                    "Up": self.move_up,
                    "Down": self.move_down,
                    "Left": self.move_left,
                    "Right": self.move_right,
                }
                moved = key_map[best_move]()

                if moved:
                    self.add_new_tile()
                    self.move_count += 1
            else:
                break

        # Find the max tile value
        max_tile = max(max(row) for row in self.grid)
        return self.score, max_tile

    def init_gui(self):
        if self.run_without_gui:
            return

        self.frame = tk.Frame(self.window, bg=BACKGROUND_COLOR)
        self.frame.grid()
        self.score_label = tk.Label(
            self.window,
            text=f"Score: {self.score}",
            font=("Verdana", 20, "bold"),
            bg=BACKGROUND_COLOR,
            padx=5,
            pady=5,
        )
        self.score_label.grid(row=2, column=0, columnspan=GRID_SIZE, sticky="nsew")
        for i in range(GRID_SIZE):
            row_cells = []
            for j in range(GRID_SIZE):
                cell = tk.Label(
                    self.frame, text="", width=4, height=2, font=FONT, bg=TILE_COLORS[0]
                )
                cell.grid(row=i + 1, column=j, padx=5, pady=5)
                row_cells.append(cell)
            self.cells.append(row_cells)
        self.window.bind("<KeyPress>", self.handle_keypress)

    def start_game(self):
        self.add_new_tile()
        self.add_new_tile()
        self.update_gui()

    def add_new_tile(self):
        empty_cells = [
            (i, j)
            for i in range(GRID_SIZE)
            for j in range(GRID_SIZE)
            if self.grid[i][j] == 0
        ]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.grid[i][j] = random.choice(NEW_TILE_VALUES)

    def update_gui(self):
        if self.run_without_gui:
            return

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                value = self.grid[i][j]
                self.cells[i][j].config(
                    text=str(value) if value else "",
                    bg=TILE_COLORS.get(value, "#edc22e"),
                )
        self.score_label.config(text=f"Score: {self.score}")
        self.window.update_idletasks()

    def take_screenshot(self, filename: str, text: str):
        if self.run_without_gui:
            return

        """Take a screenshot of the game window and add text."""
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
        draw.text((10, 10), text, fill="black", font=font)
        screenshot.save(filename)

    def autoplay(self):
        if self.run_without_gui:
            return

        if not self.can_move():
            self.game_over()
            return

        best_move = self.a_star_best_move()

        if best_move:
            key_map = {
                "Up": self.move_up,
                "Down": self.move_down,
                "Left": self.move_left,
                "Right": self.move_right,
            }
            moved = key_map[best_move]()  # Sadece bir yön hareket etsin

            if moved:  # Eğer gerçekten hareket ettiyse yeni taş ekle
                self.add_new_tile()
                self.update_gui()

                self.move_count += 1  # Increment move counter

                # Take a screenshot after a certain number of moves
                if self.move_count == 70 and not self.screenshot_taken:
                    self.take_screenshot("game_mid_astar.png", "A* Algorithm")
                    self.screenshot_taken = True

        self.window.after(100, self.autoplay)  # 500ms sonra tekrar çağır

    def handle_keypress(self, event):
        best_move = self.a_star_best_move()
        if best_move:
            key_map = {
                "Up": self.move_up,
                "Down": self.move_down,
                "Left": self.move_left,
                "Right": self.move_right,
            }
            moved = key_map[best_move]()
            if moved:
                self.add_new_tile()
                self.update_gui()
                if not self.can_move():
                    self.game_over()

    def simulate_move(self, direction):
        temp_grid = copy.deepcopy(self.grid)  # Grid'in kopyasını al
        temp_score = self.score  # Skoru da sakla

        key_map = {
            "Up": self.move_up,
            "Down": self.move_down,
            "Left": self.move_left,
            "Right": self.move_right,
        }

        moved = key_map[direction]()  # Hareketi dene
        if moved:
            new_grid = copy.deepcopy(self.grid)  # Yeni grid'i kaydet
            new_score = self.score
            self.grid = temp_grid  # **ÖNEMLİ**: Eski grid'i geri yükle
            self.score = temp_score
            return new_grid, new_score

        self.grid = temp_grid  # Eğer hareket olmadıysa yine geri yükle
        self.score = temp_score
        return None, None

    def heuristic_corner_max_tile(self):
        max_tile = max(max(row) for row in self.grid)  # Find max tile value
        corners = {
            (0, 0),
            (0, GRID_SIZE - 1),
            (GRID_SIZE - 1, 0),
            (GRID_SIZE - 1, GRID_SIZE - 1),
        }

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] == max_tile and (i, j) in corners:
                    return max_tile  # Bonus if max tile is in a corner

        return 0  # No bonus if max tile is not in a corner

    def heuristic_monotonicity(self):
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

    def heuristic_smoothness(self):
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

    def heuristic_tile_clustering(self):
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

    def a_star_best_move(self):
        possible_moves = ["Up", "Down", "Left", "Right"]
        best_move = None
        best_score = -1

        for move in possible_moves:
            new_grid, new_score = self.simulate_move(move)

            if new_grid:  # If move is possible
                # Calculate all heuristics
                corner_score = self.heuristic_corner_max_tile()
                empty_tiles = sum(
                    row.count(0) for row in new_grid
                )  # Encourage empty spaces
                mono_score = self.heuristic_monotonicity() / 1000  # Scale down
                smooth_score = self.heuristic_smoothness() / 100  # Scale appropriately
                cluster_score = (
                    self.heuristic_tile_clustering() / 100
                )  # Scale appropriately

                # Weighted sum of all heuristics
                total_score = (
                    new_score
                    + corner_score
                    + empty_tiles * 10
                    + mono_score
                    + smooth_score
                    + cluster_score
                )

                if total_score > best_score:
                    best_score = total_score
                    best_move = move

        return best_move

    def move_left(self):
        return self.move(lambda row: self.compress(row), lambda row: self.merge(row))

    def move_right(self):
        return self.move(
            lambda row: self.compress(row[::-1])[::-1],
            lambda row: self.merge(row[::-1])[::-1],
        )

    def move_up(self):
        return self.move_columns(
            lambda col: self.compress(col), lambda col: self.merge(col)
        )

    def move_down(self):
        return self.move_columns(
            lambda col: self.compress(col[::-1])[::-1],
            lambda col: self.merge(col[::-1])[::-1],
        )

    def move(self, compress_fn, merge_fn):
        moved = False
        for i in range(GRID_SIZE):
            original = self.grid[i][:]  # Store original row before move
            compressed = compress_fn(original)  # Shift tiles
            merged = merge_fn(compressed)  # Merge tiles
            new_row = compress_fn(merged)  # Shift again after merging

            if new_row != original:  # Ensure actual movement occurred
                self.grid[i] = new_row
                moved = True
        return moved

    def move_columns(self, compress_fn, merge_fn):
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

    def compress(self, row):
        if all(num == 0 for num in row):  # Eğer zaten boşsa işlem yapma
            return row
        return [num for num in row if num != 0] + [0] * row.count(0)

    def merge(self, row):
        for i in range(len(row) - 1):
            if row[i] != 0 and row[i] == row[i + 1]:  # Merge if same and nonzero
                row[i] *= 2
                self.score += row[i]
                row[i + 1] = 0  # Clear merged tile
        return row

    def can_move(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if self.grid[i][j] == self.grid[i][j + 1]:
                    return True
        for i in range(GRID_SIZE - 1):
            for j in range(GRID_SIZE):
                if self.grid[i][j] == self.grid[i + 1][j]:
                    return True
        return any(0 in row for row in self.grid)

    def game_over(self):
        if self.run_without_gui:
            return

        game_over_label = tk.Label(
            self.frame,
            text="Game Over!",
            font=("Verdana", 30, "bold"),
            bg=BACKGROUND_COLOR,
        )
        game_over_label.grid(row=0, column=0, columnspan=GRID_SIZE)
        self.take_screenshot(
            "game_over_astar.png", "A* Algorithm"
        )  # Take screenshot on game over


if __name__ == "__main__":
    Game2048()
