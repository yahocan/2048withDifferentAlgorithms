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
        """Oyunun grafik arayüzünü başlat."""
        try:
            self.frame = tk.Frame(self.window, bg=BACKGROUND_COLOR)
            self.frame.grid()

            # Skor etiketini üste taşı ve görünür yap
            self.score_label = tk.Label(
                self.frame,
                text=f"Score: {self.score}",
                font=("Verdana", 20, "bold"),
                bg=BACKGROUND_COLOR,
                padx=5,
                pady=5,
            )
            self.score_label.grid(row=0, column=0, columnspan=GRID_SIZE, sticky="nsew")

            # Oyun gridini satır 1'den başlat
            for i in range(GRID_SIZE):
                row_cells = []
                for j in range(GRID_SIZE):
                    cell_frame = tk.Frame(
                        self.frame, bg=TILE_COLORS[0], width=100, height=100
                    )
                    cell_frame.grid(
                        row=i + 1, column=j, padx=5, pady=5
                    )  # i+1 burada önemli
                    cell = tk.Label(
                        self.frame,
                        text="",
                        width=4,
                        height=2,
                        font=FONT,
                        bg=TILE_COLORS[0],
                    )
                    cell.grid(row=i + 1, column=j)  # i+1 burada önemli
                    row_cells.append(cell)
                self.cells.append(row_cells)
        except tk.TclError as e:
            print(f"GUI başlatılırken hata oluştu: {e}")
            self.game_running = False

    def start_game(self):
        """Oyunu başlat - iki tane başlangıç taşı ekle."""
        self.add_new_tile()
        self.add_new_tile()
        self.update_gui()

    def add_new_tile(self) -> bool:
        """Rastgele bir boş hücreye yeni bir taş ekle (2 veya 4).
        Başarılıysa True, boş hücre yoksa False döndürür."""
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
        """Arayüzü güncelle - grid değerlerini ve skoru göster."""
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
        """Grid'in derin bir kopyasını oluştur."""
        return copy.deepcopy(self.grid)

    def get_possible_moves(self, test_grid: List[List[int]]) -> List[Callable]:
        """Belirli bir grid durumu için tüm olası hareketleri al."""
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
        """Mevcut tahta durumunu değerlendir - basit heuristic olarak boş hücre sayısı."""
        return sum(row.count(0) for row in self.grid)

    def minimax(
        self,
        depth: int,
        alpha: int,
        beta: int,
        maximizing_player: bool,
        test_grid: List[List[int]],
    ) -> int:
        """Alpha-beta pruning ile minimax algoritması."""
        # Önce maksimum derinliğe ulaşıp ulaşmadığımızı kontrol et
        if depth == 0:
            return self.evaluate()

        original_grid = self.grid
        original_score = self.score
        self.grid = test_grid

        # Mevcut pozisyonda hiçbir hamle mümkün değilse kontrol et
        possible_moves = self.get_possible_moves(test_grid)
        if not possible_moves and not any(0 in row for row in test_grid):
            self.grid = original_grid
            self.score = original_score
            return self.evaluate()

        if maximizing_player:
            max_eval = float("-inf")
            for move in possible_moves:
                temp_grid = copy.deepcopy(test_grid)
                self.grid = temp_grid
                move()
                eval = self.minimax(depth - 1, alpha, beta, False, self.grid)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Alpha-beta kesme (pruning)
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

            for i, j in empty_cells:
                for new_value in NEW_TILE_VALUES:
                    temp_grid = copy.deepcopy(test_grid)
                    temp_grid[i][j] = new_value
                    eval = self.minimax(depth - 1, alpha, beta, True, temp_grid)
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break  # Alpha-beta kesme (pruning)

            self.grid = original_grid
            self.score = original_score
            return min_eval

    def get_best_move(self, depth: int = 3) -> Optional[Callable]:
        """Minimax algoritmasını kullanarak en iyi hamleyi bul."""
        try:
            best_move = None
            max_eval = float("-inf")
            current_grid = self.clone_grid()
            original_score = self.score

            for move in self.get_possible_moves(current_grid):
                temp_grid = copy.deepcopy(current_grid)
                temp_score = self.score
                self.grid = temp_grid
                move()
                eval = self.minimax(
                    depth - 1, float("-inf"), float("inf"), False, self.grid
                )
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                self.score = original_score
            self.grid = current_grid
            self.score = original_score
            return best_move
        except Exception as e:
            print(f"get_best_move'da hata oluştu: {e}")
            return None

    def schedule_ai_move(self):
        """Oyun devam ediyorsa bir sonraki AI hamlesini zamanla."""
        if self.game_running:
            self.ai_task = self.window.after(50, self.ai_play)

    def ai_play(self):
        """AI hamlesini yap ve oyun devam ediyorsa sonraki hamleyi zamanla."""
        if self.game_running and self.can_move():
            try:
                best_move = self.get_best_move()
                if best_move:
                    best_move()
                    self.add_new_tile()
                    self.update_gui()

                    if not self.can_move():
                        self.game_over()
                    else:
                        self.schedule_ai_move()
            except Exception as e:
                print(f"AI play'de hata oluştu: {e}")
                self.game_over()
        elif not self.can_move():
            self.game_over()

    def move_left(self) -> bool:
        """Taşları sola kaydır ve mümkünse birleştir."""
        return self.move(lambda row: self.compress(row), lambda row: self.merge(row))

    def move_right(self) -> bool:
        """Taşları sağa kaydır ve mümkünse birleştir."""
        return self.move(
            lambda row: self.compress(row[::-1])[::-1],
            lambda row: self.merge(row[::-1])[::-1],
        )

    def move_up(self) -> bool:
        """Taşları yukarı kaydır ve mümkünse birleştir."""
        return self.move_columns(
            lambda col: self.compress(col), lambda col: self.merge(col)
        )

    def move_down(self) -> bool:
        """Taşları aşağı kaydır ve mümkünse birleştir."""
        return self.move_columns(
            lambda col: self.compress(col[::-1])[::-1],
            lambda col: self.merge(col[::-1])[::-1],
        )

    def move(self, compress_fn: Callable, merge_fn: Callable) -> bool:
        """Yatay hareketler için genel hareket fonksiyonu."""
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
        """Dikey hareketler için genel hareket fonksiyonu."""
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
        """Sıfırları çıkar ve sayıları bir tarafa kaydır."""
        return [num for num in row if num != 0] + [0] * row.count(0)

    def merge(self, row: List[int]) -> List[int]:
        """Yan yana eşit sayıları birleştir."""
        for i in range(len(row) - 1):
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]  # Skoru güncelle
        return row

    def can_move(self) -> bool:
        """Herhangi bir hareketin mümkün olup olmadığını kontrol et."""
        # Boş hücreler için kontrol
        if any(0 in row for row in self.grid):
            return True

        # Yatay birleştirme imkanları için kontrol
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if self.grid[i][j] == self.grid[i][j + 1]:
                    return True

        # Dikey birleştirme imkanları için kontrol
        for i in range(GRID_SIZE - 1):
            for j in range(GRID_SIZE):
                if self.grid[i][j] == self.grid[i + 1][j]:
                    return True

        return False

    def game_over(self):
        """Oyun sona erdiğinde çalışacak fonksiyon."""
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
            )  # Grid'in altına yerleştir
        except tk.TclError as e:
            print(f"Game over gösterilirken hata oluştu: {e}")

    def on_closing(self):
        """Pencere kapatma olayını işle."""
        self.game_running = False
        if self.ai_task:
            self.window.after_cancel(self.ai_task)
        self.window.destroy()


if __name__ == "__main__":
    Game2048()
