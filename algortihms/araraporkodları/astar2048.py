# A* algoritması ile 2048 oyunu çözümü
# Heuristik: En büyük taşın köşede olmasını teşvik eder

import tkinter as tk
import random
import copy

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
        self.grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.cells = []
        self.score = 0
        self.init_gui()
        self.start_game()
        self.autoplay()
        self.window.mainloop()

    def init_gui(self):
        """Oyun arayüzünü oluştur."""
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
        """İki başlangıç taşı ekleyerek oyunu başlat."""
        self.add_new_tile()
        self.add_new_tile()
        self.update_gui()

    def add_new_tile(self):
        """Rastgele bir boş hücreye yeni bir taş ekle (2 veya 4)."""
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
        """Arayüzü güncelle - grid değerlerini ve skoru göster."""
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                value = self.grid[i][j]
                self.cells[i][j].config(
                    text=str(value) if value else "",
                    bg=TILE_COLORS.get(value, "#edc22e"),
                )
        self.score_label.config(text=f"Score: {self.score}")
        self.window.update_idletasks()

    def autoplay(self):
        """Otomatik oyun modu - A* algoritmasına göre hamle seçer."""
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
            moved = key_map[best_move]()  # Sadece bir yöne hareket et

            if moved:  # Eğer gerçekten hareket ettiyse yeni taş ekle
                self.add_new_tile()
                self.update_gui()

        self.window.after(100, self.autoplay)  # 100ms sonra tekrar çağır

    def handle_keypress(self, event):
        """Klavye tuş basımlarını yakala ve işle."""
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
        """Belirli bir yönde hareketin simülasyonunu yap, grid değişmeden."""
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
            self.grid = temp_grid  # ÖNEMLİ: Eski grid'i geri yükle
            self.score = temp_score
            return new_grid, new_score

        self.grid = temp_grid  # Eğer hareket olmadıysa yine geri yükle
        self.score = temp_score
        return None, None

    def heuristic_corner_max_tile(self):
        """En büyük taşın köşede olmasını değerlendiren heuristik fonksiyon."""
        max_tile = max(max(row) for row in self.grid)  # En yüksek değeri bul
        corners = {
            (0, 0),
            (0, GRID_SIZE - 1),
            (GRID_SIZE - 1, 0),
            (GRID_SIZE - 1, GRID_SIZE - 1),
        }

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] == max_tile and (i, j) in corners:
                    return max_tile  # En büyük taş köşedeyse bonus ver

        return 0  # En büyük taş köşede değilse bonus yok

    def a_star_best_move(self):
        """A* algoritması ile en iyi hamleyi seç."""
        possible_moves = ["Up", "Down", "Left", "Right"]
        best_move = None
        best_score = -1

        for move in possible_moves:
            new_grid, new_score = self.simulate_move(move)

            if new_grid:  # Eğer hamle mümkünse
                heuristic_score = self.heuristic_corner_max_tile()
                empty_tiles = sum(
                    row.count(0) for row in new_grid
                )  # Boş hücrelere bonus ver
                total_score = (
                    new_score + heuristic_score + empty_tiles * 5
                )  # Boş hücrelere ağırlık ver

                if total_score > best_score:
                    best_score = total_score
                    best_move = move

        return best_move

    def move_left(self):
        """Taşları sola kaydır ve birleştir."""
        return self.move(lambda row: self.compress(row), lambda row: self.merge(row))

    def move_right(self):
        """Taşları sağa kaydır ve birleştir."""
        return self.move(
            lambda row: self.compress(row[::-1])[::-1],
            lambda row: self.merge(row[::-1])[::-1],
        )

    def move_up(self):
        """Taşları yukarı kaydır ve birleştir."""
        return self.move_columns(
            lambda col: self.compress(col), lambda col: self.merge(col)
        )

    def move_down(self):
        """Taşları aşağı kaydır ve birleştir."""
        return self.move_columns(
            lambda col: self.compress(col[::-1])[::-1],
            lambda col: self.merge(col[::-1])[::-1],
        )

    def move(self, compress_fn, merge_fn):
        """Genel yatay hareket fonksiyonu - sıkıştır ve birleştir."""
        moved = False
        for i in range(GRID_SIZE):
            original = self.grid[i][:]  # Hareket öncesi satırı sakla
            compressed = compress_fn(original)  # Taşları kaydır
            merged = merge_fn(compressed)  # Taşları birleştir
            new_row = compress_fn(merged)  # Birleştirdikten sonra tekrar kaydır

            if new_row != original:  # Gerçekten hareket olduğundan emin ol
                self.grid[i] = new_row
                moved = True
        return moved

    def move_columns(self, compress_fn, merge_fn):
        """Genel dikey hareket fonksiyonu - sütunları işle."""
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
        """Sıfırları sil ve sayıları bir tarafa kaydır."""
        if all(num == 0 for num in row):  # Eğer zaten boşsa işlem yapma
            return row
        return [num for num in row if num != 0] + [0] * row.count(0)

    def merge(self, row):
        """Yan yana aynı olan sayıları birleştir."""
        for i in range(len(row) - 1):
            if (
                row[i] != 0 and row[i] == row[i + 1]
            ):  # Aynı ve sıfır olmayan değerleri birleştir
                row[i] *= 2
                self.score += row[i]  # Skora birleşen değeri ekle
                row[i + 1] = 0  # Birleştirilen taşı temizle
        return row

    def can_move(self):
        """Herhangi bir hareketin mümkün olup olmadığını kontrol et."""
        # Yatay ve dikey komşular arası birleştirme imkanı
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if self.grid[i][j] == self.grid[i][j + 1]:
                    return True
        for i in range(GRID_SIZE - 1):
            for j in range(GRID_SIZE):
                if self.grid[i][j] == self.grid[i + 1][j]:
                    return True

        # Boş hücre var mı kontrol et
        return any(0 in row for row in self.grid)

    def game_over(self):
        """Oyun bittiğinde çalışacak fonksiyon."""
        game_over_label = tk.Label(
            self.frame,
            text="Game Over!",
            font=("Verdana", 30, "bold"),
            bg=BACKGROUND_COLOR,
        )
        game_over_label.grid(row=0, column=0, columnspan=GRID_SIZE)


if __name__ == "__main__":
    Game2048()
