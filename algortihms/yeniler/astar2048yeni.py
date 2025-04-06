# A* algoritması ile 2048 oyunu çözümü
# Hedef: En büyük sayının köşede olması durumunda yüksek skor vermek

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
        self.move_count = 0  # Hareket sayısını tutan sayaç (Move counter)
        self.screenshot_taken = (
            False  # Ekran görüntüsü alınıp alınmadığını kontrol eden bayrak
        )

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
        """GUI olmadan oyunu çalıştır ve final skoru ile max değeri döndür."""
        if not self.run_without_gui:
            return self.score, max(max(row) for row in self.grid)

        # Oyun bitene kadar çalıştır (Run until game over)
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

        # En büyük değeri bul (Find the max tile value)
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
        """Oyunu başlat - 2 tane başlangıç taşı ekle."""
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
        """Oyun penceresinin ekran görüntüsünü al ve metin ekle."""
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
        draw.text((10, 10), text, fill="black", font=font)
        screenshot.save(filename)

    def autoplay(self):
        """Oyunu otomatik olarak oyna - A* algoritması kullanarak hamle seç."""
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

                self.move_count += 1  # Hamle sayısını artır

                # Belirli bir hamle sayısından sonra ekran görüntüsü al
                if self.move_count == 70 and not self.screenshot_taken:
                    self.take_screenshot("game_mid_astar.png", "A* Algorithm")
                    self.screenshot_taken = True

        self.window.after(100, self.autoplay)  # 100ms sonra tekrar çağır

    def handle_keypress(self, event):
        """Klavye tuş basımlarını işle."""
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
        """En büyük değerin köşede olmasını teşvik eden heuristik."""
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
                    return max_tile  # Eğer max değer köşedeyse bonus

        return 0  # Max değer köşede değilse bonus yok

    def heuristic_monotonicity(self):
        """Gridin ne kadar monoton (sıralı) olduğunu hesapla."""
        mono_score = 0

        # Yatay monotonikliği kontrol et (soldan sağa azalan)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if self.grid[i][j] and self.grid[i][j + 1]:
                    if self.grid[i][j] >= self.grid[i][j + 1]:
                        mono_score += self.grid[i][j]

        # Dikey monotonikliği kontrol et (yukarıdan aşağı azalan)
        for j in range(GRID_SIZE):
            for i in range(GRID_SIZE - 1):
                if self.grid[i][j] and self.grid[i + 1][j]:
                    if self.grid[i][j] >= self.grid[i + 1][j]:
                        mono_score += self.grid[i][j]

        return mono_score

    def heuristic_smoothness(self):
        """Gridin ne kadar smooth (pürüzsüz) olduğunu hesapla - komşu değerler arası fark."""
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
        """Büyük taşların ne kadar kümelenebildiğini hesapla."""
        clustering_score = 0

        # Taşların değerleriyle ağırlıklandırılmış ortalama pozisyonunu bul
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

            # Her taşın ortalama pozisyona uzaklığını hesapla
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    if self.grid[i][j] > 0:
                        # Kütle merkezine daha yakın olan büyük taşlara daha yüksek değer
                        distance = ((i - mean_y) ** 2 + (j - mean_x) ** 2) ** 0.5
                        clustering_score += self.grid[i][j] / (1 + distance)

        return clustering_score

    def a_star_best_move(self):
        """A* algoritması kullanarak en iyi hamleyi belirle."""
        possible_moves = ["Up", "Down", "Left", "Right"]
        best_move = None
        best_score = -1

        for move in possible_moves:
            new_grid, new_score = self.simulate_move(move)

            if new_grid:  # Eğer hamle mümkünse
                # Tüm heuristikleri hesapla
                corner_score = self.heuristic_corner_max_tile()
                empty_tiles = sum(
                    row.count(0) for row in new_grid
                )  # Boş alanları teşvik et
                mono_score = self.heuristic_monotonicity() / 1000  # Ölçeklendirme
                smooth_score = self.heuristic_smoothness() / 100  # Ölçeklendirme
                cluster_score = self.heuristic_tile_clustering() / 100  # Ölçeklendirme

                # Tüm heuristiklerin ağırlıklı toplamı
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
        """Genel hareket fonksiyonu - belirtilen şekilde sıkıştır ve birleştir."""
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
        """Sütunlardaki hareket için genel fonksiyon."""
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
                self.score += row[i]
                row[i + 1] = 0  # Birleştirilen taşı temizle
        return row

    def can_move(self):
        """Herhangi bir hareketin mümkün olup olmadığını kontrol et."""
        # Yatay birleştirmeleri kontrol et
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if self.grid[i][j] == self.grid[i][j + 1]:
                    return True
        # Dikey birleştirmeleri kontrol et
        for i in range(GRID_SIZE - 1):
            for j in range(GRID_SIZE):
                if self.grid[i][j] == self.grid[i + 1][j]:
                    return True
        # Boş hücre var mı kontrol et
        return any(0 in row for row in self.grid)

    def game_over(self):
        """Oyun sona erdiğinde çalışacak fonksiyon."""
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
        )  # Oyun bitiminde ekran görüntüsü al


if __name__ == "__main__":
    Game2048()
