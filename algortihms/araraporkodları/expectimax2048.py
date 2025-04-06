# Expectimax algoritması ile 2048 oyunu çözümü
# Şans faktörünü göz önünde bulunduran bir karar algoritması

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
            print(f"GUI başlatılırken hata oluştu: {e}")
            self.game_running = False

    def evaluate(self) -> float:
        """Mevcut tahta durumunu birden çok heuristic kullanarak değerlendir."""
        empty_count = sum(row.count(0) for row in self.grid) * 10000  # Boş hücre bonusu
        mono_score = self._calculate_monotonicity()  # Monotonluk skoru
        smooth_score = self._calculate_smoothness()  # Pürüzsüzlük skoru
        corner_score = self._calculate_corner_score()  # Köşe skoru
        return empty_count + mono_score + smooth_score + corner_score

    def _calculate_monotonicity(self) -> float:
        """Grid'in ne kadar monoton (sıralı) olduğunu hesapla."""
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
        """Grid'in pürüzsüzlüğünü hesapla (komşu hücreler arası fark)."""
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
        """Köşelerdeki yüksek değerlere göre skor hesapla."""
        corners = [
            self.grid[0][0],
            self.grid[0][GRID_SIZE - 1],
            self.grid[GRID_SIZE - 1][0],
            self.grid[GRID_SIZE - 1][GRID_SIZE - 1],
        ]
        return max(corners) * 2.0  # Köşedeki en yüksek değere bonus ver

    def expectimax(
        self, depth: int, agent_type: str, test_grid: List[List[int]]
    ) -> float:
        """Expectimax algoritması - maksimum düğümler ve şans düğümlerini içerir."""
        if depth == 0:
            return self.evaluate()

        original_grid = self.grid
        original_score = self.score
        self.grid = test_grid

        if agent_type == "max":
            # Maksimum ajan - en iyi hamleyi seçer
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
            # Şans ajanı - tüm olası yeni taşların beklenen değerini hesaplar
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
            probabilities = {2: 0.9, 4: 0.1}  # 2 ve 4 taşları için olasılıklar

            for i, j in empty_cells:
                for new_value in NEW_TILE_VALUES:
                    temp_grid = copy.deepcopy(test_grid)
                    temp_grid[i][j] = new_value
                    value = self.expectimax(depth - 1, "max", temp_grid)
                    # Olasılık ağırlıklı ortalama hesapla
                    avg_value += value * probabilities[new_value] / len(empty_cells)

            self.grid = original_grid
            self.score = original_score
            return avg_value

    def get_best_move(self, depth: int = 3) -> Optional[Callable]:
        """Expectimax algoritmasını kullanarak en iyi hamleyi bul."""
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
            print(f"get_best_move'da hata oluştu: {e}")
            return None

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
            # 2 gelme olasılığı %90, 4 gelme olasılığı %10
            self.grid[i][j] = random.choices(NEW_TILE_VALUES, weights=[0.9, 0.1])[0]

    def update_gui(self):
        """Grid değerlerini ve skoru arayüzde göster."""
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                value = self.grid[i][j]
                self.cells[i][j].configure(
                    text=str(value) if value != 0 else "",
                    bg=TILE_COLORS.get(value, "#edc22e"),
                )
        self.score_label.configure(text=f"Score: {self.score}")

    def clone_grid(self) -> List[List[int]]:
        """Grid'in derin bir kopyasını oluştur."""
        return copy.deepcopy(self.grid)

    def get_possible_moves(self, grid: List[List[int]]) -> List[Callable]:
        """Belirli bir grid durumu için mümkün olan hamlelerin listesini döndür."""
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
        """Otomatik oyun için AI hamlesini zamanla."""
        if self.game_running:
            self.ai_task = self.window.after(50, self.ai_play)

    def ai_play(self):
        """AI'nin hamlelerini otomatik olarak yap."""
        if self.game_running and self.can_move(self.grid):
            try:
                best_move = self.get_best_move()
                if best_move:
                    best_move()
                    self.add_new_tile()
                    self.update_gui()
                    if not self.can_move(self.grid):
                        self.game_over()
                    else:
                        self.schedule_ai_move()
            except Exception as e:
                print(f"AI hamlesi sırasında hata: {e}")
                self.game_over()
        elif not self.can_move(self.grid):
            self.game_over()

    def move_left(self):
        """Taşları sola kaydır."""
        moved = False
        for i in range(GRID_SIZE):
            new_row = [n for n in self.grid[i] if n != 0]  # Sıfırları kaldır
            # Birleştirme işlemleri
            for j in range(len(new_row) - 1):
                if new_row[j] == new_row[j + 1]:
                    new_row[j] *= 2
                    new_row[j + 1] = 0
                    self.score += new_row[j]
            # Birleştirmeden sonra sıfırları temizle
            new_row = [n for n in new_row if n != 0]
            # Satırı 4'e tamamla
            new_row.extend([0] * (GRID_SIZE - len(new_row)))
            if new_row != self.grid[i]:
                self.grid[i] = new_row
                moved = True
        return moved

    def move_right(self):
        """Taşları sağa kaydır."""
        # Grid'i yatay olarak çevir, sola kaydır, tekrar çevir
        self.grid = [row[::-1] for row in self.grid]
        moved = self.move_left()
        self.grid = [row[::-1] for row in self.grid]
        return moved

    def move_up(self):
        """Taşları yukarı kaydır."""
        # Grid'i transpoze et (çevir), sola kaydır, tekrar çevir
        self.transpose()
        moved = self.move_left()
        self.transpose()
        return moved

    def move_down(self):
        """Taşları aşağı kaydır."""
        # Transpoze et, sağa kaydır (sol->sağ-çevir->sol->çevir), tekrar transpoze
        self.transpose()
        moved = self.move_right()
        self.transpose()
        return moved

    def transpose(self):
        """Grid'i transpoze et (satırları sütunlara, sütunları satırlara çevir)."""
        self.grid = [list(col) for col in zip(*self.grid)]

    def can_move(self, grid: List[List[int]]) -> bool:
        """Herhangi bir hareket mümkün mü kontrol et."""
        # Boş hücre varsa hareket mümkündür
        if any(0 in row for row in grid):
            return True
        # Yatay birleştirmeler için kontrol
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if grid[i][j] == grid[i][j + 1]:
                    return True
        # Dikey birleştirmeler için kontrol
        for i in range(GRID_SIZE - 1):
            for j in range(GRID_SIZE):
                if grid[i][j] == grid[i + 1][j]:
                    return True
        return False

    def can_move_left(self, grid: List[List[int]]) -> bool:
        """Sola hareket mümkün mü kontrol et."""
        for i in range(GRID_SIZE):
            for j in range(1, GRID_SIZE):
                if grid[i][j] != 0 and (
                    grid[i][j - 1] == 0 or grid[i][j - 1] == grid[i][j]
                ):
                    return True
        return False

    def can_move_right(self, grid: List[List[int]]) -> bool:
        """Sağa hareket mümkün mü kontrol et."""
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if grid[i][j] != 0 and (
                    grid[i][j + 1] == 0 or grid[i][j + 1] == grid[i][j]
                ):
                    return True
        return False

    def can_move_up(self, grid: List[List[int]]) -> bool:
        """Yukarı hareket mümkün mü kontrol et."""
        for j in range(GRID_SIZE):
            for i in range(1, GRID_SIZE):
                if grid[i][j] != 0 and (
                    grid[i - 1][j] == 0 or grid[i - 1][j] == grid[i][j]
                ):
                    return True
        return False

    def can_move_down(self, grid: List[List[int]]) -> bool:
        """Aşağı hareket mümkün mü kontrol et."""
        for j in range(GRID_SIZE):
            for i in range(GRID_SIZE - 1):
                if grid[i][j] != 0 and (
                    grid[i + 1][j] == 0 or grid[i + 1][j] == grid[i][j]
                ):
                    return True
        return False

    def game_over(self):
        """Oyun sona erdiğinde çağrılacak fonksiyon."""
        self.game_running = False
        try:
            game_over_label = tk.Label(
                self.frame,
                text=f"Game Over! Your score: {self.score}",
                font=("Verdana", 30, "bold"),
                bg=BACKGROUND_COLOR,
            )
            game_over_label.grid(row=GRID_SIZE + 1, column=0, columnspan=GRID_SIZE)
        except tk.TclError as e:
            print(f"Game over gösterilirken hata: {e}")

    def on_closing(self):
        """Pencere kapatıldığında çağrılacak fonksiyon."""
        self.game_running = False
        if self.ai_task:
            self.window.after_cancel(self.ai_task)
        self.window.destroy()


if __name__ == "__main__":
    Game2048()
