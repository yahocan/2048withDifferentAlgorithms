# 2048 oyunu için minimax algoritması implementasyonu
# Alpha-beta pruning ve memoization ile optimize edilmiştir

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
        self.move_count = 0  # Hamle sayısını tutan sayaç
        self.screenshot_taken = (
            False  # Ekran görüntüsü alınıp alınmadığını kontrol eden bayrak
        )

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
        """GUI olmadan oyunu çalıştır ve final skoru ile max değeri döndür."""
        if not self.run_without_gui:
            return self.score, max(max(row) for row in self.grid)

        # Oyun bitene kadar çalıştır
        while self.can_move():
            best_move = self.get_best_move()
            if best_move:
                best_move()
                self.add_new_tile()
                self.move_count += 1
            else:
                break

        # En büyük değeri bul
        max_tile = max(max(row) for row in self.grid)
        return self.score, max_tile

    def init_gui(self):
        """Oyunun grafik arayüzünü başlat."""
        if self.run_without_gui:
            return

        try:
            self.frame = tk.Frame(self.window, bg=BACKGROUND_COLOR)
            self.frame.grid()

            # Skor etiketini en üste (satır 0) taşı ve görünür yap
            self.score_label = tk.Label(
                self.frame,
                text=f"Score: {self.score}",
                font=("Verdana", 20, "bold"),
                bg=BACKGROUND_COLOR,
                padx=5,
                pady=5,
            )
            self.score_label.grid(row=0, column=0, columnspan=GRID_SIZE, sticky="nsew")

            # Oyun grid'ini 1. satırdan başlat
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
        """Oyunu başlat - 2 tane başlangıç taşı ekle."""
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
        """Mevcut tahta durumunu birden çok heuristic kullanarak değerlendir."""
        # Temel heuristic: boş hücre sayısı
        empty_cell_score = sum(row.count(0) for row in self.grid) * 100

        # Yeni heuristikler
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
        """Grid'in ne kadar monoton (sıralı) olduğunu hesapla."""
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

    def _calculate_smoothness(self) -> float:
        """Grid'in ne kadar pürüzsüz olduğunu hesapla (komşu taşlar arasındaki fark)."""
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
        """Büyük taşların ne kadar iyi kümelendiğini hesapla."""
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
                        # Kütle merkezine daha yakın olan büyük taşlar için daha yüksek değerler
                        distance = ((i - mean_y) ** 2 + (j - mean_x) ** 2) ** 0.5
                        clustering_score += self.grid[i][j] / (1 + distance)

        return clustering_score

    def _evaluate_max_tile_placement(self) -> int:
        """Grid üzerindeki en büyük taşın konumunu değerlendir."""
        max_val = 0
        max_i, max_j = 0, 0

        # En büyük taş değerini ve konumunu bul
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] > max_val:
                    max_val = self.grid[i][j]
                    max_i, max_j = i, j

        # En büyük taşın köşede olup olmadığını kontrol et (en iyi pozisyon)
        corner_positions = [
            (0, 0),
            (0, GRID_SIZE - 1),
            (GRID_SIZE - 1, 0),
            (GRID_SIZE - 1, GRID_SIZE - 1),
        ]
        if (max_i, max_j) in corner_positions:
            return max_val  # Köşe yerleşimi için maksimum puan

        # Kenarda olup olmadığını kontrol et
        if max_i == 0 or max_i == GRID_SIZE - 1 or max_j == 0 or max_j == GRID_SIZE - 1:
            return max_val // 2  # Kenar yerleşimi için yarım puan

        return 0  # Merkez yerleşimi için bonus yok

    def is_terminal(self, grid: List[List[int]]) -> bool:
        """Grid'in terminal durumda olup olmadığını kontrol et (hamle yapılamıyor)."""
        # Boş hücre var mı kontrol et
        if any(0 in row for row in grid):
            return False

        # Yatay birleştirmeleri kontrol et
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if grid[i][j] == grid[i][j + 1]:
                    return False

        # Dikey birleştirmeleri kontrol et
        for i in range(GRID_SIZE - 1):
            for j in range(GRID_SIZE):
                if grid[i][j] == grid[i + 1][j]:
                    return False

        return True  # Hamle yapılamaz

    def evaluate_move_for_sorting(
        self, move: Callable, test_grid: List[List[int]]
    ) -> float:
        """Hamleleri sıralamak için hızlı değerlendirme yap."""
        temp_grid = copy.deepcopy(test_grid)
        original_grid = self.grid
        self.grid = temp_grid

        # Hamleyi yap
        move()

        # Hızlı heuristic: daha fazla boş hücre ve daha yüksek skorları tercih et
        empty_count = sum(row.count(0) for row in self.grid)
        max_value = max(max(row) for row in self.grid)
        corner_score = self._evaluate_max_tile_placement()

        # Orijinal grid'i geri yükle
        self.grid = original_grid

        return empty_count * 10 + max_value + corner_score

    def get_sorted_moves(self, test_grid: List[List[int]]) -> List[Callable]:
        """Heuristic değerlendirmeye göre sıralanmış olası hareketleri al (en iyisi önce)."""
        possible_moves = self.get_possible_moves(test_grid)

        # Hareketleri hızlı bir değerlendirmeye göre sırala (en iyisi önce)
        return sorted(
            possible_moves,
            key=lambda move: self.evaluate_move_for_sorting(move, test_grid),
            reverse=True,
        )

    def grid_to_tuple(self, grid: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
        """Grid'i memoization için hashable tuple formatına dönüştür."""
        return tuple(tuple(row) for row in grid)

    # Memoization için lru_cache ile minimax'ın önbelleğe alınmış versiyonu
    @lru_cache(maxsize=1000)  # 10000'den 1000'e düşürüldü, bellek optimize
    def cached_minimax(
        self,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool,
        grid_tuple: Tuple[Tuple[int, ...], ...],
        score: int,
    ) -> float:
        """Minimax'ın önbelleğe alınmış versiyonu."""
        # Tuple'ı listeye dönüştür
        grid = [list(row) for row in grid_tuple]

        # Orijinal grid ve skoru kaydet
        original_grid = self.grid
        original_score = self.score

        # Test grid'i geçici olarak kullan
        self.grid = grid
        self.score = score

        # Minimax çalıştır
        result = self.minimax(depth, alpha, beta, maximizing_player, self.grid)

        # Orijinal grid ve skoru geri yükle
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
        """Alpha-beta pruning ve erken sonlandırma ile optimize edilmiş minimax algoritması."""
        # Erken sonlandırma kontrolü
        if depth == 0 or self.is_terminal(test_grid):
            return self.evaluate()

        original_grid = self.grid
        original_score = self.score
        self.grid = test_grid

        if maximizing_player:
            max_eval = float("-inf")
            # Daha iyi alpha-beta budama için sıralanmış hamleleri kullan
            for move in self.get_sorted_moves(test_grid):
                temp_grid = copy.deepcopy(test_grid)
                self.grid = temp_grid
                move()

                # Özyinelemeli çağrılar için memoization kullan
                grid_tuple = self.grid_to_tuple(self.grid)
                eval = self.cached_minimax(
                    depth - 1, alpha, beta, False, grid_tuple, self.score
                )

                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)

                if beta <= alpha:
                    break  # Alpha-beta budama

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

            # 2 ve 4 taşları için olasılıksal ağırlıklandırma
            probabilities = {2: 0.9, 4: 0.1}

            for i, j in empty_cells:
                for new_value in NEW_TILE_VALUES:
                    temp_grid = copy.deepcopy(test_grid)
                    temp_grid[i][j] = new_value

                    # Taşın görünme olasılığına göre ağırlıklandır
                    grid_tuple = self.grid_to_tuple(temp_grid)
                    eval = self.cached_minimax(
                        depth - 1, alpha, beta, True, grid_tuple, self.score
                    )
                    weighted_eval = eval * probabilities[new_value]

                    min_eval = min(min_eval, weighted_eval)
                    beta = min(beta, weighted_eval)

                    if beta <= alpha:
                        break  # Alpha-beta budama

            self.grid = original_grid
            self.score = original_score
            return min_eval

    def parallel_process_move(self, move, grid, depth):
        """Pickle sorunları olmadan paralel hamle değerlendirmesi için değiştirilmiş yardımcı fonksiyon."""
        # Multiprocessing'e self geçmeden tek bir hamle değerlendirmesini işle
        temp_grid = copy.deepcopy(grid)

        # Bu değerlendirme için tkinter penceresi pickle sorunlarını önlemek için
        # Game2048'in yeni bir örneğini oluştur
        temp_game = Game2048(run_without_gui=True)
        temp_game.grid = temp_grid
        temp_game.score = self.score

        # Hamleyi yap
        move_func = getattr(temp_game, move.__name__)
        move_func()

        # Geçici oyun üzerinde doğrudan minimax kullanarak değerlendir
        result = temp_game.evaluate()

        return (move, result)

    def get_best_move(
        self, depth: int = 2
    ) -> Optional[Callable]:  # derinlik 3'ten 2'ye düşürüldü, performans için
        """Paralel işleme olmadan optimize edilmiş minimax kullanarak en iyi hamleyi al."""
        try:
            best_move = None
            max_eval = float("-inf")
            current_grid = self.clone_grid()
            original_score = self.score
            possible_moves = self.get_sorted_moves(current_grid)

            # Pickle hatalarını önlemek için her zaman sıralı işleme kullan
            # Sıralı işleme
            for move in possible_moves:
                temp_grid = copy.deepcopy(current_grid)
                temp_score = self.score
                self.grid = temp_grid
                move()

                # Memoization kullan
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
            print(f"get_best_move'da hata oluştu: {e}")
            return None

    def schedule_ai_move(self):
        """Oyun hala çalışıyorsa bir sonraki AI hamlesini zamanla."""
        if self.run_without_gui:
            return

        if self.game_running:
            self.ai_task = self.window.after(50, self.ai_play)

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
        draw.text((10, 50), text, fill="black", font=font)  # Pozisyon ayarlandı
        screenshot.save(filename)

    def ai_play(self):
        """AI hamlesini yap ve oyun hala çalışıyorsa bir sonrakini zamanla."""
        if self.run_without_gui:
            return

        if self.game_running and self.can_move():
            try:
                best_move = self.get_best_move()
                if best_move:
                    best_move()
                    self.add_new_tile()
                    self.update_gui()

                    self.move_count += 1  # Hamle sayacını artır

                    # Belirli bir hamle sayısından sonra ekran görüntüsü al
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
        # Boş hücreler için kontrol et
        if any(0 in row for row in self.grid):
            return True

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
            )  # Grid'in altına yerleştir
            self.take_screenshot(
                "game_over_minimax.png", "Minimax Algorithm"
            )  # Oyun bitiminde ekran görüntüsü al
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
