# Expectimax algoritması ile 2048 oyunu çözümü
# Şans faktörünü göz önüne alan bir arama algoritması

import tkinter as tk
import random
import copy
import os
import datetime
import time  # Zaman fonksiyonları için gerekli import
from typing import Callable, List, Optional, Tuple
import pyautogui  # ImageGrab yerine pyautogui kullanılıyor
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
)  # Ekran görüntülerine metin eklemek için gerekli
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
        self.move_count = 0  # Ekran görüntüleri için hamle sayısını takip et

        # Ekran görüntüleri için klasör oluştur (Create directory for screenshots)
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
        """GUI olmadan oyunu çalıştır ve final skoru ile max değeri döndür."""
        if not self.run_without_gui:
            return self.score, max(max(row) for row in self.grid)

        # Oyun bitene kadar çalıştır (Run until game over)
        while self.can_move(self.grid):
            best_move = self.get_best_move()
            if best_move:
                best_move()
                self.add_new_tile()
                self.move_count += 1
            else:
                break

        # En büyük değeri bul (Find the max tile value)
        max_tile = max(max(row) for row in self.grid)
        return self.score, max_tile

    def init_gui(self):
        """Oyunun grafik arayüzünü başlat."""
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
            print(f"GUI başlatılırken hata oluştu: {e}")
            self.game_running = False

    def evaluate(self) -> float:
        """
        Grid'in mevcut durumunu değerlendirir.
        - Boş hücre sayısı, monotonluk, pürüzsüzlük, köşe skoru gibi faktörler dikkate alınır.
        - Her bir heuristic, stratejik bir avantaj sağlar.
        """
        # Boş hücre sayısı: Daha fazla boş hücre, daha fazla hareket imkanı sağlar.
        # Monotonluk: Büyük taşların sıralı bir şekilde yerleşmesini teşvik eder.
        # Pürüzsüzlük: Komşu taşlar arasındaki farkın az olması tercih edilir.
        # Köşe skoru: En büyük taşın köşede olması avantaj sağlar.
        # Kümelenme: Büyük taşların birbirine yakın olması stratejik bir avantajdır.
        empty_count = sum(row.count(0) for row in self.grid) * 10000  # Boş hücre bonusu
        mono_score = self._calculate_monotonicity()  # Monotonluk skoru
        smooth_score = self._calculate_smoothness()  # Pürüzsüzlük skoru
        corner_score = self._calculate_corner_score()  # Köşe skoru

        # Yeni heuristikler
        clustering_score = self._calculate_tile_clustering() * 50  # Kümelenme skoru
        empty_tile_ratio = self._calculate_empty_tile_ratio() * 2000  # Boş hücre oranı
        merge_potential = (
            self._calculate_merge_potential() * 1000
        )  # Birleşme potansiyeli

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
        """
        Monotonluk heuristic'i:
        - Taşların sıralı bir şekilde yerleşmesini ölçer.
        - Büyük taşların bir köşede toplanmasını teşvik eder.
        """
        # Yatay ve dikey olarak taşların sıralı bir şekilde yerleşmesini kontrol eder.
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
        """
        Pürüzsüzlük heuristic'i:
        - Komşu taşlar arasındaki farkın az olması hedeflenir.
        - Daha pürüzsüz bir grid, daha iyi bir strateji anlamına gelir.
        """
        # Komşu taşlar arasındaki farkları hesaplar ve toplam farkı azaltmayı hedefler.
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
        return max(corners) * 2.0  # Köşedeki en yüksek değere bonus

    def _calculate_tile_clustering(self) -> float:
        """
        Kümelenme heuristic'i:
        - Büyük taşların birbirine yakın olması avantaj sağlar.
        - Taşların ağırlık merkezine olan uzaklıkları hesaplanır.
        """
        # Taşların ağırlık merkezine olan uzaklıklarını hesaplar.
        # Büyük taşların birbirine yakın olması stratejik bir avantaj sağlar.
        clustering_score = 0

        # Taşların değerleriyle ağırlıklandırılmış ortalama pozisyonu
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
                        # Kütle merkezine yakın büyük taşlar için yüksek puan
                        distance = ((i - mean_y) ** 2 + (j - mean_x) ** 2) ** 0.5
                        clustering_score += self.grid[i][j] / (1 + distance)

        return clustering_score

    def _calculate_empty_tile_ratio(self) -> float:
        """
        Boş hücre oranı:
        - Boş hücrelerin dolu hücrelere oranını hesaplar.
        - Daha fazla boş hücre, daha fazla hareket imkanı sağlar.
        """
        # Boş hücrelerin dolu hücrelere oranını hesaplar.
        empty_count = sum(row.count(0) for row in self.grid)
        non_empty_count = GRID_SIZE * GRID_SIZE - empty_count

        if non_empty_count == 0:  # Sıfıra bölünmeyi önle (Prevent division by zero)
            return GRID_SIZE * GRID_SIZE

        return empty_count / (non_empty_count + 1)  # +1 sıfıra bölünmeyi önlemek için

    def _calculate_merge_potential(self) -> float:
        """
        Birleştirme potansiyeli:
        - Yatay ve dikey olarak birleşebilecek taşları değerlendirir.
        - Birleşme potansiyeli, daha yüksek skor elde etme şansını artırır.
        """
        # Yatay ve dikey olarak birleşebilecek taşları değerlendirir.
        merge_score = 0

        # Yatay birleştirme potansiyelini kontrol et
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if self.grid[i][j] != 0 and self.grid[i][j] == self.grid[i][j + 1]:
                    merge_score += self.grid[i][j] * 2  # Birleşim sonrası değer

        # Dikey birleştirme potansiyelini kontrol et
        for i in range(GRID_SIZE - 1):
            for j in range(GRID_SIZE):
                if self.grid[i][j] != 0 and self.grid[i][j] == self.grid[i + 1][j]:
                    merge_score += self.grid[i][j] * 2  # Birleşim sonrası değer

        return merge_score

    def expectimax(
        self, depth: int, agent_type: str, test_grid: List[List[int]]
    ) -> float:
        """
        Expectimax algoritması:
        - Max ajan en iyi hamleyi seçer.
        - Chance ajan rastgele taş eklemeleri simüle eder.
        - Derinlik (depth), algoritmanın kaç adım ileriye bakacağını belirler.
        """
        # Max ajan: En iyi hamleyi seçmek için tüm olasılıkları değerlendirir.
        # Chance ajan: Rastgele taş eklemeleri simüle ederek olasılıkları hesaplar.
        # Derinlik sıfıra ulaştığında veya terminal durumda olduğunda değerlendirme yapılır.
        if depth == 0:
            return self.evaluate()

        original_grid = self.grid
        original_score = self.score
        self.grid = test_grid

        if agent_type == "max":
            # Maksimum ajan - en iyi hamleyi seç
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
            # Şans ajanı - tüm olası yeni taşların ortalamasını al
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
                    # Her boş hücre için ağırlıklı ortalama hesapla
                    avg_value += value * probabilities[new_value] / len(empty_cells)

            self.grid = original_grid
            self.score = original_score
            return avg_value

    def get_best_move(self, depth: int = 3) -> Optional[Callable]:
        """
        En iyi hamleyi bulur:
        - Expectimax algoritmasını kullanır.
        - Performans için memoization ve Monte Carlo yöntemleri kullanılır.
        """
        # En iyi hamleyi bulmak için Expectimax algoritmasını kullanır.
        # Performans için memoization ve Monte Carlo yöntemleri uygulanır.
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

        # Oyun başlangıcında ekran görüntüsü al
        if not self.run_without_gui:
            self.window.after(500, lambda: self.take_screenshot("start"))

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
            # %90 olasılıkla 2, %10 olasılıkla 4 gelir
            self.grid[i][j] = random.choices(NEW_TILE_VALUES, weights=[0.9, 0.1])[0]

    def update_gui(self):
        """Grid'in mevcut durumunu yansıtmak için GUI'yi güncelle."""
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
        """Grid'in derin bir kopyasını döndür."""
        return copy.deepcopy(self.grid)

    def get_possible_moves(self, grid: List[List[int]]) -> List[Callable]:
        """Mümkün olan hamlelerin listesini döndür."""
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
        """Oyunun mevcut durumunun ekran görüntüsünü al ve metin ekle."""
        if self.run_without_gui:
            return
        try:
            # Ekran görüntüsü almadan önce pencerenin güncellendiğinden emin ol
            self.window.update_idletasks()
            self.window.update()

            # Pencere konumu ve boyutu
            x, y, width, height = (
                self.window.winfo_rootx(),
                self.window.winfo_rooty(),
                self.window.winfo_width(),
                self.window.winfo_height(),
            )

            # Sebebe göre dosya adı belirle
            if reason == "start":
                filename = "game_start_expectimax.png"
                text = "BAŞLANGIÇ"  # "Expectimax Algorithm" yerine "BAŞLANGIÇ"
            elif reason == "mid":
                filename = "game_mid_expectimax.png"
                text = ""
            elif reason == "gameover":
                filename = "game_over_expectimax.png"
                text = "Expectimax Algorithm"
            else:
                return  # Diğer sebepler için ekran görüntüsü alma

            filepath = os.path.join(self.screenshots_dir, filename)

            # pyautogui ile ekran görüntüsü al
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            screenshot = screenshot.convert("RGB")

            # Ekran görüntüsüne algoritma adını ekle
            draw = ImageDraw.Draw(screenshot)
            try:
                font = ImageFont.truetype("arial.ttf", 36)
            except IOError:
                # arial.ttf bulunamazsa varsayılan fontu kullan
                font = ImageFont.load_default()

            draw.text((10, 50), text, fill="black", font=font)

            # Ekran görüntüsünü kaydet
            screenshot.save(filepath)
            print(f"Ekran görüntüsü kaydedildi: {filepath}")
        except Exception as e:
            print(f"Ekran görüntüsü alınırken hata oluştu: {e}")

    def schedule_ai_move(self):
        """AI'nin bir hamle yapmasını zamanla."""
        if self.run_without_gui:
            return
        if self.game_running:
            best_move = self.get_best_move()
            if best_move:
                best_move()
                self.add_new_tile()
                self.update_gui()
                self.move_count += 1

                # Sadece belirli bir hamle sayısında ekran görüntüsü al
                if self.move_count == 250:
                    self.take_screenshot("mid")

                if self.game_over():
                    self.game_running = False
                    self.take_screenshot(
                        "gameover"
                    )  # Oyun bittiğinde ekran görüntüsü al
                    print("Game Over!")
            self.window.after(50, self.schedule_ai_move)

    def ai_play(self):
        """AI'nın bir hamle yapmasını sağla."""
        best_move = self.get_best_move()
        if best_move:
            best_move()
            self.add_new_tile()
            self.update_gui()

    def move_left(self):
        """Taşları sola kaydır."""
        self.move(self.grid, self.compress, self.merge, self.compress)

    def move_right(self):
        """Taşları sağa kaydır."""
        self.move(
            self.grid,
            self.reverse,
            self.compress,
            self.merge,
            self.compress,
            self.reverse,
        )

    def move_up(self):
        """Taşları yukarı kaydır."""
        self.move(
            self.grid,
            self.transpose,
            self.compress,
            self.merge,
            self.compress,
            self.transpose,
        )

    def move_down(self):
        """Taşları aşağı kaydır."""
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
        """Grid üzerinde bir dizi adım uygula."""
        for step in steps:
            step(grid)

    def compress(self, grid):
        """Taşları sola kaydırıp sıfırları sağa topla."""
        for i in range(GRID_SIZE):
            new_row = [tile for tile in grid[i] if tile != 0]
            new_row += [0] * (GRID_SIZE - len(new_row))
            grid[i] = new_row

    def merge(self, grid):
        """Grid'deki taşları birleştir."""
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if grid[i][j] == grid[i][j + 1] and grid[i][j] != 0:
                    grid[i][j] *= 2
                    grid[i][j + 1] = 0
                    self.score += grid[i][j]

    def reverse(self, grid):
        """Grid'in satırlarını tersine çevir."""
        for i in range(GRID_SIZE):
            grid[i] = grid[i][::-1]

    def transpose(self, grid):
        """Grid'i transpoze et (satır ve sütunları değiştir)."""
        grid[:] = [list(row) for row in zip(*grid)]

    def can_move(self, grid) -> bool:
        """Herhangi bir hareketin mümkün olup olmadığını kontrol et."""
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
        """Sola hareket mümkün mü kontrol et."""
        for i in range(GRID_SIZE):
            for j in range(1, GRID_SIZE):
                if grid[i][j] != 0 and (
                    grid[i][j - 1] == 0 or grid[i][j - 1] == grid[i][j]
                ):
                    return True
        return False

    def can_move_right(self, grid) -> bool:
        """Sağa hareket mümkün mü kontrol et."""
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if grid[i][j] != 0 and (
                    grid[i][j + 1] == 0 or grid[i][j + 1] == grid[i][j]
                ):
                    return True
        return False

    def can_move_up(self, grid) -> bool:
        """Yukarı hareket mümkün mü kontrol et."""
        for i in range(1, GRID_SIZE):
            for j in range(GRID_SIZE):
                if grid[i][j] != 0 and (
                    grid[i - 1][j] == 0 or grid[i - 1][j] == grid[i][j]
                ):
                    return True
        return False

    def can_move_down(self, grid) -> bool:
        """Aşağı hareket mümkün mü kontrol et."""
        for i in range(GRID_SIZE - 1):
            for j in range(GRID_SIZE):
                if grid[i][j] != 0 and (
                    grid[i + 1][j] == 0 or grid[i + 1][j] == grid[i][j]
                ):
                    return True
        return False

    def game_over(self) -> bool:
        """Oyunun bitip bitmediğini kontrol et."""
        if not self.can_move(self.grid):
            return True
        return False

    def on_closing(self):
        """Pencere kapatma olayını işle."""
        self.game_running = False
        self.window.destroy()

    def is_terminal(self, grid: List[List[int]]) -> bool:
        """Grid'in terminal durumda olup olmadığını kontrol et (hamle yapılamıyor)."""
        return not self.can_move(grid)

    def grid_to_tuple(self, grid: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
        """Grid'i memoization için hashable tuple formatına dönüştür."""
        return tuple(tuple(row) for row in grid)

    @lru_cache(maxsize=10000)
    def cached_expectimax(
        self,
        depth: int,
        agent_type: str,
        grid_tuple: Tuple[Tuple[int, ...], ...],
        score: int,
    ) -> float:
        """
        Önbelleğe alınmış Expectimax:
        - Memoization ile daha önce hesaplanmış sonuçları saklar.
        - Performansı artırır ve tekrar hesaplamaları önler.
        """
        # Memoization: Daha önce hesaplanmış sonuçları saklayarak performansı artırır.
        # LRU Cache: En son kullanılan sonuçları saklar ve belleği verimli kullanır.
        # Tuple'ı listeye dönüştür
        grid = [list(row) for row in grid_tuple]

        # Orijinal grid ve skoru kaydet
        original_grid = self.grid
        original_score = self.score

        # Test grid'i geçici olarak kullan
        self.grid = grid
        self.score = score

        # Monte Carlo Expectimax çalıştır
        result = self.monte_carlo_expectimax(depth, agent_type, self.grid)

        # Orijinal grid ve skoru geri yükle
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
        """
        Monte Carlo Expectimax:
        - Rastgele simülasyonlar yaparak olasılıkları değerlendirir.
        - Rollout sayısı, doğruluk ve performans arasında bir denge sağlar.
        """
        # Monte Carlo yöntemi, rastgele örnekleme yaparak olasılıkları tahmin eder.
        # Rollout sayısı, daha fazla doğruluk için artırılabilir ancak işlem süresi uzar.
        # Monte Carlo rollout'lar, taşların köşelere yakın olmasını teşvik eder.
        # Bu, stratejik olarak taşların daha iyi yerleştirilmesini sağlar.
        # Erken sonlandırma
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

            # Ağırlık tabanlı Monte Carlo örnekleme kullan
            total_value = 0
            probabilities = {2: 0.9, 4: 0.1}  # Taş olasılıkları

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

                    # Değeri olasılığa göre ağırlıklandır
                    total_value += value * probabilities[new_value]

            # Sonuçların ortalamasını al
            avg_value = total_value / (actual_rollouts * len(NEW_TILE_VALUES))

            self.grid = original_grid
            self.score = original_score
            return avg_value

    def get_best_move(self, depth: int = 3) -> Optional[Callable]:
        """Optimize edilmiş Monte Carlo expectimax algoritmasını kullanarak en iyi hamleyi bul."""
        try:
            start_time = time.time()
            best_move = None
            max_value = float("-inf")
            current_grid = self.clone_grid()
            original_score = self.score

            # Rollout sayısını 10'a düşür - performans iyileştirmesi
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

            # Önbelleği sadece işlem uzun sürdüğünde ya da periyodik olarak temizle
            if time.time() - start_time > 0.5 or self.move_count % 150 == 0:
                self.cached_expectimax.cache_clear()

            return best_move
        except Exception as e:
            print(f"get_best_move'da hata oluştu: {e}")
            return None


if __name__ == "__main__":
    Game2048()
