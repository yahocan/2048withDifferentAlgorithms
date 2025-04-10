# Hill Climbing algoritması ile 2048 oyunu çözümü
# Rastgele yeniden başlatma stratejisi ile yerel maksimumlardan kaçınma

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
        self.move_count = 0  # Hamle sayısını tutan sayaç (Move counter)
        self.screenshot_taken = False  # Ekran görüntüsü alma kontrolü için flag

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
        while self.can_move():
            best_move = self.get_best_move()
            if best_move:
                best_move()
                self.add_new_tile()
                self.move_count += 1
            else:
                break

        # En büyük taş değerini bul (Find the max tile value)
        max_tile = max(max(row) for row in self.grid)
        return self.score, max_tile

    def init_gui(self):
        """Oyunun grafik arayüzünü başlat."""
        if self.run_without_gui:
            return

        try:
            self.frame = tk.Frame(self.window, bg=BACKGROUND_COLOR)
            self.frame.grid()

            # Skor etiketini üste yerleştir (Place score label at the top)
            self.score_label = tk.Label(
                self.frame,
                text=f"Score: {self.score}",
                font=("Verdana", 20, "bold"),
                bg=BACKGROUND_COLOR,
                padx=5,
                pady=5,
            )
            self.score_label.grid(row=0, column=0, columnspan=GRID_SIZE, sticky="nsew")

            # Oyun gridini 1. satırdan başlat (Start game grid from row 1)
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
        """Oyunu başlat - iki başlangıç taşı ekle."""
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
        """
        Grid'in mevcut durumunu değerlendirir.
        - Bu fonksiyon, oyunun mevcut durumunu analiz ederek bir skor döndürür.
        - Heuristicler:
          - Boş Hücre Sayısı: Daha fazla boş hücre, daha fazla hareket imkanı sağlar.
          - Monotonluk: Büyük taşların sıralı bir şekilde yerleşmesini teşvik eder.
          - Pürüzsüzlük: Komşu taşlar arasındaki farkın az olması tercih edilir.
          - Max Taşın Konumu: En büyük taşın köşede olması daha iyi bir stratejidir.
          - Birleştirme Potansiyeli: Taşların birleşme olasılığı.
        - Her bir heuristic, stratejik bir avantaj sağlar ve toplam skor hesaplanır.
        """
        # Analiz sonuçlarına göre ayarlanmış heuristik ağırlıkları
        empty_cell_score = (
            sum(row.count(0) for row in self.grid) * 200
        )  # Boş hücreler önemli
        monotonicity_score = (
            self._calculate_monotonicity() / 5
        )  # Monotonluk (sıralı olma)
        smoothness_score = (
            self._calculate_smoothness() * 2
        )  # Pürüzsüzlük (komşu değerler yakın)
        max_tile_placement_score = (
            self._evaluate_max_tile_placement() * 5
        )  # Max taşın konumu

        # Birleştirme potansiyeli (Merge potential)
        merge_potential = self._calculate_merge_potential() * 10

        return (
            empty_cell_score
            + monotonicity_score
            + smoothness_score
            + max_tile_placement_score
            + merge_potential
        )

    def _calculate_merge_potential(self) -> float:
        """
        Birleştirme Potansiyeli Heuristic'i:
        - Taşların birleşme olasılığını değerlendirir.
        - Yatay ve dikey olarak birleşebilecek taşları analiz eder.
        """
        merge_score = 0

        # Yatay birleştirme potansiyeli kontrolü
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if self.grid[i][j] != 0 and self.grid[i][j] == self.grid[i][j + 1]:
                    merge_score += self.grid[i][j] * 2  # Birleşim sonrası değer

        # Dikey birleştirme potansiyeli kontrolü
        for i in range(GRID_SIZE - 1):
            for j in range(GRID_SIZE):
                if self.grid[i][j] != 0 and self.grid[i][j] == self.grid[i + 1][j]:
                    merge_score += self.grid[i][j] * 2  # Birleşim sonrası değer

        return merge_score

    def _calculate_monotonicity(self) -> float:
        """
        Monotonluk Heuristic'i:
        - Taşların sıralı bir şekilde yerleşmesini ölçer.
        - Soldan sağa ve yukarıdan aşağıya azalan bir düzen tercih edilir.
        """
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
        """
        Pürüzsüzlük Heuristic'i:
        - Komşu taşlar arasındaki farkın az olması hedeflenir.
        - Daha pürüzsüz bir grid, daha iyi bir strateji anlamına gelir.
        """
        smoothness = 0

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] != 0:
                    if j < GRID_SIZE - 1 and self.grid[i][j + 1] != 0:
                        smoothness -= abs(self.grid[i][j] - self.grid[i][j + 1])
                    if i < GRID_SIZE - 1 and self.grid[i + 1][j] != 0:
                        smoothness -= abs(self.grid[i][j] - self.grid[i + 1][j])

        return smoothness

    def _evaluate_max_tile_placement(self) -> int:
        """
        Max Taşın Konumu Heuristic'i:
        - En büyük taşın köşede olması daha iyi bir stratejidir.
        - Köşede değilse, kenarda olması da bir miktar avantaj sağlar.
        """
        max_val = 0
        max_i, max_j = 0, 0

        # En büyük taş değeri ve konumunu bul
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] > max_val:
                    max_val = self.grid[i][j]
                    max_i, max_j = i, j

        # Köşede olma durumu kontrolü (en iyi pozisyon)
        corner_positions = [
            (0, 0),
            (0, GRID_SIZE - 1),
            (GRID_SIZE - 1, 0),
            (GRID_SIZE - 1, GRID_SIZE - 1),
        ]
        if (max_i, max_j) in corner_positions:
            return max_val  # Köşedeyse maksimum puan

        # Kenarda olma durumu kontrolü
        if max_i == 0 or max_i == GRID_SIZE - 1 or max_j == 0 or max_j == GRID_SIZE - 1:
            return max_val // 2  # Kenardaysa yarım puan

        return 0  # Merkezdeyse bonus yok

    def get_best_move(self) -> Optional[Callable]:
        """
        Hill Climbing Algoritması:
        - En iyi hamleyi seçer.
        - **Rastgele Yeniden Başlatma**: Yerel maksimumlardan kaçınmak için %15 olasılıkla rastgele seçim yapar.
        - Tüm olası hamleler değerlendirilir ve en yüksek skoru veren hamle seçilir.
        """
        best_move = None
        best_score = float("-inf")
        current_grid = self.clone_grid()
        original_score = self.score

        # Rastgele yeniden başlatma oranı %15 - yerel maksimumlara takılmamak için
        # Random restart probability 15% to avoid local maxima
        use_random = random.random() < 0.15

        possible_moves = self.get_possible_moves(current_grid)

        if use_random and possible_moves:
            # Akıllı rastgele seçim - tamamen rastgele değil, en iyi 2 hamlede biri
            # Smart randomization - choose one of the top 2 moves
            move_scores = []
            for move in possible_moves:
                temp_grid = copy.deepcopy(current_grid)
                temp_score = self.score
                self.grid = temp_grid
                move()
                score = self.evaluate()
                move_scores.append((move, score))
                self.score = original_score

            # En iyi 2 hamleyi alıp aralarından birini rastgele seç
            # Get top 2 moves and randomly choose one
            move_scores.sort(key=lambda x: x[1], reverse=True)
            top_moves = move_scores[: min(2, len(move_scores))]
            return random.choice([m[0] for m in top_moves])

        # Normal hill climbing işlemi - klasik en iyi hamleyi seç
        # Standard hill climbing - choose the best move
        for move in possible_moves:
            temp_grid = copy.deepcopy(current_grid)
            temp_score = self.score
            self.grid = temp_grid
            move()
            score = self.evaluate()
            if score > best_score:
                best_score = score
                best_move = move
            self.score = original_score

        self.grid = current_grid
        self.score = original_score
        return best_move

    def schedule_ai_move(self):
        """Oyun devam ediyorsa bir sonraki AI hamlesini zamanla."""
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
        draw.text((10, 50), text, fill="black", font=font)
        screenshot.save(filename)

    def ai_play(self):
        """AI hamlesini yap ve oyun devam ediyorsa sonraki hamleyi zamanla."""
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
                            "game_mid_hillclimbing.png", "Hill Climbing Algorithm"
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
        """
        Taşları sola kaydırır ve mümkünse birleştirir.
        - Bu işlem, grid'in her satırında yapılır.
        - Sıfırlar çıkarılır, taşlar birleştirilir ve tekrar sıkıştırılır.
        """
        return self.move(lambda row: self.compress(row), lambda row: self.merge(row))

    def move_right(self) -> bool:
        """
        Taşları sağa kaydırır ve mümkünse birleştirir.
        - Bu işlem, grid'in her satırında yapılır.
        - Satırlar ters çevrilir, işlem yapılır ve tekrar ters çevrilir.
        """
        return self.move(
            lambda row: self.compress(row[::-1])[::-1],
            lambda row: self.merge(row[::-1])[::-1],
        )

    def move_up(self) -> bool:
        """
        Taşları yukarı kaydırır ve mümkünse birleştirir.
        - Bu işlem, grid'in her sütununda yapılır.
        - Grid transpoze edilir, işlem yapılır ve tekrar transpoze edilir.
        """
        return self.move_columns(
            lambda col: self.compress(col), lambda col: self.merge(col)
        )

    def move_down(self) -> bool:
        """
        Taşları aşağı kaydırır ve mümkünse birleştirir.
        - Bu işlem, grid'in her sütununda yapılır.
        - Grid transpoze edilir, ters çevrilir, işlem yapılır ve tekrar eski haline getirilir.
        """
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
        """
        Sıfırları çıkarır ve sayıları bir tarafa kaydırır.
        - Bu işlem, bir satırdaki taşları sıkıştırarak boşlukları doldurur.
        """
        return [num for num in row if num != 0] + [0] * row.count(0)

    def merge(self, row: List[int]) -> List[int]:
        """
        Yan yana eşit sayıları birleştirir.
        - Birleştirilen taşların değeri iki katına çıkar ve skor güncellenir.
        """
        for i in range(len(row) - 1):
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]  # Skoru güncelle
        return row

    def can_move(self) -> bool:
        """
        Herhangi bir hareketin mümkün olup olmadığını kontrol eder.
        - Eğer grid'de boş hücre varsa veya birleştirilebilecek taşlar varsa, hareket mümkündür.
        """
        # Boş hücreler için kontrol
        if any(0 in row for row in self.grid):
            return True

        # Yatay birleştirme imkanı kontrolü
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if self.grid[i][j] == self.grid[i][j + 1]:
                    return True

        # Dikey birleştirme imkanı kontrolü
        for i in range(GRID_SIZE - 1):
            for j in range(GRID_SIZE):
                if self.grid[i][j] == self.grid[i + 1][j]:
                    return True

        return False

    def game_over(self):
        """
        Oyun sona erdiğinde çalışacak fonksiyon.
        - Oyun sonu mesajı gösterilir ve ekran görüntüsü alınır.
        """
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
            )  # Gridin altına yerleştir
            self.take_screenshot(
                "game_over_hillclimbing.png", "Hill Climbing Algorithm"
            )  # Oyun sonu ekran görüntüsü
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
