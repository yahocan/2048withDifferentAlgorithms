import tkinter as tk
import random

# Sabitler ve renk tanımları
GRID_SIZE = 4  # Oyun gridinin boyutu (4x4)
NEW_TILE_VALUES = [2, 4]  # Yeni taşların alabileceği değerler
BACKGROUND_COLOR = "#bbada0"  # Arka plan rengi
TILE_COLORS = {  # Taşların değerlerine göre renkleri
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
FONT = ("Verdana", 24, "bold")  # Taşların yazı tipi ve boyutu


class Game2048:
    def __init__(self):
        # Oyun sınıfının başlangıç ayarları
        self.window = tk.Tk()  # Tkinter penceresi oluşturulur
        self.window.title("2048")  # Pencere başlığı
        self.grid = [
            [0] * GRID_SIZE for _ in range(GRID_SIZE)
        ]  # Oyun gridini sıfırlarla başlat
        self.cells = []  # GUI hücreleri için liste
        self.score = 0  # Oyuncunun skoru
        self.init_gui()  # Grafik arayüzü başlat
        self.start_game()  # Oyunu başlat
        self.window.mainloop()  # Tkinter döngüsünü başlat

    def init_gui(self):
        # Grafik arayüzünü başlatır
        self.frame = tk.Frame(self.window, bg=BACKGROUND_COLOR)  # Ana çerçeve
        self.frame.grid()
        self.score_label = tk.Label(
            self.window,
            text=f"Score: {self.score}",  # Skor etiketi
            font=("Verdana", 20, "bold"),
            bg=BACKGROUND_COLOR,
        )
        self.score_label.grid(
            row=2, column=0, columnspan=GRID_SIZE
        )  # Skor etiketi konumu
        for i in range(GRID_SIZE):
            row_cells = []
            for j in range(GRID_SIZE):
                cell = tk.Label(
                    self.frame, text="", width=4, height=2, font=FONT, bg=TILE_COLORS[0]
                )  # Hücreler için etiketler
                cell.grid(row=i + 1, column=j, padx=5, pady=5)  # Hücre konumu
                row_cells.append(cell)
            self.cells.append(row_cells)
        self.window.bind("<KeyPress>", self.handle_keypress)  # Klavye tuşlarını dinle

    def start_game(self):
        # Oyunu başlatır ve başlangıçta iki taş ekler
        self.add_new_tile()
        self.add_new_tile()
        self.update_gui()

    def add_new_tile(self):
        # Rastgele bir boş hücreye yeni bir taş ekler
        empty_cells = [
            (i, j)
            for i in range(GRID_SIZE)
            for j in range(GRID_SIZE)
            if self.grid[i][j] == 0
        ]
        if empty_cells:
            i, j = random.choice(empty_cells)  # Rastgele bir boş hücre seç
            self.grid[i][j] = random.choice(NEW_TILE_VALUES)  # Yeni taş ekle

    def update_gui(self):
        # Grafik arayüzünü günceller
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                value = self.grid[i][j]
                self.cells[i][j].config(
                    text=str(value) if value else "",  # Taş değeri
                    bg=TILE_COLORS.get(value, "#edc22e"),  # Taş rengi
                )
        self.score_label.config(text=f"Score: {self.score}")  # Skoru güncelle

    def handle_keypress(self, event):
        # Klavye tuşlarına basıldığında çalışır
        key_map = {
            "Up": self.move_up,
            "Down": self.move_down,
            "Left": self.move_left,
            "Right": self.move_right,
        }
        if event.keysym in key_map:
            moved = key_map[event.keysym]()  # İlgili hareket fonksiyonunu çağır
            if moved:
                self.add_new_tile()  # Yeni taş ekle
                self.update_gui()  # Grafik arayüzünü güncelle
                if not self.can_move():  # Hareket mümkün değilse oyunu bitir
                    self.game_over()

    def move_left(self):
        # Taşları sola kaydırır ve birleştirir
        return self.move(lambda row: self.compress(row), lambda row: self.merge(row))

    def move_right(self):
        # Taşları sağa kaydırır ve birleştirir
        return self.move(
            lambda row: self.compress(row[::-1])[::-1],
            lambda row: self.merge(row[::-1])[::-1],
        )

    def move_up(self):
        # Taşları yukarı kaydırır ve birleştirir
        return self.move_columns(
            lambda col: self.compress(col), lambda col: self.merge(col)
        )

    def move_down(self):
        # Taşları aşağı kaydırır ve birleştirir
        return self.move_columns(
            lambda col: self.compress(col[::-1])[::-1],
            lambda col: self.merge(col[::-1])[::-1],
        )

    def move(self, compress_fn, merge_fn):
        # Genel hareket fonksiyonu
        moved = False
        for i in range(GRID_SIZE):
            original = self.grid[i][:]  # Satırın orijinal halini sakla
            compressed = compress_fn(original)  # Sıkıştırma işlemi
            merged = merge_fn(compressed)  # Birleştirme işlemi
            new_row = compress_fn(merged)  # Tekrar sıkıştırma
            if new_row != original:  # Eğer değişiklik olduysa
                self.grid[i] = new_row
                moved = True
        return moved

    def move_columns(self, compress_fn, merge_fn):
        # Sütunlar için genel hareket fonksiyonu
        moved = False
        for j in range(GRID_SIZE):
            original = [self.grid[i][j] for i in range(GRID_SIZE)]  # Sütunu al
            compressed = compress_fn(original)  # Sıkıştırma işlemi
            merged = merge_fn(compressed)  # Birleştirme işlemi
            new_col = compress_fn(merged)  # Tekrar sıkıştırma
            if new_col != original:  # Eğer değişiklik olduysa
                for i in range(GRID_SIZE):
                    self.grid[i][j] = new_col[i]  # Sütunu güncelle
                moved = True
        return moved

    def compress(self, row):
        # Sıfırları çıkar ve sayıları bir tarafa kaydır
        return [num for num in row if num != 0] + [0] * row.count(0)

    def merge(self, row):
        # Yan yana eşit sayıları birleştir
        for i in range(len(row) - 1):
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] *= 2  # Birleştirilen taşın değeri iki katına çıkar
                self.score += row[i]  # Skoru güncelle
                row[i + 1] = 0  # Birleştirilen taş sıfırlanır
        return row

    def can_move(self):
        # Herhangi bir hareketin mümkün olup olmadığını kontrol eder
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if self.grid[i][j] == self.grid[i][j + 1]:  # Yatay kontrol
                    return True
        for i in range(GRID_SIZE - 1):
            for j in range(GRID_SIZE):
                if self.grid[i][j] == self.grid[i + 1][j]:  # Dikey kontrol
                    return True
        return any(0 in row for row in self.grid)  # Boş hücre kontrolü

    def game_over(self):
        # Oyun sona erdiğinde çalışacak fonksiyon
        game_over_label = tk.Label(
            self.frame,
            text="Game Over!",
            font=("Verdana", 30, "bold"),
            bg=BACKGROUND_COLOR,
        )
        game_over_label.grid(
            row=0, column=0, columnspan=GRID_SIZE
        )  # "Game Over" mesajını göster


if __name__ == "__main__":
    Game2048()  # Oyunu başlat
