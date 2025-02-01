    best_move = None
        max_score = -1
        current_grid = self.clone_grid()

        # Only consider moves that are possible (change the grid)
        possible_moves = self.get_possible_moves(current_grid)

        for move in possible_moves:
            score_diff = self.evaluate_move(move)
            if score_diff > max_score:
                max_score = score_diff
                best_move = move

        return best_move