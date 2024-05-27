import random

class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]  # Initialize an empty board
        self.current_player = 'X'  # Player 'X' starts first

    def print_board(self):
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            return True
        return False

    def is_winner(self, letter):
        # Check rows, columns, and diagonals for a win
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]
        for condition in win_conditions:
            if all(self.board[i] == letter for i in condition):
                return True
        return False

    def is_board_full(self):
        return ' ' not in self.board

    def minimax(self, depth, is_maximizing):
        if self.is_winner('X'):
            return -1  # If AI wins, return a high score (negative because maximizing for 'O')
        if self.is_winner('O'):
            return 1  # If player wins, return a low score (positive because minimizing for 'X')
        if self.is_board_full():
            return 0  # Game is a draw

        if is_maximizing:
            best_score = -float('inf')
            for move in self.available_moves():
                self.board[move] = 'O'
                score = self.minimax(depth + 1, False)
                self.board[move] = ' '  # Undo move
                best_score = max(best_score, score)
            return best_score
        else:
            best_score = float('inf')
            for move in self.available_moves():
                self.board[move] = 'X'
                score = self.minimax(depth + 1, True)
                self.board[move] = ' '  # Undo move
                best_score = min(best_score, score)
            return best_score

    def get_best_move(self):
        best_move = -1
        best_score = -float('inf')
        for move in self.available_moves():
            self.board[move] = 'O'
            score = self.minimax(0, False)
            self.board[move] = ' '  # Undo move
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def play(self):
        while not self.is_board_full():
            if self.current_player == 'X':
                # Player's turn
                self.print_board()
                try:
                    player_move = int(input("Enter your move (0-8): "))
                    if player_move in self.available_moves():
                        self.make_move(player_move, 'X')
                        if self.is_winner('X'):
                            print("Congratulations! You win!")
                            break
                        self.current_player = 'O'
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            else:
                # AI's turn
                ai_move = self.get_best_move()
                self.make_move(ai_move, 'O')
                print("AI makes a move...")
                if self.is_winner('O'):
                    self.print_board()
                    print("AI wins! Better luck next time.")
                    break
                self.current_player = 'X'

        if self.is_board_full() and not self.is_winner('X') and not self.is_winner('O'):
            self.print_board()
            print("It's a draw!")

# Main game loop
if __name__ == '__main__':
    game = TicTacToe()
    game.play()