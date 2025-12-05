"""
Blind-Accessible Tic-Tac-Toe Game
Fully keyboard-driven with voice feedback for all actions
"""
import tkinter as tk
from tkinter import font
import os


class BlindAccessibleTicTacToe:
    """Tic-Tac-Toe game designed for blind users"""
    
    def __init__(self, master, tts_callback, tts_stop_callback, ai_callback, ai_cancel_callback, voice_command_callback=None):
        """
        Initialize blind-accessible game
        
        Args:
            master: Tkinter root window
            tts_callback: Function to speak text (callable with text string)
            tts_stop_callback: Function to stop/interrupt current speech
            ai_callback: Function to trigger AI description (callable)
            ai_cancel_callback: Function to cancel ongoing AI processing
            voice_command_callback: Function to trigger voice command mode (callable)
        """
        self.master = master
        self.tts = tts_callback
        self.tts_stop = tts_stop_callback
        self.ai_callback = ai_callback
        self.ai_cancel = ai_cancel_callback
        self.voice_command_callback = voice_command_callback
        
        master.title("Blind-Accessible Tic-Tac-Toe")
        master.geometry("600x700")
        master.configure(bg='#1a1a1a')
        
        # Game state
        self.board = [' ' for _ in range(9)]  # 0-8 for positions 1-9
        self.current_player = 'X'
        self.game_over = False
        self.winner = None
        
        # UI elements
        self.buttons = []
        self.status_label = None
        self.info_label = None
        
        self._create_widgets()
        self._bind_keys()
        
        # Announce game start after a short delay
        master.after(1000, lambda: self.tts(
            "Welcome to Tic Tac Toe. You are playing X. "
            "Press numbers 1 through 9 to select a square. "
            "Top left is 1, top right is 3, bottom right is 9. "
            "Press D for AI assistance. "
            "Press R to reset the game. "
            "It's your turn, Player X."
        ))
    
    def _create_widgets(self):
        """Create game UI"""
        # Title
        title = tk.Label(
            self.master,
            text="Blind-Accessible Tic-Tac-Toe",
            font=('Arial', 20, 'bold'),
            bg='#1a1a1a',
            fg='#ffffff'
        )
        title.pack(pady=20)
        
        # Status label
        self.status_label = tk.Label(
            self.master,
            text="Player X's turn",
            font=('Arial', 16, 'bold'),
            bg='#1a1a1a',
            fg='#00ff00'
        )
        self.status_label.pack(pady=10)
        
        # Info label
        self.info_label = tk.Label(
            self.master,
            text="Press 1-9 for squares | D for AI help | R to reset",
            font=('Arial', 12),
            bg='#1a1a1a',
            fg='#888888',
            wraplength=550
        )
        self.info_label.pack(pady=5)
        
        # Board frame
        board_frame = tk.Frame(self.master, bg='#1a1a1a')
        board_frame.pack(pady=20)
        
        # Create 3x3 grid with large, high-contrast buttons
        button_font = font.Font(family='Arial', size=48, weight='bold')
        for i in range(9):
            row = i // 3
            col = i % 3
            
            btn = tk.Label(
                board_frame,
                text=str(i + 1),  # Show position number
                font=button_font,
                width=3,
                height=1,
                bg='#333333',
                fg='#666666',
                relief='raised',
                borderwidth=3
            )
            btn.grid(row=row, column=col, padx=5, pady=5)
            self.buttons.append(btn)
        
        # Instructions
        instructions = tk.Label(
            self.master,
            text="Square numbering:\n1  2  3\n4  5  6\n7  8  9",
            font=('Courier', 14),
            bg='#1a1a1a',
            fg='#888888',
            justify='center'
        )
        instructions.pack(pady=20)
    
    def _bind_keys(self):
        """Bind keyboard shortcuts"""
        self.master.bind('<Key>', self._on_key_press)
        self.master.focus_set()
    
    def _on_key_press(self, event):
        """Handle keyboard input"""
        key = event.char.lower()
        
        # Number keys 1-9 for moves
        if key in '123456789':
            # Interrupt speech and cancel AI for game move
            self.tts_stop()
            if self.ai_cancel:
                self.ai_cancel()
            position = int(key) - 1
            self._make_move(position)
        
        # D for AI description
        elif key == 'd':
            # Don't interrupt - user wants AI info
            # If something is speaking, let it finish or they can press another key
            self.tts("Analyzing board...")
            self.ai_callback()
        
        # V for voice command
        elif key == 'v':
            # Interrupt for voice command
            self.tts_stop()
            if self.ai_cancel:
                self.ai_cancel()
            # Call voice command callback if available
            if self.voice_command_callback:
                self.voice_command_callback()
            else:
                self.tts("Voice commands not available.")
        
        # R for reset
        elif key == 'r':
            # Interrupt speech and cancel AI for reset
            self.tts_stop()
            if self.ai_cancel:
                self.ai_cancel()
            self._reset_game()
        
        # H for help
        elif key == 'h':
            # Interrupt for help
            self.tts_stop()
            if self.ai_cancel:
                self.ai_cancel()
            self._announce_help()
    
    def _make_move(self, position):
        """
        Attempt to make a move at the given position
        
        Args:
            position: Board position (0-8)
        """
        if self.game_over:
            self._play_error_sound()
            self.tts("Game is over. Press R to reset.")
            return
        
        if self.board[position] != ' ':
            self._play_error_sound()
            square_num = position + 1
            occupant = self.board[position]
            self.tts(f"Square {square_num} is already occupied by {occupant}. Choose another square.")
            return
        
        # Valid move
        self.board[position] = self.current_player
        square_num = position + 1
        
        # Update button display
        self.buttons[position].config(
            text=self.current_player,
            fg='#00ff00' if self.current_player == 'X' else '#ff4444',
            bg='#222222'
        )
        
        # Announce move immediately
        self.tts(f"Placed {self.current_player} on square {square_num}.")
        
        # Check for winner
        if self._check_winner():
            self.game_over = True
            self.winner = self.current_player
            self.status_label.config(text=f"Player {self.winner} wins!")
            # Queue win message immediately (TTS queue handles ordering)
            self.tts(f"Game over! Player {self.winner} wins! Press R to play again.")
            return
        
        # Check for tie
        if ' ' not in self.board:
            self.game_over = True
            self.status_label.config(text="It's a tie!")
            # Queue tie message immediately
            self.tts("Game over! It's a tie! Press R to play again.")
            return
        
        # Switch players
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        self.status_label.config(text=f"Player {self.current_player}'s turn")
        
        # Announce next turn immediately (TTS queue handles ordering)
        self.tts(f"It's now Player {self.current_player}'s turn.")
    
    def _check_winner(self):
        """Check if current player has won"""
        # Winning combinations (0-indexed)
        wins = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
        for combo in wins:
            if all(self.board[i] == self.current_player for i in combo):
                # Highlight winning combination
                for i in combo:
                    self.buttons[i].config(bg='#ffff00', fg='#000000')
                return True
        return False
    
    def _reset_game(self):
        """Reset the game"""
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'
        self.game_over = False
        self.winner = None
        
        # Reset all buttons
        for i, btn in enumerate(self.buttons):
            btn.config(
                text=str(i + 1),
                fg='#666666',
                bg='#333333'
            )
        
        self.status_label.config(text="Player X's turn")
        self.tts("Game reset. You are playing X. It's your turn.")
    
    def _announce_help(self):
        """Announce help information"""
        self.tts(
            "Press numbers 1 through 9 to place your piece. "
            "Square 1 is top left, square 3 is top right, square 9 is bottom right. "
            "Press D to hear the board state from AI. "
            "Press R to reset the game. "
            "Press H for help."
        )
    
    def _play_error_sound(self):
        """Play an error buzzer sound"""
        # Use system beep (cross-platform)
        try:
            # macOS
            os.system('afplay /System/Library/Sounds/Basso.aiff &')
        except:
            # Fallback: terminal bell
            print('\a')
    
    def get_window_bounds(self):
        """Get window position and size for screen capture"""
        self.master.update_idletasks()
        return {
            'x': self.master.winfo_x(),
            'y': self.master.winfo_y(),
            'width': self.master.winfo_width(),
            'height': self.master.winfo_height()
        }
    
    def get_board_state_description(self):
        """Get a concise text description of the current board state"""
        x_squares = []
        o_squares = []
        empty_squares = []
        
        for i in range(9):
            square_num = i + 1
            if self.board[i] == 'X':
                x_squares.append(square_num)
            elif self.board[i] == 'O':
                o_squares.append(square_num)
            else:
                empty_squares.append(square_num)
        
        # Build concise description
        parts = []
        
        if x_squares:
            parts.append(f"X on {', '.join(map(str, x_squares))}")
        if o_squares:
            parts.append(f"O on {', '.join(map(str, o_squares))}")
        if empty_squares:
            parts.append(f"Empty: {', '.join(map(str, empty_squares))}")
        
        desc = ". ".join(parts) + "."
        return desc
    
    def run(self):
        """Start the game"""
        self.master.mainloop()


