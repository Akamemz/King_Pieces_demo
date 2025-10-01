import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math


# --- Army Placement Logic ---

def get_king_army_placements(n, k):
    """
    Calculates the optimal placement and size for k peaceful armies of kings on an n x n board.
    Returns placements, army_size, and a description of the strategy.
    """
    placements = {}  # Dictionary to hold coordinates for each army
    army_size = 0
    strategy = ""

    # Define a list of colors for the armies
    colors = ['#FF4136', '#0074D9', '#2ECC40', '#FFDC00', '#B10DC9', '#FF851B', '#7FDBFF', '#F012BE']

    if k == 1:
        # Standard independence problem
        strategy = "Standard Independence: Kings are placed on every other square to maximize their number."
        placements[0] = []
        for r in range(0, n, 2):
            for c in range(0, n, 2):
                placements[0].append((r, c))
        army_size = len(placements[0])

    elif k == 2:
        # Two armies: split the board vertically with one buffer column
        strategy = "Two Armies (Strips): The board is split vertically, with a one-square buffer column separating the two armies."
        army_width = (n - 1) // 2
        if army_width > 0:
            army_size = n * army_width
            # Army 1
            placements[0] = [(r, c) for r in range(n) for c in range(army_width)]
            # Army 2
            placements[1] = [(r, c) for r in range(n) for c in range(army_width + 1, n)]
        else:  # handles small boards where no space is available
            army_size = 0


    elif k == 3:
        # Three armies: split the board vertically with two buffer columns
        strategy = "Three Armies (Strips): The board is divided into three vertical strips, separated by two buffer columns."
        army_width = (n - 2) // 3
        if army_width > 0:
            army_size = n * army_width
            # Army 1
            placements[0] = [(r, c) for r in range(n) for c in range(army_width)]
            # Army 2
            placements[1] = [(r, c) for r in range(n) for c in range(army_width + 1, 2 * army_width + 1)]
            # Army 3
            placements[2] = [(r, c) for r in range(n) for c in range(2 * army_width + 2, n)]
        else:
            army_size = 0

    elif k == 4:
        # Four armies: split the board into a 2x2 grid (quadrants)
        strategy = "Four Armies (Quadrants): The board is divided into a 2x2 grid, with a buffer row and column separating the four armies."
        army_dim = (n - 1) // 2
        if army_dim > 0:
            army_size = army_dim * army_dim
            # Army 1 (top-left)
            placements[0] = [(r, c) for r in range(army_dim) for c in range(army_dim)]
            # Army 2 (top-right)
            placements[1] = [(r, c) for r in range(army_dim) for c in range(army_dim + 1, n)]
            # Army 3 (bottom-left)
            placements[2] = [(r, c) for r in range(army_dim + 1, n) for c in range(army_dim)]
            # Army 4 (bottom-right)
            placements[3] = [(r, c) for r in range(army_dim + 1, n) for c in range(army_dim + 1, n)]
        else:
            army_size = 0

    elif k > 4:
        # General case for k > 4: Arrange as k vertical strips
        strategy = f"{k} Armies (Strips): A general strategy is to arrange the armies in {k} vertical strips, separated by {k - 1} buffer columns."
        num_buffers = k - 1
        army_width = (n - num_buffers) // k
        if army_width > 0:
            army_size = n * army_width
            for i in range(k):
                start_col = i * (army_width + 1)
                end_col = start_col + army_width
                placements[i] = [(r, c) for r in range(n) for c in range(start_col, end_col)]
        else:
            army_size = 0

    # Assign colors to each army
    for i in range(k):
        if i in placements:
            placements[i] = {'coords': placements[i], 'color': colors[i % len(colors)]}

    return placements, army_size, strategy


# --- Matplotlib Board Drawing ---

def draw_board(n, armies, light_color, dark_color):
    """Uses Matplotlib to draw the chessboard and place the armies."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([chr(65 + i) for i in range(n)])
    ax.set_yticklabels(range(1, n + 1))
    ax.tick_params(length=0)
    ax.grid(True, color='black', linewidth=1.5)
    ax.set_aspect('equal')

    # Draw squares with user-selected colors
    for r in range(n):
        for c in range(n):
            color = light_color if (r + c) % 2 == 0 else dark_color
            rect = patches.Rectangle((c, r), 1, 1, facecolor=color)
            ax.add_patch(rect)

    # Place pieces for each army
    for i, army_data in armies.items():
        color = army_data['color']
        for r, c in army_data['coords']:
            ax.text(c + 0.5, r + 0.5, '♔',
                    fontsize=40 / n * 4, ha='center', va='center', color=color,
                    bbox=dict(boxstyle='circle,pad=0.1', fc=color, alpha=0.3, ec='none'))

    ax.invert_yaxis()
    return fig


# --- Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("Peaceful Armies of Kings")
st.write(
    "This demo calculates and visualizes the maximum size of `k` equal-sized armies of kings "
    "that can be placed on an `n x n` board so that kings from different armies do not attack each other."
)

st.sidebar.header("Controls")
board_size = st.sidebar.slider("Select Board Size (n)", min_value=1, max_value=25, value=9)
num_armies = st.sidebar.slider("Select Number of Armies (k)", min_value=1, max_value=8, value=4)

st.sidebar.header("Color Customization")
light_square_color = st.sidebar.color_picker("Light Square Color", "#F0D9B5")
dark_square_color = st.sidebar.color_picker("Dark Square Color", "#B58863")

# --- Main Logic & Display ---

col1, col2 = st.columns([1, 2])

armies, army_size, strategy = get_king_army_placements(board_size, num_armies)

with col1:
    st.header("Results")
    st.metric(f"Size of Each Army", army_size)
    st.metric("Total Number of Kings", army_size * num_armies if army_size > 0 else 0)
    st.write("---")
    st.subheader("Optimal Strategy")
    st.info(strategy)

with col2:
    st.header("Board Visualization")
    if army_size == 0 and board_size > 0:
        st.warning(
            f"The {board_size}x{board_size} board is too small to fit {num_armies} armies with the required buffer zones.")
    fig = draw_board(board_size, armies, light_square_color, dark_square_color)
    st.pyplot(fig)

