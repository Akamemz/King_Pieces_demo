import streamlit as st

# --- Page Configuration ---
# This should be the first Streamlit command in your main script.
st.set_page_config(
    page_title="Chessboard Placement Puzzles",
    page_icon="♔",
    layout="wide"
)

# --- App Content ---

st.title("Welcome to Chessboard Placement Puzzles!")

st.markdown(
    """
    This application is a collection of interactive tools designed to explore mathematical and strategic
    problems on a chessboard. Each page in this app tackles a different puzzle, allowing you to visualize
    solutions and experiment with various parameters.
    """
)

st.info(
    """
    **How to Navigate:** Use the sidebar on the left to select a puzzle to explore.
    """,
    icon="👈"
)

st.header("Featured Puzzle: Peaceful Armies of Kings")
st.markdown(
    """
    The **Peaceful Armies of Kings** puzzle explores a variation of the classic chessboard independence problem.
    The goal is to determine the maximum size of `k` equal-sized armies of kings that can be placed on
    an `n x n` board such that no king from one army can attack a king from a different army.

    Navigate to the page to see the optimal placements, adjust the board size, and change the number of armies.
    """
)
st.image("https://images.unsplash.com/photo-1580541832626-2a7131ee809f?q=80&w=2070&auto=format&fit=crop",
         caption="Explore strategic placements on the board.", use_column_width=True)

st.write("---")
st.subheader("About This Project")
st.markdown(
    """
    This project was created to demonstrate Streamlit's capabilities for building interactive, data-driven
    applications for mathematical visualization. If you have ideas for other puzzles to add, feel free
    to contribute!
    """
)
