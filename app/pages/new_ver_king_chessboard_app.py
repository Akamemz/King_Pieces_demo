# Peaceful Armies of Kings — Interactive Streamlit App
# Summary moved into an expander, with an expanded "Why lattices matter" section
# Run: streamlit run app.py

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import streamlit as st

Coord = Tuple[int, int]
Region = Tuple[int, int, int, int]  # (r0, r1, c0, c1)

# ==============================
# Core helpers
# ==============================

def king_attacks(a: Coord, b: Coord) -> bool:
    """Return True iff kings on a and b attack each other (Chebyshev distance ≤ 1)."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def validate_cross_army_peace(placements: Dict[int, List[Coord]]) -> Tuple[bool, List[Tuple[int, Coord, int, Coord]]]:
    """Check that no king in one army attacks a king in another army.
    Returns (is_valid, conflicts_list).
    """
    conflicts = []
    armies = sorted(placements.keys())
    for i, ai in enumerate(armies):
        for aj in armies[i+1:]:
            for p in placements[ai]:
                for q in placements[aj]:
                    if king_attacks(p, q):
                        conflicts.append((ai, p, aj, q))
    return (len(conflicts) == 0, conflicts)


# ==============================
# Region builders with 1‑cell moats
# ==============================

def vertical_strips(n: int, k: int) -> List[Region]:
    if k <= 0:
        return []
    moats = k - 1
    usable = n - moats
    if usable <= 0:
        return []
    base_w, extra = divmod(usable, k)
    widths = [base_w + (1 if i < extra else 0) for i in range(k)]

    regions: List[Region] = []
    c = 0
    for w in widths:
        c0 = c
        c1 = c0 + w - 1
        regions.append((0, n - 1, c0, c1))
        c = c1 + 2  # moat column
    return regions


def horizontal_strips(n: int, k: int) -> List[Region]:
    moats = k - 1
    usable = n - moats
    if usable <= 0:
        return []
    base_h, extra = divmod(usable, k)
    heights = [base_h + (1 if i < extra else 0) for i in range(k)]

    regions: List[Region] = []
    r = 0
    for h in heights:
        r0 = r
        r1 = r0 + h - 1
        regions.append((r0, r1, 0, n - 1))
        r = r1 + 2  # moat row
    return regions


def grid_regions(n: int, k: int, rows: int = None, cols: int = None) -> List[Region]:
    if rows is None or cols is None:
        cols = math.ceil(math.sqrt(k)) if cols is None else cols
        rows = math.ceil(k / cols) if rows is None else rows
    if rows * cols < k:
        raise ValueError("rows*cols must be ≥ k")

    moat_rows = rows - 1
    moat_cols = cols - 1
    usable_r = n - moat_rows
    usable_c = n - moat_cols
    if usable_r <= 0 or usable_c <= 0:
        return []

    base_h, extra_h = divmod(usable_r, rows)
    base_w, extra_w = divmod(usable_c, cols)

    heights = [base_h + (1 if i < extra_h else 0) for i in range(rows)]
    widths  = [base_w + (1 if j < extra_w else 0) for j in range(cols)]

    regions: List[Region] = []
    r = 0
    for i in range(rows):
        r0 = r
        r1 = r0 + heights[i] - 1
        c = 0
        for j in range(cols):
            c0 = c
            c1 = c0 + widths[j] - 1
            regions.append((r0, r1, c0, c1))
            c = c1 + 2  # moat col
        r = r1 + 2  # moat row
    return regions[:k]


# ==============================
# Population strategies inside regions
# ==============================

def fill_every_square(region: Region) -> List[Coord]:
    r0, r1, c0, c1 = region
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def fill_lattice(region: Region, step: int) -> List[Coord]:
    """Generic lattice: place a king every `step` rows and columns (step ≥ 2 ensures no same‑army attacks)."""
    r0, r1, c0, c1 = region
    coords: List[Coord] = []
    for r in range(r0, r1 + 1, step):
        for c in range(c0, c1 + 1, step):
            coords.append((r, c))
    return coords


@dataclass
class Layout:
    regions: List[Region]
    placements: Dict[int, List[Coord]]
    per_army: int
    total: int
    strategy_text: str


def build_layout(n: int, k: int, style: str, intra_army_mode: str) -> Layout:
    if k <= 0 or n <= 0:
        return Layout([], {}, 0, 0, "Invalid input.")

    if style == "Vertical strips":
        regions = vertical_strips(n, k)
        style_note = "k vertical regions with one‑column moats."
    elif style == "Horizontal strips":
        regions = horizontal_strips(n, k)
        style_note = "k horizontal regions with one‑row moats."
    else:
        regions = grid_regions(n, k)
        style_note = "grid of near‑square regions with moats between blocks."

    if not regions or len(regions) < k:
        return Layout(regions, {}, 0, 0, "Board too small to host k regions with required moats.")

    placements: Dict[int, List[Coord]] = {}
    for i, reg in enumerate(regions):
        if intra_army_mode == "Every square (same‑army attacks allowed)":
            coords = fill_every_square(reg)
        elif intra_army_mode == "Non‑attacking: 2×2 lattice":
            coords = fill_lattice(reg, step=2)
        elif intra_army_mode == "Non‑attacking: 3×3 lattice":
            coords = fill_lattice(reg, step=3)
        else:
            coords = fill_lattice(reg, step=2)  # safe default
        placements[i] = coords

    per_army = len(next(iter(placements.values()))) if placements else 0
    total = per_army * k

    strategy = (
        f"Arrangement: {style}. Regions separated by one‑cell moats to prevent cross‑army attacks; "
        f"inside each region we use ‘{intra_army_mode}’. {style_note}"
    )

    return Layout(regions, placements, per_army, total, strategy)


# ==============================
# Visualization
# ==============================

def draw_board(n: int, armies: Dict[int, List[Coord]], colors: List[str], light_color: str, dark_color: str,
               show_regions: bool, regions: List[Region]):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([chr(65 + i) for i in range(n)])
    ax.set_yticklabels(range(1, n + 1))
    ax.tick_params(length=0)
    ax.grid(True, color='black', linewidth=1.0)
    ax.set_aspect('equal')

    # squares
    for r in range(n):
        for c in range(n):
            color = light_color if (r + c) % 2 == 0 else dark_color
            rect = patches.Rectangle((c, r), 1, 1, facecolor=color)
            ax.add_patch(rect)

    # optional region outlines
    if show_regions:
        for i, (r0, r1, c0, c1) in enumerate(regions):
            rect = patches.Rectangle((c0, r0), c1 - c0 + 1, r1 - r0 + 1,
                                     fill=False, linewidth=2, linestyle='--')
            ax.add_patch(rect)
            ax.text(c0 + (c1 - c0 + 1) / 2, r0 + (r1 - r0 + 1) / 2, f"Army {i+1}",
                    ha='center', va='center', fontsize=10)

    # kings
    for i, coords in armies.items():
        color = colors[i % len(colors)]
        for (r, c) in coords:
            ax.text(c + 0.5, r + 0.5, '♔', fontsize=40 / max(1, n) * 4,
                    ha='center', va='center', color=color,
                    bbox=dict(boxstyle='circle,pad=0.08', fc=color, alpha=0.30, ec='none'))

    ax.invert_yaxis()
    return fig


# ==============================
# Streamlit UI
# ==============================

st.set_page_config(layout="wide", page_title="Peaceful Armies of Kings")
st.title("Peaceful Armies of Kings — Interactive Explorer (Kings)")

with st.expander("What do ‘peaceful’ and ‘lattice’ mean?", expanded=False):
    st.markdown(
        """
        **Peaceful armies**: Partition kings into \(k\) color‑classes (armies) on an \(n\times n\) board so that no king from one
        army can attack a king from another army. We enforce this by splitting the board into **regions** separated by 1‑cell
        **moats**; any two squares in different regions are at Chebyshev distance ≥ 2.

        **Lattice (intra‑army structure)**: A *lattice* is a regular grid pattern used *inside a region* so that kings of the
        **same** army also don’t attack each other. Placing a king every `s` rows and `s` columns (spacing `step = s`) ensures
        same‑army Chebyshev distance ≥ `s`.

        • **2×2 lattice (step = 2)** — *Optimal non‑attacking density for kings.* Count in an \(r\times c\) region is
          \(\lceil r/2\rceil\cdot\lceil c/2\rceil\) (≈ 1/4 of the squares asymptotically).  
        • **3×3 lattice (step = 3)** — *Looser spacing.* Count is \(\lceil r/3\rceil\cdot\lceil c/3\rceil\) (≈ 1/9 density).

        **Limits & why it’s interesting**  
        • The 2×2 lattice is *density‑optimal* for non‑attacking kings on large boards; you can’t beat ~1/4 density without
          re‑introducing adjacency.  
        • Lattice spacing reveals a clean **trade‑off** between *intra‑army safety* and *army size*: larger steps (e.g., 3×3)
          provide more visual separation but reduce soldiers per army.  
        • The ideas generalize: other pieces (knights, bishops, rooks) have their own optimal spacings/tilings; comparing them
          is a fun entry point to **combinatorial packing** and **discrete geometry**.  
        • On finite boards, **boundary effects** (leftover rows/cols) slightly perturb counts via ceilings; the asymptotic
          densities (1/4 for 2×2, 1/9 for 3×3) emerge as \(n\) grows.
        """
    )

st.sidebar.header("Controls")
n = st.sidebar.slider("Board size n", 1, 40, 12)
k = st.sidebar.slider("Number of armies k", 1, 12, 4)

st.sidebar.subheader("Arrangement style")
style = st.sidebar.selectbox("Choose a partition pattern", [
    "Vertical strips", "Horizontal strips", "Grid (auto)"
])

st.sidebar.subheader("Inside each region (kings)")
intra = st.sidebar.radio("Fill rule", [
    "Every square (same‑army attacks allowed)",
    "Non‑attacking: 2×2 lattice",
    "Non‑attacking: 3×3 lattice",
], index=0)

compare = st.sidebar.checkbox("Side‑by‑side compare 2×2 vs 3×3 lattices", value=False)

st.sidebar.subheader("Board colors")
light_square_color = st.sidebar.color_picker("Light square", "#F0D9B5")
dark_square_color  = st.sidebar.color_picker("Dark square",  "#B58863")

st.sidebar.subheader("Army colors")
default_colors = ['#FF4136', '#0074D9', '#2ECC40', '#FFDC00', '#B10DC9', '#FF851B', '#7FDBFF', '#F012BE', '#2E8B57', '#8B4513', '#808000', '#4682B4']
army_colors = [st.sidebar.color_picker(f"Army {i+1}", default_colors[i % len(default_colors)], key=f"army_color_{i}") for i in range(k)]

show_regions = st.sidebar.checkbox("Outline regions", value=True)

# Main layout(s)
layout = build_layout(n, k, style, intra)

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Results")
    st.metric("Size of each army", layout.per_army)
    st.metric("Total kings", layout.total)

    valid, conflicts = validate_cross_army_peace({i: coords for i, coords in layout.placements.items()})
    if valid:
        st.success("Cross‑army peace verified: no kings of different armies attack each other.")
    else:
        st.error(f"Cross‑army conflicts detected: {len(conflicts)} pairs.")
        if st.checkbox("Show first 20 conflicts"):
            dfc = pd.DataFrame(conflicts[:20], columns=["Army A", "Pos A (r,c)", "Army B", "Pos B (r,c)"])
            st.dataframe(dfc, use_container_width=True)

    st.write("\n")
    st.subheader("Strategy")
    st.info(layout.strategy_text)

    # Simple upper bound from area (no piece constraints):
    ub = (n * n) // k
    st.caption(f"Simple area bound: per‑army soldiers ≤ ⌊n² / k⌋ = {ub}. (Kings need moats for cross‑army peace; lattices control intra‑army peace.)")

with col2:
    if compare:
        st.subheader("Board visualization — Comparison (2×2 vs 3×3)")
        cA, cB = st.columns(2)
        layout_A = build_layout(n, k, style, "Non‑attacking: 2×2 lattice")
        layout_B = build_layout(n, k, style, "Non‑attacking: 3×3 lattice")

        with cA:
            st.caption(f"2×2 lattice — per‑army: {layout_A.per_army}, total: {layout_A.total}")
            figA = draw_board(n, layout_A.placements, army_colors, light_square_color, dark_square_color, show_regions, layout_A.regions)
            st.pyplot(figA)
        with cB:
            st.caption(f"3×3 lattice — per‑army: {layout_B.per_army}, total: {layout_B.total}")
            figB = draw_board(n, layout_B.placements, army_colors, light_square_color, dark_square_color, show_regions, layout_B.regions)
            st.pyplot(figB)

        # Export both PNGs
        pngA = io.BytesIO(); figA.savefig(pngA, format='png', bbox_inches='tight', dpi=200)
        pngB = io.BytesIO(); figB.savefig(pngB, format='png', bbox_inches='tight', dpi=200)
        st.download_button("Download PNG — 2×2", data=pngA.getvalue(), file_name=f"board_n{n}_k{k}_lattice2x2.png", mime="image/png")
        st.download_button("Download PNG — 3×3", data=pngB.getvalue(), file_name=f"board_n{n}_k{k}_lattice3x3.png", mime="image/png")
    else:
        st.subheader("Board visualization")
        fig = draw_board(n, layout.placements, army_colors, light_square_color, dark_square_color, show_regions, layout.regions)
        st.pyplot(fig)
        # Export single PNG
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format='png', bbox_inches='tight', dpi=200)
        st.download_button("Download board as PNG", data=png_buf.getvalue(), file_name=f"peaceful_kings_n{n}_k{k}.png", mime="image/png")


# =================================
# Summary moved into an expander
# =================================

with st.expander("Outcome Summary with Density Considerations", expanded=False):
    if intra == "Every square (same‑army attacks allowed)":
        intra_explain = (
            "Inside each region we place a king on every square. This maximizes the per‑army count, but kings of the same "
            "army do attack each other. Cross‑army peace is still guaranteed by the moats between regions."
        )
    elif intra == "Non‑attacking: 2×2 lattice":
        intra_explain = (
            "Inside each region we use a **2×2 lattice** (place a king every second row and column). This is the **densest** "
            "non‑attacking arrangement for kings within an army, yielding \(\lceil r/2\rceil\cdot\lceil c/2\rceil\) per region of size \(r\times c\)."
        )
    else:
        intra_explain = (
            "Inside each region we use a **3×3 lattice** (every third row and column). This is a **sparser** non‑attacking pattern; "
            "counts are \(\lceil r/3\rceil\cdot\lceil c/3\rceil\). It’s useful for visual comparison and pedagogy, but not density‑optimal."
        )

    summary_md = f"""
    ### Setup
    - Board: {n}×{n}  
    - Armies: k = {k}  
    - Partition style: **{style}** with 1‑cell moats between regions  
    - Intra‑army mode: **{intra}**  

    ### Results
    - Size of each army: **{layout.per_army}**  
    - Total kings: **{layout.total}**  
    - Cross‑army peace: {'**verified**' if validate_cross_army_peace(layout.placements)[0] else '**violations detected**'}  

    ### Why lattices matter (for curious readers)
    - A **lattice** is a regular grid pattern with spacing `step = s`. Placing kings every `s` rows/columns guarantees that any two same‑army kings are at Chebyshev distance ≥ s. For kings, **s = 2** is the minimal spacing that forbids adjacency, making it the density‑optimal non‑attacking pattern inside a region.  
    - Densities (asymptotic): 2×2 ≈ **1/4** of squares; 3×3 ≈ **1/9**. This explains the visual and numerical gap you’ll see in the comparison view.  
    - Inter‑army peace is achieved at the **region level** by moats; intra‑army peace is achieved by **lattice spacing**.

    ### Limits & interesting angles
    - **Density limit:** For kings, you cannot asymptotically exceed **1/4** of the squares without creating adjacency — the 2×2 lattice is essentially optimal for large boards.  
    - **Trade‑offs:** Larger lattice steps improve visual separation but reduce army size; smaller steps increase density but risk same‑army attacks.  
    - **Finite‑board effects:** Exact counts are impacted by leftover rows/columns, hence the ceiling terms in the formulas.  
    - **Generalizations:** Different pieces lead to different optimal spacings and tilings; comparing them opens doors to **discrete geometry**, **sphere packing on grids**, and **extremal combinatorics** topics.  

    ### Notes
    {intra_explain}

    A simple area upper bound (ignoring king attack geometry) gives per‑army ≤ ⌊n²/k⌋. Our construction respects king attack
    rules by subtracting moat lines between regions and, if selected, by enforcing lattice spacing inside each region.
    """

    st.markdown(summary_md)
