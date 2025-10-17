# Peaceful Armies of Kings — Streamlit App (Lattice Removed)
# Focus on moat-based partitioning strategies and modular arithmetic analysis
# Run: streamlit run new_ver_king_chessboard_app.py

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

def vertical_strips(n: int, c: int) -> List[Region]:
    """Divide board into c vertical strips separated by 1-column moats."""
    if c <= 0:
        return []
    moats = c - 1
    usable = n - moats
    if usable <= 0:
        return []
    base_w, extra = divmod(usable, c)
    widths = [base_w + (1 if i < extra else 0) for i in range(c)]

    regions: List[Region] = []
    col = 0
    for w in widths:
        c0 = col
        c1 = c0 + w - 1
        regions.append((0, n - 1, c0, c1))
        col = c1 + 2  # skip moat column
    return regions


def horizontal_strips(n: int, c: int) -> List[Region]:
    """Divide board into c horizontal strips separated by 1-row moats."""
    moats = c - 1
    usable = n - moats
    if usable <= 0:
        return []
    base_h, extra = divmod(usable, c)
    heights = [base_h + (1 if i < extra else 0) for i in range(c)]

    regions: List[Region] = []
    row = 0
    for h in heights:
        r0 = row
        r1 = r0 + h - 1
        regions.append((r0, r1, 0, n - 1))
        row = r1 + 2  # skip moat row
    return regions


def radial_partitions(n: int, c: int) -> List[Region]:
    """
    Radial/angular partitioning from center.
    Divide board into c sectors with approximately equal angles from center.
    
    Algorithm:
    1. Calculate angle for each cell from board center
    2. Assign cells to sectors based on angle ranges (360°/c per sector)
    3. Identify boundary cells between sectors and mark as moats
    4. Build bounding box regions for each sector
    
    For c=3: three ~120° sectors
    For c=4: four ~90° sectors
    etc.
    """
    if c <= 0 or n <= 0:
        return []
    
    import numpy as np
    
    # Center point of the board (using continuous coordinates)
    center_r = (n - 1) / 2.0
    center_c = (n - 1) / 2.0
    
    # Angle per sector
    angle_per_sector = 360.0 / c
    
    # Assign each cell to a sector based on angle from center
    sector_assignment = {}  # (r, c) -> sector_id
    sector_cells = {i: [] for i in range(c)}  # sector_id -> list of cells
    
    for r in range(n):
        for col in range(n):
            # Calculate angle from center to this cell
            dr = r - center_r
            dc = col - center_c
            
            # Use atan2 to get angle in radians, then convert to degrees
            # atan2 returns [-π, π], so we convert to [0, 360)
            angle_rad = np.arctan2(dr, dc)
            angle_deg = np.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 360
            
            # Determine which sector this belongs to
            # We rotate by half a sector to make the divisions more symmetric
            adjusted_angle = (angle_deg + angle_per_sector / 2) % 360
            sector_id = int(adjusted_angle / angle_per_sector)
            
            sector_assignment[(r, col)] = sector_id
            sector_cells[sector_id].append((r, col))
    
    # Identify moat cells: cells adjacent to cells from different sectors
    moat_cells = set()
    
    for r in range(n):
        for col in range(n):
            current_sector = sector_assignment[(r, col)]
            
            # Check all 8 neighbors (including diagonals for king movement)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    nr, nc = r + dr, col + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        neighbor_sector = sector_assignment[(nr, nc)]
                        if neighbor_sector != current_sector:
                            # This cell is on a boundary - mark as moat
                            moat_cells.add((r, col))
                            break
                if (r, col) in moat_cells:
                    break
    
    # Remove moat cells from sector assignments
    for moat_cell in moat_cells:
        sector_id = sector_assignment[moat_cell]
        if moat_cell in sector_cells[sector_id]:
            sector_cells[sector_id].remove(moat_cell)
    
    # Build bounding box regions for each sector
    regions = []
    for sector_id in range(c):
        cells = sector_cells[sector_id]
        if not cells:
            continue
        
        # Find bounding box
        rows = [cell[0] for cell in cells]
        cols = [cell[1] for cell in cells]
        
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        
        # Note: This creates a rectangular region, but only the cells
        # actually in sector_cells[sector_id] should be filled
        # We'll need to filter during filling
        regions.append((r0, r1, c0, c1))
    
    # Store sector cells for later use in filling
    # We'll use a workaround: return regions but also store metadata
    return regions, sector_cells


def fill_radial_sector(region: Region, sector_id: int, sector_cells_map: dict) -> List[Coord]:
    """
    Fill a radial sector region with only the cells that actually belong to this sector.
    """
    r0, r1, c0, c1 = region
    valid_cells = sector_cells_map.get(sector_id, [])
    
    # Filter to only cells within this region's bounding box that are in the sector
    result = []
    for (r, c) in valid_cells:
        if r0 <= r <= r1 and c0 <= c <= c1:
            result.append((r, c))
    
    return result


def grid_regions(n: int, c: int, rows: int = None, cols: int = None) -> List[Region]:
    """Divide board into a grid of regions with moats between blocks."""
    if rows is None or cols is None:
        cols = math.ceil(math.sqrt(c)) if cols is None else cols
        rows = math.ceil(c / cols) if rows is None else rows
    if rows * cols < c:
        raise ValueError("rows*cols must be ≥ c")

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
        col = 0
        for j in range(cols):
            c0 = col
            c1 = c0 + widths[j] - 1
            regions.append((r0, r1, c0, c1))
            col = c1 + 2  # moat col
        r = r1 + 2  # moat row
    return regions[:c]


# ==============================
# Population strategies (fill every square)
# ==============================

def fill_every_square(region: Region) -> List[Coord]:
    """Place a piece on every square in the region."""
    r0, r1, c0, c1 = region
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


# ==============================
# Modular arithmetic analysis
# ==============================

def analyze_modular_efficiency(n: int, c: int, regions: List[Region]) -> Dict:
    """
    Analyze efficiency based on n mod c.
    Professor's conjecture: vertical moats are tight when n ≡ (c-1) (mod c).
    """
    n_mod_c = n % c
    total_area = n * n
    moat_cols = c - 1
    usable_width = n - moat_cols
    
    # Calculate wasted space
    ideal_per_army = total_area / c
    actual_per_army = sum((r1-r0+1)*(c1-c0+1) for r0,r1,c0,c1 in regions) / len(regions) if regions else 0
    
    efficiency = actual_per_army / ideal_per_army if ideal_per_army > 0 else 0
    
    is_tight = (n_mod_c == c - 1)
    
    analysis = {
        'n_mod_c': n_mod_c,
        'is_tight_config': is_tight,
        'efficiency': efficiency,
        'wasted_space': total_area - sum((r1-r0+1)*(c1-c0+1) for r0,r1,c0,c1 in regions),
        'expected_tight_at': f"n ≡ {c-1} (mod {c})",
        'current_congruence': f"n ≡ {n_mod_c} (mod {c})"
    }
    
    return analysis


@dataclass
class Layout:
    regions: List[Region]
    placements: Dict[int, List[Coord]]
    per_army: int
    total: int
    strategy_text: str
    modular_analysis: Dict


def build_layout(n: int, c: int, style: str) -> Layout:
    """Build layout with specified partitioning style."""
    if c <= 0 or n <= 0:
        return Layout([], {}, 0, 0, "Invalid input.", {})

    sector_cells_map = None  # For radial partitioning
    
    if style == "Vertical strips":
        regions = vertical_strips(n, c)
        style_note = f"{c} vertical regions with one‑column moats."
    elif style == "Horizontal strips":
        regions = horizontal_strips(n, c)
        style_note = f"{c} horizontal regions with one‑row moats."
    elif style == "Radial (experimental)":
        radial_result = radial_partitions(n, c)
        if isinstance(radial_result, tuple):
            regions, sector_cells_map = radial_result
            style_note = f"Radial partitioning into {c} sectors (~{360/c:.0f}° each)."
        else:
            regions = radial_result
            style_note = f"Radial partitioning into {c} sectors (experimental)."
    else:
        regions = grid_regions(n, c)
        style_note = f"Grid of near‑square regions with moats between blocks."

    if not regions or len(regions) < c:
        return Layout(regions, {}, 0, 0, f"Board too small to host {c} regions with required moats.", {})

    # Fill regions - handle radial specially
    placements: Dict[int, List[Coord]] = {}
    
    if sector_cells_map is not None:
        # Radial partitioning: use sector cells map
        for i, reg in enumerate(regions):
            coords = fill_radial_sector(reg, i, sector_cells_map)
            placements[i] = coords
    else:
        # Standard partitioning: fill every square in bounding box
        for i, reg in enumerate(regions):
            coords = fill_every_square(reg)
            placements[i] = coords

    per_army = len(next(iter(placements.values()))) if placements else 0
    total = per_army * c

    # Perform modular analysis
    mod_analysis = analyze_modular_efficiency(n, c, regions)

    strategy = (
        f"Arrangement: {style_note} "
        f"Regions separated by 1-cell moats to prevent cross-army attacks. "
        f"Each region is filled completely (every square occupied)."
    )

    return Layout(regions, placements, per_army, total, strategy, mod_analysis)


# ==============================
# Visualization
# ==============================

def draw_board(n: int, armies: Dict[int, List[Coord]], colors: List[str], 
               light_color: str, dark_color: str,
               show_regions: bool, regions: List[Region], show_moats: bool = True):
    """Draw the chessboard with armies and optional region outlines."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([chr(65 + i) for i in range(n)])
    ax.set_yticklabels(range(1, n + 1))
    ax.tick_params(length=0)
    ax.grid(True, color='black', linewidth=1.0)
    ax.set_aspect('equal')

    # Draw squares
    for r in range(n):
        for c in range(n):
            color = light_color if (r + c) % 2 == 0 else dark_color
            rect = patches.Rectangle((c, r), 1, 1, facecolor=color)
            ax.add_patch(rect)

    # Highlight moat areas if requested
    if show_moats:
        occupied = set()
        for r0, r1, c0, c1 in regions:
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    occupied.add((r, c))
        
        # Mark moat cells
        for r in range(n):
            for c in range(n):
                if (r, c) not in occupied:
                    rect = patches.Rectangle((c, r), 1, 1, facecolor='gray', alpha=0.5)
                    ax.add_patch(rect)

    # Draw region outlines
    if show_regions:
        for i, (r0, r1, c0, c1) in enumerate(regions):
            rect = patches.Rectangle((c0, r0), c1 - c0 + 1, r1 - r0 + 1,
                                     fill=False, linewidth=2.5, edgecolor='black', linestyle='--')
            ax.add_patch(rect)
            ax.text(c0 + (c1 - c0 + 1) / 2, r0 + (r1 - r0 + 1) / 2, f"Army {i+1}",
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Draw kings
    for i, coords in armies.items():
        color = colors[i % len(colors)]
        for (r, c) in coords:
            ax.text(c + 0.5, r + 0.5, '♔', fontsize=max(8, 40 / max(1, n) * 4),
                    ha='center', va='center', color=color,
                    bbox=dict(boxstyle='circle,pad=0.08', fc=color, alpha=0.35, ec='none'))

    ax.invert_yaxis()
    return fig


# ==============================
# Streamlit UI
# ==============================

st.set_page_config(layout="wide", page_title="Peaceful Armies of Kings")
st.title("Peaceful Armies of Kings — Moat-Based Partitioning Analysis")

with st.expander("Research Context & Professor's Insights", expanded=False):
    st.markdown("""
    ### Research Problem
    We partition an n×n chessboard into **c** armies (color classes) such that kings from different 
    armies cannot attack each other. This is achieved through **moat-based partitioning**: regions 
    are separated by 1-cell wide moats, ensuring Chebyshev distance ≥ 2 between different armies.
    
    ### Professor's Key Observations (c=3 case)
    
    #### Modular Arithmetic Conjecture
    For c=3 with vertical moats, the configuration is expected to be:
    - **Tight (optimal)** when n ≡ 2 (mod 3) — i.e., n = 3j+2
    - **Wasteful** when n ≡ 0 or 1 (mod 3) — i.e., n = 3j or n = 3j+1
    
    #### Alternative Strategy: Radial Partitioning
    Instead of vertical moats, consider slicing the board into c regions from the center at 
    approximately equal angles:
    - For c=3: three ~120° sectors
    - For c=4: four ~90° sectors
    - etc.
    
    The challenge: irregular boundaries on a discrete grid. Question: Can this reduce wasted space 
    for cases where n ≢ (c-1) (mod c)?
    
    #### Generalization Pattern
    - c=3: tight at n ≡ 2 (mod 3)
    - c=4: tight at n ≡ 3 (mod 4)
    - c=5: tight at n ≡ 4 (mod 5)
    - **General**: For c armies, vertical moats are tight when n ≡ (c-1) (mod c)
    
    The next case following this pattern would be c=6, tight at n ≡ 5 (mod 6).
    """)

st.sidebar.header("Configuration")
n = st.sidebar.slider("Board size (n)", 3, 50, 12, help="Size of n×n chessboard")
c = st.sidebar.slider("Number of armies (c)", 2, 12, 3, help="Number of peaceful armies")

st.sidebar.subheader("Partitioning Strategy")
style = st.sidebar.selectbox("Choose partition pattern", [
    "Vertical strips",
    "Horizontal strips", 
    "Radial (experimental)",
    "Grid (auto)"
])

st.sidebar.subheader("Visualization Options")
show_regions = st.sidebar.checkbox("Show region outlines", value=True)
show_moats = st.sidebar.checkbox("Highlight moat areas", value=True)

st.sidebar.subheader("Board & Army Colors")
light_square_color = st.sidebar.color_picker("Light square", "#F0D9B5")
dark_square_color  = st.sidebar.color_picker("Dark square",  "#B58863")

default_colors = ['#FF4136', '#0074D9', '#2ECC40', '#FFDC00', '#B10DC9', 
                  '#FF851B', '#7FDBFF', '#F012BE', '#2E8B57', '#8B4513', 
                  '#808000', '#4682B4']
army_colors = [st.sidebar.color_picker(f"Army {i+1}", default_colors[i % len(default_colors)], 
                                       key=f"army_color_{i}") for i in range(c)]

# Build layout
layout = build_layout(n, c, style)

# Main display
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Results")
    st.metric("Soldiers per army", layout.per_army)
    st.metric("Total soldiers", layout.total)
    
    # Modular analysis display
    st.subheader("Modular Arithmetic Analysis")
    mod = layout.modular_analysis
    
    if mod:
        st.write(f"**Board size:** n = {n}")
        st.write(f"**Number of armies:** c = {c}")
        st.write(f"**Current congruence:** {mod['current_congruence']}")
        st.write(f"**Expected tight configuration:** {mod['expected_tight_at']}")
        
        if mod['is_tight_config']:
            st.success("✓ This is expected to be a TIGHT configuration!")
        else:
            st.warning(f"⚠ Potential wasted space (n ≢ {c-1} mod {c})")
        
        st.metric("Efficiency", f"{mod['efficiency']:.2%}")
        st.metric("Wasted squares", mod['wasted_space'])

    # Validation
    st.subheader("Cross-Army Peace Verification")
    valid, conflicts = validate_cross_army_peace({i: coords for i, coords in layout.placements.items()})
    if valid:
        st.success("✓ Verified: No cross-army attacks")
    else:
        st.error(f"✗ {len(conflicts)} cross-army conflicts detected")
        if st.checkbox("Show conflicts"):
            dfc = pd.DataFrame(conflicts[:20], columns=["Army A", "Pos A", "Army B", "Pos B"])
            st.dataframe(dfc, use_container_width=True)
    
    st.subheader("Strategy")
    st.info(layout.strategy_text)
    
    # Theoretical bound
    area_bound = (n * n) // c
    st.caption(f"Simple area bound: ⌊n²/c⌋ = {area_bound} soldiers per army (ignoring moat constraints)")

with col2:
    st.subheader("Board Visualization")
    
    # Add comparison mode toggle
    compare_mode = st.checkbox("Compare Two Strategies Side-by-Side", value=False, key="compare_strategies")
    
    if compare_mode:
        st.write("**Strategy Comparison Mode**")
        comp_col1, comp_col2 = st.columns(2)
        
        strategy1 = st.selectbox("Strategy 1", ["Vertical strips", "Horizontal strips", "Grid (auto)", "Radial (experimental)"], index=0, key="strat1")
        strategy2 = st.selectbox("Strategy 2", ["Vertical strips", "Horizontal strips", "Grid (auto)", "Radial (experimental)"], index=1, key="strat2")
        
        layout1 = build_layout(n, c, strategy1)
        layout2 = build_layout(n, c, strategy2)
        
        with comp_col1:
            st.caption(f"**{strategy1}**")
            st.write(f"Soldiers per army: **{layout1.per_army}**")
            st.write(f"Efficiency: **{layout1.modular_analysis.get('efficiency', 0):.1%}**")
            fig1 = draw_board(n, layout1.placements, army_colors, light_square_color, 
                             dark_square_color, show_regions, layout1.regions, show_moats)
            st.pyplot(fig1)
            
        with comp_col2:
            st.caption(f"**{strategy2}**")
            st.write(f"Soldiers per army: **{layout2.per_army}**")
            st.write(f"Efficiency: **{layout2.modular_analysis.get('efficiency', 0):.1%}**")
            fig2 = draw_board(n, layout2.placements, army_colors, light_square_color, 
                             dark_square_color, show_regions, layout2.regions, show_moats)
            st.pyplot(fig2)
        
        # Comparison summary
        st.subheader("Comparison Summary")
        diff = layout1.per_army - layout2.per_army
        if diff > 0:
            st.success(f"✓ {strategy1} produces **{diff} more soldiers** per army than {strategy2}")
        elif diff < 0:
            st.success(f"✓ {strategy2} produces **{abs(diff)} more soldiers** per army than {strategy1}")
        else:
            st.info(f"Both strategies produce the same army size ({layout1.per_army} soldiers)")
        
        # Export comparison
        comparison_buffer = io.BytesIO()
        fig_combined, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Recreate plots for combined figure
        for ax, layout_data, strat_name in [(ax_left, layout1, strategy1), (ax_right, layout2, strategy2)]:
            ax.set_xlim(0, n)
            ax.set_ylim(0, n)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels([chr(65 + i) for i in range(n)])
            ax.set_yticklabels(range(1, n + 1))
            ax.tick_params(length=0)
            ax.grid(True, color='black', linewidth=1.0)
            ax.set_aspect('equal')
            ax.set_title(f"{strat_name}\n{layout_data.per_army} soldiers/army", fontsize=12, fontweight='bold')
            
            # Draw squares
            for r in range(n):
                for col in range(n):
                    color = light_square_color if (r + col) % 2 == 0 else dark_square_color
                    rect = patches.Rectangle((col, r), 1, 1, facecolor=color)
                    ax.add_patch(rect)
            
            # Draw kings
            for i, coords in layout_data.placements.items():
                color = army_colors[i % len(army_colors)]
                for (r, col) in coords:
                    ax.text(col + 0.5, r + 0.5, '♔', fontsize=max(8, 40 / max(1, n) * 4),
                            ha='center', va='center', color=color,
                            bbox=dict(boxstyle='circle,pad=0.08', fc=color, alpha=0.35, ec='none'))
            
            ax.invert_yaxis()
        
        plt.tight_layout()
        fig_combined.savefig(comparison_buffer, format='png', bbox_inches='tight', dpi=200)
        
        st.download_button(
            "📥 Download Comparison (PNG)",
            data=comparison_buffer.getvalue(),
            file_name=f"strategy_comparison_n{n}_c{c}.png",
            mime="image/png"
        )
        
    else:
        # Single strategy view
        fig = draw_board(n, layout.placements, army_colors, light_square_color, 
                         dark_square_color, show_regions, layout.regions, show_moats)
        st.pyplot(fig)
        
        # Export
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format='png', bbox_inches='tight', dpi=200)
        st.download_button("Download PNG", data=png_buf.getvalue(), 
                          file_name=f"peaceful_armies_n{n}_c{c}.png", mime="image/png")

# Comparative analysis section
st.header("Comparative Analysis Tool")

with st.expander("Compare Multiple Board Sizes", expanded=False):
    st.write("Analyze how efficiency changes with different board sizes for fixed c")
    
    c_fixed = st.number_input("Fix number of armies (c)", min_value=2, max_value=12, value=3)
    n_start = st.number_input("Starting board size", min_value=3, max_value=50, value=5)
    n_end = st.number_input("Ending board size", min_value=3, max_value=50, value=20)
    
    if st.button("Run Analysis"):
        results = []
        for n_test in range(n_start, n_end + 1):
            layout_test = build_layout(n_test, c_fixed, "Vertical strips")
            mod = layout_test.modular_analysis
            results.append({
                'n': n_test,
                'n mod c': mod['n_mod_c'],
                'Is Tight?': '✓' if mod['is_tight_config'] else '✗',
                'Per Army': layout_test.per_army,
                'Efficiency': f"{mod['efficiency']:.2%}",
                'Wasted': mod['wasted_space']
            })
        
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        
        # Highlight pattern
        st.write(f"**Pattern observation:** Tight configurations occur at n ≡ {c_fixed-1} (mod {c_fixed})")
        tight_values = [r['n'] for r in results if r['Is Tight?'] == '✓']
        st.write(f"Tight at n = {tight_values}")

# Efficiency Plot Section
st.header("Modular Arithmetic Efficiency Analysis")

with st.expander("📊 Efficiency vs Board Size Plot", expanded=True):
    st.write("Visualize how efficiency changes with board size and modular arithmetic patterns")
    
    plot_col1, plot_col2 = st.columns([1, 2])
    
    with plot_col1:
        st.subheader("Plot Settings")
        c_plot = st.slider("Number of armies (c) for plot", min_value=2, max_value=12, value=3, key="c_plot")
        n_min_plot = st.slider("Minimum board size", min_value=3, max_value=30, value=5, key="n_min")
        n_max_plot = st.slider("Maximum board size", min_value=5, max_value=50, value=30, key="n_max")
        
        if n_min_plot >= n_max_plot:
            st.error("Maximum must be greater than minimum!")
        else:
            show_per_army_plot = st.checkbox("Also plot soldiers per army", value=False, key="show_per_army")
            
            if st.button("Generate Efficiency Plot", type="primary"):
                # Collect data
                plot_data = []
                for n_test in range(n_min_plot, n_max_plot + 1):
                    layout_test = build_layout(n_test, c_plot, "Vertical strips")
                    mod = layout_test.modular_analysis
                    plot_data.append({
                        'n': n_test,
                        'efficiency': mod['efficiency'],
                        'n_mod_c': mod['n_mod_c'],
                        'is_tight': mod['is_tight_config'],
                        'per_army': layout_test.per_army,
                        'wasted': mod['wasted_space']
                    })
                
                # Create the plot
                if show_per_army_plot:
                    fig_plot, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                else:
                    fig_plot, ax1 = plt.subplots(1, 1, figsize=(10, 5))
                
                # Group by n mod c for color coding
                colors_map = plt.cm.Set3(range(c_plot))
                
                for mod_class in range(c_plot):
                    subset = [d for d in plot_data if d['n_mod_c'] == mod_class]
                    if subset:
                        n_vals = [d['n'] for d in subset]
                        eff_vals = [d['efficiency'] * 100 for d in subset]  # Convert to percentage
                        
                        marker = 'o' if mod_class == c_plot - 1 else 's'
                        label = f"n ≡ {mod_class} (mod {c_plot})" + (" [TIGHT]" if mod_class == c_plot - 1 else "")
                        
                        ax1.scatter(n_vals, eff_vals, c=[colors_map[mod_class]], 
                                   marker=marker, s=80, alpha=0.7, label=label, edgecolors='black', linewidth=1)
                        ax1.plot(n_vals, eff_vals, c=colors_map[mod_class], alpha=0.3, linewidth=1)
                
                ax1.set_xlabel('Board Size (n)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Efficiency (%)', fontsize=12, fontweight='bold')
                ax1.set_title(f'Efficiency vs Board Size for c={c_plot} Armies (Vertical Strips)', 
                             fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3, linestyle='--')
                ax1.legend(loc='best', fontsize=10)
                ax1.set_ylim([0, 105])
                
                # Add horizontal line at 100%
                ax1.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Perfect (100%)')
                
                # Second plot: soldiers per army (if requested)
                if show_per_army_plot:
                    for mod_class in range(c_plot):
                        subset = [d for d in plot_data if d['n_mod_c'] == mod_class]
                        if subset:
                            n_vals = [d['n'] for d in subset]
                            per_army_vals = [d['per_army'] for d in subset]
                            
                            marker = 'o' if mod_class == c_plot - 1 else 's'
                            
                            ax2.scatter(n_vals, per_army_vals, c=[colors_map[mod_class]], 
                                       marker=marker, s=80, alpha=0.7, edgecolors='black', linewidth=1)
                            ax2.plot(n_vals, per_army_vals, c=colors_map[mod_class], alpha=0.3, linewidth=1)
                    
                    ax2.set_xlabel('Board Size (n)', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Soldiers per Army', fontsize=12, fontweight='bold')
                    ax2.set_title(f'Army Size vs Board Size for c={c_plot} Armies', 
                                 fontsize=14, fontweight='bold')
                    ax2.grid(True, alpha=0.3, linestyle='--')
                
                plt.tight_layout()
                
                st.session_state['efficiency_plot'] = fig_plot
                st.session_state['plot_data'] = plot_data
    
    with plot_col2:
        if 'efficiency_plot' in st.session_state:
            st.pyplot(st.session_state['efficiency_plot'])
            
            # Download button
            plot_buffer = io.BytesIO()
            st.session_state['efficiency_plot'].savefig(plot_buffer, format='png', bbox_inches='tight', dpi=300)
            st.download_button(
                "📥 Download Plot (PNG)",
                data=plot_buffer.getvalue(),
                file_name=f"efficiency_plot_c{c_plot}_n{n_min_plot}-{n_max_plot}.png",
                mime="image/png"
            )
            
            # Summary statistics
            if 'plot_data' in st.session_state:
                st.subheader("Summary Statistics")
                
                data = st.session_state['plot_data']
                tight_configs = [d for d in data if d['is_tight']]
                non_tight_configs = [d for d in data if not d['is_tight']]
                
                col_stat1, col_stat2 = st.columns(2)
                
                with col_stat1:
                    st.metric("Tight configs", len(tight_configs))
                    if tight_configs:
                        avg_eff_tight = sum(d['efficiency'] for d in tight_configs) / len(tight_configs) * 100
                        st.metric("Avg efficiency (tight)", f"{avg_eff_tight:.1f}%")
                
                with col_stat2:
                    st.metric("Non-tight configs", len(non_tight_configs))
                    if non_tight_configs:
                        avg_eff_non_tight = sum(d['efficiency'] for d in non_tight_configs) / len(non_tight_configs) * 100
                        st.metric("Avg efficiency (non-tight)", f"{avg_eff_non_tight:.1f}%")
                
                # Key insight
                st.info(f"**Key Insight:** Configurations where n ≡ {c_plot-1} (mod {c_plot}) show {'higher' if tight_configs and non_tight_configs and avg_eff_tight > avg_eff_non_tight else 'different'} efficiency patterns.")
        else:
            st.info("Click 'Generate Efficiency Plot' to visualize the modular arithmetic patterns")

# Research notes section
with st.expander("Research Notes & Next Steps", expanded=False):
    st.markdown("""
    ### ✅ Implemented Features
    
    1. **Efficiency Plots** — Visualize how efficiency changes with board size, color-coded by n mod c
    2. **Strategy Comparison** — Side-by-side visualization of different partitioning strategies
    3. **Modular Analysis Dashboard** — Real-time calculation of tightness based on n ≡ (c-1) (mod c)
    4. **Radial Partitioning Algorithm** — Angle-based sector division from board center with automatic moat detection
    
    ### 🔬 How Radial Partitioning Works
    
    The algorithm divides the board into c sectors using angular geometry:
    
    1. **Center Point**: Uses (n-1)/2 as the continuous center of the board
    2. **Angle Assignment**: Each cell is assigned to a sector based on its angle from center (using atan2)
    3. **Sector Ranges**: 360°/c per sector (e.g., 120° for c=3, 90° for c=4)
    4. **Moat Detection**: Cells adjacent to cells from different sectors are marked as moats
    5. **Region Building**: Each sector's cells are grouped into regions
    
    **Key Characteristics:**
    - Natural rotational symmetry around the center
    - Irregular boundaries due to discrete grid (some sectors may have slightly different sizes)
    - Moats automatically form between sectors, ensuring cross-army peace
    - Works for any c ≥ 2
    
    ### 📋 Open Research Questions
    
    1. **Radial vs Vertical Efficiency**: Now that both are implemented, systematically compare them:
       - Does radial improve efficiency for n ≢ (c-1) (mod c)?
       - Or do irregular boundaries negate potential gains?
       - Use the comparison tool to test multiple board sizes
    
    2. **Asymmetry in Radial Partitioning**: Due to the discrete grid:
       - Some sectors may be slightly larger/smaller
       - Edge effects near boundaries
       - Does this matter practically?
    
    3. **Optimal Angle Offset**: Currently using angle_per_sector/2 offset
       - Does rotating the sector boundaries improve efficiency?
       - Is there an optimal alignment with the grid?
    
    4. **Numerical Verification of Tightness**: The efficiency plots should show:
       - Vertical strips: tight at n ≡ (c-1) (mod c)
       - Radial: different pattern? Or similar?
    
    5. **Theoretical Analysis**: 
       - Can we prove bounds for radial partitioning?
       - What's the expected wasted space due to irregular boundaries?
    
    6. **Hybrid Approaches**: Could we combine strategies?
       - Radial near center, vertical/horizontal near edges?
       - Adaptive strategy based on board size and c?
    
    ### 🎯 Recommended Experiments
    
    1. **Systematic Comparison** (use the comparison tool):
       - For c=3: test n=10,11,12,...,25
       - Compare radial vs vertical for each n
       - Record which strategy wins for each n mod 3 case
    
    2. **Efficiency Analysis**:
       - Generate efficiency plots for both radial and vertical
       - Compare patterns - do they follow similar modular arithmetic?
    
    3. **Visual Inspection**:
       - For small boards (n=8-15), visually compare sector shapes
       - Identify where radial "wastes" space vs where it gains
    
    4. **Scale Testing**:
       - Test on large boards (n=40-50)
       - Do patterns converge as n increases?
    
    5. **Different c values**:
       - c=3 (120° sectors)
       - c=4 (90° sectors)
       - c=6 (60° sectors)
       - c=8 (45° sectors)
       - Which c values favor radial vs vertical?
    
    ### 📊 Data Collection Template
    
    For your paper/thesis, collect:
    - Table: n, c, vertical_army_size, radial_army_size, difference, winner
    - Efficiency plots for c=3,4,5,6
    - Representative board visualizations showing sector shapes
    - Statistical analysis: average improvement (if any) by modular class
    """)