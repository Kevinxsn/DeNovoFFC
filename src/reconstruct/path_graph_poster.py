import os
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


_PALETTE = {
    # Background & structure
    "bg":             "#FFFFFF",
    "grid_clean":     "#E8EAF0",
    "grid_spurious":  "#F5DADA",
    "tick_clean":     "#4A5068",
    "tick_spurious":  "#C03030",

    # Nodes
    "node_clean":     "#B0BED8",
    "node_spurious":  "#E8BFBF",

    # Correct path (hero)
    "hero_line":      "#00946B",
    "hero_glow":      "#00C88E",
    "hero_marker":    "#FFFFFF",
    "hero_box_bg":    "#E6F7F1",
    "hero_box_ec":    "#00946B",
    "hero_text":      "#006648",

    # Candidate paths
    "cand_line":      "#6C5FD0",
    "cand_marker":    "#FFFFFF",
    "cand_box_bg":    "#EEEAFC",
    "cand_box_ec":    "#6C5FD0",
    "cand_text":      "#4A3FA0",

    # Start / End markers
    "start_fill":     "#E83050",
    "start_edge":     "#B02040",
    "end_fill":       "#F0A010",
    "end_edge":       "#C08000",

    # Residual box
    "res_box_bg":     "#FFF8E8",
    "res_box_ec":     "#E09000",
    "res_text":       "#8B5E00",

    # Accents
    "title":          "#1A1E2E",
    "label":          "#5A6078",
    "legend_bg":      "#F6F7FA",
    "legend_ec":      "#D0D4E0",
    "legend_text":    "#3A4060",
    "spine":          "#C8CCD8",
}


def _stagger_tick_labels(ax, axis='x', min_gap_pt=10):
    """
    Detect overlapping tick labels, hide the originals, and replace them
    with annotated labels connected by thin leader lines back to the
    actual tick position on the axis.

    This completely eliminates overlap by placing shifted labels at
    guaranteed-clear positions with visible connection lines.

    min_gap_pt: minimum gap in points between label bounding boxes
                before staggering kicks in.
    """
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()

    if axis == 'x':
        tick_objs = ax.xaxis.get_major_ticks()
        labels = [t.label2 for t in tick_objs]  # label2 = top
    else:
        tick_objs = ax.yaxis.get_major_ticks()
        labels = [t.label1 for t in tick_objs]  # label1 = left

    tick_positions = [t.get_loc() for t in tick_objs]

    # Get bounding boxes in display coords
    bboxes = []
    for lbl in labels:
        if lbl.get_text():
            try:
                bb = lbl.get_window_extent(renderer)
                bboxes.append(bb)
            except Exception:
                bboxes.append(None)
        else:
            bboxes.append(None)

    n = len(bboxes)
    if n < 2:
        return

    # Find crowded clusters
    overlaps_prev = [False] * n
    for i in range(1, n):
        if bboxes[i] is None or bboxes[i - 1] is None:
            continue
        prev, curr = bboxes[i - 1], bboxes[i]
        if axis == 'x':
            gap = curr.x0 - prev.x1
        else:
            gap = prev.y0 - curr.y1
        if gap < min_gap_pt:
            overlaps_prev[i] = True

    # Assign tier levels within each crowded run
    tier = [0] * n
    in_run = False
    run_counter = 0
    for i in range(n):
        if overlaps_prev[i]:
            if not in_run:
                in_run = True
                run_counter = 1
            else:
                run_counter += 1
            tier[i] = run_counter
        else:
            in_run = False
            run_counter = 0
            tier[i] = 0

    # ── Leader line style ──
    leader_style = dict(
        arrowstyle='-',
        color='#9098B0',
        lw=0.8,
        shrinkA=0,
        shrinkB=1,
    )

    # ── Offset magnitudes per tier level ──
    # Direction is computed contextually below: each shifted label moves
    # AWAY from the cluster's anchor (tier-0) label, so the leader line
    # never crosses over another label.
    if axis == 'x':
        # (|dx|, dy) — horizontal magnitude + vertical lift per level
        tier_magnitudes = [
            (0, 0),
            (18, 18),      # level 1
            (30, 32),      # level 2
            (42, 46),      # level 3
            (54, 60),      # level 4
        ]
    else:
        # (dx, |dy|) — leftward push + vertical magnitude per level
        tier_magnitudes = [
            (0, 0),
            (-22, 10),     # level 1
            (-22, 20),     # level 2
            (-38, 10),     # level 3
            (-38, 20),     # level 4
        ]

    # Find the anchor (tier-0) index for each shifted label's cluster
    # Walk backwards from each shifted label to find the first non-shifted one
    cluster_anchor_idx = [None] * n
    for i in range(n):
        if tier[i] > 0:
            # Walk backward to find the tier-0 anchor
            j = i - 1
            while j >= 0 and tier[j] > 0:
                j -= 1
            cluster_anchor_idx[i] = max(j, 0)

    for i in range(n):
        t = tier[i]
        if t == 0 or bboxes[i] is None:
            continue

        level = ((t - 1) % (len(tier_magnitudes) - 1)) + 1
        mag_a, mag_b = tier_magnitudes[level]

        # Determine direction: shift AWAY from the cluster anchor
        anchor_i = cluster_anchor_idx[i]
        anchor_pos = tick_positions[anchor_i] if anchor_i is not None else tick_positions[0]

        if axis == 'x':
            # If this tick is to the right of the anchor, shift right; else left
            if tick_positions[i] >= anchor_pos:
                dx_pt = mag_a   # shift right
            else:
                dx_pt = -mag_a  # shift left
            dy_pt = mag_b       # always lift up
        else:
            dx_pt = mag_a       # always shift left (negative value)
            # If this tick is below the anchor (larger y in inverted axis),
            # shift down; else up
            if tick_positions[i] >= anchor_pos:
                dy_pt = -mag_b  # shift down (positive y = lower on inverted)
            else:
                dy_pt = mag_b   # shift up

        # Get label properties before hiding
        text = labels[i].get_text()
        color = labels[i].get_color()
        fontweight = labels[i].get_fontweight()
        fontsize = labels[i].get_fontsize() - min(0.8, level * 0.3)
        pos = tick_positions[i]

        # Hide the original label
        labels[i].set_visible(False)

        # Place annotation with leader line using blended transform:
        # The anchor point (xy) is at the tick position on the axis edge.
        # For x-axis top: (data_x, axes_y=1.0) — top edge
        # For y-axis left: (axes_x=0.0, data_y) — left edge
        if axis == 'x':
            xy_point = (pos, 1.0)
            xy_transform = ax.get_xaxis_transform()  # x=data, y=axes
            ha, va = 'center', 'bottom'
            rotation = 90
        else:
            xy_point = (0.0, pos)
            xy_transform = ax.get_yaxis_transform()  # x=axes, y=data
            ha, va = 'right', 'center'
            rotation = 0

        # Place the annotation with leader line
        ax.annotate(
            text,
            xy=xy_point,
            xycoords=xy_transform,
            xytext=(dx_pt, dy_pt),
            textcoords='offset points',
            fontsize=fontsize,
            fontfamily='monospace',
            fontweight=fontweight,
            color=color,
            ha=ha, va=va,
            rotation=rotation,
            arrowprops=leader_style,
            zorder=100,
            annotation_clip=False,
        )


def visualize_all_paths(spectrum, spurious_masses=None,
                        candidate_paths=None, correct_path=None,
                        tolerance=0.02, aa_map=None, pep_mass=None,
                        title="Peptide Spectrum Graph",
                        save_path=None, figsize=(18, 18), dpi=300,
                        short_edge_thresh=160):
    """
    Modern visualization of peptide spectrum graph paths.

    Light scientific theme with subtle glow effects for hero paths,
    refined typography, and smart tick-label de-overlapping.

    Args:
        spectrum: List of fragment masses.
        spurious_masses: Optional set/list of masses considered spurious.
        candidate_paths: List of candidate paths, each a list of (x1, x2) tuples.
        correct_path: The correct path as a list of (x1, x2) tuples.
        tolerance: Mass tolerance for snapping and AA matching.
        aa_map: Dict mapping {mass: aa_name} for edge annotation.
        pep_mass: Total peptide mass for residual calculation at endpoints.
        title: Plot title string.
        save_path: If provided, saves figure to this path instead of showing.
        figsize: Figure dimensions.
        dpi: Resolution for saved figures.
        short_edge_thresh: Edge length (data units) below which labels are
            placed to the side with a leader line instead of centered.
    """
    P = _PALETTE

    # ── 1. Setup & Classification ──────────────────────────────────────────
    S = sorted(list(set(spectrum)))
    S_arr = np.array(S)

    spurious_set = set(spurious_masses) if spurious_masses else set()
    is_spurious = lambda val: val in spurious_set

    # ── 2. Create Figure ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, facecolor=P["bg"])
    ax.set_facecolor(P["bg"])

    # ── 3. Grid ────────────────────────────────────────────────────────────
    for mass in S:
        if is_spurious(mass):
            # Spurious lines: bold, visible, warm red
            c, a, ls, lw = '#E06060', 0.45, (0, (6, 3)), 1.2
        else:
            c, a, ls, lw = P["grid_clean"], 0.6, (0, (1, 3)), 0.4
        ax.axvline(x=mass, color=c, linestyle=ls, linewidth=lw, alpha=a)
        ax.axhline(y=mass, color=c, linestyle=ls, linewidth=lw, alpha=a)

    # ── 4. Nodes ───────────────────────────────────────────────────────────
    X, Y = np.meshgrid(S, S)
    is_noise_mask = np.array([is_spurious(m) for m in S])
    X_noise, Y_noise = np.meshgrid(is_noise_mask, is_noise_mask)

    # Three categories:
    #   both_spurious: both x and y are spurious masses
    #   one_spurious:  exactly one of x or y is spurious (cross-point)
    #   clean:         neither is spurious
    both_spurious = X_noise & Y_noise
    one_spurious = (X_noise | Y_noise) & ~both_spurious
    clean = ~(X_noise | Y_noise)

    # Clean nodes: subtle, as before
    ax.scatter(X[clean], Y[clean],
               s=14, c=P["node_clean"], alpha=0.5, zorder=2,
               edgecolors='none')

    # One-spurious cross-points: medium visibility, red-tinted with edge
    ax.scatter(X[one_spurious], Y[one_spurious],
               s=28, c='#F0C0C0', alpha=0.7, zorder=3,
               edgecolors='#D07070', linewidth=0.5)

    # Both-spurious intersections: most visible, bold red markers
    ax.scatter(X[both_spurious], Y[both_spurious],
               s=40, c='#E88080', alpha=0.8, zorder=4,
               edgecolors='#C04040', linewidth=0.8, marker='D')

    # ── 5. Helpers ─────────────────────────────────────────────────────────
    def get_closest_in_spectrum(val):
        idx = (np.abs(S_arr - val)).argmin()
        closest = S_arr[idx]
        return closest if abs(closest - val) <= tolerance else val

    def get_snapped_path(raw_path):
        return [(get_closest_in_spectrum(px), get_closest_in_spectrum(py))
                for px, py in raw_path]

    def get_anno_text(mass_diff):
        abs_diff = abs(mass_diff)
        txt = f"{mass_diff:.2f}"
        if aa_map:
            min_diff = tolerance + 1e-9
            best_name = None
            for aa_mass, name in aa_map.items():
                diff = abs(abs_diff - aa_mass)
                if diff <= tolerance and diff < min_diff:
                    min_diff = diff
                    best_name = name
            if best_name:
                txt = f"{best_name}\n({mass_diff:.2f})"
        return txt

    # ── 6. Path Drawing ───────────────────────────────────────────────────
    def draw_path(path_data, is_hero=False):
        if not path_data or len(path_data) < 2:
            return

        snapped = get_snapped_path(path_data)

        if is_hero:
            line_c   = P["hero_line"]
            marker_c = P["hero_marker"]
            box_bg   = P["hero_box_bg"]
            box_ec   = P["hero_box_ec"]
            text_c   = P["hero_text"]
            lw, alpha, ls = 2.8, 0.95, '-'
            ms, fs = 7, 13
            z = 50
        else:
            line_c   = P["cand_line"]
            marker_c = P["cand_marker"]
            box_bg   = P["cand_box_bg"]
            box_ec   = P["cand_box_ec"]
            text_c   = P["cand_text"]
            lw, alpha, ls = 1.4, 0.65, (0, (5, 3))
            ms, fs = 5, 11
            z = 20

        # Subtle glow / shadow effect for hero path
        if is_hero:
            for glow_lw, glow_a in [(9, 0.05), (6, 0.08), (4, 0.12)]:
                for i in range(len(snapped) - 1):
                    p0, p1 = snapped[i], snapped[i + 1]
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                            color=P["hero_glow"], linewidth=glow_lw,
                            alpha=glow_a, solid_capstyle='round', zorder=z - 1)

        # Start node
        start = snapped[0]
        s_size = 220 if is_hero else 80
        ax.scatter([start[0]], [start[1]],
                   color=P["start_fill"], s=s_size,
                   edgecolors=P["start_edge"], linewidth=1.2,
                   zorder=z + 5, alpha=0.9)
        if is_hero:
            ax.scatter([start[0]], [start[1]],
                       color=P["start_fill"], s=s_size * 3,
                       edgecolors='none', alpha=0.06, zorder=z + 4)

        # Edges
        short_count = 0          # counter for alternating side offsets
        for i in range(len(snapped) - 1):
            p0 = snapped[i]
            p1 = snapped[i + 1]

            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color=line_c, linewidth=lw, alpha=alpha, linestyle=ls,
                    marker='o', markersize=ms,
                    markerfacecolor=marker_c, markeredgecolor=line_c,
                    markeredgewidth=1.2, solid_capstyle='round',
                    zorder=z)

            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            mass_jump = max(abs(dx), abs(dy))
            mid_x = (p0[0] + p1[0]) / 2
            mid_y = (p0[1] + p1[1]) / 2

            # Determine edge length in data units; short edges get
            # side-placed labels with leader lines to avoid overlap.
            edge_len = np.hypot(dx, dy)
            short_thresh = short_edge_thresh if is_hero else short_edge_thresh * 0.875

            if edge_len < short_thresh:
                # ── Side label with leader line ──
                # Alternate direction each consecutive short edge so
                # labels don't stack on top of one another.
                offset_mag = 55 if is_hero else 45
                side = 1 if (short_count % 2 == 0) else -1
                short_count += 1

                if abs(dx) > abs(dy):
                    # Horizontal edge → alternate above / below
                    offset_x_pt = 0
                    offset_y_pt = offset_mag * side
                    ha = 'center'
                    va = 'top' if side > 0 else 'bottom'
                else:
                    # Vertical / diagonal edge → alternate right / left
                    offset_x_pt = offset_mag * side
                    offset_y_pt = 0
                    ha = 'left' if side > 0 else 'right'
                    va = 'center'

                ax.annotate(
                    get_anno_text(mass_jump),
                    xy=(mid_x, mid_y),
                    xytext=(offset_x_pt, offset_y_pt),
                    textcoords='offset points',
                    ha=ha, va=va, fontsize=fs,
                    color=text_c, fontweight='semibold',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.4',
                              fc=box_bg, ec=box_ec, alpha=0.92, lw=1.2),
                    arrowprops=dict(
                        arrowstyle='-',
                        color=box_ec,
                        lw=1.0,
                        shrinkA=0,
                        shrinkB=3,
                    ),
                    zorder=z + 10,
                )
            else:
                short_count = 0   # reset alternation after a long edge
                # ── Centered label (original behaviour for long edges) ──
                ax.text(mid_x, mid_y, get_anno_text(mass_jump),
                        ha='center', va='center', fontsize=fs,
                        color=text_c, fontweight='semibold',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.4',
                                  fc=box_bg, ec=box_ec, alpha=0.92, lw=1.2),
                        zorder=z + 10)

        # End node (star)
        end = snapped[-1]
        e_size = 350 if is_hero else 120
        ax.scatter([end[0]], [end[1]],
                   color=P["end_fill"], marker='*', s=e_size,
                   edgecolors=P["end_edge"], linewidth=0.8,
                   zorder=z + 15, alpha=0.9)
        if is_hero:
            ax.scatter([end[0]], [end[1]],
                       color=P["end_fill"], s=e_size * 2.5,
                       edgecolors='none', alpha=0.06, zorder=z + 14)

        # Residual annotation
        if pep_mass is not None:
            residual = pep_mass - (end[0] + end[1])
            res_text = get_anno_text(residual)
            label = f"Δ {res_text}"

            ax.text(end[0], end[1], label,
                    ha='center', va='bottom', fontsize=fs,
                    color=P["res_text"], fontweight='semibold',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.4',
                              fc=P["res_box_bg"], ec=P["res_box_ec"],
                              alpha=0.95, lw=1.2),
                    zorder=z + 20)

    # ── 7. Draw Paths ─────────────────────────────────────────────────────
    if candidate_paths:
        for path in candidate_paths:
            draw_path(path, is_hero=False)

    if correct_path:
        draw_path(correct_path, is_hero=True)

    # ── 8. Axis Formatting ────────────────────────────────────────────────
    ax.set_xticks(S)
    ax.set_yticks(S)
    ax.set_xticklabels([f"{v:.2f}" for v in S], rotation=90, fontsize=8,
                        fontfamily='monospace')
    ax.set_yticklabels([f"{v:.2f}" for v in S], fontsize=8,
                        fontfamily='monospace')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # Color tick labels
    for tick, val in zip(ax.xaxis.get_major_ticks(), S):
        lbl = tick.label2
        if is_spurious(val):
            lbl.set_color(P["tick_spurious"])
            lbl.set_fontweight('bold')
        else:
            lbl.set_color(P["tick_clean"])

    for tick, val in zip(ax.yaxis.get_major_ticks(), S):
        lbl = tick.label1
        if is_spurious(val):
            lbl.set_color(P["tick_spurious"])
            lbl.set_fontweight('bold')
        else:
            lbl.set_color(P["tick_clean"])

    # Tick & spine styling
    ax.tick_params(axis='both', which='both',
                   color=P["spine"], length=4, width=0.5)
    for spine in ax.spines.values():
        spine.set_color(P["spine"])
        spine.set_linewidth(0.6)

    pad = 5
    ax.set_xlim(min(S) - pad, max(S) + pad)
    ax.set_ylim(min(S) - pad, max(S) + pad)
    ax.invert_yaxis()

    ax.set_xlabel("Fragment mass x₁ (Da)", labelpad=14,
                  fontsize=11, color=P["label"], fontfamily='sans-serif')
    ax.set_ylabel("Fragment mass x₂ (Da)", labelpad=10,
                  fontsize=11, color=P["label"], fontfamily='sans-serif')

    # ── 9. Legend ──────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color=P["hero_line"], lw=5.5,
               label='Correct path'),
        Line2D([0], [0], color=P["cand_line"], lw=3.5,
               linestyle='--', label='Candidate path'),
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=P["start_fill"], markeredgecolor=P["start_edge"],
               markersize=16, label='Start node'),
        Line2D([0], [0], marker='*', color='none',
               markerfacecolor=P["end_fill"], markeredgecolor=P["end_edge"],
               markersize=18, label='End node (+ residual)'),
        Line2D([0], [0], color='#E06060', lw=2.8, linestyle=(0, (6, 3)),
               alpha=0.45, label='Spurious grid line'),
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor='#F0C0C0', markeredgecolor='#D07070',
               markersize=13, label='Partial spurious node'),
        Line2D([0], [0], marker='D', color='none',
               markerfacecolor='#E88080', markeredgecolor='#C04040',
               markersize=12, label='Full spurious node'),
    ]

    leg = ax.legend(handles=legend_elements,
                    loc='upper right', bbox_to_anchor=(1.0, 1.0),
                    fontsize=18, frameon=True,
                    facecolor=P["legend_bg"], edgecolor=P["legend_ec"],
                    labelcolor=P["legend_text"],
                    borderpad=1.4, handletextpad=1.2, handlelength=3.0)
    leg.get_frame().set_alpha(0.95)
    leg.get_frame().set_linewidth(1.2)

    # ── 10. Title ─────────────────────────────────────────────────────────
    ax.set_title(title, pad=36, fontsize=15, fontweight='semibold',
                 color=P["title"], fontfamily='sans-serif',
                 loc='left')

    ax.text(1.0, 1.02, "2D-PC-MS",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color=P["label"], alpha=0.35,
            fontfamily='monospace', fontweight='light')

    # ── 11. De-overlap tick labels ────────────────────────────────────────
    # Must draw canvas first so label bounding boxes are computed
    fig.canvas.draw()
    _stagger_tick_labels(ax, axis='x', min_gap_pt=6)
    _stagger_tick_labels(ax, axis='y', min_gap_pt=8)

    # ── 12. Save / Show ───────────────────────────────────────────────────
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi,
                    facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        print(f"Saved → {save_path}")
    else:
        plt.tight_layout()
        plt.show()
        
        
'''
        
if __name__ =='__main__':
    
    allowed_mass_list = list(AA_MASSES.values()) + list(DOUBLE_AA_MASSES.values()) + list(TRIPLE_AA_MASSES.values()) + list(QUADRA_AA_MASSES.values())
    
    #pep_seq = 'HADGSFSDEMNTILDNLAARDFINWLIQTKITD'
    pep_seq = 'VEADIAGHGQEVLIR'
    charge = 3

    pep = peptide.Pep(f'[{pep_seq}+{charge}H]{charge}+', end_h20=True)
    #breaks = [i for i in range(1, len(pep_seq))]
    #breaks = [2, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    breaks = [3, 4, 5, 6, 8, 9, 10, 11, 12]

    spec = [pep.ion_mass(f'b{i}') for i in breaks] + [pep.ion_mass(f'y{len(pep_seq) - i}') for i in breaks]
    spec = [i-1 for i in spec]
    spec = [0] + [18.01056] + spec + [pep.pep_mass] + [100]
    spec.sort()
    sorted_array = spec
    mid_point = len(sorted_array) // 2
    lower_half = sorted_array[:mid_point]
    upper_half = sorted_array[mid_point:]
    lower_half.sort()
    graph = construct_graph_nodes(lower_half)
    graph = build_adjacency_graph(nodes = graph, allowed_masses=allowed_mass_list, total_peptide_mass=pep.pep_mass - charge*1.007276, threshold=0.05)
    length, peptide_masses, node_path = find_longest_path(graph=graph, start_node=SOURCE_NODE)
    print(length)
    print(node_path)
    
    correct = node_path[:-1]
    candidates = [correct]
    
    amino_acid_masses_switch = {v: k for k, v in amino_acid_masses_merge.items()}
    
    visualize_all_paths(lower_half, spurious_masses=[100], 
                         candidate_paths=candidates, 
                         correct_path=correct, 
                         aa_map=amino_acid_masses_switch,
                         title = pep_seq,
                         pep_mass=pep.seq_mass + 18.01056,
                         save_path= False)
    
'''

# ─── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    spectrum = [0.00, 18.01, 299.14, 299.16, 396.16, 400.29, 414.18,
                499.36, 504.25, 527.27, 598.18, 598.30, 628.40]
    spurious = [299.14, 299.16, 396.16, 504.25, 598.18]

    aa_map = {
        299.16: "?",
        115.03: "D",
        99.07:  "V",
        113.08: "J",
        71.04:  "A",
        129.04: "E",
    }

    correct = [(18.01, 18.01), (18.01, 299.16), (299.16, 299.16),
               (299.16, 400.29), (400.29, 400.29), (400.29, 499.36),
               (499.36, 499.36), (499.36, 504.25),
               (504.25, 504.25), (504.25, 628.40)]

    visualize_all_paths(
        spectrum,
        spurious_masses=spurious,
        candidate_paths=[],
        correct_path=correct,
        tolerance=0.5,
        aa_map=aa_map,
        pep_mass=628.40,
        title="VEA3+  •  Spurious Impact Analysis",
        save_path="/home/claude/demo_spurious.png",
    )