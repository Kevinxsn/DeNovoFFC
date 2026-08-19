"""
path_finding.py — De novo sequencing via longest-path search on a 2D
fragment-correlation spectrum graph.

This module loads precomputed compound amino-acid masses from the JSON
file produced by ``edges.py`` (so you never recompute them at runtime)
and builds a directed acyclic graph whose longest source→sink path
reconstructs the peptide sequence.

Visualization is delegated to ``path_graph.visualize_all_paths``.

Typical workflow
----------------
  # 1. (once) Precompute edges
  python edges.py --max_n 4 -o edges.json

  # 2. Run sequencing
  python path_finding.py
"""

import numpy as np
from bisect import bisect_left
from pathlib import Path

# ── Import the edge loader ──────────────────────────────────────────────
from edges import load_edges, AA_MASSES

# ── Import the visualization module ─────────────────────────────────────
import path_graph
import path_graph_revised
import path_graph_poster

# ── Constants ────────────────────────────────────────────────────────────
SOURCE_NODE = (0.0, 18.01056)
SINK_NODE = "Sink"

amino_acid_masses_merge = {
    "A": 71.03711,   "R": 156.10111,  "N": 114.04293,  "D": 115.02694,
    "C": 103.00919,  "E": 129.04259,  "Q": 128.05858,  "G": 57.02146,
    "H": 137.05891,  "J": 113.08406,  "K": 128.09496,  "M": 131.04049,
    "F": 147.06841,  "P": 97.05276,   "S": 87.03203,   "T": 101.04768,
    "W": 186.07931,  "Y": 163.06333,  "V": 99.06841,
}


# ── Formatting helper ────────────────────────────────────────────────────
def format_path_string(path, with_aa=False, aa_masses=None):
    """
    Turn a list of nodes [(x1,x2), ...] into a human-readable arrow string.
    """
    if not path:
        return ""
    if aa_masses is None:
        aa_masses = amino_acid_masses_merge

    def fmt_node(n):
        return f"({round(n[0], 3)}, {round(n[1], 3)})"

    output = fmt_node(path[0])
    for i in range(len(path) - 1):
        curr, nxt = path[i], path[i + 1]
        if nxt[0] > curr[0]:
            direction = "Down ↓"
            mass = round(nxt[0] - curr[0], 3)
        else:
            direction = "Right →"
            mass = round(nxt[1] - curr[1], 3)

        if with_aa:
            aa = None
            for key, value in aa_masses.items():
                if abs(value - mass) <= 0.01:
                    aa = key
                    break
            output += f" {direction}({aa}) {fmt_node(nxt)}"
        else:
            output += f" {direction} {fmt_node(nxt)}"
    return output


def _merge_close(values, tolerance):
    """Collapse near-duplicate floats (within *tolerance*) in a sorted list."""
    merged = []
    for v in sorted(values):
        if merged and abs(v - merged[-1]) <= tolerance:
            continue
        merged.append(v)
    return merged


def format_gapped_reconstruction(path, total_mass, aa_masses=None,
                                 tolerance=0.01):
    """
    Render a hero path as a gapped N->C reconstruction string, e.g.
    ``"299.15 - DIA - 194.08 - GQEV - 382.27"``.

    A node reached by a "Down" move (x1 growing) is a breakpoint counted
    from the N-terminus (b-axis residue sum = x1 directly). A node reached
    by a "Right" move (x2 growing) is a breakpoint counted from the
    C-terminus; it is converted onto the same N-to-C scale via
    ``total_mass - WATER - (x2 - WATER)``, i.e. mirroring it through the
    total residue mass. Merging both breakpoint sets and sorting them
    gives the full N->C cleavage ladder implied by the path. The mass
    between two consecutive breakpoints is either a single resolved
    residue (rendered as its letter) or an unresolved multi-residue /
    gap segment (rendered as its mass, 2 decimals).

    Parameters
    ----------
    path : list of (x1, x2) tuples
        The hero path's node sequence (source..last node), *excluding*
        the trailing ``"Sink"`` sentinel.
    total_mass : float
        Neutral peptide mass in path_finding's residue-sum + water
        convention (i.e. the same value used as ``total_peptide_mass``
        when building the graph).
    aa_masses : dict, optional
        Single-residue mass table; defaults to ``amino_acid_masses_merge``.
    tolerance : float
        Mass tolerance (Da) for both AA matching and breakpoint merging.
    """
    if not path:
        return ""
    if aa_masses is None:
        aa_masses = amino_acid_masses_merge

    water = SOURCE_NODE[1]
    m_res = total_mass - water

    b_side = [x1 for x1, _ in path]
    y_side = [m_res - (x2 - water) for _, x2 in path]
    # The one gap that isn't covered by any b- or y-side breakpoint at all —
    # where the two ladders stop meeting evidence — is the sink/residual gap.
    sink_prev, sink_next = max(b_side), min(y_side)
    breakpoints = _merge_close([0.0, m_res] + b_side + y_side, tolerance)

    tokens = []
    letter_run = ""
    for prev, nxt in zip(breakpoints[:-1], breakpoints[1:]):
        gap = nxt - prev
        is_sink_gap = (abs(prev - sink_prev) <= tolerance
                       and abs(nxt - sink_next) <= tolerance)
        aa = None
        for key, value in aa_masses.items():
            if abs(value - gap) <= tolerance:
                aa = key
                break
        if aa:
            letter_run += f"{aa}(sink)" if is_sink_gap else aa
        else:
            if letter_run:
                tokens.append(letter_run)
                letter_run = ""
            label = f"{gap:.3f}"
            if is_sink_gap:
                label += " (sink)"
            tokens.append(label)
    if letter_run:
        tokens.append(letter_run)

    return " - ".join(tokens)


def is_reconstruction_correct(path, total_mass, pep_seq, aa_masses=None,
                              tolerance=0.001):
    """
    Check whether a hero path's gapped reconstruction is consistent with a
    known reference peptide sequence.

    Every breakpoint implied by the path (b-side x1 values and the
    complementary N-to-C conversion of y-side x2 values — the same set
    ``format_gapped_reconstruction`` collapses into segments) must land
    on an exact cumulative-residue-mass boundary of *pep_seq*. If every
    breakpoint lines up with a true residue boundary, each segment
    (letter or gap) necessarily spans a whole, correctly-placed run of
    residues, so the reconstruction is correct even where it left a
    multi-residue segment unresolved as a plain mass.

    Parameters
    ----------
    path : list of (x1, x2) tuples
        The hero path's node sequence, *excluding* the trailing "Sink".
    total_mass : float
        Neutral peptide mass, same convention as ``format_gapped_reconstruction``.
    pep_seq : str
        The reference (ground-truth) peptide sequence, one-letter codes.
    aa_masses : dict, optional
        Single-residue mass table; defaults to ``amino_acid_masses_merge``.
        "I" and "L" are treated as the isobaric "J" mass automatically.
    tolerance : float
        Mass tolerance (Da) for the boundary comparison.
    """
    if aa_masses is None:
        aa_masses = amino_acid_masses_merge
    j_mass = aa_masses.get("J", 113.08406)
    aa_masses = {**aa_masses, "I": j_mass, "L": j_mass}

    cum = 0.0
    true_breakpoints = [0.0]
    for ch in pep_seq:
        if ch not in aa_masses:
            return False
        cum += aa_masses[ch]
        true_breakpoints.append(cum)

    water = SOURCE_NODE[1]
    m_res = total_mass - water
    b_side = [x1 for x1, _ in path]
    y_side = [m_res - (x2 - water) for _, x2 in path]
    breakpoints = _merge_close([0.0, m_res] + b_side + y_side, tolerance)

    return all(
        any(abs(bp - t) <= tolerance for t in true_breakpoints)
        for bp in breakpoints
    )


# ── Graph construction ───────────────────────────────────────────────────
def construct_graph_nodes(spectrum):
    """
    Build the 2D grid nodes from the spectrum.
    Ensures the SOURCE_NODE coordinates (0, 18.01056) are present.
    """
    extended = set(spectrum)
    extended.add(SOURCE_NODE[0])
    extended.add(SOURCE_NODE[1])
    sorted_spec = sorted(extended)

    nodes = []
    for x1 in sorted_spec:
        for x2 in sorted_spec:
            if x1 != x2 or (x1 == SOURCE_NODE[0] and x2 == SOURCE_NODE[1]):
                nodes.append((x1, x2))
    return nodes


def build_adjacency_graph(nodes, allowed_masses, total_peptide_mass,
                          threshold=0.01):
    """
    Build the DAG edges (Down, Right, Sink) for the spectrum graph.

    **Optimised strategy** — instead of iterating over every allowed mass
    (potentially millions) for every node, we flip the loop:

      • The *outer* loop is over nodes  (N ~ spectrum² ≈ hundreds–thousands).
      • For each node coordinate we compute the *upper bound* on useful
        edge mass, then iterate only over *reachable spectrum coordinates*
        (≤ spectrum size ≈ tens) and check whether the gap to each one
        is a valid compound mass via binary search on the sorted edge list.

    Complexity drops from  O(N × E)  to  O(N × S × log E)
    where S = spectrum size (small) and E = edge-list size (large).

    Parameters
    ----------
    nodes : list of (x1, x2) tuples
    allowed_masses : list[float]
        Precomputed compound AA masses loaded from edges.json.
    total_peptide_mass : float
        Neutral peptide mass M.
    threshold : float
        Mass tolerance (Da).
    """
    graph = {node: [] for node in nodes}
    valid_nodes_set = set(nodes)

    # Sorted arrays for binary-search lookups
    sorted_allowed = sorted(allowed_masses)
    np_allowed = np.array(sorted_allowed)           # for fast bin-search
    min_edge = sorted_allowed[0]  if sorted_allowed else 0.0
    max_edge = sorted_allowed[-1] if sorted_allowed else 0.0

    unique_coords = sorted(set(n[0] for n in nodes) | set(n[1] for n in nodes))
    np_coords = np.array(unique_coords)
    max_coord = unique_coords[-1] if unique_coords else 0.0

    def _is_allowed_mass(gap):
        """Check (via binary search) whether *gap* matches any entry in
        sorted_allowed within ±threshold.  Returns the matched mass or None."""
        if gap < min_edge - threshold or gap > max_edge + threshold:
            return None
        idx = bisect_left(sorted_allowed, gap)
        if idx < len(sorted_allowed) and abs(sorted_allowed[idx] - gap) <= threshold:
            return sorted_allowed[idx]
        if idx > 0 and abs(sorted_allowed[idx - 1] - gap) <= threshold:
            return sorted_allowed[idx - 1]
        return None

    def _coords_above(val):
        """Return the slice of unique_coords that are > val - threshold
        (i.e. all coordinates reachable by a positive-mass edge)."""
        idx = bisect_left(unique_coords, val + min_edge - threshold)
        return unique_coords[idx:]            # already sorted ascending

    for current_node in graph:
        x1, x2 = current_node
        max_current = max(x1, x2)

        # ── Sink connection ──────────────────────────────────────────
        remainder = total_peptide_mass - x1 - x2
        if remainder > 0:
            matched = _is_allowed_mass(remainder)
            if matched is not None:
                graph[current_node].append((SINK_NODE, matched, 'sink'))

        # ── Upper bound on useful edge mass from this node ───────────
        # No edge can land beyond max_coord, so the biggest useful gap
        # for x1 is (max_coord - x1), for x2 is (max_coord - x2).
        max_gap_x1 = max_coord - x1 + threshold
        max_gap_x2 = max_coord - x2 + threshold

        # ── DOWN edges (x1 grows) ───────────────────────────────────
        # Iterate over spectrum coords *above* x1 and check the gap.
        if max_gap_x1 >= min_edge - threshold:
            for target_x1 in _coords_above(x1):
                gap = target_x1 - x1
                if gap > max_gap_x1:
                    break                     # coords are sorted, done
                if gap < min_edge - threshold:
                    continue
                matched = _is_allowed_mass(gap)
                if matched is not None:
                    tgt = (target_x1, x2)
                    if tgt in valid_nodes_set and max_current < max(target_x1, x2):
                        graph[current_node].append((tgt, matched, 'down'))

        # ── RIGHT edges (x2 grows) ──────────────────────────────────
        if max_gap_x2 >= min_edge - threshold:
            for target_x2 in _coords_above(x2):
                gap = target_x2 - x2
                if gap > max_gap_x2:
                    break
                if gap < min_edge - threshold:
                    continue
                matched = _is_allowed_mass(gap)
                if matched is not None:
                    tgt = (x1, target_x2)
                    if tgt in valid_nodes_set and max_current < max(x1, target_x2):
                        graph[current_node].append((tgt, matched, 'right'))

    return graph


# ── Longest-path search ──────────────────────────────────────────────────
def find_all_longest_paths(graph, start_node, sink_node="Sink",
                           max_paths=1000):
    """
    DFS with memoization that returns **all** paths of maximum length
    from *start_node* to *sink_node* in the DAG.

    When there are ties (multiple paths with the same number of edges),
    every tied path is collected.  A ``max_paths`` cap prevents
    combinatorial blow-up — once a node already stores that many
    solutions, no further ties are appended.

    Parameters
    ----------
    graph : dict
        Adjacency dict  {node: [(neighbor, mass, edge_type), ...]}.
    start_node : tuple
        The (0.0, 18.01056) source node.
    sink_node : str
        Identifier for the sink (default ``"Sink"``).
    max_paths : int
        Safety cap on the number of tied paths stored per node.
        Set to ``None`` or ``float('inf')`` to disable the cap.

    Returns
    -------
    (max_length, list_of_results)
        max_length : int — number of edges on the longest path (-1 if
                     no path reaches the sink).
        list_of_results : list of (mass_sequence, node_path) tuples,
                          one per tied longest path.
    """
    if max_paths is None:
        max_paths = float('inf')

    # memo[node] = (best_length, [(mass_seq, node_seq), ...])
    memo = {}

    def dfs(node):
        if node == sink_node:
            return (0, [([], [sink_node])])
        if node in memo:
            return memo[node]
        if node not in graph or not graph[node]:
            memo[node] = (-1, [])
            return memo[node]

        best_len = -1
        best_paths = []                       # list of (mass_seq, node_seq)

        for neighbor, mass, _ in graph[node]:
            nb_len, nb_paths = dfs(neighbor)
            if nb_len == -1:
                continue
            cur_len = 1 + nb_len

            if cur_len > best_len:
                # Strictly better — reset collection
                best_len = cur_len
                best_paths = []
                for m_seq, n_seq in nb_paths:
                    best_paths.append(([mass] + m_seq, [node] + n_seq))
                    if len(best_paths) >= max_paths:
                        break

            elif cur_len == best_len and len(best_paths) < max_paths:
                # Tied — accumulate
                for m_seq, n_seq in nb_paths:
                    best_paths.append(([mass] + m_seq, [node] + n_seq))
                    if len(best_paths) >= max_paths:
                        break

        if best_len == -1:
            memo[node] = (-1, [])
        else:
            memo[node] = (best_len, best_paths)
        return memo[node]

    return dfs(start_node)


def find_longest_path(graph, start_node, sink_node="Sink"):
    """
    Backward-compatible wrapper — returns a single longest path.

    Returns
    -------
    (max_length, mass_sequence, node_path)
        Identical signature to the original function.  If multiple
        tied paths exist, the first one found is returned.
    """
    length, paths = find_all_longest_paths(
        graph, start_node, sink_node, max_paths=1,
    )
    if length == -1 or not paths:
        return (-1, [], [])
    mass_seq, node_seq = paths[0]
    return (length, mass_seq, node_seq)


# ── Main demo ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # ── 1. Load precomputed edges ────────────────────────────────────────
    EDGE_FILE = "edges.json"
    if not Path(EDGE_FILE).exists():
        print(f"Edge file '{EDGE_FILE}' not found. Generating with max_n=4 ...")
        from edges import build_and_save
        build_and_save(max_n=4, path=EDGE_FILE)

    allowed_mass_list = load_edges(EDGE_FILE)
    print(f"Loaded {len(allowed_mass_list):,} allowed masses from {EDGE_FILE}")

    # ── 2. Define a test spectrum ────────────────────────────────────────
    #  Replace this block with your real spectrum / peptide object.
    #  Below is a self-contained example using hardcoded masses.

    # Example: a small synthetic spectrum (replace with your data)
    # You can plug in your `peptide.Pep` object here if available.
    try:
        # Try importing your local peptide module for a real test
        sys.path.insert(0, "..")
        import peptide as peptide_mod

        pep_seq = "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"
        #pep_seq = "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"
        #pep_seq = "YPSKPDNPGEDAPAEDMARYYSALRHYINLITRQRY"
        #pep_seq = "HADGSFSDEMNTILDNLAARDFINWLIQTKITD"
        #pep_seq = "VEADIAGHGQEVLIR"
        #pep_seq = "HGTVVLTALGGILK"
        #pep_seq = 'ADLAGHGQEVLLR'
        #pep_seq = "YLEFISDAIIHVLHSK"
        
        charge = 4
        pep = peptide_mod.Pep(f"[{pep_seq}+{charge}H]{charge}+", end_h20=True)
        #breaks = [3, 4, 5, 6, 8, 9, 10, 11, 12]
        #breaks= [3,4,8,9,10,11,12,13,15,16, 25, 26,29] #LLG
        #breaks= [4,26,27,28,29,30,31,32,33,35] #KWK
        #breaks= [2,3,6,7,8,9,10,11,12,13,15,16,28,30] #YPS
        #breaks = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] #4+new
        #breaks = [4, 6, 8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33] #kwk_new
        #breaks = [3, 7, 8, 10, 11, 12, 13, 16, 23, 25, 26, 29] #LLG new
        #breaks = [4, 6, 8, 9, 10, 11, 12, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35] #kwk_all
        #breaks = [2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] #4+all
        #breaks = [2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 23, 25, 26, 27, 29, 30] #LLG+all
        #breaks = [3, 4, 5, 6, 8, 9, 10, 11, 12]#vea
        #breaks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] #vea complete
        #breaks = [29,27,26,30,31,12,28,13,14,23,25,17,18,24,19,11,10,15,22,2,9,20,21] #HAD newest
        #breaks = [16, 10, 12, 13, 11, 4, 9, 23, 29, 25, 30, 2, 26, 14, 8, 15, 5 ] #LLG
        #breaks = [2,3,4,5,8, 9,10, 11, 12, 13, 14, 15, 16, 25, 26, 29,30] #LLG
        #breaks = [26, 10, 8, 25, 9, 11, 27, 23, 30,29, 4, 24, 12, 31, 28, 6, 16, 17] # KWK
        #breaks = [12, 6, 10, 7, 9, 11, 16, 2, 8, 13, 14, 25, 27]
        #breaks = [12, 6, 10, 7, 9, 11, 16, 2, 8, 25, 3, 13, 14, 4] #YPS6
        #breaks = [7, 28, 6, 16, 10, 15, 13, 14, 9, 8, 11, 2, 26, 12] #YPS7
        #breaks = [4,5, 6, 7, 8, 9, 10, 11,12] #HGT
        #breaks = [3, 4, 5, 6, 7, 8, 9, 11, 12, 14] #YLE
        #breaks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12] #HGT2+
        #breaks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] #ADI2+
        #breaks = [2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 , 24, 25, 26, 27, 28, 29, 30, 31, 32] #HAD more fragements
        #breaks = [1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,23,24,25,26,29,30,31] #LLG internal
        #breaks = [1, 2, 6, 7, 8, 9, 10, 11, 12, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32] #KWK internal
        breaks = [2, 9, 10, 12, 26, 32] #LLG8+        
        
        
        spec = (
            [pep.ion_mass(f"b{i}") for i in breaks]
            + [pep.ion_mass(f"y{len(pep_seq) - i}") for i in breaks]
        )
        #spu = [710.694, 3564.292, 4274.986, 954.512, 3320.695] #YPS
        #spu = [3252.991, 1305.1206, 2702.4324, 753.464] #kwk 754.4525
        #spu = [1253.5365, 3017.5415, 1201.5145, 3069.5939]
        #spu = [1410.7519, 2357.1742, 1520.679, 2247.181] #new 4+
        #spu = [798.7381, 3207.4215, 741.4235, 3264.9842, 2394.7098, 1611.9451] #kwk
        #spu = [1498.2011, 2994.3910] 
        spu = []
        #spu = [3653.7715, 842.4812, 3646.1421, 850.4832]#LLG
        spu = [i - 1 for i in spu]
        
        spec = [m - 1.007276 for m in spec]
        spec = spec + spu
        spec = sorted(set([0, 18.01056] + spec + [pep.pep_mass]))
        total_mass = pep.pep_mass - charge * 1.007276
        seq_mass_for_plot = pep.seq_mass + 18.01056
        print(spec)

        # Map each axis mass to the b/y ion that produced it, so the
        # revised graph can annotate non-spurious ticks with their ion name.
        ion_labels = {}
        for i in breaks:
            if i > 0:
                ion_labels[pep.ion_mass(f"b{i}") - 1] = f"b{i}"
            y_idx = len(pep_seq) - i
            if y_idx > 0:
                ion_labels[pep.ion_mass(f"y{y_idx}") - 1] = f"y{y_idx}"

    except Exception:
        # Fallback: a tiny toy spectrum so the script runs standalone
        print("peptide module not available — using toy spectrum for demo")
        pep_seq = "DEMO"
        spec = [0.0, 18.01056, 57.021, 71.037, 128.058, 186.079, 300.0]
        total_mass = 300.0
        charge = 1
        seq_mass_for_plot = 300.0
        ion_labels = {}

    # Take lower half of spectrum (per your existing convention)
    sorted_spec = sorted(spec)
    mid = len(sorted_spec) // 2 + 1
    lower_half = sorted(sorted_spec[:mid])

    # ── 3. Build graph & find ALL longest paths ────────────────────────────
    nodes = construct_graph_nodes(lower_half)
    graph = build_adjacency_graph(
        nodes=nodes,
        allowed_masses=allowed_mass_list,
        total_peptide_mass=total_mass,
        threshold=0.001,
    )

    max_len, all_paths = find_all_longest_paths(
        graph, start_node=SOURCE_NODE, max_paths=1000,
    )

    print(f"\nLongest path length: {max_len}")
    print(f"Number of tied paths: {len(all_paths)}")
    for i, (masses, nodes_seq) in enumerate(all_paths):
        stripped = nodes_seq[:-1]  # remove Sink sentinel
        gapped = format_gapped_reconstruction(stripped, total_mass)
        correct = is_reconstruction_correct(stripped, total_mass, pep_seq)
        print(f"\n  Path {i+1}:")
        print(f"    Nodes:  {nodes_seq}")
        print(f"    String: {format_path_string(stripped, with_aa=True)}")
        print(f"    Gapped: {gapped}{'  [CORRECT]' if correct else ''}")

    # ── 4. Visualize with path_graph ─────────────────────────────────────
    aa_map_inv = {v: k for k, v in amino_acid_masses_merge.items()}

    if all_paths:
        # First path is the "hero" (correct), rest are candidates
        #hero = all_paths[0][1][:-1]           # node_seq minus Sink
        #candidates = [p[1][:-1] for p in all_paths[1:]]
        
        
        # Suppose hero_path is already known
        hero = all_paths[0][1][:-1]   # node_seq minus Sink

        # Candidates = all other paths except the hero
        candidates = [
            p[1][:-1]
            for p in all_paths
            if p[1][:-1] != hero
        ]
        #candidates = []
        
        
        path_graph_revised.visualize_all_paths(
            lower_half,
            spurious_masses=spu,
            candidate_paths=candidates if candidates else None,
            correct_path=hero,
            aa_map=aa_map_inv,
            ion_labels=ion_labels,
            show_aa_mass=False,
            title="",
            pep_mass=seq_mass_for_plot,
            save_path='graph.png',
        )