# m_rules.py
# M intersection — 4-arm intersection with directional movement names.
#
# Arm assignment (from user-provided section map):
#   1_X  → North (N)
#   2_X  → West  (W)
#   3_X  → South (S)
#   4_X  → East  (E)
#
# Approach sections (vehicles entering the intersection from each arm):
#   N: {1_1, 1_2, 1_3}
#   W: {2_2, 2_3}
#   S: {3_4, 3_5, 3_6}
#   E: {4_1, 4_2}
#
# Departure sections (vehicles leaving the intersection to each arm):
#   N: {1_4, 1_5}
#   W: {2_1}
#   S: {3_1, 3_2, 3_3}
#   E: {4_3, 4_4}
#
# 14 directional movements:
#   From S: SN, SW, SE, SS (U-turn)
#   From N: NS, NN (U-turn), NW, NE
#   From W: WN, WE, WS
#   From E: EN, EW, ES
#
# Section paths per movement (from user specification):
#   SN:  3_4->3_5->3_6 => 1_4->1_5
#   SW:  3_4->3_5->3_6 => 2_1
#   SE:  3_4->3_5->3_6 => 4_3->4_4
#   SS:  3_4 => 3_1->3_2
#   NS:  1_1->1_2->1_3 => 3_1->3_2->3_3
#   NN:  1_1 => 1_5
#   NW:  1_1->1_2->1_3 => 2_1
#   NE:  1_1->1_2->1_3 => 4_3->4_4
#   WE:  2_2->2_3 => 4_3->4_4
#   WS:  2_2->2_3 => 3_2->3_3
#   WN:  2_2->2_3 (via 1_3) => 1_4->1_5
#   EW:  4_1->4_2 => 2_1
#   EN:  4_1 => 1_5
#   ES:  4_1->4_2 => 3_1->3_2->3_3

# Section-prefix → direction letter
ARM_PREFIX_TO_DIR = {'1': 'N', '2': 'W', '3': 'S', '4': 'E'}

DIRS = ('N', 'W', 'S', 'E')
# Backward-compat alias used by vehicle_grouping and other callers
ARMS = DIRS

M_ARM_APPROACH = {
    'N': frozenset({'1_1', '1_2', '1_3'}),
    'W': frozenset({'2_2', '2_3'}),
    'S': frozenset({'3_4', '3_5', '3_6'}),
    'E': frozenset({'4_1', '4_2'}),
}

M_ARM_DEPART = {
    'N': frozenset({'1_4', '1_5'}),
    'W': frozenset({'2_1'}),
    'S': frozenset({'3_1', '3_2', '3_3'}),
    'E': frozenset({'4_3', '4_4'}),
}

M_ARM_ALL = {
    'N': frozenset({'1_1', '1_2', '1_3', '1_4', '1_5'}),
    'W': frozenset({'2_1', '2_2', '2_3'}),
    'S': frozenset({'3_1', '3_2', '3_3', '3_4', '3_5', '3_6'}),
    'E': frozenset({'4_1', '4_2', '4_3', '4_4', '4_5'}),
}

# Ordered list of all 14 valid movements
MOVEMENTS = (
    'SN', 'SW', 'SE', 'SS',
    'NS', 'NN', 'NW', 'NE',
    'WN', 'WE', 'WS',
    'EN', 'EW', 'ES',
)

# Movement rules as (approach_set, depart_set, name) — used by vehicle_grouping
M_MOVEMENT_RULES = [
    (M_ARM_APPROACH[mov[0]], M_ARM_DEPART[mov[1]], mov)
    for mov in MOVEMENTS
]


def _section_to_dir(section: str) -> str:
    """Return direction letter ('N','W','S','E') for a section like '1_3', or ''."""
    if section is None:
        return ''
    prefix = str(section).strip().split('_')[0]
    return ARM_PREFIX_TO_DIR.get(prefix, '')


# Keep legacy alias used by some callers
_section_to_arm = _section_to_dir


def infer_m_movement(start_section, end_section) -> str:
    """Infer movement name (e.g. 'SN', 'NW') from start and end sections.

    Tier 1: approach-set origin × departure-set destination.
    Tier 2: arm-prefix fallback (any section in the arm).
    """
    if start_section is None or end_section is None:
        return 'UNASSIGNED'

    s = str(start_section).strip()
    e = str(end_section).strip()

    # Tier 1 — approach → departure matching
    origin = None
    for d in DIRS:
        if s in M_ARM_APPROACH[d]:
            origin = d
            break

    dest = None
    for d in DIRS:
        if e in M_ARM_DEPART[d]:
            dest = d
            break

    if origin and dest:
        mov = origin + dest
        if mov in MOVEMENTS:
            return mov

    # Tier 2 — arm-prefix fallback
    origin = origin or ARM_PREFIX_TO_DIR.get(s.split('_')[0] if '_' in s else '', '')
    dest = dest or ARM_PREFIX_TO_DIR.get(e.split('_')[0] if '_' in e else '', '')

    if origin and dest:
        mov = origin + dest
        if mov in MOVEMENTS:
            return mov

    return 'UNASSIGNED'


# ---------------------------------------------------------------------------
# Lane-aware movement patterns
# ---------------------------------------------------------------------------
# Each entry: (pattern_tuple_of_section_lane_states, lane_label, description)
# Section-lane states are strings like "3_4_2" (section 3_4, lane 2).

def _build_m_lane_movement_patterns() -> dict:
    """Build lane-aware path patterns for each movement.

    Patterns are derived from the user-specified section-lane paths.
    Each entry in patt[mov] is a 3-tuple:
        (pattern_tuple_of_section_lane_states, lane_label, description)

    Notation used in specifications:
        X/Y/Z  → lanes X, Y and Z of that section (listed in path order)
        A -> B → vehicle passes through A then B (same arm)
        A => B → vehicle exits approach arm and enters departure arm

    Specifications
    ──────────────
    SN: 3_4_2/3/4 -> 3_5_2/3/4 -> 3_6_3/4/5 => 1_4_1/2/3 -> 1_5_1/2/3
    SW: 3_4 -> 3_5 -> 3_6 => 2_1
    SE: 3_4 -> 3_5 -> 3_6 => 4_3 -> 4_4
    SS: 3_4_1 => 3_1_1/2 -> 3_2_1/2/3 -> 3_3_1/2/3

    NS: 1_1_5/4/3 -> 1_2_4/3/2 -> 1_3_6/5/4 => 3_1_3/2/1 -> 3_2_3/2/1 -> 3_3_3/2/1
    NN: 1_1_1/2 => 1_5_1/2/3
    NW: 1_1_5/4 -> 1_2_4/3 -> 1_3_7 => 2_1_4/3
    NE: 1_1_3/2 -> 1_2_2/1 -> 1_3_3/2/1 => 4_3_1/2/3 -> 4_4_1/2/3

    WE: 2_2_2/3/4 -> 2_3_2/3/4 => 4_3_1/2/3 -> 4_4_1/2/3
    WS: 2_2_1 -> 2_3_1 => 3_2 -> 3_3
    WN: 2_2 -> 2_3 => 1_4_1/2 -> 1_5_1/2/3

    EW: 4_1_4/3/2 -> 4_2_4/3/2 => 2_1_4/3/2/1
    EN: 4_1_4/5 => 1_5_4/3/2
    ES: 4_1_1 -> 4_2_1 => 3_1_1/2 -> 3_2_1/2/3 -> 3_3_1/2/3
    """
    patt: dict = {mov: [] for mov in MOVEMENTS}

    # ── SN ──────────────────────────────────────────────────────────────────
    # 3 lanes.  Approach offset shifts: 3_4(2,3,4) → 3_5(2,3,4) → 3_6(3,4,5)
    # Departure: 1_4(1,2,3) → 1_5(1,2,3)  (index-aligned)
    for i, (a1, a2, a3, d1, d2) in enumerate([
        (2, 2, 3, 1, 1),
        (3, 3, 4, 2, 2),
        (4, 4, 5, 3, 3),
    ]):
        patt['SN'].append((
            (f"3_4_{a1}", f"3_5_{a2}", f"3_6_{a3}", f"1_4_{d1}", f"1_5_{d2}"),
            f"lane_{i + 1}",
            f"3_4_{a1} -> 3_5_{a2} -> 3_6_{a3} => 1_4_{d1} -> 1_5_{d2}",
        ))

    # ── SW ──────────────────────────────────────────────────────────────────
    # No lane spec on approach; departure is 2_1 (4 lanes).
    # Approach 3_4 has 4 lanes; 3_6 has 6 lanes.
    # Build cross-product of all approach × departure lanes so any
    # vehicle seen in 3_4_x then 2_1_y is captured.
    for al in range(1, 5):          # 3_4 lanes 1-4
        for dl in range(1, 5):      # 2_1 lanes 1-4
            patt['SW'].append((
                (f"3_4_{al}", f"2_1_{dl}"),
                f"lane_{al}",
                f"3_4_{al} -> 3_6 => 2_1_{dl}",
            ))

    # ── SE ──────────────────────────────────────────────────────────────────
    # No lane spec on approach; departure is 4_3 -> 4_4 (each 4 lanes).
    for al in range(1, 5):
        for dl in range(1, 5):      # 4_3 lanes 1-4
            patt['SE'].append((
                (f"3_4_{al}", f"4_3_{dl}"),
                f"lane_{al}",
                f"3_4_{al} -> 3_6 => 4_3_{dl}",
            ))

    # ── SS ──────────────────────────────────────────────────────────────────
    # Only lane 1 of 3_4 performs the U-turn.
    # Departure: 3_1_1/2, 3_2_1/2/3, 3_3_1/2/3
    for dl in range(1, 3):              # 3_1 lanes 1-2
        patt['SS'].append((
            (f"3_4_1", f"3_1_{dl}"),
            "lane_1",
            f"3_4_1 => 3_1_{dl}",
        ))
    for dl in range(1, 4):              # 3_2 lanes 1-3
        patt['SS'].append((
            (f"3_4_1", f"3_2_{dl}"),
            "lane_1",
            f"3_4_1 => 3_2_{dl}",
        ))
    for dl in range(1, 4):              # 3_3 lanes 1-3
        patt['SS'].append((
            (f"3_4_1", f"3_3_{dl}"),
            "lane_1",
            f"3_4_1 => 3_3_{dl}",
        ))

    # ── NS ──────────────────────────────────────────────────────────────────
    # 3 lanes.  1_1(5,4,3) → 1_2(4,3,2) → 1_3(6,5,4) => 3_1(3,2,1)
    for i, (a1, a2, a3, d1) in enumerate([
        (5, 4, 6, 3),
        (4, 3, 5, 2),
        (3, 2, 4, 1),
    ]):
        patt['NS'].append((
            (f"1_1_{a1}", f"1_2_{a2}", f"1_3_{a3}", f"3_1_{d1}"),
            f"lane_{i + 1}",
            f"1_1_{a1} -> 1_2_{a2} -> 1_3_{a3} => 3_1_{d1}",
        ))

    # ── NN ──────────────────────────────────────────────────────────────────
    # Approach lanes 1 & 2 of 1_1; departure 1_5 lanes 1/2/3.
    for al in (1, 2):
        for dl in range(1, 4):
            patt['NN'].append((
                (f"1_1_{al}", f"1_5_{dl}"),
                f"lane_{al}",
                f"1_1_{al} => 1_5_{dl}",
            ))

    # ── NW ──────────────────────────────────────────────────────────────────
    # 2 lanes, both via 1_3_7 (the far-left lane in section 1_3).
    # lane_1: 1_1_5 -> 1_2_4 -> 1_3_7 => 2_1_4
    # lane_2: 1_1_4 -> 1_2_3 -> 1_3_7 => 2_1_3
    for i, (a1, a2, d1) in enumerate([(5, 4, 4), (4, 3, 3)]):
        patt['NW'].append((
            (f"1_1_{a1}", f"1_2_{a2}", f"1_3_7", f"2_1_{d1}"),
            f"lane_{i + 1}",
            f"1_1_{a1} -> 1_2_{a2} -> 1_3_7 => 2_1_{d1}",
        ))

    # ── NE ──────────────────────────────────────────────────────────────────
    # 2 main lanes + 1 via E-merge lane (1_3_1).
    # lane_1: 1_1_3 -> 1_2_2 -> 1_3_3 => 4_3_3 -> 4_4_3
    # lane_2: 1_1_2 -> 1_2_1 -> 1_3_2 => 4_3_2 -> 4_4_2
    for i, (a1, a2, a3, d1, d2) in enumerate([
        (3, 2, 3, 3, 3),
        (2, 1, 2, 2, 2),
    ]):
        patt['NE'].append((
            (f"1_1_{a1}", f"1_2_{a2}", f"1_3_{a3}", f"4_3_{d1}", f"4_4_{d2}"),
            f"lane_{i + 1}",
            f"1_1_{a1} -> 1_2_{a2} -> 1_3_{a3} => 4_3_{d1} -> 4_4_{d2}",
        ))
    # lane_3: via E-merge (1_3_1) => 4_3_1 -> 4_4_1
    patt['NE'].append((
        (f"1_3_1", f"4_3_1", f"4_4_1"),
        "lane_3",
        f"1_3_1 => 4_3_1 -> 4_4_1",
    ))

    # ── WE ──────────────────────────────────────────────────────────────────
    # 3 lanes.  Approach 2_2/2_3 lanes 2/3/4 → departure 4_3/4_4 lanes 1/2/3.
    for i, (al, dl) in enumerate([(2, 1), (3, 2), (4, 3)]):
        patt['WE'].append((
            (f"2_2_{al}", f"2_3_{al}", f"4_3_{dl}", f"4_4_{dl}"),
            f"lane_{i + 1}",
            f"2_2_{al} -> 2_3_{al} => 4_3_{dl} -> 4_4_{dl}",
        ))

    # ── WS ──────────────────────────────────────────────────────────────────
    # Only lane 1 of 2_2 and 2_3.  Departure 3_2 and 3_3 (no lane spec).
    for dl in range(1, 4):      # 3_2 lanes 1-3 (section has 4 lanes; spec says 1/2/3)
        patt['WS'].append((
            (f"2_2_1", f"2_3_1", f"3_2_{dl}"),
            "lane_1",
            f"2_2_1 -> 2_3_1 => 3_2_{dl}",
        ))
    for dl in range(1, 4):      # 3_3 lanes 1-3
        patt['WS'].append((
            (f"2_2_1", f"2_3_1", f"3_3_{dl}"),
            "lane_1",
            f"2_2_1 -> 2_3_1 => 3_3_{dl}",
        ))

    # ── WN ──────────────────────────────────────────────────────────────────
    # No lane spec for approach (2_2, 2_3); departure 1_4_1/2 → 1_5_1/2/3.
    # Key discriminator is the departure section (1_4/1_5).
    # 2_2 has 5 lanes; 2_3 has 4 lanes.
    for al in range(1, 5):          # 2_3 lanes 1-4 (last approach before intersection)
        for dl in range(1, 3):      # 1_4 lanes 1-2
            patt['WN'].append((
                (f"2_3_{al}", f"1_4_{dl}"),
                f"lane_{al}",
                f"2_3_{al} => 1_4_{dl}",
            ))
        for dl in range(1, 4):      # also match directly to 1_5 lanes 1-3
            patt['WN'].append((
                (f"2_3_{al}", f"1_5_{dl}"),
                f"lane_{al}",
                f"2_3_{al} => 1_5_{dl}",
            ))

    # ── EW ──────────────────────────────────────────────────────────────────
    # 3+ lanes.  Approach 4_1/4_2 lanes 4/3/2 → departure 2_1 lanes 4/3/2/1.
    for i, (al, dl) in enumerate([(4, 4), (3, 3), (2, 2)]):
        patt['EW'].append((
            (f"4_1_{al}", f"4_2_{al}", f"2_1_{dl}"),
            f"lane_{i + 1}",
            f"4_1_{al} -> 4_2_{al} => 2_1_{dl}",
        ))
    # Also cover 2_1_1 as destination (4th departure lane)
    patt['EW'].append((
        (f"4_1_2", f"4_2_2", f"2_1_1"),
        "lane_4",
        f"4_1_2 -> 4_2_2 => 2_1_1",
    ))

    # ── EN ──────────────────────────────────────────────────────────────────
    # Approach lanes 4 and 5 of 4_1; departure 1_5 lanes 4/3/2.
    for al, dl in [(5, 4), (4, 3), (4, 2)]:
        patt['EN'].append((
            (f"4_1_{al}", f"1_5_{dl}"),
            f"lane_{'1' if al == 5 else '2'}",
            f"4_1_{al} => 1_5_{dl}",
        ))

    # ── ES ──────────────────────────────────────────────────────────────────
    # Only lane 1 of 4_1 and 4_2.
    # Departure: 3_1_1/2, 3_2_1/2/3, 3_3_1/2/3.
    for dl in range(1, 3):      # 3_1 lanes 1-2
        patt['ES'].append((
            (f"4_1_1", f"4_2_1", f"3_1_{dl}"),
            "lane_1",
            f"4_1_1 -> 4_2_1 => 3_1_{dl}",
        ))
    for dl in range(1, 4):      # 3_2 lanes 1-3
        patt['ES'].append((
            (f"4_1_1", f"4_2_1", f"3_2_{dl}"),
            "lane_1",
            f"4_1_1 -> 4_2_1 => 3_2_{dl}",
        ))
    for dl in range(1, 4):      # 3_3 lanes 1-3
        patt['ES'].append((
            (f"4_1_1", f"4_2_1", f"3_3_{dl}"),
            "lane_1",
            f"4_1_1 -> 4_2_1 => 3_3_{dl}",
        ))

    return patt


M_LANE_MOVEMENT_PATTERNS = _build_m_lane_movement_patterns()


def get_lane_path_description(movement: str, lane_label: str) -> str:
    for _pattern, label, desc in M_LANE_MOVEMENT_PATTERNS.get(movement, []):
        if label == lane_label:
            return desc
    return ""


def _is_subsequence(states: list, pattern: tuple) -> bool:
    if not pattern:
        return True
    j = 0
    for s in states:
        if s == pattern[j]:
            j += 1
            if j == len(pattern):
                return True
    return False


def _count_subsequence_matches(states: list, pattern: tuple) -> int:
    if not pattern:
        return 0
    j = 0
    matched = 0
    for s in states:
        if j < len(pattern) and s == pattern[j]:
            matched += 1
            j += 1
    return matched


_MIN_PATTERN_MATCHES = 2


def infer_m_movement_from_lane_sequence(states: list) -> str:
    """Classify a vehicle's movement from its section-lane state sequence."""
    if not states:
        return 'UNASSIGNED'

    # Tier 1: exact full-pattern subsequence match
    for mov in MOVEMENTS:
        for p, _lane, _desc in M_LANE_MOVEMENT_PATTERNS.get(mov, []):
            if _is_subsequence(states, p):
                return mov

    # Tier 2: partial match (>= _MIN_PATTERN_MATCHES elements)
    best_mov, best_count, best_plen = None, 0, 0
    for mov in MOVEMENTS:
        for p, _lane, _desc in M_LANE_MOVEMENT_PATTERNS.get(mov, []):
            if len(p) < _MIN_PATTERN_MATCHES:
                continue
            n = _count_subsequence_matches(states, p)
            if n >= _MIN_PATTERN_MATCHES:
                if n > best_count or (n == best_count and len(p) > best_plen):
                    best_mov, best_count, best_plen = mov, n, len(p)
    if best_mov:
        return best_mov

    # Tier 3: set-based overlap
    states_set = set(states)
    best_mov_t3, best_overlap = None, 0
    for mov in MOVEMENTS:
        for p, _lane, _desc in M_LANE_MOVEMENT_PATTERNS.get(mov, []):
            overlap = len(states_set & set(p))
            if overlap > best_overlap:
                best_overlap, best_mov_t3 = overlap, mov
    if best_mov_t3 and best_overlap >= 1:
        return best_mov_t3

    # Tier 4: section-prefix fallback
    for st in states:
        sec = st.rsplit('_', 1)[0] if st.count('_') >= 2 else st
        d = _section_to_dir(sec)
        if d:
            for mov in MOVEMENTS:
                if mov.startswith(d):
                    return mov

    return 'UNASSIGNED'


def infer_m_lane_from_sequence(states: list, movement: str) -> str:
    """Determine which lane a vehicle belongs to given its movement."""
    if not states or movement == 'UNASSIGNED':
        return 'unknown'

    entries = M_LANE_MOVEMENT_PATTERNS.get(movement, [])

    # Tier 1: exact match
    for p, lane_label, _desc in entries:
        if _is_subsequence(states, p):
            return lane_label

    # Tier 2: partial match
    best_lane, best_count, best_plen = None, 0, 0
    for p, lane_label, _desc in entries:
        if len(p) < _MIN_PATTERN_MATCHES:
            continue
        n = _count_subsequence_matches(states, p)
        if n >= _MIN_PATTERN_MATCHES:
            if n > best_count or (n == best_count and len(p) > best_plen):
                best_lane, best_count, best_plen = lane_label, n, len(p)
    if best_lane:
        return best_lane

    # Tier 3: set-based overlap
    states_set = set(states)
    best_lane_t3, best_overlap = None, 0
    for p, lane_label, _desc in entries:
        overlap = len(states_set & set(p))
        if overlap > best_overlap:
            best_overlap, best_lane_t3 = overlap, lane_label
    if best_lane_t3 and best_overlap >= 1:
        return best_lane_t3

    # Tier 4: origin section fallback
    o_dir = movement[0] if movement and len(movement) >= 2 else ''
    if o_dir in M_ARM_APPROACH:
        for st in states:
            sec = st.rsplit('_', 1)[0] if st.count('_') >= 2 else st
            if sec in M_ARM_APPROACH[o_dir]:
                lane_num = st.rsplit('_', 1)[-1]
                return f"lane_{lane_num}"

    return 'unknown'
