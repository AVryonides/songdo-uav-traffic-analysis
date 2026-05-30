# r_rules.py
# R intersection — 4-arm intersection with arms N, W, S, E from R.csv
#
# Direction mapping (from segmentation / coordinate analysis):
#   Arm 1 (sections 1_1..1_7) — NORTH
#     Approach path (N → intersection): 1_2 → 1_3 → 1_4 (7-lane stopline)
#     Departure path (intersection → N): 1_5 → 1_6
#   Arm 2 (sections 2_1..2_4) — WEST
#     Approach path (W → intersection): 2_3 → 2_4 (5-lane stopline)
#     Departure path (intersection → W): 2_1 → 2_2
#   Arm 3 (sections 3_1..3_3) — SOUTH
#     Approach path (S → intersection): 3_3 (5-lane stopline)
#     Departure path (intersection → S): 3_1 → 3_2
#   Arm 4 (sections 4_1..4_7) — EAST
#     Approach path (E → intersection): 4_2 → 4_3 → 4_4 → 4_5 (6-lane stopline)
#     Departure path (intersection → E): 4_6
#
# Approach sections (vehicles queuing / entering the intersection box):
#   Arm 1 (N): {1_2, 1_3, 1_4}   Arm 2 (W): {2_3, 2_4}
#   Arm 3 (S): {3_3}               Arm 4 (E): {4_2, 4_3, 4_4, 4_5}
#
# Departure sections (vehicles leaving the intersection box):
#   Arm 1 (N): {1_5, 1_6}   Arm 2 (W): {2_1, 2_2}
#   Arm 3 (S): {3_1, 3_2}   Arm 4 (E): {4_6}
#
# 12 directional movements: SN SW SE NS NW NE WE WS WN EW EN ES

R_ARM_APPROACH = {
    '1': frozenset({'1_2', '1_3', '1_4'}),
    '2': frozenset({'2_3', '2_4'}),
    '3': frozenset({'3_3'}),
    '4': frozenset({'4_2', '4_3', '4_4', '4_5'}),
}

R_ARM_DEPART = {
    '1': frozenset({'1_5', '1_6'}),
    '2': frozenset({'2_1', '2_2'}),
    '3': frozenset({'3_1', '3_2'}),
    '4': frozenset({'4_6'}),
}

R_ARM_ALL = {
    '1': frozenset({'1_1', '1_2', '1_3', '1_4', '1_5', '1_6', '1_7'}),
    '2': frozenset({'2_1', '2_2', '2_3', '2_4'}),
    '3': frozenset({'3_1', '3_2', '3_3'}),
    '4': frozenset({'4_1', '4_2', '4_3', '4_4', '4_5', '4_6', '4_7'}),
}

ARMS = ('1', '2', '3', '4')

# Cardinal direction labels for each arm
ARM_DIRECTION = {'1': 'N', '2': 'W', '3': 'S', '4': 'E'}


def movement_to_name(mov: str) -> str:
    """Convert arm-pair movement '3->1' to directional name 'SN'."""
    if '->' not in mov:
        return mov
    o, d = mov.split('->')
    return ARM_DIRECTION.get(o, o) + ARM_DIRECTION.get(d, d)


# All 12 movement rules: (origin_approach, dest_depart, name)
R_MOVEMENT_RULES = [
    (R_ARM_APPROACH[o], R_ARM_DEPART[d], f"{o}->{d}")
    for o in ARMS for d in ARMS if o != d
]


def _section_to_arm(section: str) -> str:
    """Return the arm prefix ('1'..'4') for a section like '1_3', or ''."""
    if section is None:
        return ''
    return str(section).strip().split('_')[0]


def infer_r_movement(start_section, end_section) -> str:
    """Infer movement name (e.g. '2->3') from start and end sections."""
    if start_section is None or end_section is None:
        return 'UNASSIGNED'

    s_str = str(start_section).strip()
    e_str = str(end_section).strip()

    # Primary: strict approach → depart check
    for o_arm in ARMS:
        if s_str in R_ARM_APPROACH[o_arm]:
            for d_arm in ARMS:
                if o_arm == d_arm:
                    continue
                if e_str in R_ARM_DEPART[d_arm]:
                    return f"{o_arm}->{d_arm}"

    # Fallback: any section in arm → any section in different arm
    o_arm = _section_to_arm(s_str)
    d_arm = _section_to_arm(e_str)
    if o_arm and d_arm and o_arm != d_arm and o_arm in ARMS and d_arm in ARMS:
        return f"{o_arm}->{d_arm}"

    return 'UNASSIGNED'


def _build_r_lane_movement_patterns() -> dict:
    """Build lane-aware patterns for R intersection based on verified section paths.
    
    Based on detailed movement analysis from R.csv:
      SN  (S→N): 3_3_2/3/4 → 1_5_1/2/3 → 1_6_1/2/3
      SW  (S→W): 3_3_1 → 2_1_1 → 2_2_1
      SE  (S→E): 3_3_5 → 4_6_5/4
      NS  (N→S): 1_2_5/4/3 → 1_3_5/4/3 → 1_4_6/5/4 → 3_1_1/2/3 → 3_2_1/2/3
      NW  (N→W): 1_2_5 → 1_3_6 → 1_4_7 → 2_1_5/4 → 2_2_4
      NE  (N→E): 1_2_1/2 → 1_3_2/1 → 1_4_1/2/3 → 4_6_1/2/3/4
      WE  (W→E): 2_3_2/3/4/5 → 2_4_2/3/4/5 → 4_6
      WS  (W→S): 2_3_5/6 → 3_2_4/3
      WN  (W→N): 2_3_1 → 2_4_1 → 1_5_1 → 1_6_1
      EW  (E→W): 4_2_6/5/4/3 → 4_3_7/6/5/4 → 4_4_7/6/5/4 → 4_5_6/5/4/3 → 2_1 → 2_2
      EN  (E→N): 4_2_7 → 4_3_8 → 1_6_4/3
      ES  (E→S): 4_2_1/2 → 4_3_2/3 → 4_4_2/3 → 4_5_1/2 → 3_1_1/2 → 3_2_1/2
    """
    patt: dict = {f"{o}->{d}": [] for o in ARMS for d in ARMS if o != d}

    # 3→1: SN (S→N)
    patt["3->1"] = [
        (("3_3_2", "1_5_1", "1_6_1"), "lane_A", "3_3_2 → 1_5_1 → 1_6_1 (N)"),
        (("3_3_3", "1_5_2", "1_6_2"), "lane_B", "3_3_3 → 1_5_2 → 1_6_2 (N)"),
        (("3_3_4", "1_5_3", "1_6_3"), "lane_C", "3_3_4 → 1_5_3 → 1_6_3 (N)"),
    ]

    # 3→2: SW (S→W)
    patt["3->2"] = [
        (("3_3_1", "2_1_1", "2_2_1"), "lane_A", "3_3_1 → 2_1_1 → 2_2_1 (W)"),
    ]

    # 3→4: SE (S→E)
    patt["3->4"] = [
        (("3_3_5", "4_6_4"), "lane_A", "3_3_5 → 4_6_4 (E)"),
        (("3_3_5", "4_6_5"), "lane_B", "3_3_5 → 4_6_5 (E)"),
    ]

    # 1→3: NS (N→S)
    patt["1->3"] = [
        (("1_2_3", "1_3_4", "1_4_5", "3_1_1", "3_2_1"), "lane_A", "1_2_3 → 1_3_4 → 1_4_5 → 3_1_1 → 3_2_1 (S)"),
        (("1_2_4", "1_3_5", "1_4_6", "3_1_2", "3_2_2"), "lane_B", "1_2_4 → 1_3_5 → 1_4_6 → 3_1_2 → 3_2_2 (S)"),
        (("1_2_5", "1_3_5", "1_4_6", "3_1_3", "3_2_3"), "lane_C", "1_2_5 → 1_3_5 → 1_4_6 → 3_1_3 → 3_2_3 (S)"),
    ]

    # 1→2: NW (N→W)
    patt["1->2"] = [
        (("1_2_5", "1_3_6", "1_4_7", "2_1_4", "2_2_4"), "lane_A", "1_2_5 → 1_3_6 → 1_4_7 → 2_1_4 → 2_2_4 (W)"),
    ]

    # 1→4: NE (N→E)
    patt["1->4"] = [
        (("1_2_1", "1_3_1", "1_4_1", "4_6_1"), "lane_A", "1_2_1 → 1_3_1 → 1_4_1 → 4_6_1 (E)"),
        (("1_2_2", "1_3_2", "1_4_2", "4_6_2"), "lane_B", "1_2_2 → 1_3_2 → 1_4_2 → 4_6_2 (E)"),
        (("1_2_1", "1_3_2", "1_4_3", "4_6_3"), "lane_C", "1_2_1 → 1_3_2 → 1_4_3 → 4_6_3 (E)"),
        (("1_2_2", "1_3_2", "1_4_3", "4_6_4"), "lane_D", "1_2_2 → 1_3_2 → 1_4_3 → 4_6_4 (E)"),
    ]

    # 2→4: WE (W→E)
    patt["2->4"] = [
        (("2_3_2", "2_4_2", "4_6"), "lane_A", "2_3_2 → 2_4_2 → 4_6 (E)"),
        (("2_3_3", "2_4_3", "4_6"), "lane_B", "2_3_3 → 2_4_3 → 4_6 (E)"),
        (("2_3_4", "2_4_4", "4_6"), "lane_C", "2_3_4 → 2_4_4 → 4_6 (E)"),
        (("2_3_5", "2_4_5", "4_6"), "lane_D", "2_3_5 → 2_4_5 → 4_6 (E)"),
    ]

    # 2→3: WS (W→S)
    patt["2->3"] = [
        (("2_3_5", "3_2_3"), "lane_A", "2_3_5 → 3_2_3 (S)"),
        (("2_3_6", "3_2_4"), "lane_B", "2_3_6 → 3_2_4 (S)"),
    ]

    # 2→1: WN (W→N)
    patt["2->1"] = [
        (("2_3_1", "2_4_1", "1_5_1", "1_6_1"), "lane_A", "2_3_1 → 2_4_1 → 1_5_1 → 1_6_1 (N)"),
    ]

    # 4→2: EW (E→W)
    patt["4->2"] = [
        (("4_2_3", "4_3_4", "4_4_4", "4_5_3", "2_1"), "lane_A", "4_2_3 → 4_3_4 → 4_4_4 → 4_5_3 → 2_1 → 2_2 (W)"),
        (("4_2_4", "4_3_5", "4_4_5", "4_5_4", "2_1"), "lane_B", "4_2_4 → 4_3_5 → 4_4_5 → 4_5_4 → 2_1 → 2_2 (W)"),
        (("4_2_5", "4_3_6", "4_4_6", "4_5_5", "2_1"), "lane_C", "4_2_5 → 4_3_6 → 4_4_6 → 4_5_5 → 2_1 → 2_2 (W)"),
        (("4_2_6", "4_3_7", "4_4_7", "4_5_6", "2_1"), "lane_D", "4_2_6 → 4_3_7 → 4_4_7 → 4_5_6 → 2_1 → 2_2 (W)"),
    ]

    # 4→1: EN (E→N)
    patt["4->1"] = [
        (("4_2_7", "4_3_8", "1_6_3"), "lane_A", "4_2_7 → 4_3_8 → 1_6_3 (N)"),
        (("4_2_7", "4_3_8", "1_6_4"), "lane_B", "4_2_7 → 4_3_8 → 1_6_4 (N)"),
    ]

    # 4→3: ES (E→S)
    patt["4->3"] = [
        (("4_2_1", "4_3_2", "4_4_2", "4_5_1", "3_1_1", "3_2_1"), "lane_A", "4_2_1 → 4_3_2 → 4_4_2 → 4_5_1 → 3_1_1 → 3_2_1 (S)"),
        (("4_2_2", "4_3_3", "4_4_3", "4_5_2", "3_1_2", "3_2_2"), "lane_B", "4_2_2 → 4_3_3 → 4_4_3 → 4_5_2 → 3_1_2 → 3_2_2 (S)"),
    ]

    return patt


R_LANE_MOVEMENT_PATTERNS = _build_r_lane_movement_patterns()


def get_lane_path_description(movement: str, lane_label: str) -> str:
    entries = R_LANE_MOVEMENT_PATTERNS.get(movement, [])
    for _pattern, label, desc in entries:
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


def infer_r_movement_from_lane_sequence(states: list) -> str:
    """Classify a vehicle's movement from its section-lane state sequence."""
    if not states:
        return 'UNASSIGNED'

    movement_order = [f"{o}->{d}" for o in ARMS for d in ARMS if o != d]

    # Tier 1: exact full-pattern subsequence match
    for mov in movement_order:
        for p, _lane, _desc in R_LANE_MOVEMENT_PATTERNS.get(mov, []):
            if _is_subsequence(states, p):
                return mov

    # Tier 2: partial match (>= _MIN_PATTERN_MATCHES elements)
    best_mov, best_count, best_plen = None, 0, 0
    for mov in movement_order:
        for p, _lane, _desc in R_LANE_MOVEMENT_PATTERNS.get(mov, []):
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
    for mov in movement_order:
        for p, _lane, _desc in R_LANE_MOVEMENT_PATTERNS.get(mov, []):
            overlap = len(states_set & set(p))
            if overlap > best_overlap:
                best_overlap, best_mov_t3 = overlap, mov
    if best_mov_t3 and best_overlap >= 1:
        return best_mov_t3

    # Tier 4: section-prefix fallback
    for st in states:
        arm = _section_to_arm(st.rsplit('_', 1)[0] if '_' in st else st)
        if arm in ARMS:
            for mov in movement_order:
                if mov.startswith(arm + '->'):
                    return mov

    return 'UNASSIGNED'


_MOVEMENT_ORIGIN_SECTION = {
    f"{o}->{d}": next(iter(R_ARM_APPROACH[o]))
    for o in ARMS for d in ARMS if o != d
}

_ORIGIN_LANE_TO_LABEL = {
    mov: {str(i): f"lane_{i}" for i in range(1, 8)}
    for mov in [f"{o}->{d}" for o in ARMS for d in ARMS if o != d]
}


def infer_r_lane_from_sequence(states: list, movement: str) -> str:
    """Determine which lane a vehicle belongs to given its movement."""
    if not states or movement == 'UNASSIGNED':
        return 'unknown'

    entries = R_LANE_MOVEMENT_PATTERNS.get(movement, [])

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

    # Tier 4: origin-section fallback
    o_arm = movement.split('->')[0] if '->' in movement else ''
    if o_arm in R_ARM_APPROACH:
        for st in states:
            sec = st.rsplit('_', 1)[0] if st.count('_') >= 2 else st
            if sec in R_ARM_APPROACH[o_arm]:
                lane_num = st.rsplit('_', 1)[-1]
                return f"lane_{lane_num}"

    return 'unknown'
