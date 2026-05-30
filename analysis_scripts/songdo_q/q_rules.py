# q_rules.py
# Based on geometric analysis and confirmed section-to-section transitions
# Arm 1 (sections 1_1-1_5) -> NORTH
# Arm 2 (sections 2_1-2_4) -> SOUTH
# Arm 3 (sections 3_1-3_4) -> EAST

Q_ARM_DIRECTIONS = {'1': 'N', '2': 'S', '3': 'E'}

# Movement mapping: (origin_sections) -> (destination_sections) -> movement_name
Q_MOVEMENT_RULES = [
    (frozenset(['1_2', '1_3']), frozenset(['2_1']),           'NS'),  # North to South
    (frozenset(['1_2', '1_3']), frozenset(['3_3', '3_4']),    'NE'),  # North to East
    (frozenset(['2_3', '2_4']), frozenset(['1_4', '1_5']),    'SN'),  # South to North
    (frozenset(['2_3', '2_4']), frozenset(['3_3', '3_4']),    'SE'),  # South to East
    (frozenset(['3_1', '3_2']), frozenset(['1_4', '1_5']),    'EN'),  # East to North
    (frozenset(['3_1', '3_2']), frozenset(['2_1']),           'ES'),  # East to South
]

def infer_q_movement(start_section, end_section):
    """
    Infer movement for a vehicle based on start and end sections.
    Returns movement name (NS, NE, SN, SE, EN, ES) or UNASSIGNED.
    """
    if start_section is None or end_section is None:
        return 'UNASSIGNED'

    for origins, dests, movement_name in Q_MOVEMENT_RULES:
        if start_section in origins and end_section in dests:
            return movement_name

    return 'UNASSIGNED'


def _build_q_lane_movement_patterns() -> dict:
    """
    Lane-aware movement patterns based on merged section-lane paths.

    Each entry is (pattern_tuple, lane_label, path_description).
      - lane_label:       used for folder/file names (e.g. "lane_A")
      - path_description: human-readable path shown in diagram titles

    A vehicle is classified if at least 2 elements of the pattern appear
    as a subsequence in its state sequence.

    Paths (from user-verified segmentation maps):
      NS lane_A: 1_2_1 -> 1_3_3 -> 2_1_1
         lane_B: 1_2_2 -> 1_3_4 -> 2_1_2
         lane_C: 1_2_3 -> 1_3_5 -> 2_1_3

      NE lane_A: 1_2_1 -> 1_3_1 -> 3_3_x -> 3_4_x
         lane_B: 1_2_1 -> 1_3_2 -> 3_3_x -> 3_4_x

      SN lane_A: 2_3_1 -> 2_4_1 -> 1_4_1 -> 1_5_1
         lane_B: 2_3_2 -> 2_4_2 -> 1_4_2 -> 1_5_2
         lane_C: 2_3_3 -> 2_4_3 -> 1_4_3 -> 1_5_3

      SE lane_A: 2_3_3 -> 2_4_4 -> 3_3_x -> 3_4_x

      EN lane_A: 3_2_3 -> 1_4_x -> 1_5_x

      ES lane_A: 3_2_1 -> 2_1_x
         lane_B: 3_2_2 -> 2_1_x
    """
    patt = {"NS": [], "NE": [], "SN": [], "SE": [], "EN": [], "ES": []}

    # NS — 3 lane paths (straight through)
    patt["NS"] += [
        (("1_2_1", "1_3_3", "2_1_1"), "lane_A", "1_2_1 \u2192 1_3_3 \u2192 2_1_1"),
        (("1_2_2", "1_3_4", "2_1_2"), "lane_B", "1_2_2 \u2192 1_3_4 \u2192 2_1_2"),
        (("1_2_3", "1_3_5", "2_1_3"), "lane_C", "1_2_3 \u2192 1_3_5 \u2192 2_1_3"),
    ]

    # NE — 2 lane paths (from 1_2_1, via 1_3_1 or 1_3_2, turning east)
    for s33 in ["3_3_1", "3_3_2"]:
        for s34 in ["3_4_1", "3_4_2"]:
            patt["NE"].append(
                (("1_2_1", "1_3_1", s33, s34), "lane_A", "1_2_1 \u2192 1_3_1 \u2192 3_3 \u2192 3_4")
            )
            patt["NE"].append(
                (("1_2_1", "1_3_2", s33, s34), "lane_B", "1_2_1 \u2192 1_3_2 \u2192 3_3 \u2192 3_4")
            )

    # SN — 3 lane paths (straight through)
    patt["SN"] += [
        (("2_3_1", "2_4_1", "1_4_1", "1_5_1"), "lane_A", "2_3_1 \u2192 2_4_1 \u2192 1_4_1 \u2192 1_5_1"),
        (("2_3_2", "2_4_2", "1_4_2", "1_5_2"), "lane_B", "2_3_2 \u2192 2_4_2 \u2192 1_4_2 \u2192 1_5_2"),
        (("2_3_3", "2_4_3", "1_4_3", "1_5_3"), "lane_C", "2_3_3 \u2192 2_4_3 \u2192 1_4_3 \u2192 1_5_3"),
    ]

    # SE — 1 lane path (from 2_3_3, turning east)
    for s33 in ["3_3_3", "3_3_1", "3_3_2"]:
        for s34 in ["3_4_1", "3_4_2"]:
            patt["SE"].append(
                (("2_3_3", "2_4_4", s33, s34), "lane_A", "2_3_3 \u2192 2_4_4 \u2192 3_3 \u2192 3_4")
            )

    # EN — 1 lane path (from 3_2_3, turning north)
    for s14 in ["1_4_2", "1_4_3", "1_4_4"]:
        for s15 in ["1_5_1", "1_5_2", "1_5_3"]:
            patt["EN"].append(
                (("3_2_3", s14, s15), "lane_A", "3_2_3 \u2192 1_4 \u2192 1_5")
            )

    # ES — 2 lane paths (from 3_2_1 and 3_2_2, heading south)
    for s21 in ["2_1_1", "2_1_2", "2_1_3"]:
        patt["ES"].append(
            (("3_2_1", s21), "lane_A", "3_2_1 \u2192 2_1")
        )
        patt["ES"].append(
            (("3_2_2", s21), "lane_B", "3_2_2 \u2192 2_1")
        )

    return patt


Q_LANE_MOVEMENT_PATTERNS = _build_q_lane_movement_patterns()


def get_lane_path_description(movement: str, lane_label: str) -> str:
    """Return the human-readable path description for a movement+lane.

    Used in diagram titles, e.g. '1_2_1 → 1_3_3 → 2_1_1'.
    Returns empty string if not found.
    """
    entries = Q_LANE_MOVEMENT_PATTERNS.get(movement, [])
    for _pattern, label, desc in entries:
        if label == lane_label:
            return desc
    return ""


def _is_subsequence(states: list, pattern: tuple) -> bool:
    """Check if ALL elements of pattern appear in states in order."""
    if len(pattern) == 0:
        return True
    j = 0
    for s in states:
        if s == pattern[j]:
            j += 1
            if j == len(pattern):
                return True
    return False


def _count_subsequence_matches(states: list, pattern: tuple) -> int:
    """Count how many elements of pattern appear in states as a subsequence.

    Returns the number of pattern elements matched in order.
    """
    if len(pattern) == 0:
        return 0
    j = 0
    matched = 0
    for s in states:
        if j < len(pattern) and s == pattern[j]:
            matched += 1
            j += 1
    return matched


# Minimum number of pattern elements a vehicle must match to be classified.
_MIN_PATTERN_MATCHES = 2


def infer_q_movement_from_lane_sequence(states: list) -> str:
    """Classify a vehicle's movement from its section-lane state sequence.

    A vehicle is assigned to a movement if at least _MIN_PATTERN_MATCHES
    elements of any lane-path pattern appear as a subsequence in its states.

    Full-pattern matches are tried first (exact). If none, partial matches
    (>= 2 elements) are accepted, preferring the pattern with the most matches.
    """
    if states is None or len(states) == 0:
        return "UNASSIGNED"

    # Tier 1: exact full-pattern match
    movement_order = ["NS", "NE", "SN", "SE", "EN", "ES"]
    for mov in movement_order:
        entries = Q_LANE_MOVEMENT_PATTERNS.get(mov, [])
        for p, _lane, _desc in entries:
            if _is_subsequence(states, p):
                return mov

    # Tier 2: partial match (at least _MIN_PATTERN_MATCHES elements)
    best_mov = None
    best_count = 0
    best_plen = 0
    for mov in movement_order:
        entries = Q_LANE_MOVEMENT_PATTERNS.get(mov, [])
        for p, _lane, _desc in entries:
            if len(p) < _MIN_PATTERN_MATCHES:
                continue
            n = _count_subsequence_matches(states, p)
            if n >= _MIN_PATTERN_MATCHES:
                if n > best_count or (n == best_count and len(p) > best_plen):
                    best_mov = mov
                    best_count = n
                    best_plen = len(p)

    if best_mov is not None:
        return best_mov

    # Tier 3: set-based single-element match — if any state appears in
    # a pattern, assign that movement
    states_set = set(states)
    best_mov_t3 = None
    best_overlap_t3 = 0
    for mov in movement_order:
        entries = Q_LANE_MOVEMENT_PATTERNS.get(mov, [])
        for p, _lane, _desc in entries:
            overlap = len(states_set & set(p))
            if overlap > best_overlap_t3:
                best_overlap_t3 = overlap
                best_mov_t3 = mov
    if best_mov_t3 is not None and best_overlap_t3 >= 1:
        return best_mov_t3

    # Tier 4: section-only fallback — infer from section prefix
    _SEC_TO_MOV = {
        "1_2": "NS", "1_3": "NS",  # default for ARM1 origin
        "2_3": "SN", "2_4": "SN",  # default for ARM2 origin
        "3_1": "ES", "3_2": "ES",  # default for ARM3 origin
    }
    for st in states:
        sec_prefix = "_".join(st.split("_")[:2])
        if sec_prefix in _SEC_TO_MOV:
            return _SEC_TO_MOV[sec_prefix]

    return "UNASSIGNED"


# Map each movement to its origin section prefix (for origin-lane fallback).
_MOVEMENT_ORIGIN_SECTION = {
    "NS": "1_2", "NE": "1_2",
    "SN": "2_3", "SE": "2_3",
    "EN": "3_2", "ES": "3_2",
}

# Map origin-lane index to lane_A/B/C labels per movement
# (used when tier-3 fallback assigns by first origin-section state)
_ORIGIN_LANE_TO_LABEL = {
    "NS":  {"1": "lane_A", "2": "lane_B", "3": "lane_C"},
    "NE":  {"1": "lane_A"},  # both NE lanes start from 1_2_1
    "SN":  {"1": "lane_A", "2": "lane_B", "3": "lane_C"},
    "SE":  {"3": "lane_A"},
    "EN":  {"3": "lane_A"},
    "ES":  {"1": "lane_A", "2": "lane_B"},
}


def infer_q_lane_from_sequence(states: list, movement: str) -> str:
    """
    Given a vehicle's section-lane state sequence and its known movement,
    determine which lane it belongs to by matching against Q_LANE_MOVEMENT_PATTERNS.
    Returns a lane label like 'lane_A', 'lane_B', etc., or 'unknown'.

    Three-tier matching strategy:
      1) Exact full-pattern subsequence match.
      2) Partial match: at least _MIN_PATTERN_MATCHES elements matched.
         Picks the pattern with the most matches.
      3) Origin-lane fallback: assign by the first origin-section state.
    """
    if states is None or len(states) == 0 or movement == "UNASSIGNED":
        return "unknown"

    entries = Q_LANE_MOVEMENT_PATTERNS.get(movement, [])

    # --- Tier 1: exact full-pattern match ---
    for p, lane_label, _desc in entries:
        if _is_subsequence(states, p):
            return lane_label

    # --- Tier 2: partial match (>= _MIN_PATTERN_MATCHES elements) ---
    best_lane = None
    best_count = 0
    best_plen = 0
    for p, lane_label, _desc in entries:
        if len(p) < _MIN_PATTERN_MATCHES:
            continue
        n = _count_subsequence_matches(states, p)
        if n >= _MIN_PATTERN_MATCHES:
            if n > best_count or (n == best_count and len(p) > best_plen):
                best_lane = lane_label
                best_count = n
                best_plen = len(p)
    if best_lane is not None:
        return best_lane

    # --- Tier 3: origin-lane fallback ---
    origin_prefix = _MOVEMENT_ORIGIN_SECTION.get(movement)
    lane_map = _ORIGIN_LANE_TO_LABEL.get(movement, {})
    if origin_prefix:
        for st in states:
            if st.startswith(origin_prefix + "_"):
                lane_num = st.rsplit("_", 1)[-1]
                if lane_num in lane_map:
                    return lane_map[lane_num]

    # --- Tier 4: set-based matching (any pattern element in states) ---
    # For vehicles that only appear in part of the trajectory (e.g.,
    # missing the first section), check if ANY states appear in a pattern.
    states_set = set(states)
    best_lane_t4 = None
    best_overlap = 0
    for p, lane_label, _desc in entries:
        overlap = len(states_set & set(p))
        if overlap > best_overlap:
            best_overlap = overlap
            best_lane_t4 = lane_label
    if best_lane_t4 is not None and best_overlap >= 1:
        return best_lane_t4

    # --- Tier 5: destination-lane fallback ---
    # If origin-prefix didn't work, try matching by any section_lane
    # element across ALL patterns for this movement.
    _DEST_LANE_MAPS = {
        "NS": {"2_1_1": "lane_A", "2_1_2": "lane_B", "2_1_3": "lane_C",
               "1_3_3": "lane_A", "1_3_4": "lane_B", "1_3_5": "lane_C"},
        "SN": {"1_4_1": "lane_A", "1_4_2": "lane_B", "1_4_3": "lane_C",
               "1_5_1": "lane_A", "1_5_2": "lane_B", "1_5_3": "lane_C",
               "2_4_1": "lane_A", "2_4_2": "lane_B", "2_4_3": "lane_C"},
        "NE": {"1_3_1": "lane_A", "1_3_2": "lane_B"},
        "SE": {"2_4_4": "lane_A"},
        "EN": {"3_2_3": "lane_A"},
        "ES": {"3_2_1": "lane_A", "3_2_2": "lane_B"},
    }
    dest_map = _DEST_LANE_MAPS.get(movement, {})
    for st in states:
        if st in dest_map:
            return dest_map[st]

    return "unknown"
