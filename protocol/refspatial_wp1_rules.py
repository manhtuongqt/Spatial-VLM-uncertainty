"""Conservative offline candidate rules. No rule certifies visual correctness."""
import ast
import math
import re

QUESTION_TYPES = {
    'Identifying key object features, ranking objects based on those features, and then determining which specific object is being referenced.': 'object_features_ranking',
    'Identifying and selecting objects positioned along the edges of a tabletop.': 'object_at_table_edge',
    'Identifying object distances from a reference object.': 'object_distance_from_anchor',
    'Perceiving the orientation and distance from an anchor object, and then determining the specific location of a point along that direction at the given distance.': 'placement_direction_distance',
    'Localizing the positions of desktop edges.': 'placement_table_edge',
    'Determining the spatial position of a point located between two reference objects.': 'placement_between',
    'Labeling points that simultaneously satisfy directional relationships with respect to multiple anchor objects.': 'placement_multi_anchor',
    'Understanding the equidistant arrangement of objects and identifying the position where an object is missing.': 'placement_missing',
}
RELATIONS = {
    'left_pairwise': r'\b(?:left of|left side of)\b',
    'right_pairwise': r'\b(?:right of|right side of)\b',
    'front_pairwise': r'\b(?:in front of|front side of)\b',
    'behind_pairwise': r'\bbehind\b',
    'between_object': r'\bbetween\b',
    'leftmost_ranking': r'\bleftmost\b',
    'rightmost_ranking': r'\brightmost\b',
    'horizontal_ordinal_ranking': r'\b(?:left to right|right to left)\b',
    'nearest_ranking': r'\b(?:nearest|closest)\b',
    'farthest_ranking': r'\b(?:farthest|furthest)\b',
}


def normalized(text):
    return ' '.join(text.casefold().split())


def target_kind(question_type):
    canonical = QUESTION_TYPES.get(question_type, 'unknown')
    return ('object' if canonical.startswith('object_') else
            'free_space' if canonical.startswith('placement_') else 'unknown'), canonical


def point(answer):
    try:
        value = ast.literal_eval(answer)
        if (isinstance(value, (list, tuple)) and len(value) == 1
                and isinstance(value[0], (list, tuple)) and len(value[0]) == 2
                and all(type(x) in (float, int) and math.isfinite(x) and 0 <= x <= 1 for x in value[0])):
            return tuple(float(x) for x in value[0])
    except (ValueError, SyntaxError, TypeError):
        pass
    return None


def relation_tags(question, kind):
    if kind != 'object':
        return []
    query = question.split('Your answer should')[0]
    return [name for name, pattern in RELATIONS.items() if re.search(pattern, query, re.I)]


def semantic_review_requirements(question, kind, canonical_type):
    """Conservative scope guard; lexical tags are not target relation labels.

    No RGB, rationale, or proxy depth can establish an unspecified metric.
    Unknown syntax stays outside the direct horizontal audit pool.
    """
    query = normalized(question.split('Your answer should')[0])
    tags = relation_tags(question, kind)
    if kind != 'object':
        return {'scope': 'not_object', 'direct_relation': None,
                'blockers': ['not_object_qa'], 'training_eligible': False}
    blockers = []
    if 'fromt to left' in query:
        blockers.append('malformed_horizontal_direction')
    if re.search(r'ranks\s+in proximity', query):
        blockers.append('missing_distance_rank')
    distance = bool(re.search(r'\b(?:nearest|closest|farthest|furthest|nearer|farther|proximity)\b', query))
    if canonical_type == 'object_distance_from_anchor':
        scope, direct = 'distance_from_anchor', None
        blockers.append('anchor_distance_metric_unverified')
    elif distance:
        scope, direct = 'distance_ranking', None
        blockers.append('distance_reference_frame_unverified')
    elif re.search(r'\b(?:shortest|tallest|biggest|smallest|shorter|taller|bigger|smaller)\b', query):
        scope, direct = 'size_ranking', None
        blockers.append('size_definition_unverified')
    elif canonical_type == 'object_at_table_edge':
        scope, direct = 'table_edge', None
    elif len(tags) == 1 and tags[0] in {
            'leftmost_ranking', 'rightmost_ranking', 'horizontal_ordinal_ranking'}:
        scope, direct = 'direct_horizontal_ranking', tags[0]
    else:
        scope, direct = 'other_object_grounding', None
    if 'malformed_horizontal_direction' in blockers:
        scope, direct = 'malformed_query', None
    return {'scope': scope, 'direct_relation': direct, 'blockers': blockers,
            'training_eligible': False}


def fruit_evidence(label):
    """Return conservative noun evidence, not an exhaustive object taxonomy.

    Mixed relational phrases are unresolved: a fruit used as an anchor must not
    make a cup target a fruit. Container/drink context defeats orange-as-fruit.
    """
    s = normalized(label)
    if re.search(r'\b(?:near|beside|behind|between|containing|with|next to|left of|right of)\b', s):
        return 'unresolved', 'mixed_or_relational_label'
    if re.search(r'\b(?:juice|soda|drink|smoothie|flavou?r(?:ed)?)\b', s):
        return 'nonfruit', 'beverage_or_flavour'
    if re.search(r'\b(?:cup|mug|bottle|can|jar|box|carton|container|bowl|plate|basket|bag|packet|wrapper)\b', s):
        return 'nonfruit', 'container_noun'
    if re.search(r'\b(?:apple|banana|pear|plum|peach|lemon|lime|grape|strawberry|mango|pineapple|watermelon|orange)s?\b', s):
        return 'fruit', 'fruit_noun'
    if re.search(r'\bfruit\b', s):
        return 'fruit', 'generic_fruit_noun'
    return 'unresolved', 'no_supported_fruit_noun'


def corrected_categories(labels, categories):
    evidence = [fruit_evidence(label) for label in labels]
    updated = set(categories)
    # Only correct the known false fruit tag when EVERY label carrying a fruit
    # noun is explicitly a beverage/container; generic ordinal labels are neutral.
    relevant = [(l, e) for l, e in zip(labels, evidence)
                if re.search(r'\b(?:orange|fruit|apple|banana|pear|plum|peach|lemon|lime|grape|strawberry|mango|pineapple|watermelon)s?\b', l, re.I)]
    if 'fruit' in updated and relevant and all(e[0] == 'nonfruit' for _, e in relevant):
        updated.remove('fruit')
    return sorted(updated), [{'label': l, 'evidence': e[0], 'reason': e[1]} for l, e in zip(labels, evidence)]


def anchor_candidates(question, thinking, xy):
    """Offline candidates only; positions remain evaluator/annotation privileged."""
    query = normalized(question.split('Your answer should')[0])
    output = []
    for label, answer in re.findall(r'\[Position\]\s*\[([^\]]+)\]:\s*(\[\([^\]]+\)\])', thinking):
        pos = point(answer)
        if pos is not None and pos != xy and re.search(r'(?<!\w)' + re.escape(normalized(label)) + r'(?!\w)', query):
            row = {'label': label, 'xy': list(pos), 'status': 'OFFLINE_CANDIDATE_REVIEW_REQUIRED'}
            if row not in output:
                output.append(row)
    return output
