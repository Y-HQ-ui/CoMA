import copy
from collections import Counter
from planning.timeline import validate_blueprint

def intervene(blueprint, *, segment_id, field, value, track_id=None):
    result = copy.deepcopy(blueprint)
    matches = [s for s in result.segments if s.segment_id == segment_id]
    if len(matches) != 1:
        raise ValueError('segment_id must identify exactly one segment')
    segment = matches[0]
    if field in {'tempo_bpm', 'energy'}:
        if track_id is not None:
            raise ValueError('segment controls have no track_id')
        setattr(segment, field, value)
    elif field in {'density', 'register', 'pitch_range'}:
        tracks = [t for t in segment.tracks if (t.track_id or t.role) == track_id]
        if len(tracks) != 1:
            raise ValueError('track_id must identify exactly one track')
        setattr(tracks[0], field, value)
    else:
        raise ValueError('unsupported intervention field')
    validate_blueprint(result)
    return result

def score_change(before, after, *, track_id, start, end):

    def in_target(e):
        return e['track_id'] == track_id and start <= e['onset'] < end

    def key(e):
        return (e['track_id'], e['pitch'], round(e['onset'], 6), round(e['duration'], 6))
    untouched_before = Counter((key(e) for e in before if not in_target(e)))
    untouched_after = Counter((key(e) for e in after if not in_target(e)))
    changed = sum((untouched_before - untouched_after).values()) + sum((untouched_after - untouched_before).values())
    left = [e for e in before if in_target(e)]
    right = [e for e in after if in_target(e)]

    def attrs(items):
        return {'onsets_per_second': len({e['onset'] for e in items}) / (end - start), 'pitch_min': min([e['pitch'] for e in items], default=None), 'pitch_max': max([e['pitch'] for e in items], default=None)}
    return {'target_before': attrs(left), 'target_after': attrs(right), 'non_target_changed_events': changed, 'non_target_change_fraction': changed / max(1, sum(untouched_before.values()) + sum(untouched_after.values()))}
