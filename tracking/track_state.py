import numpy as np

def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def boxes_are_duplicate(a, b, iou_threshold, center_ratio):
    box_a = (a["x1"], a["y1"], a["x2"], a["y2"])
    box_b = (b["x1"], b["y1"], b["x2"], b["y2"])
    iou = calculate_iou(box_a, box_b)
    if iou >= iou_threshold:
        return True
    acx = (a["x1"] + a["x2"]) / 2
    acy = (a["y1"] + a["y2"]) / 2
    bcx = (b["x1"] + b["x2"]) / 2
    bcy = (b["y1"] + b["y2"]) / 2
    distance = np.hypot(acx - bcx, acy - bcy)
    da = np.hypot(max(1, a["x2"] - a["x1"]), max(1, a["y2"] - a["y1"]))
    db = np.hypot(max(1, b["x2"] - b["x1"]), max(1, b["y2"] - b["y1"]))
    return iou >= 0.40 and distance <= ((da + db) / 2) * center_ratio

def deduplicate_persons(persons, iou_threshold, center_ratio):
    if len(persons) <= 1:
        return persons, set()
    ordered = sorted(persons, key=lambda p: p.get("confidence", 0.0), reverse=True)
    kept = []
    suppressed = set()
    for person in ordered:
        if any(boxes_are_duplicate(person, other, iou_threshold, center_ratio) for other in kept):
            suppressed.add(person["id"])
        else:
            kept.append(person)
    return kept, suppressed

def find_previous_track(track_states, person, current_time, active_ids, max_seconds, center_ratio, iou_threshold):
    best_id = None
    best_score = -1.0
    new_box = (person["x1"], person["y1"], person["x2"], person["y2"])
    ncx = (person["x1"] + person["x2"]) / 2
    ncy = (person["y1"] + person["y2"]) / 2
    nd = np.hypot(max(1, person["x2"] - person["x1"]), max(1, person["y2"] - person["y1"]))

    for old_id, state in track_states.items():
        if old_id == person["id"] or old_id in active_ids:
            continue
        missing = current_time - state.get("last_seen", current_time)
        if missing < 0 or missing > max_seconds:
            continue
        old_box = state.get("last_box")
        if old_box is None:
            continue
        ox1, oy1, ox2, oy2 = old_box
        ocx = (ox1 + ox2) / 2
        ocy = (oy1 + oy2) / 2
        od = np.hypot(max(1, ox2 - ox1), max(1, oy2 - oy1))
        ref = (nd + od) / 2
        distance = np.hypot(ncx - ocx, ncy - ocy)
        iou = calculate_iou(new_box, old_box)
        close = distance <= ref * center_ratio
        overlapping = iou >= iou_threshold
        if not (close or overlapping):
            continue
        score = iou * 5 + max(0.0, 1.0 - distance / max(ref, 1))
        if score > best_score:
            best_score = score
            best_id = old_id
    return best_id