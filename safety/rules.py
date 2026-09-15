from tracking.track_state import deduplicate_persons, find_previous_track

class SafetyRules:
    def __init__(self, camera_id, zones_count, event_manager, config):
        self.camera_id = camera_id
        self.event_manager = event_manager
        self.multiple_limit = config.multiple_person_limit_seconds
        self.cooldown_seconds = config.multiple_person_cooldown_seconds
        self.absence_limit = config.absence_limit_seconds
        self.track_grace = config.track_grace_seconds
        self.inside_grace = config.inside_grace_seconds
        self.switch_max = config.track_id_switch_max_seconds
        self.switch_center = config.track_id_switch_center_ratio
        self.switch_iou = config.track_id_switch_iou_threshold
        self.duplicate_iou = config.duplicate_iou_threshold
        self.duplicate_center = config.duplicate_center_ratio
        self.track_states = {}
        self.multiple_limit_seconds=120
        self.multiple_start = {i: None for i in range(zones_count)}
        self.cooldown_until = {i: None for i in range(zones_count)}
        self.multiple_logged = {i: False for i in range(zones_count)}

    def update(self, persons, current_time, zone_count):
        persons, suppressed = deduplicate_persons(persons, self.duplicate_iou, self.duplicate_center)
        for track_id in suppressed:
            self.track_states.pop(track_id, None)

        active_ids = {p["id"] for p in persons}
        for person in persons:
            track_id = person["id"]
            zone = person["zone"]
            if track_id not in self.track_states:
                old_id = find_previous_track(
                    self.track_states, person, current_time, active_ids,
                    self.switch_max, self.switch_center, self.switch_iou
                )
                if old_id is not None:
                    self.track_states[track_id] = self.track_states.pop(old_id).copy()
                    print(f"[{self.camera_id}] Track ID changed {old_id} -> {track_id}")
                else:
                    self.track_states[track_id] = {
                        "last_seen": current_time,
                        "last_inside": current_time if zone is not None else None,
                        "last_zone": zone,
                        "absence_logged": False,
                        "phone_logged": False,
                        "last_box": (person["x1"], person["y1"], person["x2"], person["y2"]),
                    }
            state = self.track_states[track_id]
            state["last_seen"] = current_time
            state["last_box"] = (person["x1"], person["y1"], person["x2"], person["y2"])
            if zone is not None:
                state["last_inside"] = current_time
                state["last_zone"] = zone
                state["absence_logged"] = False

        for track_id, state in list(self.track_states.items()):
            if state["last_inside"] is None:
                continue
            if current_time - state["last_seen"] <= self.track_grace:
                continue
            away = current_time - state["last_inside"]
            if away >= self.absence_limit and not state["absence_logged"]:
                self.event_manager.log(self.camera_id, current_time, "PERSON_AWAY_OVER_5_MINUTES", track_id, f"Track {track_id} outside zone for {away:.1f} seconds")
                state["absence_logged"] = True

        inside = {i: set() for i in range(zone_count)}
        for track_id, state in self.track_states.items():
            if current_time - state["last_seen"] <= self.track_grace and state["last_inside"] is not None and current_time - state["last_inside"] <= self.inside_grace:
                z = state["last_zone"]
                if z in inside:
                    inside[z].add(track_id)

        for z in range(zone_count):
            count = len(inside[z])
            cooldown = self.cooldown_until[z]
            if cooldown is not None and current_time < cooldown:
                self.multiple_start[z] = None
                continue
            if cooldown is not None and current_time >= cooldown:
                self.cooldown_until[z] = None
                self.multiple_logged[z] = False
            if count > 1:
                if self.multiple_start[z] is None:
                    self.multiple_start[z] = current_time
                    self.multiple_logged[z] = False
                    print(f"[{self.camera_id}] Zone {z + 1}: Multiple people detected. Timer started at {self._ts(current_time)}")
                elapsed = current_time - self.multiple_start[z]
                if elapsed >= self.multiple_limit and not self.multiple_logged[z]:
                    self.multiple_logged[z] = True
                    self.cooldown_until[z] = current_time + self.cooldown_seconds
                    ids = set(inside[z])
                    text = "|".join(str(i) for i in sorted(ids))
                    self.event_manager.log(self.camera_id, current_time, "MULTIPLE_PEOPLE_OVER_2_MINUTES", text, f"{count} people inside Zone {z + 1} for {elapsed:.1f} seconds; 30 minute cooldown started")
                    self.multiple_start[z] = None
            else:
                self.multiple_start[z] = None
                self.multiple_logged[z] = False
        self.number_inside = sum(len(v) for v in inside.values())
        return persons, inside

    @staticmethod
    def _ts(seconds):
        return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"

    def phone_allowed(self, track_id):
        return not self.track_states.get(track_id, {}).get("phone_logged", False)

    def mark_phone(self, track_id):
        if track_id in self.track_states:
            self.track_states[track_id]["phone_logged"] = True

    def reset_zones(self, zone_count):
        self.multiple_start = {i: None for i in range(zone_count)}
        self.cooldown_until = {i: None for i in range(zone_count)}
        self.multiple_logged = {i: False for i in range(zone_count)}
