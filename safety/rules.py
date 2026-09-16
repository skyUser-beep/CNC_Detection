from tracking.track_state import (deduplicate_persons,find_previous_track,match_locked_person)

class SafetyRules:
    def __init__(self, camera_id, zones_count, event_manager, config):

        self.camera_id = camera_id
        self.event_manager = event_manager

        self.multiple_limit = config.multiple_person_limit_seconds
        self.multiple_limit_seconds = config.multiple_person_limit_seconds
        self.cooldown_seconds = config.multiple_person_cooldown_seconds


        self.absence_limit = config.absence_limit_seconds

        # Small grace period for temporary detector loss.
        self.track_grace = config.track_grace_seconds

        # Grace period for inside-zone state.
        self.inside_grace = config.inside_grace_seconds

        self.switch_max = config.track_id_switch_max_seconds
        self.switch_center = config.track_id_switch_center_ratio
        self.switch_iou = config.track_id_switch_iou_threshold

        self.duplicate_iou = config.duplicate_iou_threshold
        self.duplicate_center = config.duplicate_center_ratio

        self.track_states = {}

        self.multiple_start = {
            i: None for i in range(zones_count)
        }
        self.cooldown_until = {
            i: None for i in range(zones_count)
        }
        self.multiple_logged = {
            i: False for i in range(zones_count)
        }

        self.next_person_key = 1
        self.zone_locks = { i: [] for i in range(zones_count)
        }
        self.number_inside = 0

    def update(self, persons, current_time, zone_count):
        events = []
        # Make sure all required zone locks exist.
        for z in range(zone_count):
            self.zone_locks.setdefault(z, [])
        persons, suppressed = deduplicate_persons(persons,
            self.duplicate_iou,self.duplicate_center
        )

        for track_id in suppressed:
            self.track_states.pop(track_id, None)
        # Current tracker IDs
        active_ids = { p["id"] for p in persons}

        for person in persons:
            track_id = person["id"]
            zone = person["zone"]
            if track_id not in self.track_states:
                old_id = find_previous_track(self.track_states,
                    person,current_time,
                    active_ids,self.switch_max,
                    self.switch_center,self.switch_iou
                )
                if old_id is not None:
                    self.track_states[track_id] = self.track_states.pop(old_id).copy()
                    print(f"[{self.camera_id}] Track ID changed {old_id} -> {track_id}")
                else:
                    self.track_states[track_id] = {
                        "last_seen": current_time,
                        "last_inside": (
                            current_time
                            if zone is not None
                            else None
                        ),
                        "last_zone": zone,
                        "absence_logged": False,
                        "phone_logged": False,
                        "last_box": (
                            person["x1"],person["y1"],
                            person["x2"],person["y2"]
                        ),
                    }

            state = self.track_states[track_id]
            state["last_seen"] = current_time
            state["last_box"] = (person["x1"], person["y1"],
                person["x2"],person["y2"]
            )

            if zone is not None:
                state["last_inside"] = current_time
                state["last_zone"] = zone
                state["absence_logged"] = False

        inside = {z: set() for z in range(zone_count)}

        for z in range(zone_count):
            current_people = [person
                for person in persons
                if person["zone"] == z]

            used_ids = set()
            locks = self.zone_locks[z]

            for lock in locks:
                matched = match_locked_person(lock["box"],
                    current_people,used_ids=used_ids,
                    center_ratio=self.switch_center,
                    iou_threshold=self.switch_iou
                )

                if matched is not None:
                    old_track_id = lock["track_id"]
                    new_track_id = matched["id"]

                    if old_track_id != new_track_id:
                        print(f"[{self.camera_id}] Zone {z + 1}: "
                            f"Locked person ID changed {old_track_id} -> "
                            f"{new_track_id}")

                    lock["track_id"] = new_track_id
                    lock["box"] = (matched["x1"],matched["y1"],
                        matched["x2"],matched["y2"])

                    lock["last_seen"] = current_time
                    lock["inside"] = True
                    # Person is present again.
                    lock["away_start"] = None
                    lock["away_logged"] = False
                    used_ids.add(new_track_id)
                    # This person counts as present.
                    inside[z].add(new_track_id)
                    continue

                missing_for = current_time - lock["last_seen"]

                if missing_for <= self.track_grace:
                    continue

                if lock["away_start"] is None:
                    lock["away_start"] = current_time

                away_duration = current_time -lock["away_start"]

                if away_duration >= self.absence_limit and not lock["away_logged"]:
                    event_type = "PERSON_AWAY_OVER_5_MINUTES"
                    details = (f"Locked person "
                        f"{lock['person_key']} "
                        f"outside zone {z + 1} "
                        f"for {away_duration:.1f} seconds"
                    )
                    self.event_manager.log(self.camera_id,current_time,
                        event_type,lock["track_id"],
                        details
                    )
                    events.append({ "event_type": event_type,
                        "track_ids": {lock["track_id"]
                        },
                        "details": details,
                        "zone": z
                    })

                    lock["away_logged"] = True
                    print(f"[{self.camera_id}] Zone {z + 1}: "
                        f"PERSON AWAY EVENT | Person Key: "
                        f"{lock['person_key']} | Track ID: "
                        f"{lock['track_id']}"
                    )
            for person in current_people:
                track_id = person["id"]
                if track_id in used_ids:
                    continue
                # Create a new physical-person lock.
                person_key = self.next_person_key
                self.next_person_key += 1
                lock = {
                    "person_key": person_key,
                    "track_id": track_id,
                    "box": (
                        person["x1"], person["y1"],
                        person["x2"],person["y2"]
                    ),
                    "last_seen": current_time,
                    "away_start": None,
                    "away_logged": False,
                    "inside": True
                }
                locks.append(lock)
                used_ids.add(track_id)
                inside[z].add(track_id)

                print(f"[{self.camera_id}] Zone {z + 1}: "
                    f"Person locked | Person Key: {person_key} | Track ID: {track_id}")

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

                    print(f"[{self.camera_id}] Zone {z + 1}: Multiple people detected. "
                        f"Timer started at {self._ts(current_time)}")

                elapsed = current_time -  self.multiple_start[z]

                if  elapsed >= self.multiple_limit and not self.multiple_logged[z]:
                    self.multiple_logged[z] = True
                    event_type = "MULTIPLE_PEOPLE_OVER_2_MINUTES"
                    details = (f"{count} people inside "
                        f"zone {z + 1} "
                        f"for {elapsed:.1f} seconds; "
                        f"30 minutes cooldown started"
                    )

                    self.cooldown_until[z] = current_time +self.cooldown_seconds
                    ids = set(inside[z])
                    text = "|".join(str(i)
                        for i in sorted(ids)
                    )
                    self.event_manager.log( self.camera_id,
                        current_time,event_type,
                        text,details
                    )
                    events.append({"event_type": event_type,
                        "track_ids": ids,"details": details,
                        "zone": z
                    })
                    print(f"[{self.camera_id}] Zone {z + 1}: MULTIPLE PEOPLE EVENT | IDs: {text}")
                    self.multiple_start[z] = None
            else:
                self.multiple_start[z] = None
                self.multiple_logged[z] = False
        self.number_inside = sum(len(v)
            for v in inside.values()
        )
        return persons, inside, events

    @staticmethod
    def _ts(seconds):
        return (
            f"{int(seconds // 3600):02d}:"
            f"{int((seconds % 3600) // 60):02d}:"
            f"{int(seconds % 60):02d}"
        )

    def phone_allowed(self, track_id):
        return not (self.track_states
            .get(track_id, {})
            .get("phone_logged", False)
        )

    def mark_phone(self, track_id):

        if track_id in self.track_states:
            self.track_states[track_id]["phone_logged"] = True

    def reset_zones(self, zone_count):
        self.multiple_start = {i: None
            for i in range(zone_count)
        }
        self.cooldown_until = {i: None
            for i in range(zone_count)
        }
        self.multiple_logged = {i: False for i in range(zone_count)}

        # Reset physical-person locks too.
        self.zone_locks = {i: []for i in range(zone_count)}
        self.next_person_key = 1