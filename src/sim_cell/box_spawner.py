"""Wave-based random box spawning onto ConveyorTrack (loop1 zone 0).

Teleports parked `stage_setup.box_pool` prims onto the belt and re-enables
their rigid bodies; boxes `stage_setup.truck.despawn_boxes_in_truck` recycles
(already parked/disabled/hidden) become available again via `release()`. See
`stage_setup.box_pool` for why this is teleport-a-pool-prim rather than
create/delete a fresh one each time.
"""

from __future__ import annotations

import logging
import math
import os
import random

from conveyor_indexing.belt_geometry import compute_belt_bounds
from conveyor_indexing.zone import ConveyorZone
from sim_cell.faults import NEAR_FAR_LATERAL_FRACTIONS, SPAWN_LAYOUTS, near_far_lateral_offsets
from sim_cell.stage_setup.box_pool import BoxPool

logger = logging.getLogger(__name__)

# ConveyorTrack's belt (2.0 m along travel, 0.9 m across) fits 4 boxes in one
# row without overlap under the placement clearance in _spawn_wave (confirmed
# empirically - a request of 5 clamped to 4 on every wave). Waves larger than
# a row are laid out as a GRID (2026-09-10): a second lane across the belt
# takes the wave to 8. Each lane is 0.45 m wide against a 0.37 m worst-case
# diagonal footprint of the 26 cm box, and the single-row lateral jitter
# already spread boxes across the whole belt width, so two lanes stay inside
# the lateral distribution every collection was made with.
#
# Wave sizes are overridable per run (the evaluation wants a denser supply
# than the collections used); unset -> the historical 1..4 draw, so recorded
# seeds replay exactly.
INITIAL_WAVE_COUNT = 4
WAVE_COUNT_MIN = 1
WAVE_COUNT_MAX = 4
WAVE_MIN_ENV_VAR = "CONVEYOR_INDEXING_WAVE_MIN"
WAVE_MAX_ENV_VAR = "CONVEYOR_INDEXING_WAVE_MAX"
# lateral clearance between lanes of a multi-row wave (the along-travel
# clearance is SPAWN_SLOT_CLEARANCE_M as before)
LANE_CLEARANCE_M = 0.03

# ConveyorTrack must read empty this long before a new wave spawns - guards
# against spawning mid-settle, while a box from the previous wave is still
# bouncing through the occupancy column.
EMPTY_DEBOUNCE_S = 0.5
# Minimum gap between waves even if the zone reads empty sooner.
SPAWN_COOLDOWN_S = 1.0

# Dropped from just above the belt surface, inside compute_belt_bounds's
# occupancy column (OCCUPANCY_QUERY_HALF_HEIGHT above belt top) - so the very
# next control tick reads the zone occupied, and the empty-debounce can't
# double-fire a second wave on top of this one.
SPAWN_DROP_HEIGHT_M = 0.03
SPAWN_SLOT_CLEARANCE_M = 0.05

# Overridable for reproducible training runs; unset -> a fresh random seed
# each run, logged so any run can be replayed.
SEED_ENV_VAR = "CONVEYOR_INDEXING_SPAWN_SEED"
# Restrict spawning to the pool variants whose key contains this text, e.g.
# "21cm" for the small box only (demonstration takes). Unset spawns every
# variant.
BOX_VARIANT_ENV_VAR = "CONVEYOR_INDEXING_BOX_VARIANT"
# Stop spawning after this many automatic waves (demonstration takes that want
# exactly one box on the line). Unset spawns whenever the zone empties.
MAX_WAVES_ENV_VAR = "CONVEYOR_INDEXING_MAX_WAVES"


def wave_bounds_from_env(environ=None) -> tuple[int, int]:
    """(min, max) boxes per wave: WAVE_MIN_ENV_VAR / WAVE_MAX_ENV_VAR, else the
    historical 1..4. Validated so a typo cannot silently spawn nothing."""
    env = os.environ if environ is None else environ
    lo = int(env.get(WAVE_MIN_ENV_VAR, WAVE_COUNT_MIN))
    hi = int(env.get(WAVE_MAX_ENV_VAR, WAVE_COUNT_MAX))
    if lo < 1 or hi < lo:
        raise ValueError(f"{WAVE_MIN_ENV_VAR}={lo} {WAVE_MAX_ENV_VAR}={hi}: need 1 <= min <= max")
    return lo, hi


def grid_layout(count: int, travel_half_extent: float, lateral_half_extent: float, max_half_diag: float) -> list[int]:
    """How many boxes go in each lane across the belt for a wave of `count`.
    One lane holds as many along-travel slots as fit with the placement
    clearance; more boxes add lanes until the lanes themselves would
    overlap, at which point the wave is clamped. Returns the per-lane counts
    (sum <= count); the sum is the number that will actually be spawned."""
    min_slot_half = max_half_diag + SPAWN_SLOT_CLEARANCE_M
    per_lane = max(1, int(travel_half_extent // min_slot_half))
    max_lanes = max(1, int(lateral_half_extent // (max_half_diag + LANE_CLEARANCE_M)))
    lanes = min(max_lanes, math.ceil(count / per_lane))
    n = min(count, lanes * per_lane)
    base, extra = divmod(n, lanes)
    return [base + (1 if k < extra else 0) for k in range(lanes)]


class BoxSpawner:
    """Spawns random waves of pool boxes onto one ConveyorZone whenever it empties."""

    def __init__(
        self,
        zone: ConveyorZone,
        box_rigid_prims: dict,
        pool: BoxPool,
        seed: int | None = None,
        layout: str = "waves",
        near_side_sign: float = 1.0,
    ) -> None:
        """`layout` "waves" is the random wave draw; "near_far" (demonstrations,
        see sim_cell.faults) spawns one pair per wave, one box at the robot's
        belt edge and one at the far edge, with `near_side_sign` (+1/-1) the
        lateral direction toward the robot; "far" spawns the far box alone."""
        if layout not in SPAWN_LAYOUTS:
            raise ValueError(f"spawn layout {layout!r}: expected one of {SPAWN_LAYOUTS}")
        self._layout = layout
        self._near_side_sign = 1.0 if near_side_sign >= 0 else -1.0
        self._zone = zone
        self._box_rigid_prims = box_rigid_prims
        self._pool = pool
        self._available = {variant: list(paths) for variant, paths in pool.paths_by_variant.items()}
        variant_filter = os.environ.get(BOX_VARIANT_ENV_VAR, "").strip()
        if variant_filter:
            keep = {v: ps for v, ps in self._available.items() if variant_filter.lower() in v.lower()}
            if not keep:
                raise ValueError(f"{BOX_VARIANT_ENV_VAR}={variant_filter!r} matches none of {sorted(self._available)}")
            self._available = keep
            logger.info("box variants restricted to %s (%s=%s)", sorted(keep), BOX_VARIANT_ENV_VAR, variant_filter)

        belt_bounds = compute_belt_bounds(zone.belt_prim)
        self._belt_top_z = belt_bounds.belt_top_z
        travel = zone.world_travel_direction
        assert travel is not None, f"{zone.node_path}: BoxSpawner needs world_travel_direction set first"
        self._travel_axis = 0 if abs(travel[0]) >= abs(travel[1]) else 1
        self._lateral_axis = 1 - self._travel_axis
        self._bbox_center = zone.bbox_center
        self._bbox_half_extent = zone.bbox_half_extent

        if seed is None:
            env_seed = os.environ.get(SEED_ENV_VAR)
            seed = int(env_seed) if env_seed is not None else random.SystemRandom().randrange(2**32)
        self._rng = random.Random(seed)
        # Public (not just logged) so a run's ground-truth recording can carry
        # the exact seed needed to replay its box waves - see
        # sim_cell.recording.maybe_build_mcap_recorder's RunMetadata.
        self.seed = seed
        self.wave_min, self.wave_max = wave_bounds_from_env()
        # The first wave fills one row (the historical 4), but never more than
        # this run's own ceiling: a "sparse" run (1..1) asking for a 4-box
        # opening wave is not sparse for its first minute (2026-09-10).
        self.initial_wave_count = min(max(INITIAL_WAVE_COUNT, self.wave_min), self.wave_max)
        logger.info("box spawner seed=%d (override with %s); waves of %d..%d box(es) (override with %s/%s); layout %s",
                    seed, SEED_ENV_VAR, self.wave_min, self.wave_max, WAVE_MIN_ENV_VAR, WAVE_MAX_ENV_VAR, layout)

        self._empty_since: float | None = None
        self._last_spawn_time: float | None = None
        max_waves_raw = os.environ.get(MAX_WAVES_ENV_VAR, "").strip()
        self.max_waves: int | None = int(max_waves_raw) if max_waves_raw else None
        if self.max_waves is not None and self.max_waves < 1:
            raise ValueError(f"{MAX_WAVES_ENV_VAR}={max_waves_raw}: need >= 1")
        if self.max_waves is not None:
            logger.info("automatic waves capped at %d (%s)", self.max_waves, MAX_WAVES_ENV_VAR)
        self.waves_spawned = 0
        # Parking is deferred to the first update() call (see its comment) rather
        # than done here in __init__.
        self._parked = False
        # Scripted placement (2026-09-10, decision trials): an external client
        # can pause the automatic waves and place / clear boxes itself - see
        # `spawn_at`, `despawn`, and the "sim/boxes/command" topic in
        # external_command_bridge.py.
        self.auto_waves = True

    def _park_all_pool_boxes(self) -> None:
        """Disable and hide every pool box - same runtime disable mechanism
        stage_setup.truck.despawn_boxes_in_truck uses for recycled boxes (it only
        works as a toggle on an already-live PhysX actor, see box_pool.py's
        comment). No teleport needed here: box_pool.author_box_pool already
        placed each one at its own distinct park slot.

        Deliberately NOT called from __init__: disabling a rigid body before the
        physics scene has completed even one world.step() past world.reset()
        reliably throws "PxRigidDynamic::setLinearVelocity/setAngularVelocity:
        Not allowed if PxActorFlag::eDISABLE_SIMULATION is set!" for every box
        (confirmed by running the sim with an earlier version that parked eagerly
        in __init__, and by isolated repro scripts showing the same disable call
        on a freshly-reset scene is fine once at least one step has run first).
        The exact same set_enabled_rigid_bodies([False]) call made later, e.g. via
        despawn_boxes_in_truck mid-run, never hits this - so parking is deferred
        to the first update() call instead, which only ever runs from
        sim_cell.runner's main loop after world.step() has already executed at
        least once that iteration.
        """
        for path in self._pool.all_paths():
            rigid_prim = self._box_rigid_prims[path]
            rigid_prim.set_enabled_rigid_bodies([False])
            rigid_prim.set_visibilities([False])
        logger.info("parked %d pool box(es)", len(self._pool.all_paths()))

    def release(self, box_paths: list) -> None:
        """Return truck-recycled boxes (already parked by despawn_boxes_in_truck) to
        the pool so a future wave can reuse them.
        """
        for path in box_paths:
            variant = self._variant_of(path)
            if variant in self._available:  # excluded variants never spawn, so never come back
                self._available[variant].append(path)

    def update(self, sim_time: float, zone_occupied: bool) -> list:
        """Call once per control tick with ConveyorTrack's current occupancy.

        Returns this call's newly-spawned boxes as ``(path, variant, position,
        quat_wxyz)`` tuples (empty list on every tick that doesn't spawn a
        wave) - ground-truth recording (sim_cell.recording) uses this to emit
        BOX_EVENT_SPAWNED without needing its own spawn-detection logic.
        """
        if not self._parked:
            self._park_all_pool_boxes()
            self._parked = True

        if not self.auto_waves:
            return []
        if self.max_waves is not None and self.waves_spawned >= self.max_waves:
            return []

        if self._last_spawn_time is None:
            # Belt starts empty - spawn the first wave immediately rather than
            # waiting out the debounce.
            return self._spawn_wave(sim_time, self.initial_wave_count)

        if zone_occupied:
            self._empty_since = None
            return []
        if self._empty_since is None:
            self._empty_since = sim_time
        if sim_time - self._empty_since < EMPTY_DEBOUNCE_S:
            return []
        if sim_time - self._last_spawn_time < SPAWN_COOLDOWN_S:
            return []
        if not self._total_available():
            logger.debug("zone empty but pool exhausted - nothing to spawn")
            return []

        return self._spawn_wave(sim_time, self._rng.randint(self.wave_min, self.wave_max))

    def spawn_at(self, sim_time: float, x: float, y: float, yaw_rad: float = 0.0, variant: str | None = None) -> list:
        """Place ONE pool box at world (x, y) resting on the belt top, with the
        given yaw - the decision-trial protocol's exact placement. Returns the
        same `(path, variant, position, quat_wxyz)` list `update` does (one
        entry, or empty if the pool has nothing of that variant). Does not
        touch the wave state, so automatic waves (if enabled) continue as
        before. Callers normally pause them first (`auto_waves = False`)."""
        if not self._parked:
            self._park_all_pool_boxes()
            self._parked = True
        if variant is None:
            in_stock = [v for v, paths in self._available.items() if paths]
            if not in_stock:
                logger.warning("spawn_at: pool exhausted")
                return []
            variant = self._rng.choice(in_stock)
        if not self._available.get(variant):
            logger.warning("spawn_at: no %s left in the pool", variant)
            return []
        path = self._available[variant].pop()
        hz = self._pool.half_extents_by_variant[variant][2]
        position = (float(x), float(y), self._belt_top_z + hz + SPAWN_DROP_HEIGHT_M)
        quat_wxyz = (math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0))
        self._place(path, position, quat_wxyz)
        logger.info("spawn_at: %s at (%.2f, %.2f) yaw %.2f at t=%.2f", path, x, y, yaw_rad, sim_time)
        return [(path, variant, position, quat_wxyz)]

    def despawn(self, box_paths: list) -> list:
        """Park the given live boxes (same disable/hide/park mechanics as the
        truck and floor despawns) and return them to the pool. Returns the
        paths actually parked."""
        from sim_cell.stage_setup.truck import DESPAWNED_BOX_PARK_POSITION
        done = []
        for path in box_paths:
            rigid_prim = self._box_rigid_prims[path]
            rigid_prim.set_enabled_rigid_bodies([False])
            rigid_prim.set_visibilities([False])
            rigid_prim.set_world_poses(positions=[DESPAWNED_BOX_PARK_POSITION])
            done.append(path)
        self.release(done)
        if done:
            logger.info("despawned %d box(es) on request", len(done))
        return done

    def _variant_of(self, box_path: str) -> str:
        for variant, paths in self._pool.paths_by_variant.items():
            if box_path in paths:
                return variant
        raise KeyError(f"{box_path} is not a pool prim")

    def _total_available(self) -> int:
        return sum(len(paths) for paths in self._available.values())

    def _spawn_wave(self, sim_time: float, requested_count: int) -> list:
        spawned = self._spawn_wave_boxes(sim_time, requested_count)
        if spawned:
            self.waves_spawned += 1
        return spawned

    def _spawn_wave_boxes(self, sim_time: float, requested_count: int) -> list:
        if self._layout == "near_far":
            return self._spawn_lateral_row(sim_time, NEAR_FAR_LATERAL_FRACTIONS, "near/far pair")
        if self._layout == "far":
            return self._spawn_lateral_row(sim_time, NEAR_FAR_LATERAL_FRACTIONS[1:], "far box")
        count = min(requested_count, self._total_available())
        if count < requested_count:
            logger.warning("wave shrunk from %d to %d box(es) - pool exhausted", requested_count, count)
        if count == 0:
            return []

        variants_in_stock = [v for v, paths in self._available.items() if paths]
        max_half_diag = max(math.hypot(*self._pool.half_extents_by_variant[v][:2]) for v in variants_in_stock)
        travel_half_extent = self._bbox_half_extent[self._travel_axis]
        lateral_half_extent = self._bbox_half_extent[self._lateral_axis]
        lanes = grid_layout(count, travel_half_extent, lateral_half_extent, max_half_diag)
        if sum(lanes) < count:
            logger.warning(
                "clamped wave from %d to %d box(es) - not enough belt for non-overlapping slots (%d lane(s))",
                count, sum(lanes), len(lanes),
            )
            count = sum(lanes)
        lane_half = lateral_half_extent / len(lanes)

        spawned = []
        for lane_index, lane_count in enumerate(lanes):
            slot_half = travel_half_extent / lane_count
            # One lane is the historical layout exactly (same slots, same
            # jitter room, same RNG call order), so recorded seeds replay.
            if len(lanes) == 1:
                lane_center = 0.0
                lateral_room_cap = lateral_half_extent
            else:
                lane_center = -lateral_half_extent + lane_half * (2 * lane_index + 1)
                lateral_room_cap = lane_half - LANE_CLEARANCE_M
            for slot_index in range(lane_count):
                # Recomputed every slot (not just once per wave) - each pop() below
                # can exhaust a variant partway through a wave.
                variant = self._rng.choice([v for v, paths in self._available.items() if paths])
                path = self._available[variant].pop()
                hx, hy, hz = self._pool.half_extents_by_variant[variant]
                r = math.hypot(hx, hy)  # worst-case footprint radius at any yaw

                slot_center = -travel_half_extent + slot_half * (2 * slot_index + 1)
                jitter_room = max(0.0, slot_half - r - SPAWN_SLOT_CLEARANCE_M)
                travel_offset = self._rng.uniform(-jitter_room, jitter_room)
                lateral_room = max(0.0, lateral_room_cap - r)
                lateral_offset = lane_center + self._rng.uniform(-lateral_room, lateral_room)

                position_xyz = [0.0, 0.0, 0.0]
                position_xyz[self._travel_axis] = self._bbox_center[self._travel_axis] + slot_center + travel_offset
                position_xyz[self._lateral_axis] = self._bbox_center[self._lateral_axis] + lateral_offset
                position_xyz[2] = self._belt_top_z + hz + SPAWN_DROP_HEIGHT_M
                position = (position_xyz[0], position_xyz[1], position_xyz[2])

                yaw = self._rng.uniform(0.0, 2 * math.pi)
                quat_wxyz = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))

                self._place(path, position, quat_wxyz)
                spawned.append((path, variant, position, quat_wxyz))

        self._last_spawn_time = sim_time
        self._empty_since = None
        logger.info("spawned wave of %d box(es) at t=%.2f: %s", len(spawned), sim_time, spawned)
        return spawned

    def _spawn_lateral_row(self, sim_time: float, fractions: tuple, what: str) -> list:
        """One box per lateral fraction (sim_cell.faults.NEAR_FAR_LATERAL_FRACTIONS:
        a hair off the robot's belt edge, the far edge), side by side at the
        zone centre, yaw 0, so they stop together at the pick zone's stop line.
        The wave size request is ignored: the whole row or nothing."""
        n = len(fractions)
        if self._total_available() < n:
            logger.warning("%s needs %d box(es), pool has %d - nothing spawned", what, n, self._total_available())
            return []
        variants = []
        paths = []
        for _ in range(n):
            variant = self._rng.choice([v for v, ps in self._available.items() if ps])
            variants.append(variant)
            paths.append(self._available[variant].pop())
        half_extents = [self._pool.half_extents_by_variant[v] for v in variants]
        offsets = near_far_lateral_offsets(
            self._bbox_half_extent[self._lateral_axis], half_extents, self._near_side_sign, LANE_CLEARANCE_M,
            fractions=fractions,
        )
        spawned = []
        for path, variant, half, offset in zip(paths, variants, half_extents, offsets):
            position_xyz = [0.0, 0.0, 0.0]
            position_xyz[self._travel_axis] = self._bbox_center[self._travel_axis]
            position_xyz[self._lateral_axis] = self._bbox_center[self._lateral_axis] + offset
            position_xyz[2] = self._belt_top_z + half[2] + SPAWN_DROP_HEIGHT_M
            position = (position_xyz[0], position_xyz[1], position_xyz[2])
            quat_wxyz = (1.0, 0.0, 0.0, 0.0)
            self._place(path, position, quat_wxyz)
            spawned.append((path, variant, position, quat_wxyz))
        self._last_spawn_time = sim_time
        self._empty_since = None
        logger.info("spawned %s at t=%.2f: %s", what, sim_time, spawned)
        return spawned

    def _place(self, path: str, position: tuple, quat_wxyz: tuple) -> None:
        """Teleport + revive one pool box (inverse of despawn_boxes_in_truck)."""
        rigid_prim = self._box_rigid_prims[path]
        # set_world_poses is allowed on a disabled body; set_enabled_rigid_bodies
        # comes next.
        rigid_prim.set_world_poses(positions=[position], orientations=[quat_wxyz])
        rigid_prim.set_enabled_rigid_bodies([True])
        # RigidPrim has no public wake_up(): required here, not optional - without
        # it the body re-enables and its tensor/Fabric pose is correct (confirmed
        # via RigidPrim.get_world_poses()), but Hydra never picks up the new
        # transform and the box renders as if still at its pre-disable pose -
        # confirmed by rendering an actual frame with and without this call (see
        # scratchpad diagnose_render4.py / render5.py). It can occasionally log a
        # harmless internal PhysX rejection ("Not allowed if
        # PxActorFlag::eDISABLE_SIMULATION is set!" from wake_up's own internal
        # velocity-reset attempt racing the disable-flag clearing) - that's
        # cosmetic noise, not a functional issue; removing the call to silence it
        # is NOT an option, it breaks rendering.
        physics_view = rigid_prim._physics_rigid_body_view
        assert physics_view is not None, f"{path}: no physics tensor view to wake up"
        physics_view.wake_up()
        rigid_prim.set_visibilities([True])
