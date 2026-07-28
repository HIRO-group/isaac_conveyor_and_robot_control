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
from sim_cell.stage_setup.box_pool import BoxPool

logger = logging.getLogger(__name__)

# ConveyorTrack's belt only fits 4 boxes without overlap under the placement
# clearance in _spawn_wave (confirmed empirically - a request of 5 clamps to 4
# on every wave), so 4 is the real ceiling here, not just a safety clamp.
INITIAL_WAVE_COUNT = 4
WAVE_COUNT_MIN = 1
WAVE_COUNT_MAX = 4

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


class BoxSpawner:
    """Spawns random waves of pool boxes onto one ConveyorZone whenever it empties."""

    def __init__(self, zone: ConveyorZone, box_rigid_prims: dict, pool: BoxPool, seed: int | None = None) -> None:
        self._zone = zone
        self._box_rigid_prims = box_rigid_prims
        self._pool = pool
        self._available = {variant: list(paths) for variant, paths in pool.paths_by_variant.items()}

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
        logger.info("box spawner seed=%d (override with %s)", seed, SEED_ENV_VAR)

        self._empty_since: float | None = None
        self._last_spawn_time: float | None = None
        # Parking is deferred to the first update() call (see its comment) rather
        # than done here in __init__.
        self._parked = False

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
            self._available[self._variant_of(path)].append(path)

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

        if self._last_spawn_time is None:
            # Belt starts empty - spawn the first wave immediately rather than
            # waiting out the debounce.
            return self._spawn_wave(sim_time, INITIAL_WAVE_COUNT)

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

        return self._spawn_wave(sim_time, self._rng.randint(WAVE_COUNT_MIN, WAVE_COUNT_MAX))

    def _variant_of(self, box_path: str) -> str:
        for variant, paths in self._pool.paths_by_variant.items():
            if box_path in paths:
                return variant
        raise KeyError(f"{box_path} is not a pool prim")

    def _total_available(self) -> int:
        return sum(len(paths) for paths in self._available.values())

    def _spawn_wave(self, sim_time: float, requested_count: int) -> list:
        count = min(requested_count, self._total_available())
        if count < requested_count:
            logger.warning("wave shrunk from %d to %d box(es) - pool exhausted", requested_count, count)
        if count == 0:
            return []

        variants_in_stock = [v for v, paths in self._available.items() if paths]
        max_half_diag = max(math.hypot(*self._pool.half_extents_by_variant[v][:2]) for v in variants_in_stock)
        min_slot_half = max_half_diag + SPAWN_SLOT_CLEARANCE_M
        travel_half_extent = self._bbox_half_extent[self._travel_axis]
        slot_half = travel_half_extent / count
        if slot_half < min_slot_half:
            fitting_count = max(1, min(count, int(travel_half_extent // min_slot_half)))
            logger.warning(
                "clamped wave from %d to %d box(es) - not enough belt length for non-overlapping slots",
                count, fitting_count,
            )
            count = fitting_count
            slot_half = travel_half_extent / count

        lateral_half_extent = self._bbox_half_extent[self._lateral_axis]
        spawned = []
        for slot_index in range(count):
            # Recomputed every slot (not just once per wave) - each pop() below
            # can exhaust a variant partway through a wave.
            variant = self._rng.choice([v for v, paths in self._available.items() if paths])
            path = self._available[variant].pop()
            hx, hy, hz = self._pool.half_extents_by_variant[variant]
            r = math.hypot(hx, hy)  # worst-case footprint radius at any yaw

            slot_center = -travel_half_extent + slot_half * (2 * slot_index + 1)
            jitter_room = max(0.0, slot_half - r - SPAWN_SLOT_CLEARANCE_M)
            travel_offset = self._rng.uniform(-jitter_room, jitter_room)
            lateral_room = max(0.0, lateral_half_extent - r)
            lateral_offset = self._rng.uniform(-lateral_room, lateral_room)

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
