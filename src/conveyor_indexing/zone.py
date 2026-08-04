"""Single conveyor zone: bridges one ConveyorZoneStateMachine to its USD
ConveyorNode + belt bbox.
"""

from __future__ import annotations

import carb
from pxr import Gf, PhysxSchema, Usd, UsdGeom

from conveyor_indexing.belt_geometry import compute_belt_bounds
from conveyor_indexing.occupancy import overlap_box_prim_paths
from conveyor_indexing.state_machine import ConveyorZoneStateMachine

# Zone velocity at 100% speed; each loop scales its actual run speed down from
# this via a per-line run_speed_pct.
ZONE_RUN_VELOCITY = 2.0


class ConveyorZone:
    """Bridges one ConveyorZoneStateMachine to its USD ConveyorNode + belt bbox."""

    def __init__(
        self,
        index: int,
        node_path: str,
        stage: Usd.Stage,
        excluded_roots: tuple,
        run_speed_pct: int = 100,
    ) -> None:
        self.index = index
        self.node_path = node_path
        self._excluded_roots = excluded_roots
        self.node_prim = stage.GetPrimAtPath(node_path)
        if not self.node_prim.IsValid():
            raise RuntimeError(f"ConveyorNode prim not found at {node_path}")

        rel = self.node_prim.GetRelationship("inputs:conveyorPrim")
        targets = rel.GetTargets() if rel else []
        if not targets:
            raise RuntimeError(f"{node_path} has no inputs:conveyorPrim target")
        self.belt_prim = stage.GetPrimAtPath(targets[0])

        belt_bounds = compute_belt_bounds(self.belt_prim)
        self.bbox_half_extent = belt_bounds.bbox_half_extent
        self.bbox_center = belt_bounds.bbox_center
        # Belts are treated as axis-aligned and static; identity rotation.
        self._quat = carb.Float4(0.0, 0.0, 0.0, 1.0)

        # inputs:velocity is wired via a ReadVariable node to this per-track OmniGraph
        # variable rather than holding a plain value; apply_command authors it directly.
        self.velocity_var_attr = self.node_prim.GetParent().GetAttribute("graph:variable:Velocity")
        if not self.velocity_var_attr:
            raise RuntimeError(f"{node_path}'s graph has no graph:variable:Velocity")

        # inputs:direction is a unit vector for straight zones, an angular-velocity axis for
        # curved ones. The baked values aren't reliable (wrong sign/magnitude in places); see
        # conveyor_indexing.directions.fix_zone_directions, which rederives both from geometry.
        self.direction_attr = self.node_prim.GetAttribute("inputs:direction")
        baked_direction = self.direction_attr.Get()
        self.is_straight = baked_direction is not None and baked_direction[2] == 0.0

        # World-space travel direction for straight zones; set by
        # fix_zone_directions, only meaningful there (used by is_past_center).
        self.world_travel_direction: Gf.Vec3f | None = None

        self.state_machine = ConveyorZoneStateMachine(name=node_path, run_speed_pct=run_speed_pct)
        self.state_machine.start()

        # Per-tick cache: check_occupied(), the hold-zone stop-position check, and
        # (for pick zones) sim_cell.pick_dispatch.evaluate_pick_station all call
        # get_occupying_prim_paths() for the same zone within one control tick - up to
        # 3x the same PhysX overlap_box query per tick without this. Invalidated once
        # per tick by ConveyorLineController.step(), so the first call each tick still
        # queries PhysX and every later call that tick reuses its result.
        self._occupancy_cache: list | None = None

    def invalidate_occupancy_cache(self) -> None:
        self._occupancy_cache = None

    def get_occupying_prim_paths(self) -> list:
        """Return paths of every non-excluded rigid body overlapping this zone.

        Used for the boolean occupied check and, for the pick zone, to identify
        WHICH box actually triggered pick_ready rather than a hardcoded path.
        """
        if self._occupancy_cache is None:
            self._occupancy_cache = overlap_box_prim_paths(
                self.bbox_half_extent,
                self.bbox_center,
                self._quat,
                self._excluded_roots,
                zone_name=self.node_path,
            )
        return self._occupancy_cache

    def check_occupied(self) -> bool:
        return len(self.get_occupying_prim_paths()) > 0

    def is_past_center(self, world_position, stop_fraction: float = 0.5) -> bool:
        """True once world_position has passed stop_fraction of the way through this
        straight zone along world_travel_direction (0.5 = midpoint). Used by hold
        zones to settle an occupying part at a fixed, robot-reachable point.
        """
        travel = self.world_travel_direction
        assert travel is not None, f"{self.node_path}: is_past_center called before world_travel_direction was set"
        axis = 0 if abs(travel[0]) >= abs(travel[1]) else 1
        sign = 1.0 if travel[axis] > 0 else -1.0
        stop_point = self.bbox_center[axis] + sign * self.bbox_half_extent[axis] * (2 * stop_fraction - 1)
        return (world_position[axis] - stop_point) * sign >= 0.0

    def apply_command(self, run: bool, speed_pct: int) -> None:
        self.node_prim.GetAttribute("inputs:enabled").Set(run)
        if run:
            # Direction is baked in per-track; only magnitude is set here.
            self.velocity_var_attr.Set(ZONE_RUN_VELOCITY * speed_pct / 100.0)
        else:
            # enabled=False alone doesn't stop the belt (OgnIsaacConveyor leaves the
            # last nonzero surface velocity authored) - zero it directly instead.
            surface_velocity_api = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(self.belt_prim)
            zero = Gf.Vec3f(0.0, 0.0, 0.0)
            surface_velocity_api.GetSurfaceVelocityAttr().Set(zero)
            surface_velocity_api.GetSurfaceAngularVelocityAttr().Set(zero)
