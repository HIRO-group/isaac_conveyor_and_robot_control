"""Creates camera prims + render products + Replicator annotators for the
sim's camera rig, and captures RGB8 frames from them each tick.

This machine is CPU-throttled (see the top-level README's "Design" section),
so the capture path after rendering stays on the GPU wherever possible:
Replicator's "rgb" annotator is asked for its data on the `cuda` device (a
zero-copy Warp array), converted to a Torch tensor via `warp.torch.to_torch`
(also zero-copy), and the RGBA->RGB slice + contiguous-copy happen on the
GPU - only the final, already-RGB buffer crosses the PCIe bus to host
memory, once per frame. Falls back to the CPU annotator path (logged once)
if the CUDA/Torch interop isn't available in a given Isaac Sim install.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import omni.kit.app
import omni.replicator.core as rep
from pxr import Gf, Sdf, Usd, UsdGeom

from cameras.specs import CameraSpec

logger = logging.getLogger(__name__)

# Standard 35mm-style sensor width; combined with each CameraSpec's
# focal_length, this sets the field of view (see sim_cell.camera_layout's
# per-camera placement comments for the FOV math).
HORIZONTAL_APERTURE_MM = 20.955
CLIPPING_RANGE_M = (0.05, 100.0)


class CameraRig:
    """Owns the runtime-created camera prims, render products, and RGB
    annotators for every `CameraSpec` in the rig.
    """

    def __init__(self, stage: Usd.Stage, specs: list[CameraSpec]) -> None:
        # Defensive, matches sim_cell.stage_setup's pattern for
        # isaacsim.asset.gen.conveyor - not guaranteed enabled by the
        # resolved app config.
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)

        self._stage = stage
        self._entries: list[tuple[CameraSpec, rep.annotators.Annotator]] = []
        self._use_gpu = True
        self._warp_to_torch: Callable | None = None

        for spec in specs:
            self._create_camera_prim(spec)
            render_product = rep.create.render_product(spec.prim_path, (spec.width, spec.height))
            annotator = self._attach_annotator(render_product)
            self._entries.append((spec, annotator))

        logger.info("camera rig ready: %d camera(s), gpu_capture=%s", len(self._entries), self._use_gpu)

    def _attach_annotator(self, render_product):
        try:
            import warp.torch as warp_torch

            annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cuda")
            annotator.attach(render_product)
            self._warp_to_torch = warp_torch.to_torch
            return annotator
        except Exception:
            logger.warning(
                "GPU (cuda) annotator path unavailable in this Isaac Sim install - falling back to CPU "
                "capture (will use more CPU than intended, see README 'Design')",
                exc_info=True,
            )
            self._use_gpu = False
            annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            annotator.attach(render_product)
            return annotator

    def _ensure_ancestor_xform(self, prim_path: str) -> None:
        """`UsdGeom.Camera.Define` auto-creates missing ancestor prims, but
        as typeless scopes - fine for a hand cam (parent is the flange,
        which already exists via the UR20 reference) but not for a fresh
        grouping prim like `/World/Cameras` (see sim_cell.layout), which
        should behave as an ordinary, identity-transform Xform.
        """
        parent_path = Sdf.Path(prim_path).GetParentPath()
        if parent_path.isEmpty:
            return
        if not self._stage.GetPrimAtPath(parent_path).IsValid():
            UsdGeom.Xform.Define(self._stage, parent_path)

    def _create_camera_prim(self, spec: CameraSpec) -> None:
        self._ensure_ancestor_xform(spec.prim_path)
        camera = UsdGeom.Camera.Define(self._stage, spec.prim_path)

        xform_api = UsdGeom.XformCommonAPI(camera.GetPrim())
        xform_api.SetTranslate(Gf.Vec3d(*spec.translate))
        xform_api.SetRotate(Gf.Vec3f(*spec.rotation_euler_xyz_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)

        aspect = spec.height / spec.width
        camera.GetHorizontalApertureAttr().Set(HORIZONTAL_APERTURE_MM)
        camera.GetVerticalApertureAttr().Set(HORIZONTAL_APERTURE_MM * aspect)
        camera.GetFocalLengthAttr().Set(spec.focal_length)
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(*CLIPPING_RANGE_M))

    def capture_all(self) -> dict[str, bytes]:
        """Returns {serial: raw RGB8 bytes} for every camera with a ready
        frame this tick. A camera whose annotator hasn't produced its first
        frame yet (right after `attach()`) is silently skipped for that tick
        - it'll be ready on a subsequent call.
        """
        frames = {}
        for spec, annotator in self._entries:
            data = annotator.get_data()
            if data is None or tuple(data.shape) != (spec.height, spec.width, 4):
                continue

            if self._use_gpu:
                # Set together with _use_gpu in _attach_annotator - always non-None here.
                assert self._warp_to_torch is not None
                rgb = self._warp_to_torch(data)[..., :3].contiguous().cpu().numpy()
            else:
                rgb = np.ascontiguousarray(data[..., :3])
            rgb_bytes = rgb.tobytes()

            expected_len = spec.height * spec.width * 3
            if len(rgb_bytes) != expected_len:
                logger.warning(
                    "camera %s: unexpected frame size %d bytes (expected %d), dropping frame",
                    spec.serial, len(rgb_bytes), expected_len,
                )
                continue
            frames[spec.serial] = rgb_bytes
        return frames
