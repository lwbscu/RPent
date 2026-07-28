import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from robots.behavior.planner_executor import RealCuroboBackend


def _pose2mat(pose):
    position, quaternion = pose
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    rotation = np.asarray(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(position, dtype=np.float64)
    return result


def _mat2pose(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix[:3, 3].copy(), np.asarray([0.0, 0.0, 0.0, 1.0])


class _Link:
    def __init__(self, prim_path, position):
        self.prim_path = prim_path
        self.position = np.asarray(position, dtype=np.float64)
        self.pose_reads = 0
        self.collision_meshes = {}

    def get_position_orientation(self):
        self.pose_reads += 1
        return self.position.copy(), np.asarray([0.0, 0.0, 0.0, 1.0])


class _Mesh:
    geom_type = "Mesh"

    def __init__(self, prim_path, link, local_position):
        self.prim_path = prim_path
        self.link = link
        self.local_position = np.asarray(local_position, dtype=np.float64)
        self.points = np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
            dtype=np.float32,
        )
        self.faces = np.asarray([[0, 1, 2]], dtype=np.int64)
        self.pose_reads = 0

    def get_position_orientation(self):
        self.pose_reads += 1
        return (
            self.link.get_position_orientation()[0] + self.local_position,
            np.asarray([0.0, 0.0, 0.0, 1.0]),
        )

    @staticmethod
    def get_world_scale():
        return np.ones(3, dtype=np.float32)


class _SceneObject:
    def __init__(self, *, kinematic_only, links):
        self.visual_only = False
        self.kinematic_only = kinematic_only
        self.links = links


class _RigidBodyView:
    def __init__(self, prim_paths, world_poses):
        self.prim_paths = list(prim_paths)
        self.world_poses = world_poses
        self.check_result = True
        self.count_override = None
        self.transforms_override = None
        self.get_transform_calls = 0

    @property
    def count(self):
        return (
            len(self.prim_paths)
            if self.count_override is None
            else self.count_override
        )

    def check(self):
        return self.check_result

    def get_transforms(self):
        self.get_transform_calls += 1
        if self.transforms_override is not None:
            return self.transforms_override
        return np.asarray(
            [self.world_poses[path] for path in self.prim_paths],
            dtype=np.float32,
        )


class _PhysicsSimulationView:
    def __init__(self, view):
        self.view = view
        self.create_calls = []

    def create_rigid_body_view(self, prim_paths):
        self.create_calls.append(list(prim_paths))
        return self.view


def _install_fake_omnigibson(monkeypatch, physics_sim_view):
    transform_module = ModuleType("omnigibson.utils.transform_utils")
    transform_module.pose2mat = _pose2mat
    transform_module.pose_inv = np.linalg.inv
    transform_module.mat2pose = _mat2pose
    utils_module = ModuleType("omnigibson.utils")
    utils_module.__path__ = []
    utils_module.transform_utils = transform_module
    lazy_module = ModuleType("omnigibson.lazy")
    og_module = ModuleType("omnigibson")
    og_module.__path__ = []
    og_module.sim = SimpleNamespace(
        floor_plane=None,
        currently_stepping=False,
        physics_sim_view=physics_sim_view,
    )
    og_module.lazy = lazy_module
    monkeypatch.setitem(sys.modules, "omnigibson", og_module)
    monkeypatch.setitem(sys.modules, "omnigibson.lazy", lazy_module)
    monkeypatch.setitem(sys.modules, "omnigibson.utils", utils_module)
    monkeypatch.setitem(
        sys.modules,
        "omnigibson.utils.transform_utils",
        transform_module,
    )
    return og_module


def _make_scene(monkeypatch):
    dynamic_a = _Link("/World/object_a/link", [1.0, 0.0, 0.0])
    dynamic_b = _Link("/World/object_b/link", [2.0, 0.0, 0.0])
    kinematic = _Link("/World/fixed/link", [3.0, 0.0, 0.0])
    mesh_a = _Mesh("/World/object_a/link/mesh", dynamic_a, [0.1, 0.0, 0.0])
    mesh_b = _Mesh("/World/object_b/link/mesh", dynamic_b, [0.2, 0.0, 0.0])
    mesh_k = _Mesh("/World/fixed/link/mesh", kinematic, [0.3, 0.0, 0.0])
    dynamic_a.collision_meshes = {"mesh": mesh_a}
    dynamic_b.collision_meshes = {"mesh": mesh_b}
    kinematic.collision_meshes = {"mesh": mesh_k}
    dynamic_object_a = _SceneObject(
        kinematic_only=False,
        links={"link": dynamic_a},
    )
    dynamic_object_b = _SceneObject(
        kinematic_only=False,
        links={"link": dynamic_b},
    )
    fixed_object = _SceneObject(
        kinematic_only=True,
        links={"link": kinematic},
    )
    root = _Link("/World/robot/root", [0.0, 0.0, 0.0])
    robot = SimpleNamespace(root_link=root)
    robot.scene = SimpleNamespace(
        objects=[robot, dynamic_object_b, fixed_object, dynamic_object_a]
    )
    world_poses = {
        dynamic_a.prim_path: np.asarray(
            [*dynamic_a.position, 0.0, 0.0, 0.0, 1.0]
        ),
        dynamic_b.prim_path: np.asarray(
            [*dynamic_b.position, 0.0, 0.0, 0.0, 1.0]
        ),
    }
    view = _RigidBodyView(
        [dynamic_b.prim_path, dynamic_a.prim_path],
        world_poses,
    )
    physics_sim_view = _PhysicsSimulationView(view)
    og_module = _install_fake_omnigibson(monkeypatch, physics_sim_view)
    return SimpleNamespace(
        generator=SimpleNamespace(robot=robot),
        dynamic_a=dynamic_a,
        dynamic_b=dynamic_b,
        kinematic=kinematic,
        root=root,
        meshes=(mesh_a, mesh_b, mesh_k),
        view=view,
        physics_sim_view=physics_sim_view,
        world_poses=world_poses,
        og=og_module,
    )


def test_full_snapshot_is_scalar_authoritative_and_builds_exact_shadow_view(
    monkeypatch,
):
    scene = _make_scene(monkeypatch)

    snapshot = RealCuroboBackend._current_collision_mesh_snapshot(
        scene.generator,
        full_digest=True,
    )

    assert scene.physics_sim_view.create_calls == [
        [scene.dynamic_a.prim_path, scene.dynamic_b.prim_path]
    ]
    assert snapshot["pose_batch_cache"]["physics_sim_view"] is (
        scene.physics_sim_view
    )
    assert snapshot["pose_batch_cache"]["view"] is scene.view
    assert snapshot["pose_batch_cache"]["index_by_prim_path"] == {
        scene.dynamic_b.prim_path: 0,
        scene.dynamic_a.prim_path: 1,
    }
    assert snapshot["pose_read"] == {
        "mode": "scalar_authoritative_shadow_batch",
        "count": 2,
        "fallback": False,
        "fallback_reason": None,
        "dynamic_link_count": 2,
        "scalar_link_count": 3,
    }
    poses = dict(snapshot["poses"])
    assert poses[scene.meshes[0].prim_path][:3] == pytest.approx([1.1, 0.0, 0.0])
    assert poses[scene.meshes[1].prim_path][:3] == pytest.approx([2.2, 0.0, 0.0])


def test_pose_only_uses_view_prim_path_mapping_and_scalar_residual(monkeypatch):
    scene = _make_scene(monkeypatch)
    first = RealCuroboBackend._current_collision_mesh_snapshot(
        scene.generator,
        full_digest=True,
    )
    dynamic_reads = (scene.dynamic_a.pose_reads, scene.dynamic_b.pose_reads)
    kinematic_reads = scene.kinematic.pose_reads
    mesh_reads = tuple(mesh.pose_reads for mesh in scene.meshes)
    scene.dynamic_a.position[:] = [4.0, 0.0, 0.0]
    scene.dynamic_b.position[:] = [5.0, 0.0, 0.0]
    scene.kinematic.position[:] = [6.0, 0.0, 0.0]
    scene.world_poses[scene.dynamic_a.prim_path] = np.asarray(
        [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )
    scene.world_poses[scene.dynamic_b.prim_path] = np.asarray(
        [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )

    second = RealCuroboBackend._current_collision_mesh_snapshot(
        scene.generator,
        full_digest=False,
        kinematic_cache=first["kinematic_cache"],
        pose_batch_cache=first["pose_batch_cache"],
    )

    assert scene.physics_sim_view.create_calls == [
        [scene.dynamic_a.prim_path, scene.dynamic_b.prim_path]
    ]
    assert (scene.dynamic_a.pose_reads, scene.dynamic_b.pose_reads) == dynamic_reads
    assert scene.kinematic.pose_reads == kinematic_reads + 1
    assert tuple(mesh.pose_reads for mesh in scene.meshes) == mesh_reads
    assert second["pose_read"] == {
        "mode": "batch_dynamic_scalar_residual",
        "count": 2,
        "fallback": False,
        "fallback_reason": None,
        "dynamic_link_count": 2,
        "scalar_link_count": 1,
    }
    poses = dict(second["poses"])
    assert poses[scene.meshes[0].prim_path][:3] == pytest.approx([4.1, 0.0, 0.0])
    assert poses[scene.meshes[1].prim_path][:3] == pytest.approx([5.2, 0.0, 0.0])
    assert poses[scene.meshes[2].prim_path][:3] == pytest.approx([6.3, 0.0, 0.0])


def test_invalid_cached_view_falls_back_to_current_scalar_pose(monkeypatch):
    scene = _make_scene(monkeypatch)
    first = RealCuroboBackend._current_collision_mesh_snapshot(
        scene.generator,
        full_digest=True,
    )
    scene.dynamic_a.position[:] = [8.0, 0.0, 0.0]
    scene.dynamic_b.position[:] = [9.0, 0.0, 0.0]
    scene.view.check_result = False

    second = RealCuroboBackend._current_collision_mesh_snapshot(
        scene.generator,
        full_digest=False,
        kinematic_cache=first["kinematic_cache"],
        pose_batch_cache=first["pose_batch_cache"],
    )

    assert second["pose_batch_cache"] is None
    assert second["pose_read"]["mode"] == "scalar_fallback"
    assert second["pose_read"]["fallback"] is True
    assert "rigid-body view is invalid" in second["pose_read"]["fallback_reason"]
    poses = dict(second["poses"])
    assert poses[scene.meshes[0].prim_path][:3] == pytest.approx([8.1, 0.0, 0.0])
    assert poses[scene.meshes[1].prim_path][:3] == pytest.approx([9.2, 0.0, 0.0])


def test_invalid_cached_view_rebuilds_after_one_scalar_fallback(monkeypatch):
    scene = _make_scene(monkeypatch)
    first = RealCuroboBackend._current_collision_mesh_snapshot(
        scene.generator,
        full_digest=True,
    )
    scene.view.check_result = False
    fallback = RealCuroboBackend._current_collision_mesh_snapshot(
        scene.generator,
        full_digest=False,
        kinematic_cache=first["kinematic_cache"],
        pose_batch_cache=first["pose_batch_cache"],
    )
    assert fallback["pose_batch_cache"] is None
    assert fallback["pose_read"]["mode"] == "scalar_fallback"

    scene.view.check_result = True
    scene.dynamic_a.position[:] = [10.0, 0.0, 0.0]
    scene.dynamic_b.position[:] = [11.0, 0.0, 0.0]
    scene.world_poses[scene.dynamic_a.prim_path] = np.asarray(
        [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )
    scene.world_poses[scene.dynamic_b.prim_path] = np.asarray(
        [11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )
    rebuilt = RealCuroboBackend._current_collision_mesh_snapshot(
        scene.generator,
        full_digest=False,
        kinematic_cache=fallback["kinematic_cache"],
        pose_batch_cache=fallback["pose_batch_cache"],
    )
    assert rebuilt["pose_batch_cache"] is not None
    assert rebuilt["pose_read"]["mode"] == (
        "scalar_authoritative_shadow_batch_rebuild"
    )
    assert rebuilt["pose_read"]["fallback"] is False

    scalar_reads = (scene.dynamic_a.pose_reads, scene.dynamic_b.pose_reads)
    scene.world_poses[scene.dynamic_a.prim_path] = np.asarray(
        [12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )
    scene.world_poses[scene.dynamic_b.prim_path] = np.asarray(
        [13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )
    recovered = RealCuroboBackend._current_collision_mesh_snapshot(
        scene.generator,
        full_digest=False,
        kinematic_cache=rebuilt["kinematic_cache"],
        pose_batch_cache=rebuilt["pose_batch_cache"],
    )
    assert recovered["pose_read"]["mode"] == "batch_dynamic_scalar_residual"
    assert recovered["pose_read"]["fallback"] is False
    assert (scene.dynamic_a.pose_reads, scene.dynamic_b.pose_reads) == scalar_reads
    poses = dict(recovered["poses"])
    assert poses[scene.meshes[0].prim_path][:3] == pytest.approx([12.1, 0.0, 0.0])
    assert poses[scene.meshes[1].prim_path][:3] == pytest.approx([13.2, 0.0, 0.0])


def test_fast_refresh_state_reuses_pose_batch_cache_and_reports_metrics(
    monkeypatch,
    tmp_path,
):
    scene = _make_scene(monkeypatch)

    class Checker:
        def __init__(self):
            self.tensor_args = object()
            self._env_mesh_names = [
                [[mesh.prim_path for mesh in scene.meshes][index] for index in range(3)]
            ]
            self._env_n_mesh = [3]
            self.pose_updates = []

        def update_obstacle_pose(self, **kwargs):
            self.pose_updates.append(kwargs)

    class MotionGenerator:
        def __init__(self, checker):
            self.world_coll_checker = checker
            self.graph_planner = SimpleNamespace(reset_buffer=lambda: None)

        @staticmethod
        def clear_world_cache():
            return None

    checker = Checker()
    scene.generator.mg = {"default": MotionGenerator(checker)}
    scene.generator.full_updates = 0

    def full_update(ignore_objects=None):
        assert ignore_objects is None
        scene.generator.full_updates += 1

    scene.generator.update_obstacles = full_update
    backend = RealCuroboBackend(None, output_dir=tmp_path)
    backend._install_fast_obstacle_refresh(
        scene.generator,
        world_collision_type=Checker,
        pose_factory=lambda pose, _tensor_args: tuple(pose),
    )

    scene.generator.update_obstacles()
    first_metrics = backend._obstacle_refresh_metrics(scene.generator)
    scene.dynamic_a.position[:] = [4.0, 0.0, 0.0]
    scene.dynamic_b.position[:] = [5.0, 0.0, 0.0]
    scene.world_poses[scene.dynamic_a.prim_path] = np.asarray(
        [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )
    scene.world_poses[scene.dynamic_b.prim_path] = np.asarray(
        [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )
    scene.generator.update_obstacles()
    second_metrics = backend._obstacle_refresh_metrics(scene.generator)

    assert scene.generator.full_updates == 1
    assert scene.physics_sim_view.create_calls == [
        [scene.dynamic_a.prim_path, scene.dynamic_b.prim_path]
    ]
    assert first_metrics["pose_read"]["mode"] == (
        "scalar_authoritative_shadow_batch"
    )
    assert second_metrics["pose_read"] == {
        "mode": "batch_dynamic_scalar_residual",
        "count": 2,
        "fallback": False,
        "fallback_reason": None,
        "dynamic_link_count": 2,
        "scalar_link_count": 1,
    }
    assert len(checker.pose_updates) == 3


def test_pose_batch_read_during_physics_step_fails_before_pose_reads(monkeypatch):
    scene = _make_scene(monkeypatch)
    first = RealCuroboBackend._current_collision_mesh_snapshot(
        scene.generator,
        full_digest=True,
    )
    scene.dynamic_a.position[:] = [7.0, 0.0, 0.0]
    scene.og.sim.currently_stepping = True
    reads = (
        scene.root.pose_reads,
        scene.dynamic_a.pose_reads,
        scene.dynamic_b.pose_reads,
        scene.kinematic.pose_reads,
    )

    with pytest.raises(RuntimeError, match="during a physics step"):
        RealCuroboBackend._current_collision_mesh_snapshot(
            scene.generator,
            full_digest=False,
            kinematic_cache=first["kinematic_cache"],
            pose_batch_cache=first["pose_batch_cache"],
        )
    assert reads == (
        scene.root.pose_reads,
        scene.dynamic_a.pose_reads,
        scene.dynamic_b.pose_reads,
        scene.kinematic.pose_reads,
    )


def test_pose_batch_read_on_worker_thread_fails_before_pose_reads(monkeypatch):
    scene = _make_scene(monkeypatch)
    first = RealCuroboBackend._current_collision_mesh_snapshot(
        scene.generator,
        full_digest=True,
    )
    scene.dynamic_b.position[:] = [11.0, 0.0, 0.0]
    result = {}

    def capture():
        try:
            RealCuroboBackend._current_collision_mesh_snapshot(
                scene.generator,
                full_digest=False,
                kinematic_cache=first["kinematic_cache"],
                pose_batch_cache=first["pose_batch_cache"],
            )
        except Exception as exc:
            result["error"] = exc

    worker = threading.Thread(target=capture)
    worker.start()
    worker.join(timeout=5.0)

    assert not worker.is_alive()
    assert isinstance(result["error"], RuntimeError)
    assert "main-thread dispatch" in str(result["error"])


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda view: setattr(view, "count_override", 1), "exact requested paths"),
        (
            lambda view: setattr(view, "prim_paths", ["/World/a", "/World/a"]),
            "exact requested paths",
        ),
        (
            lambda view: setattr(view, "transforms_override", np.zeros((2, 6))),
            r"shape \[N,7\]",
        ),
        (
            lambda view: setattr(
                view,
                "transforms_override",
                np.asarray(
                    [
                        [0.0, 0.0, 0.0, np.nan, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ),
            "not finite",
        ),
        (
            lambda view: setattr(
                view,
                "transforms_override",
                np.asarray(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ),
            "non-unit quaternion",
        ),
    ],
)
def test_pose_batch_rejects_invalid_view_results(monkeypatch, mutate, error):
    world_poses = {
        "/World/a": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        "/World/b": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    }
    view = _RigidBodyView(["/World/b", "/World/a"], world_poses)
    simulation_view = _PhysicsSimulationView(view)
    _install_fake_omnigibson(monkeypatch, simulation_view)
    mutate(view)

    with pytest.raises(RuntimeError, match=error):
        RealCuroboBackend._collision_link_pose_batch(
            physics_sim_view=simulation_view,
            requested_prim_paths=("/World/a", "/World/b"),
            pose_batch_cache=None,
            create=True,
        )
