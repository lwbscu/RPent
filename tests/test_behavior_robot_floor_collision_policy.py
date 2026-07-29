from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from robots.behavior.env_server import (
    _BEHAVIOR_NONFLOOR_COLLISION_GROUP,
    _BEHAVIOR_ROBOT_COLLISION_GROUP,
    _R1PRO_FLOOR_SUPPORT_LINK_NAMES,
    BehaviorEnvFacade,
)

GROUND_CATEGORIES = frozenset({"floors"})


class _Path:
    def __init__(self, value: str):
        self.pathString = value


class _Relation:
    def __init__(self, targets=()):
        self.targets = list(targets)

    def GetTargets(self):
        return [_Path(value) for value in self.targets]


class _Collection:
    def __init__(self):
        self.includes = _Relation()

    def GetIncludesRel(self):
        return self.includes


class _Group:
    def __init__(self, name: str, *, filter_self_collisions: bool):
        self.path = f"/World/collision_groups/{name}"
        self.collection = _Collection()
        self.filters = _Relation(
            [self.path] if filter_self_collisions else []
        )

    def GetCollidersCollectionAPI(self):
        return self.collection

    def GetFilteredGroupsRel(self):
        return self.filters


class _CollisionAPI:
    ACTIVE_COLLISION_GROUPS = {}

    @classmethod
    def reset(cls):
        cls.ACTIVE_COLLISION_GROUPS = {}

    @classmethod
    def create_collision_group(cls, name, filter_self_collisions=False):
        assert name not in cls.ACTIVE_COLLISION_GROUPS
        cls.ACTIVE_COLLISION_GROUPS[name] = _Group(
            name,
            filter_self_collisions=filter_self_collisions,
        )

    @classmethod
    def add_to_collision_group(cls, name, prim_path):
        cls.ACTIVE_COLLISION_GROUPS[
            name
        ].collection.includes.targets.append(prim_path)

    @classmethod
    def add_group_filter(cls, name, filter_name):
        cls.ACTIVE_COLLISION_GROUPS[name].filters.targets.append(
            f"/World/collision_groups/{filter_name}"
        )


class _Mesh:
    def __init__(self, enabled: bool):
        self.collision_enabled = enabled


class _Link:
    def __init__(self, prim_path: str, *, enabled: bool):
        self.prim_path = prim_path
        self.collision_meshes = {"collision": _Mesh(enabled)}
        self.enable_calls = 0
        self.disable_calls = 0

    def enable_collisions(self):
        self.enable_calls += 1
        for mesh in self.collision_meshes.values():
            mesh.collision_enabled = True

    def disable_collisions(self):
        self.disable_calls += 1
        for mesh in self.collision_meshes.values():
            mesh.collision_enabled = False


class R1Pro:
    name = "r1pro"
    prim_path = "/World/scene_0/robots/r1"

    def __init__(self):
        self.self_collisions = True
        self.floor_touching_base_link_names = list(
            _R1PRO_FLOOR_SUPPORT_LINK_NAMES
        )
        self.links = {
            "wheel_motor_link1": _Link(
                f"{self.prim_path}/wheel_motor_link1",
                enabled=False,
            ),
            "wheel_motor_link2": _Link(
                f"{self.prim_path}/wheel_motor_link2",
                enabled=False,
            ),
            "wheel_motor_link3": _Link(
                f"{self.prim_path}/wheel_motor_link3",
                enabled=False,
            ),
            "base_link": _Link(
                f"{self.prim_path}/base_link",
                enabled=True,
            ),
            "left_arm_link1": _Link(
                f"{self.prim_path}/left_arm_link1",
                enabled=True,
            ),
        }


def _object(path: str, category: str, *, collision_enabled: bool = True):
    return SimpleNamespace(
        prim_path=path,
        category=category,
        links={
            "root": _Link(
                f"{path}/root",
                enabled=collision_enabled,
            )
        },
    )


def _scene(robot: R1Pro, *, ground_collision_enabled: bool = True):
    floor = _object(
        "/World/scene_0/floors-main",
        "floors",
        collision_enabled=ground_collision_enabled,
    )
    carpet = _object("/World/scene_0/carpet-main", "carpet")
    table = _object("/World/scene_0/table-main", "breakfast_table")
    trash = _object("/World/scene_0/trash-main", "trash")
    return SimpleNamespace(
        objects=[robot, floor, carpet, table, trash],
    )


def test_collision_policy_keeps_ground_and_filters_nonfloor_world():
    _CollisionAPI.reset()
    robot = R1Pro()
    scene = _scene(robot)
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)

    report = facade._apply_behavior_robot_collision_policy(
        robot,
        scene=scene,
        collision_api=_CollisionAPI,
        ground_categories=GROUND_CATEGORIES,
    )

    assert robot.self_collisions is False
    support_links = {
        name: link
        for name, link in robot.links.items()
        if name in _R1PRO_FLOOR_SUPPORT_LINK_NAMES
    }
    non_support_links = {
        name: link
        for name, link in robot.links.items()
        if name not in _R1PRO_FLOOR_SUPPORT_LINK_NAMES
    }
    assert all(link.enable_calls == 1 for link in support_links.values())
    assert all(link.disable_calls == 0 for link in support_links.values())
    assert all(
        mesh.collision_enabled
        for link in support_links.values()
        for mesh in link.collision_meshes.values()
    )
    assert all(link.enable_calls == 0 for link in non_support_links.values())
    assert all(link.disable_calls == 1 for link in non_support_links.values())
    assert all(
        not mesh.collision_enabled
        for link in non_support_links.values()
        for mesh in link.collision_meshes.values()
    )
    robot_group = _CollisionAPI.ACTIVE_COLLISION_GROUPS[
        _BEHAVIOR_ROBOT_COLLISION_GROUP
    ]
    world_group = _CollisionAPI.ACTIVE_COLLISION_GROUPS[
        _BEHAVIOR_NONFLOOR_COLLISION_GROUP
    ]
    assert set(robot_group.collection.includes.targets) == {
        "/World/scene_0/robots/r1/wheel_motor_link1",
        "/World/scene_0/robots/r1/wheel_motor_link2",
        "/World/scene_0/robots/r1/wheel_motor_link3",
    }
    assert set(world_group.collection.includes.targets) == {
        "/World/scene_0/carpet-main",
        "/World/scene_0/table-main",
        "/World/scene_0/trash-main",
    }
    assert "/World/scene_0/floors-main" not in world_group.collection.includes.targets
    assert set(robot_group.filters.targets) == {
        f"/World/collision_groups/{_BEHAVIOR_ROBOT_COLLISION_GROUP}",
        f"/World/collision_groups/{_BEHAVIOR_NONFLOOR_COLLISION_GROUP}",
    }
    assert world_group.filters.targets == []
    assert report == {
        "support_links": 3,
        "support_collision_meshes": 3,
        "disabled_non_support_links": 2,
        "disabled_non_support_collision_meshes": 2,
        "ground_objects": 1,
        "ground_collision_meshes": 1,
        "nonfloor_objects": 3,
    }


def test_collision_policy_rejects_unverified_support_link_identity():
    _CollisionAPI.reset()
    robot = R1Pro()
    robot.floor_touching_base_link_names = [
        "wheel_motor_link1",
        "wheel_motor_link2",
    ]
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)

    with pytest.raises(RuntimeError, match="verified three-wheel set"):
        facade._apply_behavior_robot_collision_policy(
            robot,
            scene=_scene(robot),
            collision_api=_CollisionAPI,
            ground_categories=GROUND_CATEGORIES,
        )

    assert _CollisionAPI.ACTIVE_COLLISION_GROUPS == {}


def test_collision_policy_fails_without_enabled_ground_support():
    _CollisionAPI.reset()
    robot = R1Pro()
    scene = _scene(robot, ground_collision_enabled=False)
    scene.objects[2].links["root"].collision_meshes[
        "collision"
    ].collision_enabled = False
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)

    with pytest.raises(RuntimeError, match="no enabled collision mesh"):
        facade._apply_behavior_robot_collision_policy(
            robot,
            scene=scene,
            collision_api=_CollisionAPI,
            ground_categories=GROUND_CATEGORIES,
        )

    assert _CollisionAPI.ACTIVE_COLLISION_GROUPS == {}


def test_collision_policy_fails_without_ground_category_object():
    _CollisionAPI.reset()
    robot = R1Pro()
    scene = SimpleNamespace(
        objects=[
            robot,
            _object("/World/scene_0/table-main", "breakfast_table"),
        ]
    )
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)

    with pytest.raises(RuntimeError, match="no OmniGibson ground-category"):
        facade._apply_behavior_robot_collision_policy(
            robot,
            scene=scene,
            collision_api=_CollisionAPI,
            ground_categories=GROUND_CATEGORIES,
        )

    assert _CollisionAPI.ACTIVE_COLLISION_GROUPS == {}


def test_collision_policy_fails_closed_on_inexact_group_membership():
    class _DroppingCollisionAPI(_CollisionAPI):
        @classmethod
        def add_to_collision_group(cls, name, prim_path):
            if prim_path.endswith("/trash-main"):
                return
            super().add_to_collision_group(name, prim_path)

    _DroppingCollisionAPI.reset()
    robot = R1Pro()
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)

    with pytest.raises(RuntimeError, match="membership verification failed"):
        facade._apply_behavior_robot_collision_policy(
            robot,
            scene=_scene(robot),
            collision_api=_DroppingCollisionAPI,
            ground_categories=GROUND_CATEGORIES,
        )


def test_collision_configuration_uses_stopped_simulator_transaction(monkeypatch):
    _CollisionAPI.reset()
    robot = R1Pro()
    scene = _scene(robot)

    class _Simulator:
        state = "playing"
        calls = []

        @classmethod
        def is_playing(cls):
            return cls.state == "playing"

        @classmethod
        def is_paused(cls):
            return cls.state == "paused"

        @classmethod
        def is_stopped(cls):
            return cls.state == "stopped"

        @classmethod
        def stop(cls):
            cls.calls.append("stop")
            cls.state = "stopped"

        @classmethod
        def play(cls):
            cls.calls.append("play")
            cls.state = "playing"

        @classmethod
        def pause(cls):
            cls.calls.append("pause")
            cls.state = "paused"

    _Simulator.state = "playing"
    _Simulator.calls = []
    og_module = ModuleType("omnigibson")
    og_module.sim = _Simulator
    utils_module = ModuleType("omnigibson.utils")
    utils_module.__path__ = []
    constants_module = ModuleType("omnigibson.utils.constants")
    constants_module.GROUND_CATEGORIES = GROUND_CATEGORIES
    usd_utils_module = ModuleType("omnigibson.utils.usd_utils")
    usd_utils_module.CollisionAPI = _CollisionAPI
    monkeypatch.setitem(sys.modules, "omnigibson", og_module)
    monkeypatch.setitem(sys.modules, "omnigibson.utils", utils_module)
    monkeypatch.setitem(
        sys.modules,
        "omnigibson.utils.constants",
        constants_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "omnigibson.utils.usd_utils",
        usd_utils_module,
    )

    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._env = SimpleNamespace(
        omnigibson_env=SimpleNamespace(scene=scene),
    )
    facade._configure_behavior_robot_collisions(robot)

    assert _Simulator.calls == ["stop", "play"]
    assert _Simulator.state == "playing"
    assert robot.self_collisions is False
    assert (
        _BEHAVIOR_NONFLOOR_COLLISION_GROUP
        in _CollisionAPI.ACTIVE_COLLISION_GROUPS
    )


def test_collision_configuration_failure_keeps_simulator_stopped(monkeypatch):
    _CollisionAPI.reset()
    robot = R1Pro()
    scene = SimpleNamespace(
        objects=[
            robot,
            _object("/World/scene_0/table-main", "breakfast_table"),
        ]
    )

    class _Simulator:
        state = "playing"
        calls = []

        @classmethod
        def is_playing(cls):
            return cls.state == "playing"

        @classmethod
        def is_paused(cls):
            return cls.state == "paused"

        @classmethod
        def is_stopped(cls):
            return cls.state == "stopped"

        @classmethod
        def stop(cls):
            cls.calls.append("stop")
            cls.state = "stopped"

        @classmethod
        def play(cls):
            cls.calls.append("play")
            cls.state = "playing"

        @classmethod
        def pause(cls):
            cls.calls.append("pause")
            cls.state = "paused"

    og_module = ModuleType("omnigibson")
    og_module.sim = _Simulator
    utils_module = ModuleType("omnigibson.utils")
    utils_module.__path__ = []
    usd_utils_module = ModuleType("omnigibson.utils.usd_utils")
    usd_utils_module.CollisionAPI = _CollisionAPI
    monkeypatch.setitem(sys.modules, "omnigibson", og_module)
    monkeypatch.setitem(sys.modules, "omnigibson.utils", utils_module)
    monkeypatch.setitem(
        sys.modules,
        "omnigibson.utils.usd_utils",
        usd_utils_module,
    )

    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._env = SimpleNamespace(
        omnigibson_env=SimpleNamespace(scene=scene),
    )

    with pytest.raises(
        RuntimeError,
        match="BEHAVIOR R1Pro collision configuration failed",
    ):
        facade._configure_behavior_robot_collisions(robot)

    assert _Simulator.calls == ["stop"]
    assert _Simulator.state == "stopped"
