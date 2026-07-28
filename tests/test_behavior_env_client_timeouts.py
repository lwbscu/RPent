import pytest

from robots.behavior.env_client import BehaviorEnvClient


class _Rpc:
    def __init__(self):
        self.calls = []

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append((method, args, kwargs or {}, timeout_s))
        if method == "env.get_env_meta":
            return {"seed": 211}
        return {
            "primitive_success": False,
            "task_success": False,
            "stop_reason": "timeout",
        }

    def close(self):
        pass


class _Pi0Rpc(_Rpc):
    def __init__(self, results):
        super().__init__()
        self.results = list(results)

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append((method, args, kwargs or {}, timeout_s))
        if method == "env.get_env_meta":
            return {"seed": 211}
        if method == "env.pi0_nav_pick_chunk_step":
            return self.results.pop(0)
        return {
            "primitive_success": False,
            "task_success": False,
            "stop_reason": "timeout",
        }


def _visual_hand_check(hand: str) -> dict[str, str]:
    return {
        "camera": "head",
        "frame_id": f"head:current:{hand}",
        "selected_hand": hand,
        "assessment": "selected_hand_visually_confirmed",
    }


def _depth_probe() -> dict[str, object]:
    return {
        "frame_id": "head:12:selected-target",
        "u": 320,
        "v": 240,
        "depth_window_px": 7,
        "assessment": "target_point_visually_confirmed",
    }


def test_observe_depth_probe_uses_read_only_rpc_timeout_and_exact_kwargs():
    rpc = _Rpc()
    client = BehaviorEnvClient(rpc, expected_meta={"seed": 211})

    client.observe(camera="head", depth_probe=_depth_probe())

    assert rpc.calls[1] == (
        "env.observe",
        (),
        {
            "camera": "head",
            "depth_probe": _depth_probe(),
        },
        120.0,
    )


def test_observe_client_rejects_review_and_depth_probe_before_rpc():
    rpc = _Rpc()
    client = BehaviorEnvClient(rpc, expected_meta={"seed": 211})

    try:
        client.observe(
            camera="head",
            frame_review={
                "frame_id": "head:12:selected-target",
                "assessment": "side_or_indeterminate",
            },
            depth_probe=_depth_probe(),
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("observe accepted frame_review and depth_probe together")

    assert len(rpc.calls) == 1


def test_pi0_raw_success_marks_episode_done_and_rejects_second_chunk_before_rpc():
    observation = {"states": [0.0]}
    rpc = _Pi0Rpc(
        [
            (
                observation,
                1.0,
                False,
                False,
                {
                    "done": {
                        "success": True,
                        "termination_conditions": {
                            "timeout": {"done": False, "success": False},
                            "predicate": {"done": True, "success": True},
                        },
                    },
                    "_rpent": {"total_env_steps": 32},
                },
            ),
            (
                observation,
                0.0,
                True,
                False,
                {
                    "done": {
                        "success": False,
                        "termination_conditions": {
                            "timeout": {"done": False, "success": False},
                            "predicate": {"done": True, "success": False},
                        },
                    },
                    "_rpent": {"total_env_steps": 37},
                },
            ),
        ]
    )
    client = BehaviorEnvClient(rpc, expected_meta={"seed": 211})
    actions = [[0.0] * 23 for _ in range(32)]

    client.pi0_nav_pick_chunk_step(actions, chunk_index=1)

    assert client.episode_done is True
    assert client._official_success_latched is True
    assert client.total_env_steps == 32

    with pytest.raises(RuntimeError, match="called after episode done"):
        client.pi0_nav_pick_chunk_step(actions, chunk_index=2)

    assert client.episode_done is True
    assert client.total_env_steps == 32
    assert len(rpc.results) == 1
    assert [method for method, *_rest in rpc.calls].count(
        "env.pi0_nav_pick_chunk_step"
    ) == 1


@pytest.mark.parametrize(
    "invoke",
    [
        lambda client: client.dashboard_prepare_manual_command(
            target="chassis",
            action="forward",
        ),
        lambda client: client.dashboard_execute_prepared_command(
            plan_id="plan-1",
            command_id="command-1",
        ),
        lambda client: client.dashboard_capture_views(
            command_id="capture-1",
        ),
    ],
)
def test_raw_success_rejects_dashboard_pipeline_before_transport(invoke):
    rpc = _Rpc()
    client = BehaviorEnvClient(rpc, expected_meta={"seed": 211})
    client._official_success_latched = True
    client.episode_done = True

    with pytest.raises(RuntimeError, match="terminal"):
        invoke(client)

    assert [method for method, *_rest in rpc.calls] == ["env.get_env_meta"]


def test_neutral_rpc_timeout_tracks_deadline_with_bounded_grace():
    rpc = _Rpc()
    client = BehaviorEnvClient(rpc, expected_meta={"seed": 211})

    client.move_to(
        hand="left",
        target={"delta_xyz": [0.0, 0.0, 0.1], "frame": "world"},
        visual_hand_check=_visual_hand_check("left"),
        timeout_s=45,
    )
    client.press(
        hand="right",
        projection_id="fresh-projection",
        travel_m=0.03,
        visual_hand_check=_visual_hand_check("right"),
        timeout_s=90,
    )

    assert rpc.calls[1] == (
        "env.move_to",
        (),
        {
            "hand": "left",
            "target": {"delta_xyz": [0.0, 0.0, 0.1], "frame": "world"},
            "visual_hand_check": _visual_hand_check("left"),
            "position_tolerance_m": 0.02,
            "max_travel_m": 0.25,
            "timeout_s": 45,
        },
        105.0,
    )
    assert rpc.calls[2] == (
        "env.press",
        (),
        {
            "hand": "right",
            "projection_id": "fresh-projection",
            "travel_m": 0.03,
            "visual_hand_check": _visual_hand_check("right"),
            "timeout_s": 90,
        },
        150.0,
    )


def test_navigate_to_rpc_is_projection_bound_base_only_and_has_bounded_grace():
    rpc = _Rpc()
    client = BehaviorEnvClient(rpc, expected_meta={"seed": 211})
    visual_check = {
        "camera": "head",
        "frame_id": "head:12:navigation-target",
        "assessment": "navigation_target_visually_confirmed",
    }

    client.navigate_to(
        projection_id=" projection-current ",
        navigation_visual_check=visual_check,
        standoff_m=0.9,
        max_travel_m=1.25,
        timeout_s=300,
    )

    assert rpc.calls[1] == (
        "env.navigate_to",
        (),
        {
            "projection_id": "projection-current",
            "navigation_visual_check": visual_check,
            "standoff_m": 0.9,
            "max_travel_m": 1.25,
            "timeout_s": 300.0,
        },
        360.0,
    )
    assert {
        "hand",
        "role",
        "target_xyz",
        "delta_xyz",
        "frame",
        "chunks",
        "max_chunks",
    }.isdisjoint(rpc.calls[1][2])


def test_navigate_to_rpc_supports_relative_base_rotation_without_projection():
    rpc = _Rpc()
    client = BehaviorEnvClient(rpc, expected_meta={"seed": 211})

    client.navigate_to(
        relative_motion={
            "kind": "rotation",
            "direction": "left",
            "angle_deg": 90,
        },
        timeout_s=120,
    )

    assert rpc.calls[1] == (
        "env.navigate_to",
        (),
        {
            "relative_motion": {
                "kind": "rotation",
                "direction": "left",
                "angle_deg": 90.0,
            },
            "timeout_s": 120.0,
        },
        180.0,
    )


def test_close_open_use_neutral_rpc_names_and_transport_close_is_separate():
    rpc = _Rpc()
    client = BehaviorEnvClient(rpc, expected_meta={"seed": 211})

    client.close(
        hand="right",
        visual_hand_check=_visual_hand_check("right"),
        timeout_s=10,
    )
    client.open(
        hand="left",
        visual_hand_check=_visual_hand_check("left"),
        release_visual_check={
            "camera": "head",
            "frame_id": "head:release:left",
            "selected_hand": "left",
            "assessment": "attached_object_fully_inside_receptacle_opening",
        },
        timeout_s=20,
    )
    client.close_transport()

    assert rpc.calls[1][0] == "env.close"
    assert rpc.calls[1][2] == {
        "hand": "right",
        "visual_hand_check": _visual_hand_check("right"),
        "timeout_s": 10,
    }
    assert rpc.calls[1][3] == 70.0
    assert rpc.calls[2][0] == "env.open"
    assert rpc.calls[2][2] == {
        "hand": "left",
        "visual_hand_check": _visual_hand_check("left"),
        "release_visual_check": {
            "camera": "head",
            "frame_id": "head:release:left",
            "selected_hand": "left",
            "assessment": "attached_object_fully_inside_receptacle_opening",
        },
        "timeout_s": 20,
    }
    assert rpc.calls[2][3] == 80.0


def test_legacy_role_and_semantic_hand_values_are_rejected_without_rpc():
    rpc = _Rpc()
    client = BehaviorEnvClient(rpc, expected_meta={"seed": 211})
    baseline_calls = len(rpc.calls)

    for legacy_hand in ("held", "press"):
        try:
            client.close(
                hand=legacy_hand,
                visual_hand_check=_visual_hand_check("left"),
            )
        except ValueError as exc:
            assert "hand must be" in str(exc)
        else:
            raise AssertionError(f"legacy hand {legacy_hand!r} was accepted")
    try:
        client.close(
            role="held",
            visual_hand_check=_visual_hand_check("left"),
        )
    except TypeError:
        pass
    else:
        raise AssertionError("legacy role keyword was accepted")

    assert len(rpc.calls) == baseline_calls
