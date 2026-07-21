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


def test_planner_rpc_timeout_tracks_primitive_deadline_with_bounded_grace():
    rpc = _Rpc()
    client = BehaviorEnvClient(rpc, expected_meta={"seed": 211})

    client.move_to(hand="left", target_xyz=[0.0, 0.0, 0.0], timeout_s=45)
    client.navigate_to(hand="right", target_xyz=[1.0, 2.0, 3.0], timeout_s=90)

    assert rpc.calls[1][3] == 105.0
    assert rpc.calls[2][3] == 150.0
