from conveyor_indexing.topics import PREFIX_ENV, Topics


def test_default_prefix_keys():
    t = Topics()
    assert t.arm_state(1) == "sim/arm/1/state"
    assert t.arm_action(2) == "sim/arm/2/action_command"
    assert t.camera_color("SIM1-PICK") == "sim/camera/SIM1-PICK/color"
    assert t.conveyor_state == "sim/conveyor/state"
    assert t.boxes_state == "sim/boxes/state"
    assert t.clock == "sim/clock"


def test_prefix_from_env(monkeypatch):
    monkeypatch.setenv(PREFIX_ENV, "/cell7/")
    assert Topics.from_env().camera_list == "cell7/camera/list"
