from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_xhand_zeva import (
    action_policy_xhand_zeva,
)
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_xhand_zeva_tactile import (
    action_policy_xhand_zeva_tactile,
)
from cosmos_framework.model.zeva.tactile_encoder import FrozenTactileEncoderWithProjector
from cosmos_framework.model.zeva.tactile_memory import TactileBehaviorHead


def test_tactile_train_and_validation_share_the_input_contract() -> None:
    recipe = action_policy_xhand_zeva_tactile
    for loader_name in ("dataloader_train", "dataloader_val"):
        dataset = recipe[loader_name]["dataloader"]["datasets"]["xhand"]["dataset"]
        assert dataset["use_tactile"] is True
        assert dataset["tactile_memory_steps"] == recipe["model"]["config"]["behavior_stage2"]["tactile_memory_steps"]
        assert dataset["state_mode"] == "joint18"
        baseline = action_policy_xhand_zeva[loader_name]["dataloader"]["datasets"]["xhand"]["dataset"]
        assert not baseline.get("use_tactile", False)
    assert recipe["dataloader_val"]["restart_on_iter"] is True
    assert recipe["model"]["config"]["proprio_condition"]["input_dim"] == 18


def test_optimizer_only_selects_tactile_modules_with_an_objective() -> None:
    keys = action_policy_xhand_zeva_tactile["optimizer"]["keys_to_select"]

    def selected(name: str) -> bool:
        return any(key in name for key in keys)

    head = TactileBehaviorHead()
    for name, _ in head.named_parameters():
        assert selected(f"tactile_behavior_head.{name}") == name.startswith(("finger_projection.", "effect_head."))
    encoder_projector = FrozenTactileEncoderWithProjector()
    for name, _ in encoder_projector.named_parameters():
        assert selected(f"tactile_encoder_projector.{name}") == name.startswith("projector.")
    assert selected("tactile_effect_gate")
    assert not selected("tactile_phase_gate")
    assert not selected("proprio_projector.weight")


def test_stage2_preserves_the_trained_joint_state_projector() -> None:
    skips = action_policy_xhand_zeva_tactile["checkpoint"]["keys_to_skip_loading"]
    assert not any(key in "net.proprio_projector.weight" for key in skips)
