"""W&B metric axis: one row per epoch, keyed on `frame`.

rl_games logs every metric to TensorBoard three times (`/step`|`/frame`, `/iter`, `/time`)
and relies on TensorBoard honouring the per-tag step. W&B's `sync_tensorboard` drops that
step, so the three families collapse onto W&B's row counter and become identical curves on
a meaningless x-axis. `wandb_utils.enable_frame_axis_logging` replaces that mirror; these
tests pin the behaviour it has to keep.
"""
import os

import pytest
import torch

wandb = pytest.importorskip("wandb")

# SummaryWriter must come from a2c_common: rl_games writes through tensorboardX's class,
# not torch.utils.tensorboard's. Importing the wrong one makes these tests pass against a
# hook that captures nothing in a real run.
from rl_games.common.a2c_common import A2CBase, SummaryWriter  # noqa: E402
from rl_games.common.diagnostics import PpoDiagnostics  # noqa: E402

from rl_training.rl_games import wandb_utils  # noqa: E402

BATCH = 8192
EPOCHS = 3


@pytest.mark.parametrize("tag,expected", [
    ("rewards/step", "rewards"),
    ("episode_lengths/step", "episode_lengths"),
    ("reward/p_jerk/frame", "reward/p_jerk"),
    ("success_rate/frame", "success_rate"),
    ("rewards/iter", None),
    ("rewards/time", None),
    ("metrics/v_horizontal/iter", None),
    ("scores/iter", None),
    # not duplicates: the suffix is part of the metric's own name
    ("losses/a_loss", "losses/a_loss"),
    ("performance/step_time", "performance/step_time"),
    ("performance/rl_update_time", "performance/rl_update_time"),
    ("performance/step_inference_time", "performance/step_inference_time"),
    ("diagnostics/clip_frac/0", "diagnostics/clip_frac/0"),
    ("scores/mean", "scores/mean"),
    ("Episode/len", "Episode/len"),
    ("distill/kd_scale", "distill/kd_scale"),
])
def test_tag_mapping(tag, expected):
    assert wandb_utils._wandb_key(tag) == expected


def test_hook_targets_the_writer_rl_games_uses():
    """The failure this guards against is silent: patching torch.utils.tensorboard's
    SummaryWriter raises nothing, captures nothing, and loses a whole run's metrics."""
    import rl_games.common.a2c_common as a2c_common

    assert wandb_utils.SummaryWriter is a2c_common.SummaryWriter


class _Observer:
    """Stands in for IsaacAlgoObserver.after_print_stats, which write_stats calls."""

    def __init__(self, writer):
        self.writer = writer

    def after_print_stats(self, frame, epoch_num, total_time):
        for key, value in (("success_rate", 0.1 * epoch_num),
                           ("reward/r_progress", -0.5 * epoch_num)):
            self.writer.add_scalar(f"{key}/frame", value, frame)
            self.writer.add_scalar(f"{key}/iter", value, epoch_num)
            self.writer.add_scalar(f"{key}/time", value, total_time)
        self.writer.add_scalar("Episode/len", 42.0 + epoch_num, epoch_num)


class _StubAgent:
    """The attributes A2CBase.write_stats touches -- building a real agent needs a sim."""

    e_clip = 0.2

    def __init__(self, writer):
        self.writer = writer
        self.diagnostics = PpoDiagnostics()
        self.algo_observer = _Observer(writer)


@pytest.fixture
def logged_rows(tmp_path, monkeypatch):
    """Run EPOCHS of rl_games' write pattern and return what reached wandb.log."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setattr(wandb_utils, "_EPOCH_METRICS", wandb_utils._EpochMetrics())

    original_add_scalar = SummaryWriter.add_scalar
    original_write_stats = A2CBase.write_stats

    wandb.init(dir=str(tmp_path), project="axis-test")
    rows = []
    real_log = wandb.log  # wandb.init rebinds wandb.log; capture the bound one

    def spy_log(data, *args, **kwargs):
        rows.append((kwargs.get("step"), dict(data)))
        return real_log(data, *args, **kwargs)

    monkeypatch.setattr(wandb, "log", spy_log)
    wandb_utils.enable_frame_axis_logging()

    writer = SummaryWriter(os.path.join(str(tmp_path), "summaries"))
    agent = _StubAgent(writer)
    total_time = 0.0
    try:
        for epoch in range(1, EPOCHS + 1):
            frame = (epoch - 1) * BATCH  # rl_games reads frame before incrementing it
            total_time += 100.0 * epoch
            agent.diagnostics.current_epoch = epoch
            agent.diagnostics.diag_dict["diagnostics/exp_var"] = torch.tensor(0.8 + 0.01 * epoch)
            A2CBase.write_stats(
                agent, total_time, epoch, 1.0, 1.0, 1.0,
                [torch.tensor(0.1 * epoch)], [torch.tensor(0.2 * epoch)],
                [torch.tensor(5.0)], [torch.tensor(0.01)], 1e-4, 1.0,
                frame, 1.0, 1.0, BATCH)
            # written after write_stats returns, still part of this epoch
            writer.add_scalar("rewards/step", -10.0 * epoch, frame)
            writer.add_scalar("rewards/iter", -10.0 * epoch, epoch)
            writer.add_scalar("rewards/time", -10.0 * epoch, total_time)
        wandb_utils.flush_final_metrics()
    finally:
        writer.close()
        wandb.finish()
        SummaryWriter.add_scalar = original_add_scalar
        A2CBase.write_stats = original_write_stats
    return rows


def test_one_row_per_epoch(logged_rows):
    assert len(logged_rows) == EPOCHS


def test_x_axis_is_frame(logged_rows):
    assert [step for step, _ in logged_rows] == [e * BATCH for e in range(EPOCHS)]


def test_epoch_and_time_ride_along_as_columns(logged_rows):
    for (_, payload), epoch in zip(logged_rows, range(1, EPOCHS + 1)):
        assert payload["epoch"] == epoch
        assert payload["elapsed_s"] > 0
        assert payload["frame"] == (epoch - 1) * BATCH


def test_duplicate_families_are_dropped(logged_rows):
    for _, payload in logged_rows:
        assert not [k for k in payload if k.endswith(("/iter", "/time"))]


def test_epoch_metrics_land_in_the_same_row(logged_rows):
    """The rewards rl_games writes after write_stats belong to the epoch that just ended,
    not the next one -- an off-by-one here silently shifts every reward curve."""
    for (_, payload), epoch in zip(logged_rows, range(1, EPOCHS + 1)):
        assert payload["rewards"] == pytest.approx(-10.0 * epoch)
        assert payload["success_rate"] == pytest.approx(0.1 * epoch)
        assert payload["losses/c_loss"] == pytest.approx(0.2 * epoch)
        assert payload["diagnostics/exp_var"] == pytest.approx(0.8 + 0.01 * epoch)
        assert "Episode/len" in payload
