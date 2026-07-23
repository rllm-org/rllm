from unittest.mock import Mock

from rllm.utils.tracking import Tracking


def test_tracking_commits_explicit_wandb_step() -> None:
    tracking = Tracking.__new__(Tracking)
    tracking._finished = True
    wandb_logger = Mock()
    console_logger = Mock()
    tracking.logger = {
        "wandb": wandb_logger,
        "console": console_logger,
    }

    data = {"reward/mean": 0.5}
    tracking.log(data=data, step=8)

    wandb_logger.log.assert_called_once_with(data=data, step=8, commit=True)
    console_logger.log.assert_called_once_with(data=data, step=8)
