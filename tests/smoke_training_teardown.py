import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.distributed_utils import finish_training


def _accelerator():
    accelerator = Mock()
    accelerator.device = SimpleNamespace(index=0)
    return accelerator


def test_nccl_barrier_precedes_process_group_destruction():
    accelerator = _accelerator()
    with patch("utils.distributed_utils.torch.distributed") as dist:
        dist.is_available.return_value = True
        dist.is_initialized.return_value = True
        dist.get_backend.return_value = "nccl"

        order = Mock()
        order.attach_mock(dist.barrier, "barrier")
        order.attach_mock(accelerator.end_training, "end_training")

        finish_training(accelerator)

        assert order.mock_calls == [
            call.barrier(device_ids=[0]),
            call.end_training(),
        ]
        accelerator.wait_for_everyone.assert_not_called()


def test_no_collective_after_process_group_is_already_destroyed():
    accelerator = _accelerator()
    with patch("utils.distributed_utils.torch.distributed") as dist:
        dist.is_available.return_value = True
        dist.is_initialized.return_value = False

        finish_training(accelerator)

        dist.barrier.assert_not_called()
        accelerator.wait_for_everyone.assert_not_called()
        accelerator.end_training.assert_called_once_with()


def test_non_nccl_barrier_uses_accelerator_before_end_training():
    accelerator = _accelerator()
    with patch("utils.distributed_utils.torch.distributed") as dist:
        dist.is_available.return_value = True
        dist.is_initialized.return_value = True
        dist.get_backend.return_value = "gloo"

        order = Mock()
        order.attach_mock(accelerator.wait_for_everyone, "wait_for_everyone")
        order.attach_mock(accelerator.end_training, "end_training")

        finish_training(accelerator)

        assert order.mock_calls == [
            call.wait_for_everyone(),
            call.end_training(),
        ]
        dist.barrier.assert_not_called()


def main():
    test_nccl_barrier_precedes_process_group_destruction()
    test_no_collective_after_process_group_is_already_destroyed()
    test_non_nccl_barrier_uses_accelerator_before_end_training()
    print("[smoke_training_teardown] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
