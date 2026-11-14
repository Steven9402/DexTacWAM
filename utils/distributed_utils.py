import torch


def finish_training(accelerator) -> None:
    """Synchronize before Accelerate destroys the default process group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_backend() == "nccl":
            device_index = accelerator.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            torch.distributed.barrier(device_ids=[device_index])
        else:
            accelerator.wait_for_everyone()

    accelerator.end_training()
