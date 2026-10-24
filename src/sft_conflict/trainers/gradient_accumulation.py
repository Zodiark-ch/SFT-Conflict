"""Accumulate already-reduced CPU shards outside FSDP's backward lifecycle."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CPUGradientAccumulator:
    """Keep microbatch sums independent of FSDP's reusable CPU gradient buffer.

    FSDP1 CPU offload overwrites its CPU gradient on each synchronized backward.
    Hold a clone between microbatches and restore the sum only at the optimizer
    boundary, before Trainer's ordinary clipping/step/zero_grad. Backward has
    already synchronized the device-to-host copies; never accumulate GPU grads
    here or change loss scaling. Handles both flat and original parameters.
    """

    def __init__(self) -> None:
        self.buffers: dict[nn.Parameter, Tensor] = {}

    @torch.no_grad()
    def accumulate(self, model: nn.Module, *, optimizer_boundary: bool) -> None:
        parameters = tuple(model.parameters())
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is not None and gradient.device.type != "cpu":
                raise RuntimeError("FSDP CPU accumulation requires synchronized CPU gradients")
        for parameter in parameters:
            gradient = parameter.grad
            previous = self.buffers.get(parameter)
            if optimizer_boundary:
                if previous is not None:
                    if gradient is None:
                        parameter.grad = previous
                    else:
                        gradient.add_(previous)
            elif gradient is not None:
                if previous is None:
                    self.buffers[parameter] = gradient.detach().clone()
                else:
                    previous.add_(gradient)
        if optimizer_boundary:
            self.buffers.clear()
        else:
            # Do not let a prior CPU .grad enter FSDP's next backward.
            model.zero_grad(set_to_none=True)
