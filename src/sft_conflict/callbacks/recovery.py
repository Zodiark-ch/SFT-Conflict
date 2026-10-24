"""Additional restart checkpoints without changing evaluation/best-model cadence."""

from transformers import TrainerCallback


class RecoveryCheckpointCallback(TrainerCallback):
    def __init__(self, interval: int) -> None:
        if type(interval) is not int or interval <= 0:
            raise ValueError("recovery_save_steps must be a positive integer")
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        # Called only after an optimizer step; never save a partial accumulation.
        # OR with existing epoch/step decisions; do not request extra evaluation.
        if state.global_step > 0 and state.global_step % self.interval == 0:
            control.should_save = True
        return control
