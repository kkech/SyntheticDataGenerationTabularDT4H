"""Interface every pipeline step implements."""

from abc import ABC, abstractmethod

from pipeline.config import PipelineConfig


class PipelineStep(ABC):
    #: Unique id used for status tracking and the --only/--force-step CLI flags.
    name: str

    #: False (default): a rerun always wipes this step's output directory
    #: first (run_pipeline()'s ordinary "clean rerun" semantics) -- correct
    #: for steps that are fast and atomic. True opts a step OUT of that
    #: wipe when it is being re-attempted merely because it was never
    #: marked completed (interrupted, crashed, or never run) rather than
    #: EXPLICITLY forced (--force/--force-step) -- the step's own run()
    #: is then responsible for reconciling with whatever partial output
    #: already sits in its directory. See GenerateStep for the only
    #: current user: a multi-hour run plan should not have to redo
    #: already-succeeded runs after a crash or reboot.
    resumable: bool = False

    @abstractmethod
    def run(self, config: PipelineConfig) -> None:
        """Execute the step. Raise on failure -- the runner records it and stops."""
        ...
