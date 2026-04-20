from pydantic import BaseModel
import os

class KernelExecResult(BaseModel):
    """
    Single Kernel Execution
    """

    compiled: bool = False
    correctness: bool = False
    metadata: dict = {}
    runtime: float = -1.0  # in us, only recorded if we decide to measure performance
    runtime_stats: dict = {}  # only recorded if we decide to measure performance

    def to_dict(self) -> dict:
        return dict(
            compiled=self.compiled,
            correctness=self.correctness,
            metadata=self.metadata,
            runtime=self.runtime,
            runtime_stats=self.runtime_stats,
        )

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)