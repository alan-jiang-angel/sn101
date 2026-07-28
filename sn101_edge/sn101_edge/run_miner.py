"""Miner entrypoint with a non-blocking forward path.

The stock miner calls `handler.solve_problem(...)` synchronously from inside an
async `forward`, which blocks uvicorn's event loop for the whole solve. With
several validators querying at once, requests serialise and the later ones can
blow through their wire timeout -- each one a zero that dilutes the 24h rolling
average.

Subclassing keeps the fix entirely outside the tag101 checkout, so auto-update
never reverts it.
"""

from __future__ import annotations

import asyncio
import time

from tag101._logging import get_logger
from tag101.miner import SolverMiner
from tag101.protocol import TaskEnvelope

from .encoder import get_encoder
from .tagger import get_tagger


class EdgeMiner(SolverMiner):
    async def forward(self, synapse: TaskEnvelope) -> TaskEnvelope:
        started = time.perf_counter()
        try:
            handler = self.registry.handler_for(synapse.task_kind)
            synapse.answer = await asyncio.to_thread(
                handler.solve_problem, synapse, self.runtime
            )
        except Exception as exc:  # noqa: BLE001
            self.log.warning(f"forward failed: {type(exc).__name__}: {exc}")
            synapse.answer = {}
        elapsed = time.perf_counter() - started
        self.log.info(
            f"MINER_SOLVED_TASK task={synapse.task_id} kind={synapse.task_kind} "
            f"elapsed={elapsed:.3f}s answer={synapse.answer}"
        )
        return synapse


def main() -> None:
    log = get_logger("miner")
    miner = EdgeMiner()

    # Load the embedding checkpoint before serving. A cold first task would
    # otherwise spend several seconds importing torch and miss its deadline.
    log.info("warming embedding model...")
    try:
        get_encoder().warm()
        get_tagger()
        log.info("embedding model ready")
    except Exception as exc:  # noqa: BLE001
        log.warning(
            f"model warmup failed ({type(exc).__name__}: {exc}); "
            "miner will run on the local fallback tagger"
        )

    miner.run()


if __name__ == "__main__":
    main()
