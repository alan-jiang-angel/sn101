"""Bittensor miner entry point."""

import asyncio
import time
from typing import Any, Tuple

from ._bt import require_bittensor
from .chain.runtime import ChainRuntime
from .chain.settings import build_config
from .protocol import TaskEnvelope
from .tasks import TaskRegistry, TaskServerClient, build_task_registry


class SolverMiner:
    def __init__(self, *, config: Any | None = None, registry: TaskRegistry | None = None):
        self.bt = require_bittensor()
        self.config = config or build_config("miner")
        self.runtime = ChainRuntime(self.config, role="miner")
        self.registry = registry or self._task_registry()
        self.client = self._task_server_client()
        self.axon = self.runtime.axon()
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        self.should_exit = False

    async def forward(self, synapse: TaskEnvelope) -> TaskEnvelope:
        started = time.perf_counter()
        try:
            handler = self.registry.handler_for(synapse.task_kind)
            synapse.answer = handler.solve_problem(synapse, self.runtime)
        except Exception:
            synapse.answer = {}
        elapsed = time.perf_counter() - started
        self.bt.logging.info(
            f"MINER_SOLVED_TASK task={synapse.task_id} kind={synapse.task_kind} "
            f"elapsed={elapsed:.3f}s answer_keys={list(synapse.answer)}"
        )
        return synapse

    async def blacklist(self, synapse: TaskEnvelope) -> Tuple[bool, str]:
        dendrite = getattr(synapse, "dendrite", None)
        hotkey = getattr(dendrite, "hotkey", None)
        if not hotkey:
            allow_empty = bool(getattr(self.config.miner, "allow_empty_hotkey", False))
            return (not allow_empty), "missing caller hotkey"

        hotkeys = list(getattr(self.runtime.metagraph, "hotkeys", []))
        if hotkey not in hotkeys:
            if bool(getattr(self.config.blacklist, "allow_non_registered", False)):
                return False, "unregistered caller allowed by config"
            return True, "caller is not registered"

        uid = hotkeys.index(hotkey)
        if bool(getattr(self.config.blacklist, "force_validator_permit", False)):
            permits = getattr(self.runtime.metagraph, "validator_permit", [])
            if uid >= len(permits) or not bool(permits[uid]):
                return True, "caller has no validator permit"
        return False, "accepted"

    async def priority(self, synapse: TaskEnvelope) -> float:
        dendrite = getattr(synapse, "dendrite", None)
        hotkey = getattr(dendrite, "hotkey", None)
        hotkeys = list(getattr(self.runtime.metagraph, "hotkeys", []))
        if not hotkey or hotkey not in hotkeys:
            return 0.0
        stakes = getattr(self.runtime.metagraph, "S", [])
        uid = hotkeys.index(hotkey)
        return float(stakes[uid]) if uid < len(stakes) else 0.0

    def run(self) -> None:
        self.runtime.ensure_registered()
        self.runtime.serve_axon(self.axon)
        self.axon.start()
        self.bt.logging.info(f"miner serving at block {self.runtime.block}")
        self._announce_private_axon_at_startup()
        try:
            while not self.should_exit:
                self.runtime.sync_metagraph()
                time.sleep(12)
        except KeyboardInterrupt:
            self.bt.logging.info("miner interrupted")
        finally:
            self.axon.stop()

    def _task_server_client(self) -> TaskServerClient | None:
        url = str(getattr(self.config.task_server, "url", "") or "").strip()
        if not url or bool(getattr(self.config.miner, "axon_to_public_metagraph", False)):
            return None
        return TaskServerClient(
            url,
            timeout=float(getattr(self.config.task_server, "timeout", 30.0)),
            verify_ssl=bool(getattr(self.config.task_server, "verify_ssl", True)),
        )

    def _task_registry(self) -> TaskRegistry:
        miner_module = str(getattr(getattr(self.config, "task", None), "miner_module", "") or "")
        override_modules = [
            module.strip()
            for module in miner_module.split(",")
            if module.strip()
        ]
        return build_task_registry(override_modules=override_modules)

    def _announce_private_axon_at_startup(self) -> None:
        if self.client is None:
            return

        try:
            result = asyncio.run(self._announce_private_axon_once())
        except Exception as exc:
            self.bt.logging.warning(
                "private miner axon startup announcement failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return

        if not result.get("accepted") or not result.get("updated", True):
            self.bt.logging.warning(
                "private miner axon announcement returned unexpected response "
                f"response={result}"
            )
        else:
            self.bt.logging.info(
                "MINER_PRIVATE_AXON_ANNOUNCED "
                f"netuid={int(self.config.netuid)} block={self.runtime.block}"
            )

    async def _announce_private_axon_once(self) -> dict[str, Any]:
        if self.client is None:
            return {"accepted": False, "updated": False}
        info = self.axon.info()
        ip = getattr(info, "ip", None)
        port = getattr(info, "port", None)
        if not ip or not port:
            raise ValueError("private axon announcement requires external ip and port")
        return await self.client.announce_miner_axon(
            wallet=self.runtime.wallet,
            netuid=int(self.config.netuid),
            uid=int(self.runtime.uid),
            block=self.runtime.block,
            ip=str(ip),
            port=int(port),
            ip_type=None,
            protocol=int(getattr(info, "protocol", 4) or 4),
            version=int(getattr(info, "version", 0) or 0),
        )

    def __enter__(self) -> "SolverMiner":
        self._task = asyncio.get_event_loop().run_in_executor(None, self.run)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.should_exit = True


def main() -> None:
    SolverMiner().run()


if __name__ == "__main__":
    main()
