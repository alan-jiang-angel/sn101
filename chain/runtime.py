"""Construction and synchronization of Bittensor runtime objects."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .._bt import bittensor_attr, require_bittensor


@dataclass
class ChainRuntime:
    config: Any
    role: str
    wallet: Any = field(init=False)
    subtensor: Any = field(init=False)
    metagraph: Any = field(init=False)
    uid: int | None = field(default=None, init=False)
    _cached_block: int | None = field(default=None, init=False)
    _cached_at: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        bt = require_bittensor()
        if hasattr(bt.logging, "set_config"):
            bt.logging.set_config(config=self.config.logging)

        wallet_cls = bittensor_attr("wallet", "Wallet")
        subtensor_cls = bittensor_attr("subtensor", "Subtensor")
        self.wallet = wallet_cls(config=self.config)
        network_arg = _subtensor_network_arg(self.config)
        try:
            self.subtensor = subtensor_cls(network=network_arg, config=self.config)
        except TypeError:
            self.subtensor = subtensor_cls(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        if not getattr(self.config.neuron, "no_registration_check", False):
            self.ensure_registered()
        self.uid = self.find_uid()
        bt.logging.info(
            f"{self.role} ready: netuid={self.config.netuid} uid={self.uid} "
            f"endpoint={getattr(self.subtensor, 'chain_endpoint', None)}"
        )

    @property
    def hotkey(self) -> str:
        return self.wallet.hotkey.ss58_address

    @property
    def block(self) -> int:
        ttl = float(getattr(self.config.neuron, "block_cache_ttl", 0.0))
        now = time.time()
        if ttl <= 0 or self._cached_block is None or now - self._cached_at >= ttl:
            self._cached_block = int(self.subtensor.get_current_block())
            self._cached_at = now
        return self._cached_block

    def ensure_registered(self) -> None:
        registered = self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.hotkey,
        )
        if not registered:
            raise RuntimeError(
                f"Hotkey {self.hotkey} is not registered on netuid {self.config.netuid}."
            )

    def find_uid(self) -> int | None:
        hotkeys = list(getattr(self.metagraph, "hotkeys", []))
        if self.hotkey not in hotkeys:
            return None
        return hotkeys.index(self.hotkey)

    def sync_metagraph(self) -> None:
        self.metagraph.sync(subtensor=self.subtensor)
        self.uid = self.find_uid()

    def axon(self) -> Any:
        axon_cls = bittensor_attr("axon", "Axon")
        return axon_cls(wallet=self.wallet, config=self.config)

    def serve_axon(self, axon: Any) -> Any:
        response = self.subtensor.serve_axon(
            netuid=self.config.netuid,
            axon=axon,
            wait_for_inclusion=True,
            wait_for_finalization=False,
            wait_for_revealed_execution=False,
            period=None,
        )
        if not getattr(response, "success", False):
            raise RuntimeError(f"failed to serve axon: {getattr(response, 'message', response)}")
        return response

    def dendrite(self) -> Any:
        dendrite_cls = bittensor_attr("dendrite", "Dendrite")
        return dendrite_cls(wallet=self.wallet)


def _subtensor_network_arg(config: Any) -> str | None:
    if hasattr(config, "is_set") and config.is_set("subtensor.chain_endpoint"):
        return config.subtensor.chain_endpoint
    if hasattr(config, "is_set") and config.is_set("subtensor.network"):
        return config.subtensor.network
    return getattr(config.subtensor, "network", None)
