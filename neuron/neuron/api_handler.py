#  The MIT License (MIT)
#  Copyright © 2023 Yuma Rao
#  Copyright © 2024 WOMBO
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the “Software”), to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#  and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#  the Software.
#
#  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
#  THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#
#

from asyncio import Lock
from time import monotonic_ns

import bittensor as bt
from bittensor.utils.networking import get_external_ip
from grpc import unary_unary_rpc_method_handler, StatusCode
from grpc.aio import Metadata
from substrateinterface import Keypair

_MAX_ALLOWED_NONCE_DELTA = 4_000_000

NONCE_HEADER = "bt_header_dendrite_nonce"
HOTKEY_HEADER = "bt_header_dendrite_hotkey"
SIGNATURE_HEADER = "bt_header_dendrite_signature"


def request_error(status_code: StatusCode, detail: str):
    return unary_unary_rpc_method_handler(lambda _, context: context.abort(status_code, detail))


def serve_ip(config: bt.config, subtensor: bt.subtensor, wallet: bt.wallet):
    """Serve axon to enable external connections."""

    bt.logging.info("serving ip to chain...")

    external_ip = config.axon.external_ip or get_external_ip()
    external_port = config.axon.external_port or config.axon.port

    subtensor.serve(
        wallet=wallet,
        ip=external_ip,
        port=external_port,
        protocol=4,
        netuid=config.netuid,
    )


class RequestVerifier:
    nonces: dict[str, set[int]]

    def __init__(self, hotkey: str):
        super().__init__()

        self.nonces = {}
        self.nonce_lock = Lock()
        self.hotkey = hotkey

    async def verify(self, invocation_metadata: Metadata):
        hotkey = invocation_metadata[HOTKEY_HEADER]
        nonce = int(invocation_metadata[NONCE_HEADER])
        signature = invocation_metadata[SIGNATURE_HEADER]

        # Build the keypair from the dendrite_hotkey
        keypair = Keypair(ss58_address=hotkey)

        # Build the signature messages.
        message = f"{nonce}.{hotkey}.{self.hotkey}"

        if not keypair.verify(message, signature):
            return request_error(
                StatusCode.UNAUTHENTICATED,
                f"Signature mismatch with {message} and {signature}",
            )

        async with self.nonce_lock:
            nonces = self.nonces.get(hotkey)

            # Ensure this is not a repeated request.
            if nonces:
                if nonce in nonces:
                    return request_error(StatusCode.UNAUTHENTICATED, "Duplicate nonce")
            else:
                nonces = set[int]()
                self.nonces[hotkey] = nonces

            if monotonic_ns() - nonce > _MAX_ALLOWED_NONCE_DELTA:
                return request_error(StatusCode.UNAUTHENTICATED, "Nonce is too old")

            nonces.add(nonce)
