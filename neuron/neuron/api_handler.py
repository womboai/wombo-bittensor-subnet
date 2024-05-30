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
from typing import Callable, Awaitable

from grpc import ServerInterceptor, HandlerCallDetails, unary_unary_rpc_method_handler, StatusCode
from grpc.aio import Metadata
from substrateinterface import Keypair

_MAX_ALLOWED_NONCE_DELTA = 4_000_000

NONCE_HEADER = "bt_header_dendrite_nonce"
HOTKEY_HEADER = "bt_header_dendrite_hotkey"
SIGNATURE_HEADER = "bt_header_dendrite_signature"


class RequestVerifier(ServerInterceptor):
    def __init__(self, hotkey: str):
        super().__init__()

        self.nonces = {}
        self.nonce_lock = Lock()
        self.hotkey = hotkey

    def intercept_service(
        self,
        continuation: Callable[[HandlerCallDetails], Awaitable[None] | None],
        handler_call_details: HandlerCallDetails,
    ):
        metadata: Metadata = handler_call_details.invocation_metadata

        hotkey = metadata[HOTKEY_HEADER]
        nonce = int(metadata[NONCE_HEADER])
        signature = metadata[SIGNATURE_HEADER]

        # Build the keypair from the dendrite_hotkey
        keypair = Keypair(ss58_address=hotkey)

        # Build the signature messages.
        message = f"{nonce}.{hotkey}.{self.hotkey}"

        if not keypair.verify(message, signature):
            return unary_unary_rpc_method_handler(
                lambda _, context: context.abort(
                    StatusCode.UNAUTHENTICATED,
                    f"Signature mismatch with {message} and {signature}",
                )
            )

        async with self.nonce_lock:
            nonces = self.nonces.get(hotkey)

            # Ensure this is not a repeated request.
            if nonces:
                if nonce in nonces:
                    return unary_unary_rpc_method_handler(
                        lambda _, context: context.abort(
                            StatusCode.UNAUTHENTICATED,
                            "Duplicate nonce",
                        )
                    )
            else:
                nonces = set[int]()
                self.nonces[hotkey] = nonces

            if monotonic_ns() - nonce > _MAX_ALLOWED_NONCE_DELTA:
                return unary_unary_rpc_method_handler(
                    lambda _, context: context.abort(
                        StatusCode.UNAUTHENTICATED,
                        "Nonce is too old",
                    )
                )

            nonces.add(nonce)


class RequestBlackLister(ServerInterceptor):
    def __init__(self, test: Callable[[str]]):
        super().__init__()

        self.nonces = {}
        self.nonce_lock = Lock()
        self.hotkey = hotkey

    def intercept_service(
        self,
        continuation: Callable[[HandlerCallDetails], Awaitable[None] | None],
        handler_call_details: HandlerCallDetails,
    ):
        metadata: Metadata = handler_call_details.invocation_metadata

        hotkey = metadata[HOTKEY_HEADER]
        nonce = int(metadata[NONCE_HEADER])
        signature = metadata[SIGNATURE_HEADER]

        # Build the keypair from the dendrite_hotkey
        keypair = Keypair(ss58_address=hotkey)

        # Build the signature messages.
        message = f"{nonce}.{hotkey}.{self.hotkey}"

        if not keypair.verify(message, signature):
            return unary_unary_rpc_method_handler(
                lambda _, context: context.abort(
                    StatusCode.UNAUTHENTICATED,
                    f"Signature mismatch with {message} and {signature}",
                )
            )

        async with self.nonce_lock:
            nonces = self.nonces.get(hotkey)

            # Ensure this is not a repeated request.
            if nonces:
                if nonce in nonces:
                    return unary_unary_rpc_method_handler(
                        lambda _, context: context.abort(
                            StatusCode.UNAUTHENTICATED,
                            "Duplicate nonce",
                        )
                    )
            else:
                nonces = set[int]()
                self.nonces[hotkey] = nonces

            if monotonic_ns() - nonce > _MAX_ALLOWED_NONCE_DELTA:
                return unary_unary_rpc_method_handler(
                    lambda _, context: context.abort(
                        StatusCode.UNAUTHENTICATED,
                        "Nonce is too old",
                    )
                )

            nonces.add(nonce)
