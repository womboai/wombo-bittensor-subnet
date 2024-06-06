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
from typing import Callable, Awaitable

import bittensor as bt
from grpc import HandlerCallDetails, RpcMethodHandler, unary_unary_rpc_method_handler
from grpc.aio import ServerInterceptor, ServicerContext

from neuron.api_handler import get_metadata, HOTKEY_HEADER
from tensor.response import RequestT, ResponseT


class LoggingInterceptor(ServerInterceptor):
    async def intercept_service(
        self,
        continuation: Callable[[HandlerCallDetails], Awaitable[RpcMethodHandler]],
        handler_call_details: HandlerCallDetails,
    ) -> RpcMethodHandler:
        handler = await continuation(handler_call_details)

        async def invoke_and_log(request: RequestT, context: ServicerContext):
            hotkey = get_metadata(context).get(HOTKEY_HEADER)

            try:
                response: ResponseT = handler.unary_unary(request, context)
            except Exception:
                bt.logging.trace(
                    f"Failed request {handler_call_details.method} <- {hotkey}, "
                    f"status code {context.code()} with error {context.details()}",
                    exc_info=True,
                )

                raise

            bt.logging.trace(
                f"Successful request {handler_call_details.method} <- {hotkey}, "
                f"status code {context.code()} with details {context.details()}"
            )

            return response

        return unary_unary_rpc_method_handler(
            invoke_and_log,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )
