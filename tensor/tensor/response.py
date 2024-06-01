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
from asyncio import CancelledError
from time import perf_counter
from typing import Literal, TypeVar, Generic, TypeAlias, Annotated, Callable, cast, Awaitable

from bittensor import AxonInfo
from google.protobuf.message import Message
from grpc import StatusCode, insecure_channel, RpcError
from grpc.aio import Channel, AioRpcError, UnaryUnaryCall
from pydantic import BaseModel, Field

ResponseT = TypeVar("ResponseT")
MessageT = TypeVar("MessageT", bound=Message)


class SuccessfulResponseInfo(BaseModel):
    axon: AxonInfo
    process_time: float


class FailedResponseInfo(BaseModel):
    axon: AxonInfo
    status: StatusCode
    detail: str | None


class SuccessfulResponse(Generic[ResponseT], SuccessfulResponseInfo):
    data: ResponseT
    successful: Literal[True] = True

    @property
    def info(self):
        return SuccessfulResponseInfo(axon=self.axon, process_time=self.process_time)


class FailedResponse(FailedResponseInfo):
    successful: Literal[False] = False

    @property
    def info(self):
        return FailedResponseInfo(axon=self.axon, status=self.status, detail=self.detail)


ResponseInfo = SuccessfulResponseInfo | FailedResponseInfo

Response: TypeAlias = Annotated[
    SuccessfulResponse | FailedResponse,
    Field(discriminator="successful"),
]


def axon_channel(axon: AxonInfo):
    return insecure_channel(f"{axon.ip}:{axon.port}")


async def create_request(
    axon: AxonInfo,
    request: MessageT,
    invoker: Callable[[Channel], Callable[[MessageT], Awaitable[ResponseT]]],
) -> Response:
    async with axon_channel(axon) as channel:
        return await call_request(axon, request, invoker(channel))


async def call_request(
    axon: AxonInfo,
    request: MessageT,
    invoker: Callable[[MessageT], UnaryUnaryCall[MessageT, ResponseT]],
) -> Response:
    try:
        start = perf_counter()

        call = invoker(request)

        try:
            response = await call
        except CancelledError:
            call.cancel()
            raise

        process_time = perf_counter() - start

        return SuccessfulResponse(
            data=response,
            process_time=process_time,
            axon=axon,
        )
    except RpcError as error:
        grpc_error = cast(AioRpcError, error)

        return FailedResponse(
            axon=axon,
            status=grpc_error.code(),
            detail=grpc_error.details(),
        )
