import asyncio
import os
from argparse import ArgumentParser
from asyncio import Task
from typing import AsyncGenerator

import bittensor as bt
import grpc
from bittensor import AxonInfo
from grpc.aio import ServicerContext

from client.protos.client_pb2 import UserRequest, GenerationResponse, NeuronInfo
from client.protos.client_pb2_grpc import ClientServicer, add_ClientServicer_to_server
from neuron_selector.protos.forwarding_validator_pb2 import ValidatorGenerationResponse
from neuron_selector.protos.forwarding_validator_pb2_grpc import ForwardingValidatorStub
from neuron_selector.uids import get_best_uids
from tensor.config import config, add_args
from tensor.interceptors import LoggingInterceptor
from tensor.neuron_info import sync_neuron_info
from tensor.protos.inputs_pb2 import NeuronCapabilities
from tensor.response import FailedResponse, create_request, Response, SuccessfulResponse


class ValidatorQueryException(Exception):
    def __init__(
        self,
        failed_responses: list[FailedResponse],
    ):
        super().__init__(f"Failed to query subnetwork, responses {failed_responses}")
        self.failed_responses = failed_responses


async def get_responses(
    axons: list[AxonInfo],
    inputs: UserRequest,
) -> AsyncGenerator[Response[ValidatorGenerationResponse], None]:
    responses = asyncio.as_completed(
        [
            create_request(axon, inputs.inputs, lambda channel: ForwardingValidatorStub(channel).Generate)
            for axon in axons
        ]
    )

    for future in responses:
        yield await future


async def process_responses(
    responses: AsyncGenerator[Response[ValidatorGenerationResponse], None],
) -> SuccessfulResponse[ValidatorGenerationResponse]:
    bad_responses: list[FailedResponse] = []

    async for response in responses:
        if response.successful:
            return response
        else:
            bad_responses.append(response)

    raise ValidatorQueryException(bad_responses)


class WomboSubnetAPI:
    def __init__(self):
        client_config = self.client_config()

        self.config = client_config
        bt.logging.check_config(client_config)

        # Set up logging with the provided configuration and directory.
        bt.logging(config=client_config, logging_dir=client_config.full_path)

        # Log the configuration for reference.
        bt.logging.info(client_config)

        self.wallet = bt.wallet(config=client_config)

        self.subtensor = bt.subtensor(config=client_config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        # The metagraph holds the state of the network, letting us know about other validators and miners.
        self.metagraph = self.subtensor.metagraph(client_config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        self.server = grpc.aio.server(interceptors=[LoggingInterceptor()])

        add_ClientServicer_to_server(ClientRequestService(self), self.server)

        self.server.add_insecure_port(f"0.0.0.0:{os.getenv('PORT', str(8000))}")

        self.last_metagraph_sync = self.subtensor.get_current_block()

        self.periodic_metagraph_resync: Task
        self.neuron_info = {}

    async def resync_metagraph(self):
        bt.logging.info("resync_metagraph()")

        try:
            # Sync the metagraph.
            self.metagraph.sync(subtensor=self.subtensor)
            self.neuron_info = await sync_neuron_info(self.metagraph, self.wallet)
        except Exception as exception:
            bt.logging.error("Failed to sync client metagraph", exc_info=exception)

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        add_args(parser, "cpu")

    @classmethod
    def client_config(cls):
        return config(cls.add_args)

    async def run(self):
        await self.resync_metagraph()

        bt.logging.info("Started")
        await self.server.start()

        try:
            while True:
                block = self.subtensor.get_current_block()

                if block - self.last_metagraph_sync <= self.config.neuron.epoch_length:
                    await asyncio.sleep(max(self.config.neuron.epoch_length - block + self.last_metagraph_sync, 0) * 12)
                    continue

                await self.resync_metagraph()

        # If someone intentionally stops the client, it'll safely terminate operations.
        except KeyboardInterrupt:
            await self.server.stop()
            await self.server.wait_for_termination()
            bt.logging.success("Client killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as exception:
            bt.logging.error("Caught exception in client loop", exc_info=exception)


class ClientRequestService(ClientServicer):
    def __init__(self, api: WomboSubnetAPI):
        self.api = api

    async def Generate(self, request: UserRequest, context: ServicerContext):
        validator_uids = (
            get_best_uids(
                self.api.config.blacklist,
                self.api.metagraph,
                self.api.neuron_info,
                self.api.metagraph.total_stake,
                lambda uid, info: (
                    NeuronCapabilities.FORWARDING_VALIDATOR in info.capbilities and
                    self.api.metagraph.validator_permit[uid].item()
                )
            )
            if request.validator_uid is None
            else [request.validator_uid]
        )

        if not len(validator_uids):
            raise RuntimeError("No suitable validators found")

        axons: list[AxonInfo] = [self.api.metagraph.axons[uid] for uid in validator_uids]

        axon_uids = {
            axon.hotkey: uid.item()
            for uid, axon in zip(validator_uids, axons)
        }

        response_generator = get_responses(axons, request)

        response = await process_responses(response_generator)

        return GenerationResponse(
            image=response.data.image,
            miner_info=NeuronInfo(
                hotkey=self.api.metagraph.hotkeys[response.data.miner_uid],
                uid=response.data.miner_uid,
                processing_time=response.data.generation_time,
            ),
            validator_info=NeuronInfo(
                hotkey=response.axon.hotkey,
                uid=axon_uids[response.axon.hotkey],
                processing_time=response.process_time,
            )
        )


if __name__ == "__main__":
    asyncio.run(WomboSubnetAPI().run())
