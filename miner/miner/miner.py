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

import asyncio
import base64
import traceback
from typing import cast, Tuple

import bittensor as bt
from aiohttp import ClientSession, MultipartReader, BodyPartReader
from bittensor import SynapseDendriteNoneException
from substrateinterface import Keypair

from image_generation_protocol.io_protocol import ImageGenerationOutput, ImageGenerationRequest
from neuron.neuron import BaseNeuron
from tensor.config import add_args, check_config
from tensor.protocol import ImageGenerationSynapse, NeuronInfoSynapse
from tensor.timeouts import AXON_REQUEST_TIMEOUT, KEEP_ALIVE_TIMEOUT


def miner_forward_info(synapse: NeuronInfoSynapse):
    synapse.is_validator = False

    return synapse


class Miner(BaseNeuron):
    nonces: dict[str, set[int]]
    nonce_lock: asyncio.Lock
    last_metagraph_sync: int

    def __init__(self):
        super().__init__()

        # Warn if allowing incoming requests from anyone.
        if not self.config.blacklist.force_validator_permit:
            bt.logging.warning(
                "You are allowing non-validators to send requests to your miner. This is a security risk."
            )
        if self.config.blacklist.allow_non_registered:
            bt.logging.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk."
            )

        # The axon handles request processing, allowing validators to send this miner requests.
        self.axon = bt.axon(wallet=self.wallet, config=self.config)

        # Attach determiners which functions are called when servicing a request.
        bt.logging.info(f"Attaching forward function to miner axon.")

        self.axon.attach(forward_fn=miner_forward_info)

        self.axon.attach(
            forward_fn=self.forward_image,
            blacklist_fn=self.blacklist_image,
            priority_fn=self.priority_image,
            verify_fn=self.verify_image,
        )

        self.axon.fast_config.timeout_keep_alive = KEEP_ALIVE_TIMEOUT
        self.axon.fast_config.timeout_notify = AXON_REQUEST_TIMEOUT

        bt.logging.info(f"Axon created: {self.axon}")

        self.nonces = {}
        self.nonce_lock = asyncio.Lock()

    @classmethod
    def check_config(cls, config: bt.config):
        check_config(config, "miner")

    @classmethod
    def add_args(cls, parser):
        add_args(parser)

        parser.add_argument(
            "--generation_endpoint",
            type=str,
            help="The endpoint to call for miner generation",
            default="http://localhost:8001/api/generate",
        )

        parser.add_argument(
            "--blacklist.force_validator_permit",
            action="store_true",
            help="If set, we will force incoming requests to have a permit.",
            default=False,
        )

        parser.add_argument(
            "--blacklist.allow_non_registered",
            action="store_true",
            help="If set, miners will accept queries from non registered entities. (Dangerous!)",
            default=False,
        )

    async def run(self):
        # Check that miner is registered on the network.
        await self.sync()

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        bt.logging.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        # Start the miner's axon, making it active on the network.
        self.axon.start()

        bt.logging.info(f"Miner starting at block: {self.block}")

        # This loop maintains the miner's operations until intentionally stopped.
        try:
            while True:
                while not self.should_sync_metagraph():
                    # Wait before checking again.
                    await asyncio.sleep(1)

                # Sync metagraph and potentially set weights.
                await self.sync()

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as _:
            bt.logging.error(traceback.format_exc())

    async def resync_metagraph(self):
        if not self.should_sync_metagraph():
            return

        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        self.last_metagraph_sync = self.block

    def should_sync_metagraph(self):
        return self.block - self.last_metagraph_sync > self.config.neuron.epoch_length

    async def forward_image(
        self,
        synapse: ImageGenerationSynapse,
    ) -> ImageGenerationSynapse:
        async with ClientSession() as session:
            request = ImageGenerationRequest(
                inputs=synapse.inputs,
                step_indices=synapse.step_indices,
            )

            async with session.post(
                self.config.generation_endpoint,
                json=request.dict(),
            ) as response:
                response.raise_for_status()

                reader = MultipartReader.from_response(response)

                frames_tensor: bytes | None = None
                images: list[bytes] = []

                while True:
                    part = cast(BodyPartReader, await reader.next())

                    if part is None:
                        break

                    name = part.name

                    if not name:
                        continue

                    if name == "frames":
                        frames_tensor = await part.read(decode=True)
                    elif name.startswith("image_"):
                        index = int(name[len("image_"):])

                        while len(images) <= index:
                            # This is assuming that it will be overridden when the actual index is found
                            images.append(b"")

                        images[index] = base64.b64encode(await part.read(decode=True))

            synapse.output = ImageGenerationOutput(
                frames=base64.b64encode(frames_tensor) if frames_tensor else None,
                images=images,
            )

        return synapse

    async def blacklist_image(self, synapse: ImageGenerationSynapse) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority_image(self, synapse: ImageGenerationSynapse) -> float:
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", priority
        )
        return priority

    async def verify_image(self, synapse: ImageGenerationSynapse) -> None:
        if synapse.dendrite is None:
            raise SynapseDendriteNoneException()

        # Build the keypair from the dendrite_hotkey
        keypair = Keypair(ss58_address=synapse.dendrite.hotkey)

        # Build the signature messages.
        message = f"{synapse.dendrite.nonce}.{synapse.dendrite.hotkey}.{self.wallet.hotkey.ss58_address}.{synapse.dendrite.uuid}.{synapse.computed_body_hash}"

        if not keypair.verify(message, synapse.dendrite.signature):
            raise Exception(
                f"Signature mismatch with {message} and {synapse.dendrite.signature}"
            )

        # Build the unique endpoint key.
        endpoint_key = f"{synapse.dendrite.hotkey}:{synapse.dendrite.uuid}"

        async with self.nonce_lock:
            known_key = endpoint_key in self.nonces.keys()

            # Check the nonce from the endpoint key.
            if (
                known_key
                and self.nonces[endpoint_key] is not None
                and synapse.dendrite.nonce is not None
                and synapse.dendrite.nonce in self.nonces[endpoint_key]
            ):
                raise Exception("Duplicate nonce")

            if known_key:
                nonces = self.nonces[endpoint_key]
            else:
                nonces = set()
                self.nonces[endpoint_key] = nonces

            nonces.add(cast(int, synapse.dendrite.nonce))
