import argparse
from typing import Dict, Any, List, Tuple

import bittensor as bt
import torch

from base.protocol import ImageGenerationSynapse
from base.utils.uids import get_random_uids


class Client:
    def __init__(self):
        parser = argparse.ArgumentParser()

        bt.subtensor.add_args(parser)

        # Netuid Arg: The netuid of the subnet to connect to.
        parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)

        self.config = bt.config(parser)

        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        # The metagraph holds the state of the network, letting us know about other validators and miners.
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")


def query_synapse(
    client: Client,
    input_parameters: Dict[str, Any],
    wallet_name: str,
    hotkey: str,
    network: str,
) -> List[Any]:
    uid = get_random_uids(client, k=1)[0]

    # create a wallet instance with provided wallet name and hotkey
    wallet = bt.wallet(name=wallet_name, hotkey=hotkey)

    # instantiate the metagraph with provided network and netuid
    metagraph = bt.metagraph(
        netuid=client.config.netuid, network=network, sync=True, lite=False
    )

    # Grab the axon you're serving
    axon = metagraph.axons[uid]

    # Create a Dendrite instance to handle client-side communication.
    dendrite = bt.dendrite(wallet=wallet)

    resp: Tuple[torch.Tensor, List[Any]] = dendrite.query(
        # Send the query to selected miner axons in the network.
        axons=[axon],
        # Construct a dummy query. This simply contains a single integer.
        synapse=ImageGenerationSynapse(input_parameters=input_parameters),
        # All responses have the deserialize function called on them before returning.
        # You are encouraged to define your own deserialization function.
        deserialize=True,
    )[0]

    return resp[1]


if __name__ == "__main__":
    client = Client()

    # TODO fastapi here
