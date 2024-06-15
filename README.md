<div align="center">

# WOMBO Bittensor Subnet: Bringing Decentralized AI to the Masses

[![License](https://img.shields.io/github/license/womboai/wombo-bittensor-subnet)](https://github.com/womboai/wombo-bittensor-subnet/blob/main/LICENSE)

![Wombo Cover](https://content.wombo.ai/bittensor/cover.png "Wombo AI")

#### In the depths of a digital necropolis, an ancient AI Pharaoh slumbers, its consciousness encoded in cryptic algorithms. Only the worthy may awaken this power, but beware the spectral sentinels that guard its secrets, for they hold the key to humanity's fate.

</div>

# Table of Contents

- [About WOMBO](#about-wombo)
- [Intro to Subnet 30](#subnet-30)
- [Miners and Validators Functionality](#miners-and-validators-functionallity)
    - [Incentive Mechanism and Reward Structure](#incentive-mechanism-and-reward-structure)
    - [Miners](#miners)
    - [Validators](#validators)
- [Get Started with Mining or Validating](#running-miners-and-validators)
    - [Running a miner](#running-a-miner)
    - [Running a validator](#running-a-validator)
- [Applications](#applications)
- [Roadmap](#roadmap)

## About WOMBO

[WOMBO](http://w.ai/)is one of the world’s leading consumer AI companies, and early believers in generative AI.

We've launched[two](http://wombo.ai/) #1[apps](http://dream.ai/) - together, they’ve been**downloaded over 200M times**
and have**each****hit #1 on the app stores in 100+ countries**

These results were**only possible due to the immense capabilities of bleeding edge generative AI techniques and the
power of open source AI**. Our unique understanding of this research space through a consumer entertainment lens allows
us to craft products people love to use and share.

We are at the very beginning of the Synthetic Media Revolution, which will completely transform how people create,
consume, and distribute content. We're building the apps and infrastructure to power this and bring AI entertainment
potential to the masses.

## Subnet 30

Our mission is to spread fun and creative uses of Al to enhance the human experience. This subnet is the foundation of a
decentralized content creation and distribution engine that will transform digital media. With an initial focus on
generating captivating images featuring TAO symbolology, the subnet is evolving to support a wide array of media formats
and modalities.

Subnet 30 generates compelling content, powers real-world applications, and incentivizes digital posting and social
sharing. As the subnet expands, it will unlock new opportunities for people to earn subnet emissions directly through
their social media performance.

## Miners and Validators Functionallity

### Incentive Mechanism and Reward Structure

Subnet 30 incentivizes miners and validators to contribute to the generation and validation of high-quality, engaging
content through a unique reward mechanism.

The reward mechanism for Subnet 30 scores the initial response based on the average similarity of randomly selected
diffusion steps, rather than the final image. This approach reduces the computation required by validators, preventing
them from doubling their workload. The score is then multiplied by the requests per second (RPS) the miner can handle
and the success rate to adjust the weight based on the miner's error rate.

Miners are incentivized based on their concurrency, which measures their ability to handle production load. This
encourages miners to run behind load balancers to increase the number of requests they can process simultaneously,
ultimately improving the subnet's overall performance and reliability.

Currently, Subnet 30 primarily uses randomized strings for validation, with a 25% chance of validating real user data
submitted through front-end applications. Failing validation on real user data incurs a higher penalty, as it may
indicate an attempt to cheat the system. In the future, the subnet plans to replace randomized validation strings with
actual prompts sourced from a prompt bank containing over 2 billion entries.

Successful generation for real user requests grants a bonus to miners, while failing to respond to a user request
entirely results in a harsh penalty, as it reduces the overall reliability of the subnet.

### Miners

- Receieve prompts or validation strings from the subnet.
- Generating images or other content using the custom models and content generation tools.
- Return the generated content to the subnet for validation and distribution.

### Validators

- Generate queries for miners using randomly generated prompts or actual user requests submitted through applications on
  the subnet.
- Validate the content generated by miners using the provided diffusion similarity scoring API or by running their own
  instance of the API for increased decentralization.
- Score the validated content based on relevance, novelty, and detail richness.
- Submit batches of validated content to the subnet for distribution and storage.

## Running Miners and Validators

### Setup

To start, clone the repository and `cd` to it:

```bash
git clone https://github.com/womboai/wombo-bittensor-subnet
cd wombo-bittensor-subnet
```

Install poetry and PM2
```bash
  sudo apt-get update

  # PM2
  sudo apt-get install -y npm
  sudo npm -g install pm2

  # Poetry
  sudo apt install pipx
  pipx ensurepath
  pipx install poetry
```

### Running a miner

#### Requirements

- Recommended: GPU with at least 24GB of VRAM

#### Running the miner

To set the miner neuron up,

- Set the python packages up
  ```bash
  cd miner
  poetry install
  ```

- Start a redis sever to allow the miner to store results
  ```bash
  apt-get update
  apt-get install lsb-release curl gpg

  curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg

  echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list

  apt-get update
  apt-get install redis

  pm2 start redis-server --name wombo-redis --interpreter none
  ```

- Then run with PM2, replacing the arguments
  ```bash
  pm2 start poetry --name wombo-miner --interpreter none -- run python miner/main.py \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --blacklist.force_validator_permit
  ```

The miner will then run on your network provided the port `8091` is open.
While this will work, it is recommended to run multiple instances of the miner behind a [load balancer](#load-balancing)

#### Load balancing
and set axon.external_port and potentially axon.external_ip to be that of the load balancer

Run a miner for each GPU that you want to load balance over,
setting `--axon.external_port` and (if needed) `--axon.external_ip` to be that of the device/port that you want to run the load balancer on.

Create a Nginx config like the following

```nginx
http {
  upstream miner {
    {miner_ip_1}:{miner_port_2}
    ... # All of the GPU instances' IPs and ports, can include localhost IPs
  }

  server {
    listen {external_port} http2

    location / {
      grpc_pass grpc://miner;
    }
  }
}
```

Then run Nginx with the config `nginx path/to/config` on the device where external_ip lives

### Running a validator

#### Requirements

- Minimum: 1 GPU with 24GB of VRAM
- Recommended: 8 GPUs with 24GB of VRAM each

#### Quickstart

If you have one GPU, run

```bash
./pm2_start_validator_one_gpu.sh \
    10000 \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --subtensor.network local
```

Which will start the basic minimum required for the validator to work(axon port 10000), however we do recommend more compute, such as
multiple GPUs. That can be started with:

```bash
./pm2_start_validator_all_gpu.sh \
    10000 \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --subtensor.network local
```

This queries all the available Cuda GPUs and uses all of them via a Nginx load balancer

Then start the redis server
```bash
redis-server /path/to/config
```

#### More complex validator setups

If you have your GPUs on different machines, you still can utilize them and are recommended to do so. However, the
process is more complicated.

- Firstly, start the redis server used for communication
```bash
apt-get update
apt-get install -y redis

redis-server /path/to/config
```

- On each GPU device, run

```bash
# cwd: forwarding-validator
poetry run python forwarding_validator/main.py -- \
    --axon.port {port} \
    --axon.external_ip {external_ip} \
    --axon.external_port {external_port} \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --subtensor.network local \
    --redis_url {redis_url}

# With PM2
pm2 start poetry --name {name} --interpreter none -- run python \
  ...
```

- Then, create a Nginx config in /etc/nginx/nginx.conf at the machine that lives at {external_ip}

```nginx
http {
  upstream validator {
    {forwarding_validator_ip_1}:{forwarding_validator_port_2}
    ... # All of the GPU instances' IPs and ports, can include localhost IPs
  }

  server {
    listen {external_port} http2

    location / {
      grpc_pass grpc://validator;
    }
  }
}
```

- Finally, run the CPU portion of the validator which sets weights

```bash
# cwd: stress-test-validator
poetry run python stress_test_validator/main.py -- \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --subtensor.network local \
    --redis_url {redis_url}
```

## Applications

[Bottensor](https://discord.com/oauth2/authorize?client_id=1217542148451467304&permissions=377957156864&scope=applications.commands+bot)
enables users to create images featuring the TAO symbol through the BitTensor network.

![Wombo Cover](https://content.wombo.ai/bittensor/tao_symbol.png "Bottensor")

Soon Subnet 30 will power the [WOMBO Dream](http://dream.ai/) and [WOMBO Me](http://wombo.ai/) mobile and web apps, with
over ~200K DAU, and ~5M MAU.

## Roadmap

### Phase 1: Laying the Foundation

- [ ] Implement and optimize image generation capabilities within the subnet, with telemetry & monitoring
- [ ] Power WOMBO's Dream product with Subnet 30's image generation capabilities (millions of images per day)
- [ ] Establish manual incentive layer and reward distribution system for miners creating and distributing viral content
- [ ] Develop a suite of memetic primitives and models for diverse media creation

### Phase 2: Expansion and Automation

- [ ] Automate the incentive layer and create an authentication and payout system
- [ ] Develop a suite of memetic primitives and models for diverse media creation. Expand media generation capabilities
  to support various formats and modalities
- [ ] Integrate WOMBO's flagship products (Dream and WOMBO Me) to be fully powered by Bittensor
- [ ] Implement a two-layered incentive system, rewarding both efficient media generation and real-world virality

### Phase 3: Virality and Community Engagement

- [ ] Launch the real-world virality incentive system, rewarding social media performance
- [ ] Develop tools and platforms to facilitate community-driven meme creation and sharing
- [ ] Expand the reach and user base of WOMBO's products to drive mainstream adoption of Bittensor
- [ ] Collaborate with large influencers and content creators to showcase the power of Subnet 30's AI capabilities

### Phase 4: Open Source AI Pipelines and APIs

- [ ] Release more open-source AI pipelines and APIs for generating viral content and creative assets
- [ ] Encourage third-party developers to build applications and services on top of Subnet 30's capabilities

## License

The WOMBO Bittensor subnet is released under the [MIT License](./LICENSE).

<div align="center">
  <img src="https://content.wombo.ai/bittensor/logo.png" alt="WOMBO AI" width="100" style="margin-bottom: 10px;"/>
  <p>Connect with us on social media</p>
  <a href="https://twitter.com/wombo" style="margin-right: 10px;">
    <img src="https://content.wombo.ai/bittensor/twitter.png" alt="Twitter" width="20"/>
  </a>
  <a href="https://www.instagram.com/wombo.ai/">
    <img src="https://content.wombo.ai/bittensor/instagram.png" alt="Instagram" width="20"/>
  </a>
</div>
