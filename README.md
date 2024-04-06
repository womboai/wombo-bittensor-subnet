# WOMBO Bittensor Subnet

[![License](https://img.shields.io/github/license/womboai/wombo-bittensor-subnet)](https://github.com/womboai/wombo-bittensor-subnet/blob/master/LICENSE)

![Wombo Cover](https://content.wombo.ai/bittensor/cover.png "Wombo AI")

# Table of Contents 

- [About WOMBO](#about-wombo)
- [Subnet 30](#subnet-30)
- [Running a miner](#running-a-miner)
- [Running a validator](#running-a-validator)

    
## About WOMBO
[WOMBO](http://w.ai/) is one of the world’s leading consumer AI companies, and earlier believers in generative AI.

We've launched [two](http://wombo.ai/) #1 [apps](http://wombo.art/) - together, they’ve been **downloaded over 200M times** and have **each** **hit #1 on the app stores in 100+ countries**

These results were **only possible due to the immense capabilities of bleeding edge generative AI techniques and the power of open source AI**. Our unique understanding of this research space through a consumer entertainment lens allows us to craft products people love to use and share.

We are at the very beginning of the Synthetic Media Revolution, which will completely transform how people create, consume, and distribute content. We're building the apps and infrastructure to power this and bring AI entertainment potential to the masses.


## Running a miner

To start, clone the repository and `cd` to it:
```bash
git clone https://github.com/womboai/wombo-bittensor-subnet
cd wombo-bittensor-subnet
```

### The image generator API
Miners by default use an API for image generation, this can be set up as follows, with either PM2 or docker:

#### PM2
Set the python packages up

```bash
cd image-generator
./setup.sh
```

Then run with PM2
```bash
pm2 start run.sh --name wombo-image-generator --interpreter bash
```

#### Docker
Create a .env file from example.env

```bash
cd image-generator
# Copy the example environment file
cp example.env .env
```

Then simply run the image generator.
```bash
./run_docker.sh
```

### Running the miner neuron

To set the miner neuron up,

#### PM2
Set the python packages up

```bash
cd miner
./setup.sh
```

Then run with PM2, replacing the arguments 
```bash
pm2 start run.sh --name wombo-miner --interpreter bash -- \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --generation_endpoint http://localhost:8001/api/generate \
    --blacklist.force_validator_permit
```

#### Docker
Create a .env file from example.env
```bash
cd miner
# Copy the example environment file and edit it
cp example.env .env
$EDITOR .env
```

Then simply run the registered miner
```bash
./run_docker.sh
```

The miner will then run on your network provided the port `8091` is open.
The responsibility of keeping the miner/repo up-to-date falls on you, hence you should `git pull` every once in a while.

## Running a validator
Running a validator is similar to running a miner, with either PM2 or Docker:

#### PM2
Set the python packages up

```bash
cd validator
./setup.sh
```

Then run with PM2, replacing the arguments 
```bash
run_pm2.sh wombo-validator \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey}
```

(wombo-validator is the PM2 process name)

#### Docker
Create a .env file from example.env

```bash
cd validator
# Copy the example environment file and edit it
cp example.env .env
$EDITOR .env
```

Then simply run the registered validator with
```bash
./run_docker.sh
```

The run script will keep your validator up to date, so as long as the .env file is correct, everything should run properly.

### Validator Decentralization
To increase decentralization, running validator-api locally is recommended.
This requires a GPU.

#### PM2
Set the python packages up

```bash
cd validator-api
./setup.sh
```

Run the validator API
```bash
NETWORK={network} NETUID={netuid} pm2 start run.sh --name wombo-validator-api --interpreter bash
```

#### Docker
Create a .env file from example.env
```bash
cd validator-api
# Copy the example environment file and edit it
cp example.env .env
$EDITOR .env
```

Run the validator API
```bash
./run_docker.sh
```

Then finally add `--validation_endpoint http://localhost:8001/api/validate` to the validator arguments, changing the port and host as necessary.
