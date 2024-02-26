# wombo-bittensor-subnet

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
run_pm2.sh \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey}
```

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
