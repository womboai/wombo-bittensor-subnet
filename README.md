# wombo-bittensor-subnet

## Running a miner

To start, clone the repository and `cd` to it:
```bash
git clone https://github.com/womboai/wombo-bittensor-subnet
cd wombo-bittensor-subnet
```

### The image generator API
Miners by default use an API for image generation, the image generator to set the image generator up, create a .env file from example.env
Copy the example environment file and edit it
```bash
cd image-generator
# Copy the example environment file
cp example.env .env
```

Then simply run the image generator.
```bash
./run.sh
```

### Running the miner neuron
To set the miner neuron up,
```bash
cd miner
# Copy the example environment file and edit it
cp example.env .env
$EDITOR .env
```

Then simply run the registered miner
```bash
./run.sh
```

The miner will then run on your network provided the port `8091` is open.
The responsibility of keeping the miner/repo up-to-date falls on you, hence you should `git pull` every once in a while.

## Running a validator
Running a validator is similar to running a miner, start by being in the `validator` directory and then editing the .env file

```bash
cd validator
# Copy the example environment file and edit it
cp example.env .env
$EDITOR .env
```

Then simply run the registered validator with
```bash
./run.sh
```

The run script will keep your validator up to date, so as long as the .env file is correct, everything should run properly.
