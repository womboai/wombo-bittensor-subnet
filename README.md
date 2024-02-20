# wombo-bittensor-subnet

## Running a miner

### The image generator API
Miners by default use an API for image generation, the image generator is simple to run and does not require any special configuration
```bash
cd image-generator
./run.sh
```

### Running the miner neuron
To actually run a registered miner, go to the working directory `miner`
```bash
cd miner
```

Copy the example environment file and edit it
```bash
cp example.env .env
$EDITOR .env
```

Then simply run the miner
```bash
./run.sh
```

The miner will then run on your network provided the port `8091` is open.
The responsibility of keeping the miner/repo up-to-date falls on you, hence you should `git pull` every once in a while.

## Running a validator
Running a validator is similar to running a miner, start by being in the `validator` directory and then editing the .env file

```bash
cd validator
cp example.env .env
$EDITOR .env
```

Then simply run with
```bash
./run.sh
```

The run script will keep your validator up to date, so as long as the .env file is correct, everything should run properly.
