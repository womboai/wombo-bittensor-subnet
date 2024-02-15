FROM wombo-subnet:tensor

WORKDIR /app/

RUN pip config set global.find-links "$(pip config get global.find-links) /app/neuron"

COPY ./neuron ./neuron/

RUN pip install --no-cache-dir ./neuron
