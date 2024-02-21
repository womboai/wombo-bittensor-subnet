FROM wombo-subnet:tensor

COPY ./neuron-selector ./neuron-selector/

WORKDIR /app/client/

RUN pip install --no-cache-dir -e ./neuron-selector
