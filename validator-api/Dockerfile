FROM wombo-subnet:gpu-pipeline

COPY ./validator-api ./validator-api/

WORKDIR /app/validator-api/

RUN pip install --no-cache-dir -e .

ENTRYPOINT python -m validator_api.main $EXTRA_ARGS
