FROM wombo-subnet:gpu-pipeline

COPY ./validator-api ./validator-api/

WORKDIR /app/validator-api/

RUN pip install -e .

EXPOSE 8001

ENTRYPOINT python -m validator_api.main
