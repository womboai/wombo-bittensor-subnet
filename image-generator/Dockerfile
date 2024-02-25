FROM wombo-subnet:gpu-pipeline

COPY ./image-generator ./image-generator/

WORKDIR /app/image-generator/

RUN pip install --no-cache-dir -e .

ENTRYPOINT python -m image_generator.main $EXTRA_ARGS
