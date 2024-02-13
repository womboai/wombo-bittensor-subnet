FROM wombo-subnet:gpu-pipeline

COPY ./image-generator ./image-generator/

WORKDIR /app/image-generator/

RUN pip install -e .

EXPOSE 8001

ENTRYPOINT python -m image_generator.main
