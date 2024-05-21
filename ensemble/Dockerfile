FROM nvcr.io/nvidia/tritonserver:23.12-py3

#Install any additional libraries
RUN echo "Adding conda package to Docker image"
RUN mkdir -p /home/condpackenv/

# Copy conda env
COPY transformers_env.tar.gz /home/condpackenv/transformers_env.tar.gz
