FROM public.ecr.aws/lambda/python:3.12

# Install system-level dependencies required for dlib and other Python packages
RUN microdnf update -y && \
    microdnf install -y \
    gcc \
    gcc-c++ \
    cmake \
    make \
    libX11-devel \
    libXext-devel \
    libSM-devel \
    libXrender-devel \
    openblas-devel \
    lapack-devel \
    boost-devel \
    openssl-devel && \
    microdnf clean all

# Set the working directory
WORKDIR /var/task

# configure AWS as remote storage
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
# aws credentials
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

# Set language environment variables
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Copy the requirements and install dependencies
COPY requirements.txt /var/task
RUN pip install --no-cache-dir -r /var/task/requirements.txt

# Install DVC with S3 support
RUN pip install "dvc[s3]"   # since s3 is the remote storage

# Copy the rest of the application files to /app
COPY . /var/task

# Initialize DVC and configure remote storage
RUN dvc init --no-scm -f
RUN dvc remote add -d model-ckpt s3://models-dvc-remote/trained_models_ckpt/

# Set up permissions and pull the model from s3 
RUN dvc pull models/best_checkpoint.ckpt.dvc

# Run lambda_handler
CMD [ "lambda_handler.lambda_handler"]