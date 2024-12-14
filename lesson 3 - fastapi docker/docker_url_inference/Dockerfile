# Start from the base image
FROM python:3.11

# Install libGL for OpenCV
RUN apt-get update && apt-get install -y libgl1

# Set the working directory
WORKDIR /app

# Install CMake and other necessary build tools
RUN apt-get update && \
    apt-get install -y cmake && \
    apt-get install -y build-essential && \
    apt-get clean

# Copy the requirements and install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application files
COPY . /app

# Command to run your application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]