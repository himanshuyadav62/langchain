# Use Python 3.12.4 as the base image
FROM python:3.12.4-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir flask requests

# Install the latest stable CPU-only version of PyTorch
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install transformers
RUN pip install --no-cache-dir transformers

# Install Sentence Transformers
RUN pip install --no-cache-dir sentence-transformers

# Install Flask-Cors
RUN pip install --no-cache-dir flask-cors

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]