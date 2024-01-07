# Use an official Python runtime as a parent image
FROM python:3.11.4-slim

# Update and install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app
# Copy the setup script into the container
COPY setup.sh .

# Give execute permission to the script and run it
RUN chmod +x ./setup.sh && ./setup.sh

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Make port 80 available to the world outside this container
EXPOSE 80

# The container starts with a bash shell by default
CMD ["/bin/bash"]
