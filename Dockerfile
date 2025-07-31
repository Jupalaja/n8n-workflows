# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code to the working directory
COPY . .

# The command to run when the container starts.
# Binds to 0.0.0.0 to be accessible from outside the container.
CMD ["python", "run.py", "--host", "0.0.0.0"]
