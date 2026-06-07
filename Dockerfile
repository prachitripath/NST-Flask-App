# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy requirements first to leverage Docker caching
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Grant full read/write permissions to the static directory
RUN chmod -R 777 /code/static

# Set environment variable to ensure logs are printed instantly
ENV PYTHONUNBUFFERED=1

# Expose Hugging Face's default port
EXPOSE 7860

# Run the Flask app using python
CMD ["python", "app.py"]