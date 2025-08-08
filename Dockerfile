ENV NUMBA_DISABLE_CACHE=1

# Start from an official Python base image.
FROM python:3.9

# Set the working directory inside the container.
WORKDIR /code

# Copy the requirements file into the container.
COPY ./requirements.txt /code/requirements.txt

# Install the Python dependencies from the requirements file.
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy our main application code into the container.
COPY ./main.py /code/main.py

# Expose port 7860. Hugging Face Spaces uses this port by default.
EXPOSE 7860

# The command to run when the container starts.
# This tells `uvicorn` (our server) to run the `app` from `main.py`.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]