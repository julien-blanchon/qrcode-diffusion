FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements.txt

# Copy source code
WORKDIR /app
RUN mkdir /app/cache
COPY ./app.py /app/app.py

# Run the application
CMD ["python", "app.py"]