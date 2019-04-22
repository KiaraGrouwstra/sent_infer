FROM pytorch/pytorch:nightly-runtime-cuda10.0-cudnn7

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /tmp

RUN conda env create -f /tmp/environment.yml
RUN source activate sent
RUN pip install -r requirements.txt

# CMD ["python", "/app/src/train.py"]
