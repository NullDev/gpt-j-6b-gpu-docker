FROM nvidia/cuda:12.1.0-base-ubuntu20.04
RUN apt update\
  && apt install -y python3 python3-pip wget git zstd curl\
  && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y nvidia-cuda-toolkit
RUN wget -c https://the-eye.eu/public/AI/GPT-J-6B/step_383500_slim.tar.zstd\
  && tar -I zstd -xf step_383500_slim.tar.zstd\
  && rm step_383500_slim.tar.zstd
RUN git clone https://github.com/kingoflolz/mesh-transformer-jax.git
RUN pip3 install -r mesh-transformer-jax/requirements.txt
RUN pip3 install torch mesh-transformer-jax/ jax==0.2.12 jaxlib==0.1.68 -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN mkdir gpt-j-6B &&\
  curl https://gist.githubusercontent.com/NullDev/d494a309221abbedf77af87305c52402/raw/1e03002091322c7290d7ddeac3f9194df1ceebd1/gpt-j-6b.json > gpt-j-6B/config.json
COPY converttotorch.py ./
RUN python3 converttotorch.py
RUN pip3 install fastapi pydantic uvicorn && pip3 install numpy --upgrade && pip3 install git+https://github.com/finetuneanon/transformers@gpt-j
RUN rm -rf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1
RUN pip3 install protobuf==3.20.*
COPY web.py ./
COPY model.py ./
CMD uvicorn web:app --port 8080 --host 0.0.0.0
