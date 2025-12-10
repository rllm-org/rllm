FROM verlai/verl:vllm011.latest

WORKDIR /workspace

RUN git clone https://github.com/volcengine/verl.git
RUN cd verl && \
    git checkout v0.6.1 && \
    pip install -e .

# 2) Install rllm (editable)
RUN git clone https://github.com/rllm-org/rllm.git
RUN cd rllm && \
    pip install -e .

# 3) Install playwright
RUN pip install playwright && \
    playwright install chromium && \
    playwright install-deps

CMD ["/bin/bash"]

# Docker Usage
# docker build -t rllm .
# docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/rllm -v /tmp:/tmp --name rllm-container rllm sleep infinity
# docker start rllm-container
# docker exec -it rllm-container bash
