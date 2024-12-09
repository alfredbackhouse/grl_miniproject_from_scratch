FROM python:3.9

WORKDIR /app

ARG USERNAME=appuser
ARG UID=1000
ARG GID=1000

RUN groupadd -g $GID $USERNAME && \
    useradd -u $UID -g $GID -ms /bin/bash $USERNAME

USER $USERNAME

COPY requirements.txt /app

RUN pip3 install -r requirements.txt

ENV HF_HUB_CACHE="/app/cache"

ENV HF_HOME="/app/cache"

ENV PYTHONPATH="/app"

EXPOSE 8888

COPY . .

CMD ["python", "deepdiffusion.py"]

# docker build --build-arg USERNAME=$(whoami) --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t ${USER}/grl-miniproject .

# docker run -it --rm --gpus '"device=0"' -v $(pwd):/app --name ${USER}_diffusercontainer --entrypoint python ${USER}_diffusers --train --evaluate
# options to set checkpionts and skip full evaluation loss

# docker run -d --rm --gpus '"device=0"' \
#   -p 8888:8888 \
#   -v $(pwd):/app \
#   --entrypoint jupyter \
#   alfie_diffusers \
#   notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# docker run -it --rm --gpus '"device=1"' \
#   -v $(pwd):/app \
#   --name ${USER}_grlcontainer \
#   ${USER}/grl-miniproject \
#   /bin/bash