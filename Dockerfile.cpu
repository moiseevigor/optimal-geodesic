FROM pytorch/pytorch

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
        #  python-tk && \
     rm -rf /var/lib/apt/lists/*

RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime
RUN pip install git+https://github.com/rtqichen/torchdiffeq

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-tk && \
        # python3-pyqt5 && \
     rm -rf /var/lib/apt/lists/*

WORKDIR /app
