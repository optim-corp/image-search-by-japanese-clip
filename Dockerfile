FROM continuumio/miniconda3

ARG CONDA_ENV=clip
RUN conda create -n ${CONDA_ENV} python==3.9

ENV CONDA_DEFAULT_ENV=${CONDA_ENV}
RUN echo "conda activate ${CONDA_ENV}" >> ~/.bashrc
ENV PATH /opt/conda/envs/${CONDA_ENV}/bin:$PATH
RUN conda install pytorch==1.11.0 torchvision==0.12.0  faiss-cpu==1.7.2 cpuonly -c pytorch -c conda-forge --override-channels \
    && conda clean --all \
    && rm -rf /opt/conda/pkgs/*

ADD requirements.txt /opt/requirements.txt
RUN cd /opt && pip install -U pip \
    && pip install --no-cache-dir -U -r /opt/requirements.txt \
    && pip cache purge

COPY . /clip
WORKDIR /clip

RUN python create_index.py

EXPOSE 8501
CMD streamlit run main.py
