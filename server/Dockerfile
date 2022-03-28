FROM georgepar/pykaldi:latest

SHELL ["/bin/bash", "-c"]

COPY . /kaldigrpc
WORKDIR /kaldigrpc

RUN mkdir /data && \
    apt-get update -y && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi && \
    pip3 install . && \
    apt-get remove --purge -y python3-pip && \
    apt-get autoremove -y && \
    apt-get clean && \
    apt-get autoclean

RUN sed -i 's/^\(--mfcc-config=\).*/\1\/kaldigrpc\/model\/conf\/mfcc.conf/' /kaldigrpc/model/conf/online.conf
RUN sed -i 's/^\(--ivector-extraction-config=\).*/\1\/kaldigrpc\/model\/conf\/ivector_extractor.conf/' /kaldigrpc/model/conf/online.conf
RUN sed -i 's/^\(--cmvn-config=\).*/\1\/kaldigrpc\/model\/conf\/online_cmvn.conf/' /kaldigrpc/model/conf/online.conf
RUN sed -i 's/^\(--global-cmvn-stats=\).*/\1\/kaldigrpc\/model\/global_cmvn.stats/' /kaldigrpc/model/conf/online.conf

RUN sed -i 's/^\(--splice-config=\).*/\1\/kaldigrpc\/model\/conf\/splice.conf/' /kaldigrpc/model/conf/ivector_extractor.conf
RUN sed -i 's/^\(--cmvn-config=\).*/\1\/kaldigrpc\/model\/conf\/online_cmvn.conf/' /kaldigrpc/model/conf/ivector_extractor.conf
RUN sed -i 's/^\(--lda-matrix=\).*/\1\/kaldigrpc\/model\/ivector_extractor\/final.mat/' /kaldigrpc/model/conf/ivector_extractor.conf
RUN sed -i 's/^\(--global-cmvn-stats=\).*/\1\/kaldigrpc\/model\/ivector_extractor\/global_cmvn.stats/' /kaldigrpc/model/conf/ivector_extractor.conf
RUN sed -i 's/^\(--diag-ubm=\).*/\1\/kaldigrpc\/model\/ivector_extractor\/final.dubm/' /kaldigrpc/model/conf/ivector_extractor.conf
RUN sed -i 's/^\(--ivector-extractor=\).*/\1\/kaldigrpc\/model\/ivector_extractor\/final.ie/' /kaldigrpc/model/conf/ivector_extractor.conf

ENTRYPOINT ["kaldigrpc-serve", "--model-dir=/kaldigrpc/model"]
