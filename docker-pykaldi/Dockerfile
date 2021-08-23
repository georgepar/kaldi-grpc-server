# syntax=docker/dockerfile:1


#####################################################################################
#####################################################################################
##                                                                                 ##
##                      PYKALDI MULTISTAGE BUILD CONTAINER                         ##
##          FIRST BUILD OPENFST, PROTOBUF AND CLIF IN SEPARATE CONTAINERS          ##
##           AND THEN USE THE BUILD ARTIFACTS TO BUILD KALDI AND PYKALDI           ##
##                  THE RESULTING CONTAINER IS MINIMIZED (~4GB)                    ##
##                                                                                 ##
#####################################################################################
#####################################################################################


#####################################################################################
##                     BUILD OPENFST AND CREATE DEB FILE                           ##
#####################################################################################

FROM ubuntu:20.04 AS openfst_build
LABEL stage=builder

ARG OPENFST_VERSION=1.6.7
ARG NUM_BUILD_CORES=12
ENV OPENFST_VERSION ${OPENFST_VERSION}
ENV NUM_BUILD_CORES ${NUM_BUILD_CORES}


RUN apt-get update && \
    apt-get install -y wget make gcc g++ python-dev libz-dev checkinstall && \
    wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-${OPENFST_VERSION}.tar.gz && \
    tar -xvf openfst-${OPENFST_VERSION}.tar.gz -C / && \
    cd /openfst-${OPENFST_VERSION} && \
    ./configure --enable-static --enable-shared --with-pic --enable-far --enable-ngram-fsts --enable-lookahead-fsts && \
    make -j${NUM_BUILD_CORES} && \
    mkdir /packages && \
    checkinstall -Dy --install=no && mv *.deb /packages && \
    DESTDIR=/openfstinst make install && \
    apt-get remove --purge -y make gcc g++ python-dev libz-dev && \
    apt-get -y autoremove


#####################################################################################
##                  BUILD PROTOBUF AND CLIF DEB FILES AND WHEELS                   ##
#####################################################################################


FROM ubuntu:20.04 as protobufclif_build
LABEL stage=builder

COPY install_protobuf.sh /
COPY install_clif.sh /
COPY find_python_library.py /
RUN chmod +x /install_protobuf.sh
RUN chmod +x /install_clif.sh
RUN chmod +x /find_python_library.py

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y autoconf automake cmake curl g++ git checkinstall \
        graphviz libatlas3-base libtool make pkg-config subversion unzip wget zlib1g-dev \
        python3-dev python3-pip && \
    pip3 install -U pip setuptools numpy pydub pyparsing grpcio grpcio-tools && \
    /install_protobuf.sh $(which python3) && \
    /install_clif.sh $(which python3) &&  \
    apt-get remove --purge -y autoconf automake cmake curl g++ git graphviz libtool pkg-config make \
                            libatlas3-base subversion unzip wget zlib1g-dev checkinstall


#####################################################################################
##                      BUILD AND INSTALL KALDI AND PYKALDI                        ##
#####################################################################################


FROM ubuntu:20.04

COPY --from=protobufclif_build /packages /packages/
RUN mkdir /ofst/
COPY --from=openfst_build /packages/ /ofst/

ENV CLIF_MATCHER=/usr/clang/bin/clif-matcher
ENV PYCLIF=/usr/local/bin/pyclif
ENV PATH=/usr/local/bin:/usr/clang/bin/:$PATH
ENV KALDI_ROOT=/kaldi
ENV KALDI_DIR=/kaldi
ENV LC_ALL=C
ENV PATH=${KALDI_ROOT}/src/ivectorbin:${KALDI_ROOT}/src/online2bin:${KALDI_ROOT}/src/featbin:${PATH}

RUN apt-get update && \
   	DEBIAN_FRONTEND="noninteractive" apt-get install -y python3 python3-pip python2.7 autoconf automake \
    	cmake curl g++ git graphviz libatlas3-base libtool make \
    	pkg-config sox subversion unzip wget zlib1g-dev gfortran && \
	for debf in $(ls /packages/*.deb); do dpkg -i $debf; done && \
	for debf in $(ls /ofst/*.deb); do dpkg -i $debf; done && \
	for wheelf in $(ls /packages/*.whl); do pip install $wheelf; done && \
    git clone --depth 1 -b pykaldi_02 https://github.com/pykaldi/kaldi.git /kaldi && \
    cd /kaldi/tools && git pull && mkdir -p python && touch python/.use_default_python && \
    /kaldi/tools/extras/check_dependencies.sh && \
    /kaldi/tools/extras/install_mkl.sh && \
    cd /kaldi/src && \
    ./configure --shared --use-cuda=no --static-math --fst-root=/usr/local/ --fst-version=1.6.7 && \
    make -j$(cat /proc/cpuinfo | grep proc | wc -l) feat nnet3 ivector online2 rnnlm sgmm2 kws ivectorbin online2bin featbin rnnlmbin && \
    pip3 install -U pip setuptools numpy pydub pyparsing grpcio grpcio-tools dataclasses && \
    git clone https://github.com/pykaldi/pykaldi /pykaldi && cd /pykaldi && python3 setup.py install && \
   	apt-get remove --purge -y autoconf automake python3-pip \
        cmake curl g++ git graphviz libtool make gfortran \
        pkg-config sox subversion unzip wget zlib1g-dev && \
    apt-get autoremove -y && apt-get clean -y && apt-get autoclean -y && \
    find /kaldi -type f \( -name "*.o" -o -name "*.la" -o -name "*.a" \) -exec rm {} \; && \
    find /opt/intel -type f -name "*.a" -exec rm {} \; && \
    find /opt/intel -type f -regex '.*\(_mc.?\|_mic\|_thread\|_ilp64\)\.so' -exec rm {} \; && \
    rm -rf /kaldi/.git && rm -rf /pykaldi && \
    rm -rf /packages && rm -rf /ofst/
