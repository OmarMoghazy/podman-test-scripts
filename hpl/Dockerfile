FROM krccl/openmpi_base:403

LABEL Description="This is a cross-platform HPL container built with OpenMPI. HPL or High Performance Linpack is a known benchmark to test the capability of a computer. For more information, please refer to https://www.netlib.org/benchmark/hpl" Version="2.3.0"

MAINTAINER mohsin.shaikh@kaust.edu.sa

RUN apt-get install -y libatlas-base-dev gfortran openssh-client

WORKDIR /hpl
RUN curl -o hpl-2.3.tar.gz -L http://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz 
RUN tar xvf hpl-2.3.tar.gz -C /hpl --strip-component=1
ADD ./Make.ubuntu /hpl
RUN make arch=ubuntu
WORKDIR /
CMD ["/hpl/bin/ubuntu/xhpl"]

