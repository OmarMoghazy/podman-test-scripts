FROM registry.fedoraproject.org/fedora:30

RUN dnf -y install openmpi openssh && \
    dnf clean all

COPY ring /home/ring

RUN chmod +x /home/ring
