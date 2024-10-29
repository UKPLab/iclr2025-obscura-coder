FROM determinedai/environments:py-3.10-base-cpu-mpi-0.31.1

SHELL ["/bin/bash", "-c"]

ENV PYTHONUNBUFFERED=1 \
    GOPATH="/container/go" \
    GO111MODULE="off" \
    CARGO_HOME="/container/cargo" \
    RUSTUP_HOME="/container/rustup"

# Setup System Utilities and Languages: C, C++, Java, Lua, Perl, R, Ruby, Scala and lang-specific dependencies like Boost (C++)
RUN apt-get update --yes --quiet \
    && apt-get upgrade --yes --quiet \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
        apt-utils \
        autoconf \
        automake \
        bc \
        build-essential \
        ca-certificates \
        check \
        cmake \
        curl \
	openjdk-21-jdk-headless \
	openjdk-21-jre-headless \
	dmidecode \
        emacs \
        g++\
        gcc \
        git \
	htop \
	ifstat \
        iproute2 \
	iptables \
	iputils-ping \
        jq \
        kmod \
        libaio-dev \
        libboost-all-dev \
        libcurl4-openssl-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        libibverbs-dev \
        libnuma-dev \
        libnuma1 \
        libomp-dev \
        libsm6 \
        libssl-dev \
        libsubunit-dev \
        libsubunit0 \
        libtest-deep-perl \
        libtool \
        libxext6 \
        libxrender-dev \
        lsof \
        lua5.3 \
        lua-unit \
        make \
        moreutils \
	netcat \
        net-tools \
        ninja-build \
        openssh-client \
        openssh-server \
        openssl \
        php-cli \
        pkg-config \
        python3-dev \
        r-base \
        ruby \
        scala \
        software-properties-common \
        sudo \
	traceroute \
        unzip \
        util-linux \
        vim \
        wget \
        zlib1g-dev \
    && apt-get autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setup Go and its testing dependencies
RUN add-apt-repository --yes ppa:longsleep/golang-backports \
    && apt-get update --yes --quiet \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends golang-1.18 \
    && ln -s /usr/lib/go-1.18/bin/go /usr/bin/go \
    && go get github.com/stretchr/testify/assert

# Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH=/container/cargo/bin:$PATH

# JS/TS
RUN curl -fsSL https://deb.nodesource.com/setup_current.x | bash -
RUN apt-get install -y nodejs
RUN npm install -g typescript

# Dlang
RUN wget https://netcologne.dl.sourceforge.net/project/d-apt/files/d-apt.list -O /etc/apt/sources.list.d/d-apt.list
RUN apt-get update --allow-insecure-repositories
RUN apt-get -y --allow-unauthenticated install --reinstall d-apt-keyring
RUN apt-get update && apt-get install -yqq dmd-compiler dub

# C#
RUN apt install gnupg ca-certificates
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF
RUN echo "deb https://download.mono-project.com/repo/ubuntu stable-focal main" | tee /etc/apt/sources.list.d/mono-official-stable.list
RUN apt update
RUN apt install -yqq mono-devel

# Swift
RUN curl https://download.swift.org/swift-5.10.1-release/ubuntu2004/swift-5.10.1-RELEASE/swift-5.10.1-RELEASE-ubuntu20.04.tar.gz | tar xz -C /container/
ENV PATH="/container/swift-5.10.1-RELEASE-ubuntu20.04/usr/bin:${PATH}"

# Julia
RUN curl https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.4-linux-x86_64.tar.gz | tar xz -C /container/
ENV PATH="/container/julia-1.10.4/bin:${PATH}"

# JavaTuples
RUN mkdir /container/multipl-e && wget https://repo.mavenlibs.com/maven/org/javatuples/javatuples/1.2/javatuples-1.2.jar -O /container/multipl-e/javatuples-1.2.jar

