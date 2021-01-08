FROM nvcr.io/nvidia/tensorrt:19.10-py3

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO melodic

###################################### user #####################################

ENV SHELL=/bin/bash \
    USER=arg-medical \
    UID=1000 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV HOME=/home/${USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${UID} \
    ${USER} 

RUN echo "root:root" | chpasswd
RUN echo "${USER}:111111" | chpasswd

###################################### basic tools #####################################

RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -yq dist-upgrade \
    && apt-get -o Acquire::ForceIPv4=true install -yq --no-install-recommends \
    locales \
    cmake \
    make \
    git \
    vim \
    gedit \
    wget \
    sudo \
    lsb-release \
    build-essential \
    net-tools \
    dirmngr \
    gnupg2 \
    mercurial \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    libpcap-dev \
    python-gtk2 \
    python-gobject \
    python-tk \
    python-pip \
    python3-pip \
    python3-setuptools \
    python3-opencv \
    python3-numpy \
    python3-empy  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

###################################### ROS #####################################

# setup sources.list
# RUN echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list \
#    && echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654


# install bootstrap tools
RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -o Acquire::ForceIPv4=true install --no-install-recommends -y \
    ros-$ROS_DISTRO-desktop-full \
    ros-$ROS_DISTRO-serial \
    ros-$ROS_DISTRO-soem \
    ros-$ROS_DISTRO-openslam-gmapping \
    ros-$ROS_DISTRO-geodesy \
    ros-$ROS_DISTRO-cartographer-* \
    ros-$ROS_DISTRO-ddynamic-reconfigure \
    ros-$ROS_DISTRO-perception \
    ros-$ROS_DISTRO-rgbd-launch \
    ros-$ROS_DISTRO-joystick-drivers \
    ros-$ROS_DISTRO-pointcloud-to-laserscan \
    ros-$ROS_DISTRO-robot-localization \
    ros-$ROS_DISTRO-spacenav-node \
    ros-$ROS_DISTRO-tf2-sensor-msgs \
    ros-$ROS_DISTRO-controller-manager \
    ros-$ROS_DISTRO-twist-mux \
    ros-$ROS_DISTRO-velodyne-simulator \
    ros-$ROS_DISTRO-gazebo-ros \
    ros-$ROS_DISTRO-move-base-msgs \
    ros-$ROS_DISTRO-lms1xx \
    ros-$ROS_DISTRO-rviz-imu-plugin \
    ros-$ROS_DISTRO-interactive-marker-twist-server \
    ros-$ROS_DISTRO-teleop-twist-joy \
    ros-$ROS_DISTRO-rosserial-arduino \
    ros-$ROS_DISTRO-rosserial \
    ros-$ROS_DISTRO-socketcan-interface \
    ros-$ROS_DISTRO-moveit-planners-ompl \
    ros-$ROS_DISTRO-moveit-* \
    ros-$ROS_DISTRO-moveit \
    ros-$ROS_DISTRO-moveit-visual-tools \
    ros-$ROS_DISTRO-velocity-controllers \
    ros-$ROS_DISTRO-joint-state-controller \
    ros-$ROS_DISTRO-joint-trajectory-controller \
    ros-$ROS_DISTRO-gazebo-ros-control \
    ros-$ROS_DISTRO-geographic-msgs \
    ros-$ROS_DISTRO-force-torque-sensor-controller \
    ros-$ROS_DISTRO-industrial-msgs \
    ros-$ROS_DISTRO-effort-controllers \
    ros-$ROS_DISTRO-nmea-msgs \
    ros-$ROS_DISTRO-rosserial-server \
    ros-$ROS_DISTRO-dynamixel-workbench-toolbox \
    ros-$ROS_DISTRO-trac-ik \
    python-rosdep \
    python-rosinstall \
    python-rosinstall-generator \
    python-wstool \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init \
    && rosdep update

##################################### PIP3 ######################################

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py --force-reinstall

RUN pip3 install \
    digi-xbee==1.3.0 \
    pyyaml \
    rospkg \
    cython \
    catkin-tools \
    matplotlib \
    pandas \
    pypozyx \
    requests \
    scikit-image \
    scikit-learn \
    gdown \
    seaborn \
    modern_robotics \ 
    chumpy \ 
    trimesh \
    pycpd \
    easydict \
    shapely \
    lmdb \ 
    colour \
    tqdm \
    pycocotools

RUN pip3 install \
    torch==1.5.0+cu101 \
    torchvision==0.6.0+cu101 \
    -f https://download.pytorch.org/whl/torch_stable.html

##################################### PIP2 ######################################

RUN pip2 install --upgrade pip setuptools

RUN pip2 install \
    modern_robotics

#################################### GTSAM ####################################

RUN cd ${HOME}/ \
    && git clone https://github.com/borglab/gtsam.git \
    && cd gtsam/ \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install

####################################### procman ###########################################

RUN cd ${HOME} && git clone https://github.com/lcm-proj/lcm \
    && cd lcm \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install

RUN cd ${HOME} && git clone http://github.com/ARG-NCTU/procman \
    && cd procman \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install

######################################## realsense ########################################

RUN apt-get update && apt-get install -y \
    libsuitesparse-dev\
    software-properties-common \
    libusb-1.0-0-dev \
    libx11-dev \
    xorg-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# set the version of the realsense library
ENV LIBREALSENSE_VERSION 2.36.0
RUN wget https://github.com/IntelRealSense/librealsense/archive/v${LIBREALSENSE_VERSION}.tar.gz && \
    tar -xvzf v${LIBREALSENSE_VERSION}.tar.gz && \
    rm v${LIBREALSENSE_VERSION}.tar.gz && \
    mkdir -p librealsense-${LIBREALSENSE_VERSION}/build && \
    cd librealsense-${LIBREALSENSE_VERSION}/build && \
    cmake .. && \
    make && \
    make install && \
    rm -rf librealsense-${LIBREALSENSE_VERSION}

######################################## torch2trt ########################################

RUN cd ${HOME} && git clone https://github.com/NVIDIA-AI-IOT/torch2trt \
    && cd torch2trt \
    && python3 setup.py install --plugins \
    && cd ${HOME} \
    && rm -rf torch2trt

######################################## trt_pose ########################################

RUN cd ${HOME} && git clone https://github.com/NVIDIA-AI-IOT/trt_pose \
    && cd trt_pose \
    && python3 setup.py install \
    && cd ${HOME} \
    && rm -rf trt_pose

##################################### setting #####################################
RUN chown -R ${USER}:${USER} ${HOME}/
RUN echo "arg-medical ALL=(ALL)  ALL" > /etc/sudoers

# Support of nvidia-docker 2.0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# setup entrypoint
COPY ./ros_entrypoint.sh /

ENTRYPOINT ["/ros_entrypoint.sh"]

# Pytorch model
COPY ./resnet18-5c106cde.pth /home/arg-medical/.cache/torch/checkpoints/

USER ${USER}

WORKDIR ${HOME}
