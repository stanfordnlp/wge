FROM kelvinguu/pytorch:1.3

# Add the PostgreSQL PGP key to verify their Debian packages.
# It should be the same key as https://www.postgresql.org/media/keys/ACCC4CF8.asc
RUN apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys B97B0AFCAA1A47F044F244A07FCC7D46ACCC4CF8

# Add PostgreSQL's repository. It contains the most recent stable release of PostgreSQL, ``9.3``.
RUN echo "deb http://apt.postgresql.org/pub/repos/apt/ precise-pgdg main" > /etc/apt/sources.list.d/pgdg.list

# Install ``python-software-properties``, ``software-properties-common`` and PostgreSQL 9.3
# There are some warnings (in red) that show up during the build. You can hide
# them by prefixing each apt-get statement with DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python-software-properties software-properties-common postgresql-9.3 postgresql-client-9.3 postgresql-contrib-9.3

RUN apt-get update
RUN apt-get --yes --force-yes install libffi6 libffi-dev libssl-dev libpq-dev git

RUN pip install --upgrade pip
RUN pip install jupyter
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension  # add Jupyter notebook extension

RUN pip install fabric
RUN pip install pyOpenSSL==16.2.0
RUN pip install psycopg2==2.6.1
RUN pip install SQLAlchemy==1.1.0b3
RUN pip install cherrypy==8.1.2
RUN pip install bottle==0.12.10
RUN pip install boto==2.43.0

RUN pip install requests
RUN pip install nltk==3.2.3
RUN python -m nltk.downloader punkt  # download tokenizer data

RUN pip install keras==1.1.0
RUN pip install pyhocon line_profiler pytest tqdm faulthandler python-Levenshtein gitpython futures jsonpickle prettytable tensorboard_logger
RUN pip install Pillow==4.1
RUN pip install selenium==3.4.3

RUN apt-get install -y vim less tmux nmap wget unzip
COPY .tmux.conf /root

# vim bindings for Jupyter
# https://github.com/lambdalisue/jupyter-vim-binding
RUN mkdir -p $(jupyter --data-dir)/nbextensions
RUN git clone https://github.com/lambdalisue/jupyter-vim-binding $(jupyter --data-dir)/nbextensions/vim_binding
RUN jupyter nbextension enable vim_binding/vim_binding

# autoreload for Jupyter
RUN ipython profile create
RUN echo 'c.InteractiveShellApp.exec_lines = []' >> ~/.ipython/profile_default/ipython_config.py
RUN echo 'c.InteractiveShellApp.exec_lines.append("%load_ext autoreload")' >> ~/.ipython/profile_default/ipython_config.py
RUN echo 'c.InteractiveShellApp.exec_lines.append("%autoreload 2")' >> ~/.ipython/profile_default/ipython_config.py

#===========================================================================================================
# COPIED FROM: https://github.com/SeleniumHQ/docker-selenium/blob/3.4.0-einsteinium/Base/Dockerfile
#===========================================================================================================

#================================================
# Customize sources for apt-get
#================================================
RUN  echo "deb http://archive.ubuntu.com/ubuntu xenial main universe\n" > /etc/apt/sources.list \
  && echo "deb http://archive.ubuntu.com/ubuntu xenial-updates main universe\n" >> /etc/apt/sources.list \
  && echo "deb http://security.ubuntu.com/ubuntu xenial-security main universe\n" >> /etc/apt/sources.list

# No interactive frontend during docker build
ENV DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NONINTERACTIVE_SEEN=true

#========================
# Miscellaneous packages
# Includes minimal runtime used for executing non GUI Java programs
#========================
RUN apt-get -qqy update \
  && apt-get -qqy --no-install-recommends install \
    bzip2 \
    ca-certificates \
    openjdk-8-jre-headless \
    tzdata \
    sudo \
    unzip \
    wget \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

#===================
# Timezone settings
# Possible alternative: https://github.com/docker/docker/issues/3359#issuecomment-32150214
#===================
ENV TZ "UTC"
RUN echo "${TZ}" > /etc/timezone \
  && dpkg-reconfigure --frontend noninteractive tzdata

#==========
# Selenium
#==========
RUN  sudo mkdir -p /opt/selenium \
  && wget --no-verbose https://selenium-release.storage.googleapis.com/3.4/selenium-server-standalone-3.4.0.jar \
    -O /opt/selenium/selenium-server-standalone.jar

#===========================================================================================================
# COPIED FROM: https://github.com/SeleniumHQ/docker-selenium/blob/3.4.0-einsteinium/NodeBase/Dockerfile
#===========================================================================================================

#==============
# VNC and Xvfb
#==============
RUN apt-get update -qqy \
  && apt-get -qqy install \
    locales \
    xvfb \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

#============================
# Some configuration options
#============================
ENV SCREEN_WIDTH 1360
ENV SCREEN_HEIGHT 1020
ENV SCREEN_DEPTH 24
ENV DISPLAY :99.0

#========================
# Selenium Configuration
#========================
# As integer, maps to "maxInstances"
ENV NODE_MAX_INSTANCES 1
# As integer, maps to "maxSession"
ENV NODE_MAX_SESSION 1
# As integer, maps to "port"
ENV NODE_PORT 5555
# In milliseconds, maps to "registerCycle"
ENV NODE_REGISTER_CYCLE 5000
# In milliseconds, maps to "nodePolling"
ENV NODE_POLLING 5000
# In milliseconds, maps to "unregisterIfStillDownAfter"
ENV NODE_UNREGISTER_IF_STILL_DOWN_AFTER 60000
# As integer, maps to "downPollingLimit"
ENV NODE_DOWN_POLLING_LIMIT 2
# As string, maps to "applicationName"
ENV NODE_APPLICATION_NAME ""

# Following line fixes https://github.com/SeleniumHQ/docker-selenium/issues/87
ENV DBUS_SESSION_BUS_ADDRESS=/dev/null

#===========================================================================================================
# COPIED FROM: https://github.com/SeleniumHQ/docker-selenium/blob/3.4.0-einsteinium/NodeChrome/Dockerfile
#===========================================================================================================

#============================================
# Google Chrome
#============================================
# can specify versions by CHROME_VERSION;
#  e.g. google-chrome-stable=53.0.2785.101-1
#       google-chrome-beta=53.0.2785.92-1
#       google-chrome-unstable=54.0.2840.14-1
#       latest (equivalent to google-chrome-stable)
#       google-chrome-beta  (pull latest beta)
#============================================
ARG CHROME_VERSION="google-chrome-stable"
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
  && echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
  && apt-get update -qqy \
  && apt-get -qqy install \
    ${CHROME_VERSION:-google-chrome-stable} \
  && rm /etc/apt/sources.list.d/google-chrome.list \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

#==================
# Chrome webdriver
#==================
ARG CHROME_DRIVER_VERSION=2.30
RUN wget --no-verbose -O /tmp/chromedriver_linux64.zip https://chromedriver.storage.googleapis.com/$CHROME_DRIVER_VERSION/chromedriver_linux64.zip \
  && rm -rf /opt/selenium/chromedriver \
  && unzip /tmp/chromedriver_linux64.zip -d /opt/selenium \
  && rm /tmp/chromedriver_linux64.zip \
  && mv /opt/selenium/chromedriver /opt/selenium/chromedriver-$CHROME_DRIVER_VERSION \
  && chmod 755 /opt/selenium/chromedriver-$CHROME_DRIVER_VERSION \
  && sudo ln -fs /opt/selenium/chromedriver-$CHROME_DRIVER_VERSION /usr/bin/chromedriver

#=================================
# Chrome Launch Script Modification
#=================================
COPY chrome_launcher.sh /opt/google/chrome/google-chrome

# just installing so we can get tensorboard
RUN pip install tensorflow

# add missing tokenizer package
RUN python -m nltk.downloader perluniprops

RUN pip install regex Twisted service_identity