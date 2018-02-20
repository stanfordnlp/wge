FROM python:2.7-slim
RUN pip install selenium

RUN apt-get update && apt-get install wget unzip xvfb -y

RUN wget -q -O /tmp/linux_signing_key.pub https://dl-ssl.google.com/linux/linux_signing_key.pub \
  && apt-key add /tmp/linux_signing_key.pub \
  && echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
  && apt-get update && apt-get install google-chrome-stable -y

RUN wget -q https://chromedriver.storage.googleapis.com/2.30/chromedriver_linux64.zip -O /tmp/chromedriver.zip \
  && mkdir -p /opt/bin/ \
  && unzip /tmp/chromedriver.zip -d /opt/bin/
ENV PATH /opt/bin/:$PATH

VOLUME ["/miniwob"]
