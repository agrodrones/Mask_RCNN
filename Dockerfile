FROM waleedka/modern-deep-learning

RUN mkdir /host
WORKDIR /host

ADD requirements.txt .

RUN pip install -r requirements.txt

ADD . .