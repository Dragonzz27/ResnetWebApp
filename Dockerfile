FROM python:3.11.5

WORKDIR /ResnetWebApp

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

RUN cd web

CMD [ "python", "-m", "flask", "run" ]