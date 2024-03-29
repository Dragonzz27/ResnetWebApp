FROM python:3.10

WORKDIR /ResnetWebApp

COPY requirements.txt ./

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

WORKDIR /ResnetWebApp/web

CMD [ "python", "-m", "flask", "run", "--host=0.0.0.0" ]