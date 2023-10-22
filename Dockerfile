FROM python:3.9

WORKDIR /code
COPY ./requirements.txt /code/requirments.txt

RUN pip install basicsr
RUN pip install facexlib
RUN pip install gfpgan
RUN pip install --no-cache-dir --upgrade -r /code/requirments.txt
RUN python ./RealESRGAN/setup.py develop

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
