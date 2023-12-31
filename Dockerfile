FROM python:3.9

WORKDIR /code
COPY ./requirements.txt /code/requirments.txt
COPY ./RealESRGAN /code/RealESRGAN

RUN pip install basicsr
RUN pip install facexlib
RUN pip install gfpgan
RUN pip install --no-cache-dir --upgrade -r /code/requirments.txt
RUN python /code/RealESRGAN/setup.py develop

COPY ./main.py /code/main.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
