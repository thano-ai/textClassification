FROM python:3.9

WORKDIR /app

COPY . .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
