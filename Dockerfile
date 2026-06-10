FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["deepeval", "test", "run"]
CMD ["test_prompts.py", "test_tools.py", "test_conversation.py"]
