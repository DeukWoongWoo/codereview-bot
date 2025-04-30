FROM python:3.11-slim

WORKDIR /app

# Copy only the dependency files first
COPY pyproject.toml .

# Install build dependencies and project dependencies
RUN pip install --no-cache-dir pip-tools && \
    pip install --no-cache-dir .

# Copy the rest of the application
COPY . .

# Run the bot
CMD ["python", "bot.py"]
