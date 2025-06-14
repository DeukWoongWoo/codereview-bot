FROM python:3.11-slim

WORKDIR /app

# Copy only the dependency files first
COPY pyproject.toml .

# Install uv and project dependencies
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.cargo/bin/uv venv /app/.venv && \
    /root/.cargo/bin/uv pip install --system --no-cache pip setuptools wheel && \
    /root/.cargo/bin/uv pip freeze > requirements.txt

RUN uv pip install --system --no-cache -r requirements.txt && rm requirements.txt

# Copy the rest of the application
COPY . .

# Run the bot
CMD ["python", "bot.py"]
