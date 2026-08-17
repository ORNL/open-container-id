FROM python:3.11-slim

# Create non-root user
RUN useradd -m -s /bin/bash container_id_user
WORKDIR /app

# Install uv and dependencies
RUN pip install uv
COPY pyproject.toml .
RUN uv pip install --system -e .[runtime,train]

# Copy application code
COPY . .
RUN chown -R container_id_user:container_id_user /app

# Switch to non-root user
USER container_id_user
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["./deployment/entrypoint.sh"]
