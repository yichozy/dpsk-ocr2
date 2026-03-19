# OOM Detection and Restart Guide

This document describes the Out Of Memory (OOM) detection and automatic restart features added to the DeepSeek OCR service.

## Overview

The service now includes:

1. **Memory Monitoring** - Continuous monitoring of system and GPU memory usage
2. **OOM Detection** - Automatic detection when memory usage reaches critical levels
3. **Graceful Shutdown** - Graceful shutdown before OOM causes crashes
4. **Automatic Restart** - Optional watchdog process for automatic restart on failure
5. **Health Check Endpoint** - Enhanced health check with memory statistics

## Configuration

Memory monitoring is configured in `config.py`:

```python
# Memory monitoring thresholds (percentage of total system memory)
MEMORY_WARNING_THRESHOLD = 80.0   # Log warning when memory usage exceeds this
MEMORY_CRITICAL_THRESHOLD = 90.0  # Trigger shutdown/restart when memory usage exceeds this
MEMORY_CHECK_INTERVAL = 30        # Seconds between memory checks
```

### Adjusting Thresholds

- **Systems with more RAM**: Can increase thresholds (e.g., 85%/95%)
- **Systems with less RAM**: Should decrease thresholds (e.g., 70%/85%)
- **Frequent OOM**: Decrease `MEMORY_CRITICAL_THRESHOLD` to trigger shutdown earlier

## Memory Monitor Features

### 1. Continuous Monitoring

The memory monitor runs in the background, checking memory usage every 30 seconds (configurable).

**What it monitors:**
- System memory (RAM) usage percentage
- Available and used memory in MB
- GPU memory (if CUDA is available)

**What it does:**
- Logs warnings when usage exceeds warning threshold (default: 80%)
- Triggers cleanup when usage exceeds critical threshold (default: 90%)
- Initiates graceful shutdown if memory remains critical after cleanup

### 2. Automatic Cleanup

When memory usage exceeds the critical threshold, the service:

1. **Forces garbage collection** - `gc.collect()` to free Python objects
2. **Clears CUDA cache** - `torch.cuda.empty_cache()` to free GPU memory
3. **Logs critical alert** - Detailed log entry with memory statistics
4. **Waits and rechecks** - Checks if cleanup helped
5. **Initiates shutdown** - If still critical, starts graceful shutdown

### 3. Graceful Shutdown Process

When OOM is detected and cleanup doesn't help:

1. **Log critical event** - Record OOM with memory statistics
2. **Set shutdown flag** - Prevent new tasks from starting
3. **Wait 10 seconds** - Allow current requests to complete
4. **Exit with code 137** - Standard OOM exit code (triggers container restart)

## Health Check Endpoint

The `/health` endpoint now returns detailed memory information:

```bash
curl http://localhost:8000/health
```

**Response example:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "memory": {
    "system": {
      "total_mb": 32768.0,
      "used_mb": 8192.0,
      "available_mb": 24576.0,
      "percent_used": 25.0
    },
    "gpu": {
      "total_mb": 24576.0,
      "used_mb": 4096.0,
      "free_mb": 20480.0
    }
  },
  "queue": {
    "size": 0,
    "active_tasks": 0
  },
  "alerts": []
}
```

**Status values:**
- `"healthy"` - Memory usage below warning threshold
- `"warning"` - Memory usage above warning threshold
- `"critical"` - Memory usage above critical threshold
- `"degraded"` - Memory monitoring error

## Automatic Restart

### Option 1: Watchdog Script

Run the service with the watchdog for automatic restart:

```bash
python watchdog_restart.py
```

**Watchdog features:**
- Automatically restarts service if it crashes or exits due to OOM
- Configurable maximum restart count (default: unlimited)
- Configurable restart delay (default: 10 seconds)
- Logs all restart events with reasons
- Handles clean shutdown signals

**Custom watchdog options:**

```bash
# Limit to 5 restarts
python watchdog_restart.py --max-restarts 5

# 30 second restart delay
python watchdog_restart.py --restart-delay 30

# Custom service command
python watchdog_restart.py --command "uvicorn serve_pdf:app --host 0.0.0.0 --port 8000"
```

### Option 2: Docker Auto-Restart

When running in Docker, use the restart policy:

```bash
docker run --gpus all \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  --restart on-failure:5 \
  deepseek-ocr2
```

**Restart policies:**
- `--restart no` - Don't restart (default)
- `--restart on-failure` - Restart only if container exits with non-zero code
- `--restart on-failure:5` - Restart max 5 times
- `--restart always` - Always restart

### Option 3: Systemd Service

For production deployments, use the provided systemd service file:

```bash
# Install
sudo cp deepseek-ocr.service /etc/systemd/system/
sudo systemctl daemon-reload

# Enable and start
sudo systemctl enable deepseek-ocr
sudo systemctl start deepseek-ocr

# Check status
sudo systemctl status deepseek-ocr

# View logs
journalctl -u deepseek-ocr -f
```

**Service features:**
- Automatic restart on failure
- Resource limits (triggers graceful shutdown before OOM)
- Logging to systemd journal
- Runs as dedicated user for security

## Monitoring and Alerts

### Check Memory Usage

```bash
# Via health endpoint
curl http://localhost:8000/health | jq '.memory'

# Via system tools
free -h
nvidia-smi
```

### Watch Logs

```bash
# Application logs
tail -f app.log

# Watchdog logs (if using)
tail -f watchdog.log

# Systemd logs (if using)
journalctl -u deepseek-ocr -f
```

### Alerting Integration

The health endpoint can be integrated with monitoring tools:

**Prometheus example:**

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'deepseek-ocr'
    metrics_path: '/health'
    static_configs:
      - targets: ['localhost:8000']
```

**Grafana dashboard queries:**

```promql
# Memory usage percent
deepseek_ocr_memory_percent_used

# Alert on high memory
deepseek_ocr_memory_percent_used > 80
```

## Troubleshooting

### Frequent OOM Restarts

**Symptoms:** Service restarts frequently with OOM errors

**Solutions:**

1. **Reduce batch size** in `config.py`:
   ```python
   PDF_BATCH_SIZE = 2  # Default is 4
   ```

2. **Lower critical threshold** for earlier shutdown:
   ```python
   MEMORY_CRITICAL_THRESHOLD = 85.0  # Default is 90
   ```

3. **Reduce worker count** for preprocessing:
   ```python
   NUM_WORKERS = 32  # Default is 64
   ```

4. **Increase system RAM** or reduce concurrent processing:
   ```python
   MAX_CONCURRENCY = 50  # Default is 100
   ```

### Memory Leak Still Occurs

**If you still see memory leaks after the fixes:**

1. **Check for other leaks** - Use memory profiling tools:
   ```bash
   pip install memory-profiler
   python -m memory_profiler serve_pdf.py
   ```

2. **Monitor batch processing** - Add logging to track memory per batch:
   ```python
   import tracemalloc
   tracemalloc.start()
   # ... process batch ...
   snapshot = tracemalloc.take_snapshot()
   top_stats = snapshot.statistics('lineno')
   ```

3. **Reduce check interval** for faster OOM detection:
   ```python
   MEMORY_CHECK_INTERVAL = 15  # Check every 15 seconds
   ```

### Watchdog Won't Start Service

**Common issues:**

1. **Python not found** - Use full path to Python:
   ```bash
   /opt/deepseek-ocr/venv/bin/python watchdog_restart.py
   ```

2. **Wrong working directory** - Check path in watchdog command

3. **Port already in use** - Check if service is already running:
   ```bash
   lsof -i :8000
   ```

## Testing OOM Detection

### Simulate High Memory Usage

To test the OOM detection without actually running out of memory:

```python
# Add this to serve_pdf.py temporarily for testing
@app.post("/test_memory")
async def test_memory():
    """Consume memory to test OOM detection."""
    import numpy as np

    # Allocate 1GB
    data = []
    for i in range(10):
        data.append(np.zeros((128, 1024, 1024), dtype=np.float32))

    return {"status": "allocated", "gb": len(data)}
```

### Monitor During Test

```bash
# Terminal 1: Watch memory
watch -n 1 'curl -s http://localhost:8000/health | jq ".memory"'

# Terminal 2: Trigger memory allocation
curl -X POST http://localhost:8000/test_memory

# Terminal 3: Watch logs
tail -f watchdog.log
```

## Summary

**Key files:**
- `memory_monitor.py` - Memory monitoring implementation
- `watchdog_restart.py` - Automatic restart script
- `deepseek-ocr.service` - Systemd service file
- `config.py` - Memory threshold configuration
- `serve_pdf.py` - Integration with health check

**Key environment variables:**
- `MEMORY_WARNING_THRESHOLD` - Warning threshold (default: 80%)
- `MEMORY_CRITICAL_THRESHOLD` - Critical threshold (default: 90%)
- `MEMORY_CHECK_INTERVAL` - Check interval in seconds (default: 30)
- `PDF_BATCH_SIZE` - Batch size for PDF processing (default: 4)

For questions or issues, check the logs:
- Application logs: `watchdog.log`
- Docker logs: `docker logs <container>`
- Systemd logs: `journalctl -u deepseek-ocr`
