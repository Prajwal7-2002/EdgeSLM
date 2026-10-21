# Docker & Air-Gapped Deployment 🐳

For enterprise compliance, EdgeSLM is containerized as an air-gapped microservice with zero external network connectivity.

---

## 1. Quick Launch with Docker Compose

To build and start the EdgeSLM container locally:

```bash
docker compose up -d
```

Check logs and server readiness:
```bash
docker compose logs -f
```

The service is accessible at:
- **Swagger UI**: `http://localhost:8000/docs`
- **Classify Endpoint**: `http://localhost:8000/classify`
- **File Upload Endpoint**: `http://localhost:8000/classify-file`

---

## 2. Building the Production Docker Image

Build the lightweight multi-stage image manually:

```bash
docker build -t edgeslm:latest .
```

Run on CPU:
```bash
docker run -d \
  -p 8000:8000 \
  --name edgeslm-container \
  edgeslm:latest
```

---

## 3. Strict Air-Gapped Operation

In high-security enterprise environments where documents are classified under HIPAA, GDPR, or Defense security clearance:

1. **Build image** on a connected workstation:
   ```bash
   docker save edgeslm:latest | gzip > edgeslm-airgap.tar.gz
   ```
2. **Transfer archive** to air-gapped machine and load:
   ```bash
   docker load < edgeslm-airgap.tar.gz
   ```
3. **Execute with ZERO network interfaces** (`--network none`):
   ```bash
   docker run -d \
     --network none \
     -v /local/secure_docs:/secure_docs:ro \
     edgeslm:latest
   ```
EdgeSLM will execute deterministic security scans and edge model inference with absolute physical assurance of zero data egress.
