# Guía de despliegue — Asistente RAG UAO

> Alcance de esta guía: poner en producción la solución completa de la
> **Fase 8** del [`plan-trabajo-tecnico.md`](../plan-trabajo-tecnico.md)
> (backend REST + gRPC, frontend Streamlit, caché semántica en Redis y proxy
> inverso con TLS) sobre **Docker Compose**, en una sola máquina (servidor
> interno, VM o equipo de laboratorio).
>
> Documentos relacionados: [`README.md`](../README.md) (visión técnica y
> comandos), [`plan-trabajo-tecnico.md`](../plan-trabajo-tecnico.md)
> (contratos, decisiones y riesgos), [`asistente-uao-rag.md`](../asistente-uao-rag.md)
> (alcance del proyecto).

---

## 1. Arquitectura desplegada

```text
        Internet / intranet UAO
                │
        host :80  →  308 redirect a :443 (y validación ACME)
        host :443 ─────────────┐  (único puerto público)
                               ▼
                    ┌────────────────────────────────────────────┐
                    │  proxy (Caddy, TLS)                        │
                    │                                            │
                    │   /               → frontend               │
                    │   /api/*          → api                    │
                    │   /healthz        → (proxy)                │
                    │   mlflow.<SITE>   → mlflow                 │
                    └───────────┬────────────────────────────────┘
                                │
        ┌────────────┬──────────┼───────────┬─────────────┐
        ▼            ▼          ▼           ▼             ▼
   frontend     api (REST)  gRPC Index    redis        mlflow
  (Streamlit   + gRPC emb.)  Admin       (caché)     (tracking)
    :8501         :8000      :50051      :6379         :5000
```

| Servicio | Imagen / build | Puertos | Persistencia |
|---|---|---|---|
| `proxy` | `caddy:2-alpine` | **80, 443 (publicados)** | volúmenes `caddy_data`, `caddy_config` (certificados) |
| `frontend` | `docker/Dockerfile.frontend` | `8501` (interno) | — |
| `api` | `docker/Dockerfile` | `8000` y `50051` (internos) | bind `./Data` (índice + corpus) y volumen `models_cache` (modelo E5) |
| `redis` | `redis:7-alpine` | `6379` (interno) | volumen `redis_data` (AOF) |
| `mlflow` (F9) | `ghcr.io/mlflow/mlflow` | `5000` (interno; dashboard vía proxy) | volumen `mlflow_data` (SQLite: experimentos y trazas) |

Decisiones relevantes (plan §8 y F9):

- **Solo el proxy expone puertos.** La API REST, gRPC, Redis y MLflow son
  alcanzables únicamente dentro de la red interna del compose; el dashboard de
  MLflow, además, a través del proxy (`https://mlflow.<SITE_ADDRESS>`).
- **El índice no se construye en el build.** `./Data` se monta desde el host:
  se reutiliza el índice ya generado (`Data/chroma`), el corpus
  (`Data/Documentos/*.pdf`) y el markdown (`Data/Documentos_MD/*.md`), de modo
  que `docker compose down && docker compose up` **conserva el índice**.
- **El modelo de embeddings se descarga una sola vez** y se cachea en el
  volumen `models_cache` (`HF_HOME=/models`), no en cada build.
- **Los contenedores corren sin privilegios** (usuario `app`, uid/gid = los del
  host por defecto) para que los bind mounts sean escribibles.

---

## 2. Requisitos previos

| Requisito | Detalle |
|---|---|
| Docker Engine ≥ 24 con Compose ≥ 2.20 | Se usa `docker compose up --wait` (healthchecks) y `depends_on: condition: service_healthy`. |
| Permisos de Docker para el usuario de despliegue | `sudo usermod -aG docker "$USER"` y volver a iniciar sesión (evita usar `sudo` en cada comando). |
| Git + acceso al repositorio | `git clone https://github.com/MarcAragon/Asistente-agentico-UAO.git`. |
| Claves de API | `CEREBRAS_API_KEY` (obligatoria para responder) y `LLAMA_CLOUD_API_KEY` (solo si se re-parsean PDFs). |
| Disco libre | ≈ 8-10 GB: imágenes (~4-5 GB; `torch` CUDA del `uv.lock`), modelo E5 (~1,2 GB), índice (~25 MB) y caché de build. Se comprueba con `make disk`. |
| Red | Salida a Internet (Cerebras, Hugging Face; LlamaCloud si se parsea). Si hay proxy corporativo, ver §11. |
| (Producción con dominio) | DNS: registro `A` del dominio → IP del servidor, y puertos 80/443 abiertos desde Internet. |
| (Opcional) GPU | Driver NVIDIA + NVIDIA Container Toolkit; ver §10. |

> **Nota sobre el tamaño de la imagen**: `uv.lock` fija `torch` con wheels
> CUDA de PyPI (no hay build CPU-only en el lock). El contenedor **no
> necesita GPU**: sin driver, `torch.cuda.is_available()` es `False` y los
> embeddings corren en CPU. Si el disco es crítico, ver el plan §6 (riesgos)
> y la optimización opcional al final de §10.

---

## 3. Preparación (primera vez)

```bash
# 1) Código
git clone https://github.com/MarcAragon/Asistente-agentico-UAO.git
cd Asistente-agentico-UAO

# 2) Secretos: se crea .env (gitignored) a partir de la plantilla
make env-init
${EDITOR:-nano} .env       # completar CEREBRAS_API_KEY y LLAMA_CLOUD_API_KEY

# 3) Copiar/verificar el índice vectorial (ver §4). Si viene del repositorio
#    de trabajo, basta con que exista Data/chroma/ con la colección.
ls -la Data/chroma        # debe contener chroma.sqlite3

# 4) Validar la configuración de compose antes de construir
make config
```

`make up` construye las imágenes con los uid/gid del usuario actual
(`APP_UID`/`APP_GID` exportados por el `Makefile`) para que el contenedor pueda
leer y escribir `./Data`. Si el despliegue lo ejecuta otro usuario, no hay nada
que ajustar: el `Makefile` lo detecta automáticamente. Con `docker compose`
directo se puede forzar: `APP_UID=$(id -u) APP_GID=$(id -g) docker compose up -d`.

### 3.1 Si el índice no existe todavía

```bash
# Opción A: construirlo en el host (recomendado; el entorno uv ya está listo)
make install              # uv sync
make parse                # solo si faltan Data/Documentos_MD/*.md (LlamaCloud)
make ingest               # Data/Documentos_MD → embeddings E5 → Data/chroma

# Opción B: construirlo dentro del contenedor (tras `make up`)
make ingest-docker
```

Ambas rutas escriben en el mismo `Data/chroma` (bind mount), así que el
resultado es idéntico.

---

## 4. Variables de entorno

### 4.1 Aplicación (prefijo `UAO_RAG__`, leídas por `core/config.py`)

| Variable | Default | Uso en despliegue |
|---|---|---|
| `CEREBRAS_API_KEY` | — (obligatoria) | Generación de respuestas. Sin ella `/ask` → `503`. |
| `CEREBRAS_API_KEYS` | — | Claves extra separadas por coma; rotación ante cuota agotada (429). |
| `LLAMA_CLOUD_API_KEY` | — | Solo para `make parse` (Fase 1). No la necesita el runtime de la API. |
| `UAO_RAG__LLM_MODEL` | `qwen-3.8-27b` | Modelo de Cerebras. |
| `UAO_RAG__LLM_DISABLE_REASONING` | `1` | Evita que el modelo agote `max_tokens` en razonamiento (hallazgo F4). |
| `UAO_RAG__TOP_K` / `UAO_RAG__MIN_SIMILARITY` | `5` / `0.35` | Recuperación (recalibración pendiente en F6). |
| `UAO_RAG__EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Modelo E5 (se descarga al volumen `models_cache`). |
| `UAO_RAG__EMBEDDING_DEVICE` | vacío (auto) | `cuda`/`cpu` para forzar; vacío = CUDA → ROCm → CPU. En el contenedor resuelve CPU. |
| `UAO_RAG__REDIS_URL` | `redis://localhost:6379/0` | **El compose lo sobreescribe** a `redis://redis:6379/0`. |
| `UAO_RAG__CACHE_ENABLED` | `1` | `0` desactiva el caché (la API responde igual, más lenta). |
| `UAO_RAG__CACHE_SIMILARITY` / `UAO_RAG__CACHE_TTL_SECONDS` | `0.97` / `86400` | Umbral de reutilización y vigencia (24 h). |
| `UAO_RAG__GRPC_ENABLED` / `UAO_RAG__GRPC_PORT` | `1` / `50051` | Control plane embebido en el proceso de la API. |

### 4.2 Despliegue (leídas por `docker compose`, sin prefijo)

| Variable | Default | Uso |
|---|---|---|
| `SITE_ADDRESS` | `localhost` | Dominio del proxy: `localhost` (demo/intranet) o `asistente.uao.edu.co`. |
| `TLS_DIRECTIVE` | `internal` | `internal` = CA propia de Caddy; o el correo de ACME, p. ej. `soporte.ti@uao.edu.co`. |
| `IMAGE_TAG` | `0.1.0` | Etiqueta de las imágenes construidas localmente. |
| `APP_UID` / `APP_GID` | `id -u` / `id -g` | Dueño del usuario sin privilegios dentro de los contenedores (los exporta el `Makefile`). |

> `.env` es **gitignored** y se inyecta en el contenedor con `env_file`, nunca
> se copia a la imagen (`.dockerignore` lo excluye). Verificar que no está
> versionado: `git check-ignore -v .env`.

> **Observabilidad (Fase 9)**: `docker-compose.yml` fija en el servicio `api`
> `MLFLOW_TRACKING_URI=http://mlflow:5000` y
> `MLFLOW_EXPERIMENT_NAME=asistente-uao` (no son variables de `Settings`, no
> llevan prefijo `UAO_RAG__` y no se configuran en `.env`). Con eso la API
> instrumenta las llamadas al LLM (`mlflow.openai.autolog()`) y el dashboard se
> sirve por el proxy en `https://mlflow.<SITE_ADDRESS>`, sin publicar puertos.

---

## 5. Despliegue paso a paso

```bash
make up      # construye, arranca, espera healthchecks, precarga el modelo y muestra URLs
```

Desglose de lo que hace (equivalente manual entre paréntesis):

1. `docker compose up -d --build --wait` — construye las imágenes y arranca
   Redis, API, frontend y proxy esperando a que cada uno esté `healthy`.
2. `make models-prefetch` — descarga el modelo E5 al volumen `models_cache`
   (1-3 min la primera vez). Así la primera pregunta del usuario no paga la
   descarga.
3. `make urls` — imprime las URLs y los puertos publicados.

### 5.1 Verificación

```bash
make ps                      # redis/api/frontend/proxy/mlflow en estado "healthy"
make health                  # {"status":"ok","index_chunks":1284,"device":"cpu",...}
make grpc-status             # control plane interno: chunks y documentos
make ask                     # POST /ask a través del proxy TLS (respuesta + fuentes)
curl -sk https://localhost/healthz          # el proxy responde "ok"
curl -sk https://localhost/api/documents    # documentos indexados
curl -sk https://mlflow.localhost/health    # tracking server de MLflow (200)
```

En el navegador: **https://localhost/** → chat; pregunta de ejemplo
«¿Qué pasa si repruebo tres veces una misma asignatura?» debe responder con
fuentes citadas. Una pregunta fuera de dominio (p. ej. «¿receta de arepas?»)
debe mostrar el aviso de no-información **sin** fuentes. El dashboard de trazas
del LLM está en **https://mlflow.localhost/** (experimento `asistente-uao`).

Con `SITE_ADDRESS=localhost` el certificado lo emite la CA interna de Caddy, así
que el navegador mostrará un aviso de certificado desconocido la primera vez
(§7.1). Con `curl` se usa `-k` (o `--cacert` con la CA exportada).

### 5.2 Comprobar que no hay puertos internos publicados

```bash
docker compose ps                 # la columna PORTS solo debe mostrar 80/443 para el proxy
ss -tlnp | grep -E ':(8000|50051|6379|8501|5000)'   # sin resultados = correcto
```

---

## 6. Persistencia y respaldos

| Dato | Ubicación | Contenido | Respaldo |
|---|---|---|---|
| Índice vectorial | bind `./Data/chroma` | Colección `uao_normativa` (chunks + embeddings + metadatos) | `make index-backup` / `make index-restore FILE=…` |
| Corpus y markdown | bind `./Data/Documentos`, `./Data/Documentos_MD` | PDFs oficiales y su markdown | Git / copia del directorio |
| Caché semántico | volumen `redis_data` | Preguntas + respuestas (AOF). Regenerable | No crítico; `make cache-flush` lo invalida |
| Modelo E5 | volumen `models_cache` | Pesos descargados (~1,2 GB) | No requiere respaldo (se re-descarga) |
| Trazas del LLM | volumen `mlflow_data` | SQLite de MLflow: experimentos, trazas (prompt, contexto, respuesta), tokens y latencia | Recomendado si las trazas son valiosas; regenerable |
| Certificados TLS | volumen `caddy_data` | Certificados y claves ACME/CA interna | Recomendado con ACME (§7.2) |

### 6.1 Respaldo del índice

```bash
docker compose stop api          # respaldo consistente (evita escribir durante el tar)
make index-backup                # → Data/chroma_backup/chroma-<fecha>.tar.gz
docker compose start api
```

### 6.2 Restauración del índice

```bash
make down                                            # detener todo
make index-restore FILE=Data/chroma_backup/chroma-20260917-182350.tar.gz
make up
```

El target guarda automáticamente el índice actual como
`chroma-pre-restore-<fecha>.tar.gz` antes de sobrescribirlo.

### 6.3 Respaldo de los certificados (producción con ACME)

```bash
docker compose exec -T proxy tar czf - -C /data . > caddy-data-$(date +%F).tar.gz
# Restauración (con el stack detenido):
#   docker compose run --rm -T -v asistente-uao_caddy_data:/data proxy \
#     sh -c 'tar xzf - -C /data' < caddy-data-2026-09-17.tar.gz
```

Ese archivo contiene **claves privadas**: guárdalo cifrado y con permisos
restringidos (`chmod 600`), y nunca lo subas al repositorio.

### 6.4 Cambiar el índice a un volumen nombrado (alternativa al bind mount)

Si el servidor no debe exponer `./Data` en el host, en `docker-compose.yml`:

```yaml
    volumes:
      - chroma_data:/app/Data/chroma     # en lugar de ./Data:/app/Data
# …
volumes:
  chroma_data:
```

El volumen se inicializa con el dueño correcto (el Dockerfile crea
`/app/Data` como el usuario `app`), pero hay que **indexar dentro del
contenedor** (`make ingest-docker`), porque el `./Data` del host deja de estar
visible.

---

## 7. TLS y certificados

### 7.1 Modo local / intranet (default: CA interna de Caddy)

```env
SITE_ADDRESS=localhost
TLS_DIRECTIVE=internal
```

- Caddy emite un certificado firmado por su **CA local** (sin Internet ni DNS)
  y lo guarda en el volumen `caddy_data`.
- El navegador mostrará un aviso de «certificado no confiable» la primera vez.
  Para eliminarlo, exportar la CA raíz e importarla en el sistema/navegador:

```bash
docker compose exec -T proxy cat /data/caddy/pki/authorities/local/root.crt > caddy-root.crt
# Linux (Debian/Ubuntu):
sudo cp caddy-root.crt /usr/local/share/ca-certificates/asistente-uao-local.crt
sudo update-ca-certificates
```

- Con `curl`: `curl -sk …` o `curl --cacert caddy-root.crt …`.

### 7.2 Modo producción con dominio real (ACME / Let's Encrypt)

Requisitos: DNS público apuntando al servidor y puertos 80/443 accesibles.

```env
SITE_ADDRESS=asistente.uao.edu.co
TLS_DIRECTIVE=soporte.ti@uao.edu.co
```

- Caddy solicita el certificado automáticamente (HTTP-01), lo almacena en
  `caddy_data` y lo **renueva solo** (~30 días antes de vencer); no hay que
  programar cron ni reiniciar nada.
- La renovación es automática pero **la persistencia es obligatoria**: no borrar
  el volumen `caddy_data` (`make clean-volumes` lo elimina) para evitar chocar
  con los límites de emisión de Let's Encrypt.
- Forzar HTTPS en el cliente y verificar la cadena:

```bash
curl -sS https://asistente.uao.edu.co/healthz            # "ok"
curl -sSI https://asistente.uao.edu.co/ | head -5        # HSTS y cabeceras de seguridad
openssl s_client -connect asistente.uao.edu.co:443 -servername asistente.uao.edu.co </dev/null 2>/dev/null | openssl x509 -noout -subject -dates
```

- `:80` solo se usa para la redirección a HTTPS y para la validación ACME.

### 7.3 Cabeceras y rutas del proxy

Definidas en `docker/Caddyfile`: `Strict-Transport-Security`,
`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, ocultación de
`Server`; `/` → Streamlit (websockets incluidos), `/api/*` → API REST
(sin el prefijo `/api`), `/healthz` → 200 del proxy. Caddy detecta los cambios
del `Caddyfile` montado y recarga solo; para forzarlo:
`docker compose restart proxy`.
---

## 8. Operación diaria

### 8.1 Ciclo de vida

| Acción | Comando |
|---|---|
| Arrancar | `make up` |
| Apagar (conserva datos) | `make down` |
| Reiniciar | `make restart` |
| Estado y salud | `make ps` |
| Logs (por servicio) | `make logs`, `make logs-api`, `make logs-frontend`, `make logs-proxy`, `make logs-redis` |
| Consumir recursos | `docker stats`, `make disk` |
| Arrancar con el host | `restart: unless-stopped` ya está definido en todos los servicios |

### 8.2 Actualizar a una nueva versión

```bash
git pull                       # nuevo código / configuración
make build                     # reconstruye imágenes (uv.lock cachea dependencias)
make up                        # recrea contenedores con healthchecks + prefetch
make ps                        # verificar "healthy"
```

El índice y la caché sobreviven porque el primero es un bind mount del host y
la segunda un volumen nombrado. Si el corpus cambió: `make ingest-rebuild-docker`
y luego `make cache-flush` (un `rebuild` vía gRPC ya invalida la caché solo).

### 8.3 Reindexar y verificar

```bash
make ingest-docker            # incremental, idempotente (IDs sha256)
make ingest-rebuild-docker    # reconstrucción limpia de la colección
make grpc-status              # chunks/documentos por el control plane gRPC
make health                   # chunks indexados + contadores de caché
```

### 8.4 Caché semántico

```bash
make cache-stats              # hits / misses acumulados
make cache-flush              # invalidar todo (tras reindexar)
```

`GET /health` expone `cache_hits`, `cache_misses` y `cache_hit_ratio`. Si Redis
se detiene, la API **sigue respondiendo** (el caché es degradable, plan §0.4.10):
solo aumenta la latencia.

### 8.5 Copia de seguridad programada (sugerencia)

```cron
# Respaldo diario del índice a las 03:00 y limpieza de respaldos > 30 días
0 3 * * * cd /srv/Asistente-agentico-UAO && make index-backup >> /var/log/uao-backup.log 2>&1
```

---

## 9. GPU opcional (acelerar embeddings)

El contenedor funciona en CPU sin configuración extra. Para usar una GPU NVIDIA
(requiere NVIDIA Container Toolkit en el host):

```yaml
# docker-compose.yml → servicio api
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      UAO_RAG__EMBEDDING_DEVICE: cuda     # opcional: forzar GPU; vacío = auto
```

```bash
nvidia-smi                      # el host debe ver la GPU
make build && make up
docker compose exec api python -c "import torch; print(torch.cuda.is_available())"
```

Requiere que `torch` (fijado en `uv.lock`, build CUDA) y el driver del host sean
compatibles. Sin GPU, no cambiar `UAO_RAG__EMBEDDING_DEVICE` (auto resuelve CPU).

**Nota de tamaño (verificada)**: `uv sync` de este proyecto instala `torch` con
las wheels CUDA fijadas en `uv.lock` (~4-5 GB de imagen), y `uv sync` no admite
seleccionar el backend de torch (`--torch-backend` existe solo en la interfaz
`uv pip`). Una variante CPU-only exigiría añadir un índice de PyTorch CPU al
`pyproject.toml` (`[tool.uv.index]` + `[tool.uv.sources]` para `torch`) y
re-lockear; queda registrado como optimización futura en el registro de riesgos
del plan (§6). El contenedor funciona igual en CPU, solo ocupa más disco.

---

## 10. Seguridad y privacidad

- **Superficie mínima**: solo 80/443 del proxy. API (`:8000`), gRPC (`:50051`),
  Streamlit (`:8501`), Redis (`:6379`) y MLflow (`:5000`) no se publican;
  verificar con `docker compose ps` y `ss -tlnp` (§5.2).
- **Secretos**: viven en `.env` (gitignored) e ingresan por `env_file`; no se
  copian a las imágenes (`.dockerignore`). Rotarlos es editar `.env` y
  `make up`. `tests/test_deployment.py` falla si se detecta una clave con
  formato `csk-…`/`llx-…` en archivos versionados.
- **Contenedores sin privilegios**: usuario `app` (uid/gid del host), sin
  capabilities extra; no se monta el socket de Docker.
- **Redis interno**: sin autenticación porque no está publicado; si se expone
  a otra red, añadir `--requirepass` y `UAO_RAG__REDIS_URL` con contraseña.
- **Privacidad (Ley 1581 de 2012)**: el chat no solicita datos personales; la
  caché guarda preguntas y respuestas, no identidades. El corpus es normativa
  pública. Al usar LlamaCloud Parse, el archivo remoto se elimina tras el
  parseo (plan §6).
- **Trazas de MLflow (F9)**: el tracking server queda en la red interna y solo
  se alcanza por el proxy; las trazas se guardan en el volumen `mlflow_data`
  (prompt, contexto recuperado, respuesta, tokens y latencia). Contienen
  preguntas y normativa pública, nunca identidades. Para apagarlas, basta con
  no definir `MLFLOW_TRACKING_URI` en el servicio `api`.
- **Aviso al usuario**: el frontend muestra que las respuestas son orientativas
  y citan la fuente oficial; no sustituyen la asesoría de Secretaría Académica.
- **Datos en tránsito**: TLS obligatorio hacia el usuario (HSTS) y HTTPS hacia
  Cerebras/Hugging Face.
---

## 11. Solución de problemas

### 11.1 Docker

| Síntoma | Causa probable | Solución |
|---|---|---|
| `permission denied while trying to connect to the Docker API` | El usuario no está en el grupo `docker` | `sudo usermod -aG docker "$USER"`, cerrar sesión y volver a entrar (o usar `sudo docker …`). |
| `make up` falla con `--wait` / contenedor `unhealthy` | Healthcheck no pasa en el tiempo previsto | `make logs-api` / `make logs-frontend`; el api tiene `start_period: 40 s`. Si el disco es lento, subir `--wait-timeout`. |
| `bind: address already in use` en 80/443 | Otro servicio ocupa el puerto | `sudo ss -tlnp \| grep -E ':80|:443'`; detenerlo o cambiar los puertos publicados del proxy. |
| Cambios en el código no aparecen | Imagen antigua | `make build && make up` (o `make up`, que ya reconstruye). |
| `no space left on device` | Imágenes CUDA + modelo + caché de build | `make disk`, `docker builder prune`, `make clean-images`. |
| El proxy redirige a HTTPS pero el navegador avisa del certificado | CA interna de Caddy (modo local) | Importar la CA (§7.1) o usar un dominio real con ACME (§7.2). |
| `docker compose config` avisa de `.env` ausente | Clon recién hecho | `make env-init` (el compose lo tolera con `required: false`). |

### 11.2 Aplicación

| Síntoma | Causa probable | Solución |
|---|---|---|
| `POST /ask` responde `503` | `CEREBRAS_API_KEY` no definida en `.env` | Completar `.env` y `make up` (o `docker compose restart api`). |
| Primeras preguntas tardan minutos | Descarga del modelo E5 (~1,2 GB) | `make models-prefetch` (lo hace `make up`); después ~0,3-0,6 s por respuesta del LLM. |
| `index_chunks: 0` en `/health` | `Data/chroma` vacío o sin montar | Verificar `ls Data/chroma` en el host y `docker compose exec api ls /app/Data/chroma`; construir con `make ingest` o `make ingest-docker`. |
| Errores de permisos al escribir el índice desde el contenedor | `./Data` pertenece a otro uid | Reconstruir con `APP_UID=$(id -u) APP_GID=$(id -g) docker compose build` o ajustar el dueño del directorio. |
| Chroma con errores de bloqueo (`database is locked`) | Se está ejecutando la app local y el contenedor sobre el mismo `Data/chroma` | Usar solo uno a la vez, o cambiar `UAO_RAG__CHROMA_DIR` en el entorno local. |
| Respuestas siempre «no tengo información suficiente» | Índice desactualizado, umbral de similitud o contexto insuficiente | `make smoke`, `make grpc-status`; recalibrar `UAO_RAG__MIN_SIMILARITY` (pendiente F6) o reindexar. |
| gRPC no responde | `UAO_RAG__GRPC_ENABLED=0` o puerto ocupado | `make logs-api` (el fallo de gRPC no tumba la REST); el puerto es interno, no publicar. |
| El chat muestra «No se pudo conectar con la API» | El contenedor `api` no está `healthy` o `UAO_RAG__API_BASE_URL` mal definida | `make ps`, `make logs-frontend`; en compose debe ser `http://api:8000`. |
| El chat se queda cargando (websocket) | Proxy sin soporte de `Upgrade` | Usar el `docker/Caddyfile` del repositorio (Caddy lo hace automáticamente) y no publicar Streamlit directamente. |
| Redis caído y la API sigue funcionando | Comportamiento esperado (caché degradable) | `make logs-redis`; la latencia sube, no hay error. |
| El dashboard de MLflow no muestra trazas | La API arrancó sin `MLFLOW_TRACKING_URI` o `mlflow` no está `healthy` | `make ps` y `make logs-api` (busca «mlflow»); `make logs-mlflow`. El compose fija `MLFLOW_TRACKING_URI=http://mlflow:5000`. |
| Parseo con LlamaCloud falla | Clave ausente, sin Internet o cuota | Verificar `LLAMA_CLOUD_API_KEY` y conectividad; el markdown ya generado no se re-parsea sin `make parse-redo`. |
| Salida a Internet a través de proxy corporativo | La API no llega a Cerebras/HF | Definir `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` en el servicio `api` de `docker-compose.yml`. |

---

## 12. Checklist de despliegue

- [ ] Docker Engine + Compose instalados y usuario con acceso al socket.
- [ ] Repositorio clonado en la versión deseada (`git log -1`).
- [ ] `.env` creado (`make env-init`) con `CEREBRAS_API_KEY` (y `LLAMA_CLOUD_API_KEY` si se parsea).
- [ ] `make config` sin errores.
- [ ] `Data/chroma` presente y verificado (o construido con `make ingest`).
- [ ] `SITE_ADDRESS`/`TLS_DIRECTIVE` definidos según el entorno (§7).
- [ ] `make up` termina con los 5 servicios `healthy` (`make ps`).
- [ ] `make health` devuelve `status: ok` con el número esperado de chunks.
- [ ] `make ask` responde con fuentes; una pregunta fuera de dominio no inventa.
- [ ] Chat accesible por HTTPS en el navegador con websockets funcionando.
- [ ] Dashboard de trazas accesible en `https://mlflow.<SITE_ADDRESS>` (experimento `asistente-uao`).
- [ ] `docker compose ps` / `ss -tlnp`: solo 80 y 443 publicados.
- [ ] Respaldo del índice probado (`make index-backup` + restauración en un entorno de prueba).
- [ ] (ACME) Certificado emitido, renovación automática y respaldo de `caddy_data`.
- [ ] (Opcional) Cron de respaldo diario configurado (§8.5).
- [ ] Aviso de privacidad visible en el chat y corpus autorizado por la institución.

---

## 13. Referencias

- [`plan-trabajo-tecnico.md`](../plan-trabajo-tecnico.md) — Fases 7, 8 y 9, contratos (§3), riesgos (§6).
- [`README.md`](../README.md) — arquitectura, comandos y contratos de la API.
- [`docker-compose.yml`](../docker-compose.yml), [`docker/Dockerfile`](../docker/Dockerfile),
  [`docker/Dockerfile.frontend`](../docker/Dockerfile.frontend), [`docker/Caddyfile`](../docker/Caddyfile).
- [Caddy — Automatic HTTPS](https://caddyserver.com/docs/automatic-https) ·
  [uv — Using uv in Docker](https://docs.astral.sh/uv/guides/integration/docker/) ·
  [Streamlit — Docker](https://docs.streamlit.io/deploy/tutorials/docker).
