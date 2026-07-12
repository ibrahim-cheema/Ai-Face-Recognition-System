# AI Employee Face Recognition System

An enterprise-ready prototype for employee face registration and real-time detection, built for internal textile-company onboarding and monitoring. All authoritative biometric evaluation — face detection, pose/blur/duplicate quality checks, embedding generation, and identity matching — runs **server-side**. The browser never performs recognition; it only runs a lightweight client-side detector for live visual guidance during registration.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Technology Stack](#technology-stack)
3. [Models](#models)
4. [Project Structure](#project-structure)
5. [Setup & Installation](#setup--installation)
6. [API Reference](#api-reference)
7. [Database Schema](#database-schema)
8. [Configuration](#configuration)
9. [Core Algorithms](#core-algorithms)
10. [Challenges Encountered & How They Were Solved](#challenges-encountered--how-they-were-solved)
11. [Biometric Data Compliance](#-biometric-data-compliance-note)
12. [Known Limitations](#known-limitations)

---

## Architecture

```
  Browser (Next.js 14)                         FastAPI Backend
 ────────────────────────                      ─────────────────────────────────────
  Register page                                 POST /api/employees/register/
   - webcam + MediaPipe "short"        HTTP        check-frame   (per-frame QA)
     model, LIVE oval guidance  ───────────▶    POST /api/employees/register
     only (never authoritative)                    (batch → averaged embedding)
                                                 GET/DELETE /api/employees
  Detect page                                   GET /api/logs
   - webcam frames streamed            WS
     ~4 fps, no local logic    ◀──────────▶    WS /api/recognize/stream
                                                    SCRFD detect → ArcFace embed
  Logs page                                        → cosine-match vs SQLite
   - search / filter / paginate        HTTP         → log + snapshot
     employee directory        ───────────▶
                                                 face_service.py
                                                    SCRFDDetector + ArcFaceEmbedder
                                                    (onnxruntime, no insightface pkg)
                                                 database.py
                                                    SQLite, embeddings as float32 BLOBs
```

**Design principle: client suggests, server decides.** The browser's MediaPipe overlay exists purely to give the user real-time visual feedback (oval turns green when well-framed) so registration feels responsive. Every frame that actually counts toward a captured pose, and every recognition decision on the Detect page, is independently re-validated by the FastAPI backend using SCRFD + ArcFace — the client never gets to assert "this is employee X" on its own.

### Data flow: registration
1. User fills employee ID/name/department, webcam starts.
2. Client-side MediaPipe (`@mediapipe/face_detection`, loaded from CDN) tracks the face locally and drives the oval overlay (gray → yellow → green) plus the 5-step prompt sequence (front, left, right, up, down).
3. When the oval is green, the browser throttles calls (~2/sec) to `POST /api/employees/register/check-frame`, sending a JPEG frame as base64.
4. The backend runs real SCRFD detection + blur (Laplacian variance) + pose estimation (landmark geometry) and returns whether the frame is acceptable and what pose it actually shows.
5. Once a step's frame quota is met, the flow advances to the next pose; after all 5 steps (20 frames total) the batch is submitted to `POST /api/employees/register`.
6. The backend re-runs detection on every frame, discards blurry frames, discards near-duplicate frames (cosine similarity > 0.95 against already-accepted frames in the batch), embeds the rest with ArcFace, averages and L2-normalizes the embeddings into one 512-dim vector, and stores it (plus a cropped thumbnail) in SQLite.

### Data flow: live detection
1. Detect page starts the webcam and opens a WebSocket to `/api/recognize/stream`.
2. The client sends a JPEG frame (base64) roughly every 250ms — no local recognition logic at all.
3. The backend decodes the frame, runs SCRFD to find the largest face, embeds it with ArcFace, and computes cosine similarity against every stored employee embedding.
4. If the best match's similarity ≥ `SIMILARITY_THRESHOLD` (default 0.55), it returns `VERIFIED` with the employee's name/department/confidence; otherwise `UNKNOWN`.
5. Logging is **presence-based, not per-frame**: the backend tracks who is currently in frame and only writes a log entry (+ snapshot) when that changes — a new person steps in, or the same person leaves for more than `PRESENCE_TIMEOUT_SECONDS` (default 4s) and returns. A person standing continuously in front of the camera is logged once for that visit, not once per frame.

---

## Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Frontend framework | Next.js (App Router) | 14.2.35 | Pages, routing, dev-server API proxy |
| UI runtime | React | 18.3.1 | Component rendering |
| Language | TypeScript | ^5.4.5 | Type-checked frontend |
| Styling | Tailwind CSS | ^3.4.4 | Utility-first styling |
| Client-side face guidance | MediaPipe Face Detection | CDN (`@mediapipe/face_detection`, `@mediapipe/camera_utils`) | Live oval-alignment overlay only — not used for actual recognition |
| Backend framework | FastAPI | 0.111.0 | REST + WebSocket API |
| ASGI server | Uvicorn | 0.30.1 | Runs the FastAPI app |
| Inference runtime | ONNX Runtime | 1.18.0 | Runs SCRFD + ArcFace `.onnx` models directly (CPU by default, GPU-capable) |
| Image processing | OpenCV (`opencv-python-headless`) | 4.9.0.80 | Decode/resize/blob-prep, Laplacian blur score, affine face alignment |
| Numerics | NumPy | 1.26.4 | Array math, embedding vectors |
| Config | Pydantic / pydantic-settings | 2.7.4 / 2.3.4 | Typed env-var configuration |
| Database | SQLite (stdlib `sqlite3`) | — | Employee records + recognition logs |
| Realtime transport | WebSockets (`websockets` lib via FastAPI) | 12.0 | Live recognition stream |

**Deliberately not used:** the `insightface` PyPI package. See [Challenges](#challenges-encountered--how-they-were-solved) below — it requires an MSVC/C++ build toolchain on Windows, so the backend instead loads the same underlying `.onnx` model files directly through `onnxruntime` and reimplements SCRFD's output decoding and ArcFace's face-alignment preprocessing itself.

---

## Models

Both models come from the **`buffalo_l`** pack in the [InsightFace model zoo](https://github.com/deepinsight/insightface) (downloaded by `backend/download_models.py`) and are loaded as raw ONNX graphs:

| Model file | Role | Architecture | Output |
|---|---|---|---|
| `det_10g.onnx` | Face **detection** | SCRFD (Sample and Computation Redistribution for Face Detection), 3 FPN scales (strides 8/16/32), 2 anchors/location | Per-anchor confidence score, bbox offsets, and 5-point landmark offsets (left eye, right eye, nose, left mouth corner, right mouth corner) |
| `w600k_r50.onnx` | Face **recognition** | ArcFace embedding head on a ResNet-50 backbone | 512-dim float32 embedding, L2-normalized so cosine similarity = dot product |

Also present in the downloaded `buffalo_l` pack but **unused** by this project (harmless to leave on disk): `1k3d68.onnx` (3D landmarks), `2d106det.onnx` (dense 2D landmarks), `genderage.onnx` (gender/age estimation).

**Client-side only:** MediaPipe's bundled "short-range" `BlazeFace`-derived face detection model, loaded from `cdn.jsdelivr.net` at runtime — used exclusively to drive the registration page's live oval overlay; its output never reaches the server or influences any stored data.

---

## Project Structure

```
/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app: REST endpoints + WebSocket
│   │   ├── config.py                # Pydantic settings (env-var driven)
│   │   ├── database.py              # SQLite schema + CRUD helpers
│   │   └── services/
│   │       └── face_service.py      # SCRFDDetector, ArcFaceEmbedder, FaceService
│   ├── tests/
│   │   └── test_face_service.py     # Unit tests (pose-estimation geometry)
│   ├── download_models.py           # Fetches & extracts the buffalo_l ONNX pack
│   ├── requirements.txt             # Pinned Python dependencies
│   └── data/                        # Generated at runtime (gitignored-worthy):
│       ├── employees.db             #   SQLite database
│       ├── models/buffalo_l/        #   Downloaded .onnx model files
│       ├── thumbnails/               #   Per-employee profile thumbnail JPEGs
│       └── snapshots/                #   Per-recognition-event snapshot JPEGs
└── frontend/
    ├── src/app/
    │   ├── layout.tsx               # Sidebar nav + page shell
    │   ├── page.tsx                 # Dashboard (stats + nav cards)
    │   ├── register/page.tsx        # Guided 5-pose capture flow
    │   ├── detect/page.tsx          # Live WebSocket recognition + result card
    │   └── logs/page.tsx            # Filterable recognition log + employee directory
    ├── next.config.mjs              # Dev-server proxy: /api, /static → backend
    ├── tailwind.config.js
    └── package.json
```

---

## Setup & Installation

### 1. Backend

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows PowerShell
# source venv/bin/activate       # Linux/macOS

pip install -r requirements.txt
```

No C++/MSVC build toolchain is required — every dependency (`onnxruntime`, `opencv-python-headless`, `numpy`, `fastapi`, ...) ships as a prebuilt wheel.

### 2. Download the models

```powershell
python download_models.py
```

Downloads `buffalo_l.zip` (~275MB) from the InsightFace GitHub releases and extracts `det_10g.onnx` + `w600k_r50.onnx` (plus the unused extras) into `backend/data/models/buffalo_l/`. Safe to re-run — it skips the download if the required files already exist.

### 3. Start the backend

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

The SQLite database and `data/thumbnails` / `data/snapshots` folders are created automatically on first run.

### 4. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). `/api/*` and `/static/*` requests are proxied to the backend on port 8000 via `next.config.mjs` rewrites, and WebSocket connections (`/api/recognize/stream`) go **directly** to `ws://127.0.0.1:8000` from the browser, since Next's dev-server rewrites don't proxy WebSocket upgrades. Both the rewrite destinations and the WebSocket URL deliberately use `127.0.0.1`, not `localhost` — see [Challenges #14](#challenges-encountered--how-they-were-solved) if you're curious why that matters (short version: it avoids a real, multi-second-per-request lag on Windows).

> If port 8000 or 3000 is already in use on your machine, start the backend/frontend on different ports and update `next.config.mjs`'s rewrite destinations and the `wsUrl` in `detect/page.tsx` to match.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/api/employees/register/check-frame` | Validates one frame: face presence, blur score, estimated pose. Returns `{accepted, reason, pose, blur_score, bbox, ...}` |
| `POST` | `/api/employees/register` | Registers an employee from a batch of ≥5 base64 frames. Rejects blurry/near-duplicate frames server-side, averages the rest into one embedding. Returns frame-acceptance stats |
| `GET` | `/api/employees` | Lists all registered employees (id, name, department, thumbnail URL) |
| `GET` | `/api/employees/{id}` | Single employee detail |
| `DELETE` | `/api/employees/{id}` | Deletes an employee record and their thumbnail file |
| `GET` | `/api/logs?search=&status=&start_date=&end_date=` | Filterable recognition log |
| `WS` | `/api/recognize/stream` | Send a base64 JPEG per message; receive `{status: VERIFIED\|UNKNOWN\|NO_FACE\|ERROR, ...}` |
| `GET` | `/static/thumbnails/*`, `/static/snapshots/*` | Static image serving |

---

## Database Schema

**`employees`**
| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | Employee ID (user-supplied) |
| `name` | TEXT | |
| `department` | TEXT | |
| `thumbnail_path` | TEXT | Relative `/static/...` URL |
| `embedding` | BLOB | 512 × float32, raw bytes (2048 bytes), L2-normalized |

**`recognition_logs`**
| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | |
| `timestamp` | TEXT | ISO 8601 |
| `employee_id` | TEXT, nullable | NULL for unknown-person events |
| `employee_name` | TEXT, nullable | |
| `status` | TEXT | `VERIFIED` or `UNKNOWN` |
| `confidence` | REAL | Cosine similarity of the best match (0–1) |
| `snapshot_path` | TEXT | Server filesystem path; served via `/static/snapshots/...` |

---

## Configuration

Environment variables (create `backend/.env` to override), read via `pydantic-settings` in `app/config.py`:

```ini
PROJECT_NAME="AI Employee Face Recognition"
DATABASE_URL="sqlite:///./data/employees.db"
MODEL_DIR="./data/models/buffalo_l"

SIMILARITY_THRESHOLD=0.55         # Cosine similarity threshold to call a match VERIFIED
MIN_CAPTURE_FRAMES=20             # Total frames required across all 5 registration steps
LAPLACIAN_THRESHOLD=80.0          # Blur rejection threshold (lower = blur allowed through)
PRESENCE_TIMEOUT_SECONDS=4.0      # How long someone must be gone before their next detection counts as a new visit
```

### CPU vs. GPU inference
By default both ONNX sessions use `CPUExecutionProvider`. To use an NVIDIA GPU:
1. `pip uninstall onnxruntime && pip install onnxruntime-gpu`
2. In `backend/app/services/face_service.py`, change `providers=["CPUExecutionProvider"]` (in both `SCRFDDetector.__init__` and `ArcFaceEmbedder.__init__`) to `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]`.

If CUDA drivers aren't present, `onnxruntime-gpu` falls back to CPU automatically.

---

## Core Algorithms

- **SCRFD decoding** (`SCRFDDetector.detect`): letterbox-resizes the frame to 640×640, runs the ONNX graph, and manually decodes its 9 raw outputs (score/bbox/landmark tensors × 3 FPN strides) using anchor-center generation, `distance2bbox`/`distance2kps` transforms, and greedy NMS (IoU 0.4) — the same decoding logic InsightFace's own `model_zoo.SCRFD` implements internally.
- **ArcFace alignment** (`ArcFaceEmbedder.align`): estimates a similarity transform (`cv2.estimateAffinePartial2D`) mapping the detected 5-point landmarks onto the canonical 112×112 ArcFace reference template, warps the crop, then runs it through the ResNet-50 embedding head and L2-normalizes the output.
- **Blur detection**: variance of the Laplacian of the grayscale face crop; below `LAPLACIAN_THRESHOLD` the frame is rejected as too blurry.
- **Pose estimation**: purely geometric, from the 5 landmarks — `yaw_ratio = (nose.x − left_eye.x) / (right_eye.x − nose.x)` classifies left/right, `pitch_ratio = (nose.y − eye_midpoint.y) / (mouth_midpoint.y − nose.y)` classifies up/down. Thresholds were calibrated against the ArcFace reference template's own geometry (a true front-facing face has pitch_ratio ≈ 0.98), not arbitrary guesses.
- **Duplicate rejection**: within one registration batch, a candidate frame's embedding is compared (cosine similarity) against every already-accepted embedding; anything above 0.95 similarity is treated as a near-duplicate and dropped so the averaged embedding isn't skewed by many near-identical frames.

---

## Challenges Encountered & How They Were Solved

This project went through a real debugging pass, not just a paper design — several issues only surfaced by actually installing dependencies, downloading real models, and driving the UI with a live webcam.

1. **`insightface` package vs. Windows.** The original scaffold imported the full `insightface` Python package (`FaceAnalysis`), which needs an MSVC/C++ compiler to build on Windows and wasn't even listed in `requirements.txt` — so face recognition could never actually initialize. **Fix:** rewrote `face_service.py` to load `det_10g.onnx` and `w600k_r50.onnx` directly via `onnxruntime`, reimplementing SCRFD's multi-scale output decoding and ArcFace's landmark-based alignment from scratch. Verified end-to-end against a real photo (unit-norm 512-dim embedding, self-similarity ≈ 1.0, flipped-face similarity ≈ 0.91).

2. **Pose-estimation thresholds didn't match reality.** The pitch-ratio band classifying "front" excluded the pitch ratio a genuinely front-facing face actually produces (~0.98, confirmed against the ArcFace reference template's own landmark geometry) — a symmetric face was misclassified as "down." Caught by the project's own unit test once it was actually run. Fixed the thresholds (0.6–1.5 instead of 0.35–0.85).

3. **`numpy.float32` isn't JSON-serializable.** `check-frame` returned raw NumPy scalars (blur score, yaw/pitch ratios) straight into the response dict; FastAPI's encoder can't serialize them, so every real call 500'd. Only surfaced by hitting the live endpoint. Fixed by casting to native Python `float` at the source in `face_service.py`.

4. **Built but never wired up.** `FaceService.is_duplicate()` existed from the start but nothing called it, so the "reject near-duplicate frames" requirement wasn't actually enforced. Wired it into `POST /api/employees/register`'s accumulation loop.

5. **Windows file lock on the downloaded model zip.** `download_models.py` opened each zip member via `zip_ref.open(member)` without closing the handle, which on Windows kept the archive locked so the subsequent `os.remove()` cleanup failed with `WinError 32`. Fixed by wrapping the extraction in a `with` block.

6. **Critical Next.js CVE.** `npm audit` flagged 14.2.4 for a critical cache-poisoning vulnerability. Bumped to the latest 14.2.x patch (14.2.35) rather than jumping to Next 16 (which would mean a React 19 / App Router breaking upgrade out of scope for a bug-fix pass).

7. **Oval alignment stuck on "move closer" despite a well-framed face.** MediaPipe's bounding-box width is a fraction of the *raw, uncropped* camera frame, but the video preview is displayed with CSS `object-cover`, which visually crops/zooms that frame to fill its container. A face that looked correctly sized on screen could still measure as "too small" against the uncropped frame whenever the camera's native aspect ratio wasn't exactly 4:3. Fixed by replicating the `object-cover` transform (scale + crop offset) in the alignment math, so the measurement matches what the user actually sees.

8. **Registration would never advance past "Front-facing."** Root cause: MediaPipe's `boundingBox` reports `xCenter`/`yCenter` (the box's center point), not `xMin`/`yMin` (top-left corner) — the original code read a property that doesn't exist on the object, producing silent `NaN` positions that made the oval's centering check mathematically impossible to satisfy, forever. Diagnosed by adding a temporary on-screen debug readout (`dx:NaNpx`) rather than guessing at thresholds again, then fixed the coordinate math to use the correct fields.

9. **After that fix, capture counts stopped advancing steps but kept climbing (up to 114/6).** `detector.onResults(onFaceResults)` is registered exactly once, when the camera starts — the MediaPipe SDK then holds that one JavaScript closure forever, permanently seeing `currentStepIndex = 0` from the render it was created in, no matter how many times React's own state actually advanced. **Fix:** replaced closure-captured state with `useRef`s (`currentStepIndexRef`, `capturedImagesRef`, `stepImagesCountRef`) as the single source of truth that the callback reads/writes directly, immune to which stale closure holds a reference to the ref *object* (only `.current` needs to be current).

10. **Counts were exactly double the real capture count (18 vs. 9).** The original code nested a `setStepImagesCount(...)` call inside `setCapturedImages(...)`'s updater function — calling `setState` from inside another `setState`'s updater is unsafe under React 18 StrictMode, which double-invokes updater functions in development to catch exactly this kind of impurity; the *nested* call isn't idempotent like the outer one, so it fired twice per real event. Fixed by moving to direct ref mutation + a single mirroring `setState` call per event, eliminating the nested-updater pattern entirely.

11. **Dev-environment port collisions.** The sandbox this was built and tested in already had unrelated services bound to ports 8000, 3000, and 3001. Rather than kill processes the assistant didn't start, the app was run on alternate ports (8002/3002) for testing — a purely environmental issue, not a bug in the app. If you hit `[Errno 10048] only one usage of each socket address...` on your own machine, something else is bound to that port; either free it or run this project's servers on different ports.

12. **The live recognition stream could stall the entire backend, not just the video.** `recognize_stream`'s WebSocket handler is `async def`, but it called the CPU-bound `service.detect_faces(...)` (SCRFD + ArcFace, ~300-500ms/frame on CPU) directly inline. A blocking call inside `async def` doesn't yield back to FastAPI's single asyncio event loop, so for the duration of every frame's inference, *every other request* — other browser tabs' dashboard/logs fetches, other connections — would stall, not just the Detect page. Fixed by offloading the call with `await asyncio.to_thread(service.detect_faces, img)`, freeing the event loop to keep serving everything else while inference runs on a worker thread.

13. **Registration's per-frame quality check was computing (and discarding) a full face embedding it never used.** `check_frame` only needs blur score, pose, and bbox to validate a frame — never identity — but `detect_faces()` unconditionally ran the ArcFace embedding step for every detected face regardless of whether the caller needed it. Benchmarked on the dev machine: SCRFD detection alone ≈200ms, ArcFace embedding ≈250ms on top of that — so `check_frame` was paying for embedding computation that made every single poll during the 5-step guided capture (fired every ~500ms) roughly twice as slow as necessary. Fixed by adding a `compute_embeddings` flag to `detect_faces()` and skipping it in `check_frame` and in the registration thumbnail-crop lookup (also identity-agnostic), roughly halving their latency.

14. **A ~2 second phantom delay on nearly every request, only on Windows.** Measured server-side timing showed `check_frame` actually completing in ~140ms, but the client (and, critically, the browser through Next.js's dev-server proxy) was seeing ~2.2-3s round trips. Root cause: resolving the hostname `"localhost"` returns the IPv6 loopback (`::1`) *before* the IPv4 loopback (`127.0.0.1`) on this system, and since uvicorn only binds to the IPv4 address, any client that tries addresses in that order (confirmed for both Python's `urllib` and, very plausibly, Node.js - which is what actually performs the proxied request for Next's `next.config.mjs` rewrites) stalls waiting for the IPv6 connection attempt to fail before falling back to IPv4. This single issue plausibly accounted for the bulk of reported "sometimes laggy" / "registration is slow and hectic" symptoms, since the register page polls `check-frame` roughly twice a second - each poll paying this ~2s tax. **Fix:** use `127.0.0.1` explicitly instead of `"localhost"` in `next.config.mjs`'s rewrite destinations and in the Detect page's WebSocket URL, bypassing hostname resolution entirely. Confirmed via direct measurement: identical requests dropped from ~2200ms to ~150-190ms after the change.

---

## 🔒 Biometric Data Compliance Note

> [!WARNING]
> **Production Compliance Requirements**
>
> Because this system processes and stores biometric data (face embeddings and images), deploying it in production requires adherence to applicable legal frameworks — e.g. **GDPR Article 9** (EU), **BIPA** (Illinois, USA), or local labor law equivalents. At minimum, a production release should add:
>
> 1. **Explicit consent handling** — signed, written consent from each employee describing how their face data is captured, analyzed, and verified.
> 2. **Data retention policy** — embeddings and images deleted promptly on termination or after a defined retention window.
> 3. **Encryption at rest and in transit** — encrypt the SQLite file/backups; run all traffic over HTTPS/WSS, not plain HTTP/WS.
> 4. **Data portability/erasure** — a clear mechanism for an employee to request deletion or export of their own biometric record.

---

## Known Limitations

- **No authentication/authorization** on any API endpoint — anyone who can reach the backend can register, delete employees, or read logs. Add an auth layer before exposing this beyond localhost.
- **CORS is wide open** (`allow_origins=["*"]`) — fine for local development, not for production.
- **No rate limiting** on the WebSocket stream or registration endpoints.
- **SQLite** is fine for a single-instance prototype; a multi-instance/production deployment would need a real database and a proper vector index (e.g. FAISS/pgvector) instead of a linear per-request scan over all employee embeddings.
- **CPU inference by default** — SCRFD + ArcFace on CPU is fine for a handful of concurrent recognitions but will bottleneck at scale; see the GPU section above.
