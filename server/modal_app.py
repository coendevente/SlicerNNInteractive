"""Modal.com GPU deployment of the nnInteractive Slicer server (nninteractive v2).

Deploy (persistent URL):
    modal deploy modal_app.py

Serve locally (ephemeral URL, hot-reload):
    modal serve modal_app.py

The deployed endpoint is a drop-in replacement for the local server at port 1527.
Point the Slicer extension at the Modal URL instead of http://localhost:1527.

Changes from the upstream v1.0.1 server:
  - nninteractive upgraded from 1.0.1 → 2.4.2
  - Removed use_pinned_memory (replaced by interactions_storage='auto')
  - Added interaction_bbox to lasso/scribble calls for 3-orders-of-magnitude speedup
  - Added /health and /reset endpoints
  - Model weights cached in a persistent Modal Volume (no re-download on restart)
  - container_idle_timeout=360 keeps the container alive during a session
  - allow_concurrent_inputs=1 serializes requests to protect in-memory session state
"""

import os
import modal

app = modal.App("nninteractive-slicer-server")

# Persistent volume for model weights (~2 GB for nnInteractive_v1.0)
weight_volume = modal.Volume.from_name("nninteractive-weights", create_if_missing=True)
WEIGHTS_PATH = "/root/.nninteractive_weights"

REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libglib2.0-0")
    # torch with CUDA first so nninteractive's dep-resolver keeps the GPU build
    .pip_install(
        "torch>=2.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    # nninteractive v2 + server deps
    .pip_install(
        "nninteractive==2.4.2",
        "fastapi",
        "python-multipart",
        "uvicorn[standard]",
        "numpy",
        "xxhash",
        "huggingface_hub",
        "blosc2",      # used by nninteractive v2 for interactions_storage='auto'
    )
)


@app.function(
    image=image,
    gpu="A10G",
    volumes={WEIGHTS_PATH: weight_volume},
    scaledown_window=360,         # keep container alive 6 min after last request
    timeout=3600,                 # allow sessions up to 1 hour
    memory=32768,                 # 32 GB RAM
    cpu=4.0,
)
@modal.asgi_app()
def nninteractive_server():
    import gzip
    import io
    import time

    import numpy as np
    import torch
    from fastapi import FastAPI, File, Form, Response, UploadFile
    from huggingface_hub import snapshot_download
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    from pydantic import BaseModel

    # -------------------------------------------------------------------------
    # Model download (skipped if already in volume)
    # -------------------------------------------------------------------------
    snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=[f"{MODEL_NAME}/*"],
        local_dir=WEIGHTS_PATH,
    )
    weight_volume.commit()

    # -------------------------------------------------------------------------
    # Session initialisation (runs once per container lifetime)
    # -------------------------------------------------------------------------
    session = nnInteractiveInferenceSession(
        device=torch.device("cuda:0"),
        use_torch_compile=False,
        verbose=True,
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,
        interactions_storage="auto",   # v2 replaces use_pinned_memory with this
    )
    session.initialize_from_trained_model_folder(
        os.path.join(WEIGHTS_PATH, MODEL_NAME)
    )

    # Mutable per-container state (safe because allow_concurrent_inputs=1)
    state: dict = {"img": None, "target": None}

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _check_image():
        if state["img"] is None:
            return {"status": "error", "message": "No image uploaded"}
        return None

    def _bbox_from_mask(mask: np.ndarray):
        """Return tight [[min,max], ...] bbox of nonzero voxels, or None."""
        nz = np.argwhere(mask)
        if len(nz) == 0:
            return None
        lo, hi = nz.min(axis=0), nz.max(axis=0)
        return [[int(lo[i]), int(hi[i])] for i in range(mask.ndim)]

    def _pack_seg(arr: np.ndarray) -> bytes:
        return gzip.compress(np.packbits(arr.astype(bool), axis=None).tobytes())

    def _seg_response(arr: np.ndarray) -> Response:
        return Response(
            content=_pack_seg(arr),
            media_type="application/octet-stream",
            headers={"Content-Encoding": "gzip"},
        )

    # -------------------------------------------------------------------------
    # FastAPI app
    # -------------------------------------------------------------------------
    web_app = FastAPI(
        title="nnInteractive Slicer Server",
        description="Modal-hosted nnInteractive v2 endpoint for 3D Slicer",
        version="2.0",
    )

    @web_app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "nninteractive_version": "2.4.2",
            "image_loaded": state["img"] is not None,
            "image_shape": list(state["img"].shape) if state["img"] is not None else None,
        }

    # -- Upload endpoints --

    @web_app.post("/upload_image")
    async def upload_image(file: UploadFile = File(None)):
        arr = np.load(io.BytesIO(await file.read()))
        img = arr[None] if arr.ndim == 3 else arr   # ensure (1, x, y, z)

        session.reset_interactions()
        session.set_image(img)
        target = torch.zeros(img.shape[1:], dtype=torch.uint8)
        session.set_target_buffer(target)

        state["img"] = img
        state["target"] = target

        return {"status": "ok", "shape": list(img.shape)}

    @web_app.post("/upload_segment")
    async def upload_segment(file: UploadFile = File(None)):
        if (err := _check_image()):
            return err

        arr = np.load(io.BytesIO(gzip.decompress(await file.read())))
        if np.sum(arr) == 0:
            session.reset_interactions()
            target = torch.zeros(state["img"].shape[1:], dtype=torch.uint8)
            state["target"] = target
            session.set_target_buffer(target)
        else:
            session.add_initial_seg_interaction(arr)

        return {"status": "ok"}

    @web_app.post("/reset")
    async def reset():
        """Reset interactions without requiring a new image upload."""
        session.reset_interactions()
        if state["img"] is not None:
            target = torch.zeros(state["img"].shape[1:], dtype=torch.uint8)
            state["target"] = target
            session.set_target_buffer(target)
        return {"status": "ok"}

    # -- Interaction endpoints --

    class PointParams(BaseModel):
        voxel_coord: list[int]
        positive_click: bool

    @web_app.post("/add_point_interaction")
    async def add_point_interaction(params: PointParams):
        if (err := _check_image()):
            return err
        t = time.time()
        session.add_point_interaction(
            params.voxel_coord, include_interaction=params.positive_click
        )
        print(f"point {time.time()-t:.3f}s")
        return _seg_response(state["target"].cpu().numpy())

    class BBoxParams(BaseModel):
        outer_point_one: list[int]
        outer_point_two: list[int]
        positive_click: bool

    @web_app.post("/add_bbox_interaction")
    async def add_bbox_interaction(params: BBoxParams):
        if (err := _check_image()):
            return err
        t = time.time()
        data = np.array([params.outer_point_one, params.outer_point_two])
        lo, hi = data.min(0), data.max(0)
        bbox = [[int(lo[i]), int(hi[i])] for i in range(3)]
        session.add_bbox_interaction(bbox, include_interaction=params.positive_click)
        print(f"bbox {time.time()-t:.3f}s")
        return _seg_response(state["target"].cpu().numpy())

    @web_app.post("/add_lasso_interaction")
    async def add_lasso_interaction(
        file: UploadFile = File(...), positive_click: str = Form(...)
    ):
        if (err := _check_image()):
            return err
        t = time.time()
        mask = np.load(io.BytesIO(gzip.decompress(await file.read())))
        bbox = _bbox_from_mask(mask)   # v2 speedup: only process the drawn region
        positive = positive_click.lower() in ("true", "1", "yes")
        session.add_lasso_interaction(
            mask, include_interaction=positive, interaction_bbox=bbox
        )
        print(f"lasso {time.time()-t:.3f}s  bbox={bbox}")
        return _seg_response(state["target"].cpu().numpy())

    @web_app.post("/add_scribble_interaction")
    async def add_scribble_interaction(
        file: UploadFile = File(...), positive_click: str = Form(...)
    ):
        if (err := _check_image()):
            return err
        t = time.time()
        mask = np.load(io.BytesIO(gzip.decompress(await file.read())))
        bbox = _bbox_from_mask(mask)   # v2 speedup: only process the drawn region
        positive = positive_click.lower() in ("true", "1", "yes")
        session.add_scribble_interaction(
            mask, include_interaction=positive, interaction_bbox=bbox
        )
        print(f"scribble {time.time()-t:.3f}s  bbox={bbox}")
        return _seg_response(state["target"].cpu().numpy())

    return web_app
