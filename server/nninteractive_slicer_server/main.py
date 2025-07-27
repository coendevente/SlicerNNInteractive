import time
import warnings
import os
import io
import gzip
import hashlib
import argparse
import gc

import numpy as np
import torch
import uvicorn

import xxhash

from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import (
    snapshot_download,
)

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession


from fastapi import FastAPI, Response, UploadFile, File, Form


###############################################################################
# Global constants & FastAPI app
###############################################################################
REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"  # Updated models may be available in the future
DOWNLOAD_DIR = ".nninteractive_weights"  # Specify the download directory

app = FastAPI()

###############################################################################
# Utility / helper functions
###############################################################################


def calculate_md5_array(image_data, xx=False):
    """
    Calculate either an xxHash (if xx=True) or MD5 hash of a NumPy array's bytes.
    """
    if xx:
        xh = xxhash.xxh64()
        xh.update(image_data.tobytes())

        out_hash = xh.hexdigest()
    else:
        md5_hash = hashlib.md5()
        md5_hash.update(image_data.tobytes())
        out_hash = md5_hash.hexdigest()

    return out_hash


def unpack_binary_segmentation(binary_data, vol_shape):
    """
    Unpacks binary data (1 bit per voxel) into a full 3D numpy array (bool type).
    """
    total_voxels = np.prod(vol_shape)
    unpacked_bits = np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8))
    unpacked_bits = unpacked_bits[:total_voxels]
    segmentation_mask = (
        unpacked_bits.reshape(vol_shape).astype(np.bool_).astype(np.uint8)
    )

    return segmentation_mask


def segmentation_binary(seg_in, compress=False):
    """
    Convert a (boolean) segmentation array into packed bits and optionally compress.
    """
    seg_result = seg_in.astype(bool)  # Convert to bool type if not already
    packed_segmentation = np.packbits(seg_result, axis=None)  # Pack into 1D byte array
    packed_segmentation = packed_segmentation.tobytes()
    if compress:
        packed_segmentation = gzip.compress(packed_segmentation)
    return packed_segmentation  # Convert to bytes for transmission


def process_mask_and_click_input(file_bytes, positive_click):
    """
    Helper that decompresses file_bytes, loads the numpy mask, and interprets
    the positive_click string as a boolean.
    """
    positive_click_bool = positive_click.lower() in ["true", "1", "yes"]
    t = time.time()

    error = get_error_if_img_not_set()
    if error is not None:
        return error

    try:
        decompressed = gzip.decompress(file_bytes)
    except Exception as e:
        return {"status": "error", "message": f"Decompression failed: {e}"}

    # Load the numpy mask.
    mask = np.load(io.BytesIO(decompressed))

    return mask, positive_click_bool


def get_error_if_img_not_set():
    if PROMPT_MANAGER.img is None:
        warnings.warn("There is no image in the server. Be sure to send it before")
        return {"status": "error", "message": "No image uploaded"}
    
    return


###############################################################################
# PromptManager class
###############################################################################
class PromptManager:
    """
    Manages the image, target tensor, and runs inference sessions for point, bbox,
    lasso, and scribble interactions.
    """

    def __init__(self):
        self.img = None
        self.target_tensor = None

        self.download_weights()
        self.session = self.make_session()

    def download_weights(self):
        """
        Downloads only the files matching 'MODEL_NAME/*' into DOWNLOAD_DIR.
        """
        snapshot_download(
            repo_id=REPO_ID, allow_patterns=[f"{MODEL_NAME}/*"], local_dir=DOWNLOAD_DIR
        )

    def make_session(self):
        """
        Creates an nnInteractiveInferenceSession, points it at the downloaded model.
        """
        session = nnInteractiveInferenceSession(
            device=torch.device("cuda:0"),  # Set inference device
            use_torch_compile=False,  # Experimental: Not tested yet
            verbose=True,
            torch_n_threads=os.cpu_count(),  # Use available CPU cores
            do_autozoom=True,  # Enables AutoZoom for better patching
            use_pinned_memory=True,  # Optimizes GPU memory transfers
        )

        # Load the trained model
        model_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)
        session.initialize_from_trained_model_folder(model_path)

        return session

    def set_image(self, input_image):
        """
        Loads the user-provided 3D image into the session, resets interactions.
        """
        self.session.reset_interactions()

        self.img = input_image[None]  # Ensure shape (1, x, y, z)
        self.session.set_image(self.img)

        print("self.img.shape:", self.img.shape)

        # Validate input dimensions
        if self.img.ndim != 4:
            raise ValueError("Input image must be 4D with shape (1, x, y, z)")

        self.target_tensor = torch.zeros(
            self.img.shape[1:], dtype=torch.uint8
        )  # Must be 3D (x, y, z)
        self.session.set_target_buffer(self.target_tensor)

    def set_segment(self, mask):
        """
        Sets or resets a segmentation (mask) on the server side.
        If mask is empty, resets the session's interactions.
        """
        if np.sum(mask) == 0:
            self.session.reset_interactions()
            self.target_tensor = torch.zeros(
                self.img.shape[1:], dtype=torch.uint8
            )  # Must be 3D (x, y, z)
            self.session.set_target_buffer(self.target_tensor)
        else:
            self.session.add_initial_seg_interaction(mask)

    def add_point_interaction(self, point_coordinates, include_interaction):
        """
        Process a point-based interaction (positive or negative).
        """
        self.session.add_point_interaction(
            point_coordinates, include_interaction=include_interaction
        )

        return self.target_tensor.clone().cpu().detach().numpy()

    def add_bbox_interaction(
        self, outer_point_one, outer_point_two, include_interaction
    ):
        """
        Process bounding box-based interaction.
        """
        print("outer_point_one, outer_point_two:", outer_point_one, outer_point_two)

        data = np.array([outer_point_one, outer_point_two])
        _min = np.min(data, axis=0)
        _max = np.max(data, axis=0)

        bbox = [
            [int(_min[0]), int(_max[0])],
            [int(_min[1]), int(_max[1])],
            [int(_min[2]), int(_max[2])],
        ]

        # Call the session's bounding box interaction function.
        self.session.add_bbox_interaction(bbox, include_interaction=include_interaction)

        return self.target_tensor.clone().cpu().detach().numpy()

    def add_lasso_interaction(self, mask, include_interaction):
        """
        Process lasso-based interaction using a 3D mask.
        """
        print("Lasso mask received with shape:", mask.shape)
        self.session.add_lasso_interaction(
            mask, include_interaction=include_interaction
        )
        return self.target_tensor.clone().cpu().detach().numpy()

    def add_scribble_interaction(self, mask, include_interaction):
        """
        Process scribble-based interaction using a 3D mask.
        """
        print("Scribble mask received with shape:", mask.shape)
        self.session.add_scribble_interaction(
            mask, include_interaction=include_interaction
        )
        return self.target_tensor.clone().cpu().detach().numpy()


###############################################################################
# Global prompt manager instance
###############################################################################
PROMPT_MANAGER = PromptManager()


###############################################################################
# FastAPI endpoints
###############################################################################


#
# -- Upload endpoints
#
@app.post("/upload_image")
async def upload_image(
    file: UploadFile = File(None),
):
    """
    Receive a npy file from the client and set it as the main image in PromptManager.
    """
    file_bytes = await file.read()
    arr = np.load(io.BytesIO(file_bytes))
    PROMPT_MANAGER.set_image(arr)

    return {"status": "ok"}


@app.post("/upload_segment")
async def upload_segment(
    file: UploadFile = File(None),
):
    """
    Receive a gzipped npy file from the client and set it as the segmentation in PromptManager.
    """
    error = get_error_if_img_not_set()
    if error is not None:
        return error
    
    file_bytes = await file.read()
    decompressed = gzip.decompress(file_bytes)
    arr = np.load(io.BytesIO(decompressed))

    PROMPT_MANAGER.set_segment(arr)
    return {"status": "ok"}


#
# -- Point interaction endpoint
#
class PointParams(BaseModel):
    voxel_coord: list[int]
    positive_click: bool


@app.post("/add_point_interaction")
async def add_point_interaction(params: PointParams):
    """
    Receives a point (voxel_coord) + positive/negative. Updates the model & returns a binary mask.
    """
    try:
        error = get_error_if_img_not_set()
        if error is not None:
            return error
        
        t = time.time()

        seg_result = PROMPT_MANAGER.add_point_interaction(
            point_coordinates=params.voxel_coord, include_interaction=params.positive_click
        )
        compressed_bin = segmentation_binary(seg_result, compress=True)
        print(f"Server whole infer function time: {time.time() - t}")

        return Response(
            content=compressed_bin,
            media_type="application/octet-stream",
            headers={"Content-Encoding": "gzip"},
        )
    finally:
        gc.collect()
        torch.cuda.empty_cache()


#
# -- Bounding Box interaction endpoint
#
class BBoxParams(BaseModel):
    outer_point_one: list[int]
    outer_point_two: list[int]
    positive_click: bool


@app.post("/add_bbox_interaction")
async def add_bbox_interaction(params: BBoxParams):
    """
    Receives bounding box corners + positive/negative. Updates model & returns a mask.
    """
    try:
        error = get_error_if_img_not_set()
        if error is not None:
            return error
        
        t = time.time()

        seg_result = PROMPT_MANAGER.add_bbox_interaction(
            params.outer_point_one,
            params.outer_point_two,
            include_interaction=params.positive_click,
        )

        segmentation_binary_data = segmentation_binary(seg_result, compress=True)
        print(f"Server whole infer function time: {time.time() - t}")

        return Response(
            content=segmentation_binary_data,
            media_type="application/octet-stream",
            headers={"Content-Encoding": "gzip"},
        )
    finally:
        gc.collect()
        torch.cuda.empty_cache()


#
# -- Lasso interaction endpoint
#


@app.post("/add_lasso_interaction")
async def add_lasso_interaction(
    file: UploadFile = File(...), positive_click: str = Form(...)
):
    """
    Receives a gzipped npy mask + positive/negative. Treated as a 'lasso' 3D mask.
    """
    try:
        error = get_error_if_img_not_set()
        if error is not None:
            return error
        
        file_bytes = await file.read()
        mask, positive_click_bool = process_mask_and_click_input(file_bytes, positive_click)

        # Process the lasso interaction.
        seg_result = PROMPT_MANAGER.add_lasso_interaction(
            mask, include_interaction=positive_click_bool
        )

        # Convert the segmentation result to compressed binary data.
        segmentation_binary_data = segmentation_binary(seg_result, compress=True)

        return Response(
            content=segmentation_binary_data,
            media_type="application/octet-stream",
            headers={"Content-Encoding": "gzip"},
        )
    finally:
        gc.collect()
        torch.cuda.empty_cache()


#
# -- Scribble interaction endpoint
#
@app.post("/add_scribble_interaction")
async def add_scribble_interaction(
    file: UploadFile = File(...), positive_click: str = Form(...)
):
    """
    Receives a scribble mask + positive/negative. Updates model, returns updated segmentation.
    """
    try:
        error = get_error_if_img_not_set()
        if error is not None:
            return error
        
        # Read the uploaded file bytes and decompress.
        file_bytes = await file.read()

        mask, positive_click_bool = process_mask_and_click_input(file_bytes, positive_click)

        seg_result = PROMPT_MANAGER.add_scribble_interaction(
            mask, include_interaction=positive_click_bool
        )

        # Convert the segmentation result to compressed binary data.
        segmentation_binary_data = segmentation_binary(seg_result, compress=True)

        return Response(
            content=segmentation_binary_data,
            media_type="application/octet-stream",
            headers={"Content-Encoding": "gzip"},
        )
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Run the nnInteractive Slicer server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind to.")
    parser.add_argument("--port", type=int, default=1527, help="Port to listen on.")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
