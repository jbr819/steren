from steren.network import Detector
import rasterio
import math
import os
import numpy as np
import cv2
import torch
from torchvision.ops import batched_nms
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine, xy
from shapely.geometry import box as shapely_box
import geopandas as gpd
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("Ultralytics not found. Install with 'pip install ultralytics'.")





def make_tiles(img_w, img_h, tile_size, overlap):
    """Return list of (x1,y1,x2,y2) pixel coords covering image with overlap (pixels)."""
    step = tile_size - overlap
    nx = max(1, math.ceil((img_w - overlap) / step))
    ny = max(1, math.ceil((img_h - overlap) / step))

    tiles = []
    for iy in range(ny):
        for ix in range(nx):
            x1 = ix * step
            y1 = iy * step
            x2 = x1 + tile_size
            y2 = y1 + tile_size

            if x2 > img_w:
                x2 = img_w
                x1 = max(0, x2 - tile_size)
            if y2 > img_h:
                y2 = img_h
                y1 = max(0, y2 - tile_size)

            tiles.append((int(x1), int(y1), int(x2), int(y2)))
    # remove duplicates
    tiles = list(dict.fromkeys(tiles))
    return tiles


def read_tile_rgb(dataset, x1, y1, x2, y2, out_size=None):
    """
    Read window [x1,x2) x [y1,y2) from rasterio dataset and return HWC RGB uint8 array.
    out_size: (height, width) to rescale the read tile to (optional).
    Assumes raster has at least 3 bands (RGB). If >3 bands, uses first 3.
    """
    w = x2 - x1
    h = y2 - y1
    window = Window(x1, y1, w, h)
    # Read bands (1-indexed). Read minimum of 3 bands or all if fewer
    count = dataset.count
    bands_to_read = min(3, count)
    arr = dataset.read(range(1, bands_to_read + 1), window=window)  # shape (bands, h, w)

    # If bands < 3, replicate or pad
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    elif arr.shape[0] == 2:
        arr = np.vstack([arr, arr[0:1, :, :]])

    # convert to HWC
    tile = np.transpose(arr, (1, 2, 0))  # H, W, C

    # Raster values may be not in 0-255. Attempt rescale if dtype != uint8
    if tile.dtype != np.uint8:
        # scale using min/max per-tile (safe heuristic). If you want to use dataset-specific scaling, change here.
        tmin, tmax = tile.min(), tile.max()
        if tmax > tmin:
            tile = ((tile - tmin) / (tmax - tmin) * 255.0).astype(np.uint8)
        else:
            tile = np.clip(tile, 0, 255).astype(np.uint8)
    else:
        tile = tile.copy()

    # If out_size specified, resize tile to that size (width,height)
    if out_size is not None:
        tile = cv2.resize(tile, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)

    # Ensure channel order is RGB (rasterio gives bands in dataset order)
    return tile




class YOLOv8(Detector):
    def __init__(self, 
                 weights :str, 
                 tile_size: int, 
                 overlap:int,
                 conf_thresh :float,
                 iou_thresh:float,
                 device:int = None
                 ):

        self.weights = weights
        self.tile_size = tile_size
        self.overlap = overlap
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh

        self.model = YOLO(weights)
        if device:
            self.model.model.to(device)
    def set_device(self):
        pass 

    def load_weights(self):
        pass


    def inference(self,raster_path : str):
        
        if not os.path.exists(raster_path):
            raise FileNotFoundError(raster_path)

        with rasterio.open(raster_path) as ds:
            img_w = ds.width
            img_h = ds.height
            transform = ds.transform
            crs = ds.crs  # may be None

            tiles = make_tiles(img_w, img_h, self.tile_size, self.overlap)
            print(f"[info] image: {img_w}x{img_h}, tiles: {len(tiles)}, tile={self.tile_size}x{self.tile_size}, overlap={self.overlap}")
            all_boxes = []   # pixel-space boxes [x1,y1,x2,y2]
            all_scores = []
            all_classes = []

            for t_idx, (x1, y1, x2, y2) in enumerate(tqdm(tiles)):
                # Read tile into RGB np.uint8. Resize to model-friendly size if desired (we let model auto-resize)
                tile_rgb = read_tile_rgb(ds, x1, y1, x2, y2, out_size=None)  # HWC uint8
                if tile_rgb is None:
                    continue

                # Ultralytics model.predict accepts numpy HWC (RGB)
                results = self.model.predict(source=tile_rgb, 
                                             conf=self.conf_thresh, 
                                             imgsz=min(self.tile_size, 2048), 
                                             verbose=False)
                r = results[0]

                #if r.boxes is None or len(r.boxes) == 0:
                    #if debug:
                     #   print(f" tile {t_idx+1}/{len(tiles)}: 0 boxes")
                    #continue

                boxes = r.boxes.xyxy.cpu().numpy()  # coordinates in tile pixel space
                scores = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)

                # map tile coords -> full image pixel coords by adding offsets
                boxes[:, 0] += x1  # x1
                boxes[:, 1] += y1  # y1
                boxes[:, 2] += x1  # x2
                boxes[:, 3] += y1  # y2

                for b, s, c in zip(boxes, scores, classes):
                    all_boxes.append(b.tolist())
                    all_scores.append(float(s))
                    all_classes.append(int(c))

                #if debug:
               # print(f" tile {t_idx+1}/{len(tiles)}: {len(boxes)} boxes")

        # no detections
        if len(all_boxes) == 0:
            print("[info] no detections found")
            return

        # convert to tensors and clamp within image bounds
        boxes = torch.tensor(all_boxes, dtype=torch.float32)
        scores = torch.tensor(all_scores, dtype=torch.float32)
        labels = torch.tensor(all_classes, dtype=torch.int64)

        # clamp boxes to image
        boxes[:, 0].clamp_(0, img_w - 1)
        boxes[:, 1].clamp_(0, img_h - 1)
        boxes[:, 2].clamp_(0, img_w - 1)
        boxes[:, 3].clamp_(0, img_h - 1)

        # Perform class-aware NMS
        keep = batched_nms(boxes, scores, labels, self.iou_thresh)
        boxes_kept = boxes[keep].cpu().numpy()
        scores_kept = scores[keep].cpu().numpy()
        labels_kept = labels[keep].cpu().numpy()

        print(f"[info] {len(all_boxes)} raw boxes -> {len(keep)} after global NMS")

        
        return boxes_kept, scores_kept, labels_kept, crs, transform




