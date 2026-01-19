import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
from steren.network import Detector
import geopandas as gpd
from shapely.geometry import box
import os
from shapely.geometry import Point
import math
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import numpy as np
import cv2


headers = {
    "User-Agent": "Mozilla/5.0 (compatible; research-scraper/1.0)"
}

def clean_text(s):
    if s is None:
        return None
    return (
        s.replace("\xa0", " ")
         .replace("°", "")
         .strip()
    )


def extract_all_metadata(soup):
    """
    Extracts all metadata fields from a HiRISE details page.
    Returns a dict: {label: value}
    """
    metadata = {}
    
    # Find all <strong> tags inside the main details container
    for strong in soup.find_all("strong"):
        label = clean_text(strong.get_text())
        
        # Skip empty labels
        if not label:
            continue
        
        # Find the value after the <strong>
        value = None
        for sib in strong.next_siblings:
            if isinstance(sib, str):
                value_candidate = sib.strip()
                if value_candidate:
                    value = value_candidate
                    break
            elif hasattr(sib, "get_text"):
                value_candidate = sib.get_text(strip=True)
                if value_candidate:
                    value = value_candidate
                    break
        
        if value:
            metadata[label] = clean_text(value)
    
    return metadata

def human_size(size_bytes):
    """Convert size in bytes (int) to human-readable format"""
    for unit in ['B','KB','MB','GB','TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}PB"

class HRimage:
    
    def __init__(self, identifier : str = None, img_path : str = None):

        self.identifier = identifier 
        if identifier:
            self.mission_phase, orbit_number, latitude10 = identifier.split('_')
            self.orbit_number = int(orbit_number)
            self.latitude = int(latitude10)/10
            self.url = "https://uahirise.org/" + identifier

        if img_path:
            self.img_path = img_path


    def meta(self):
        """ Scrapes web for image metadata"""
        resp = requests.get(self.url,headers = headers, timeout = 30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text,'lxml')
        title_tag = soup.select_one("span.observation-title-milo")
        self.title = title_tag.get_text(strip=True) if title_tag else None
        
        desc_raw = soup.select_one("div.caption-text-no-extras")
        self.description = clean_text(desc_raw.get_text(" ", strip=True) if desc_raw else None)
        
        raw_metadata = extract_all_metadata(soup)

        self.date = raw_metadata.get("Acquisition date", None)
        self.latitude = float(raw_metadata.get("Latitude (centered)", None))
        self.longitude = float(raw_metadata.get("Longitude (East)", None))
        self.altitude = raw_metadata.get("Spacecraft altitude", None)



        return {
            "observations_id" : self.identifier,
            "title" : self.title,
            "description": self.description,
            "acquisition date" : self.date,
            "latitude" : self.latitude,
            "longitude": self.longitude,
            "spacecraft altitude": self.altitude
            }

    def download(self, output_dir: str):
        """List PDS files with size and date, let user pick one"""
        orbit_range = f"{self.orbit_number - (self.orbit_number % 100):06d}_{self.orbit_number - (self.orbit_number % 100) + 99:06d}"
        url = f"https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/RDR/{self.mission_phase}/ORB_{orbit_range}/{self.identifier}/"
        print(f"Fetching: {url}\n")

        r = requests.get(url)
        if r.status_code != 200:
            print("Failed to access PDS directory.")
            return None

        soup = BeautifulSoup(r.text, "html.parser")

        # PDS directory listing is inside <pre>
        pre = soup.find("pre")
        if not pre:
            print("No directory listing found.")
            return None

        files = []
        for line in pre.text.splitlines():
            parts = line.split()
            if len(parts) < 2 or line.startswith("../"):
                continue
            filename = parts[0]
            date = parts[1] + " " + parts[2] if len(parts) >= 3 else ""
            size = None
            # last numeric part is usually the size
            for p in reversed(parts):
                if p.isdigit():
                    size = int(p)
                    break
            files.append((filename, date, human_size(size)))

        # print list
        for i, (fname, date, size) in enumerate(files, 1):
            print(f"{i}. {fname:40} {date:20} {size:>8}")

        choice = input(f"\nSelect a file (1-{len(files)}): ")
        if not choice.isdigit() or not (1 <= int(choice) <= len(files)):
            print("Invalid selection.")
            return None

        selected = files[int(choice)-1][0]
        full_url = url + selected
        print(f"\nSelected: {selected}\nURL: {full_url}")
        return full_url

    def detect(self, detector: Detector):
        """Runs a detector on the image and stores the results"""
        # add a condition here where self.img_path must have a value and exist? 
        if self.img_path is None:
            raise ValueError("Please provide an image path.")

        img_path = os.path.exists(self.img_path)
        if not os.path.exists(self.img_path):
            raise FileNotFoundError(f"Image path does not exist: {img_path}")


        boxes, scores, classes, crs, transform = detector.inference(self.img_path)
        self.boxes = boxes
        self.scores = scores
        self.classes = classes
        self.crs = crs
        self.transform = transform
    
    def export_detections(self, output_path):

        if not hasattr(self, "boxes") or self.boxes is None:
            raise RuntimeError("No detections found. Run detect() first.")

        # Check file extension
        if not output_path.lower().endswith(".gpkg"):
            raise ValueError("Output path must end with .gpkg")


        geometries = []
        diameters = []
        for xmin, ymin, xmax, ymax in self.boxes:
            # pixel → map coords
            x1, y1 = self.transform * (xmin, ymin)
            x2, y2 = self.transform * (xmax, ymax)

            # center of box (map coords)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # average radius in map units
            radius = (abs(x2-x1)+abs(y2-y1))/4
            diameters.append(2*radius)

            # create circular polygon
            geometries.append(Point(cx, cy).buffer(radius))

        self.diameters = diameters



        gdf = gpd.GeoDataFrame(
            {
                "class": self.classes,
                "score": self.scores,
                "diameter": self.diameters,
            },
            geometry=geometries,
            crs=self.crs,
        )

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Write GeoPackage
        gdf.to_file(
            output_path,
            layer="detections",
            driver="GPKG") 



    def search(self):
        hirise_catalog = "https://www.uahirise.org/catalog/index.php?page="
        page_num = 0
        while True:
            page_num += 1 
            url = hirise_catalog + str(page_num)
            r = requests.get(url)
            
            if r.status_code != 200:
                print("Failed to access HiRISE catalog.")
                return None
            
            html = r.text
            codes = sorted(set(re.findall(r'ESP_\d{6}_\d{4}', html)))
            for identifier in codes:
                hrimage = HRimage(identifier)
                hrimage.meta()
                print(f"{hrimage.identifier:20} {hrimage.title:60} {hrimage.date:>10}")
                
    def slice(self, out_dir: str, tile_size:int = 512, overlap:int = 0): 
        """
        Slice a georeferenced image into overlapping tiles and save as PNG.

        Args:
            out_dir (str): Directory to save tiles.
            tile_size (int): Tile width/height in pixels.
            overlap (int): Number of overlapping pixels between tiles.
        """
        os.makedirs(out_dir, exist_ok=True)
        img_path = self.img_path

        # Open image
        with rasterio.open(img_path) as ds:
            img_w, img_h = ds.width, ds.height

            step = tile_size - overlap
            nx = max(1, math.ceil((img_w - overlap) / step))
            ny = max(1, math.ceil((img_h - overlap) / step))

            total_tiles = nx * ny
            print(f"[info] Image size: {img_w}x{img_h}, generating {total_tiles} tiles...")

            tile_count = 0
            for iy in tqdm(range(ny), desc="Rows"):
                for ix in range(nx):
                    x1 = ix * step
                    y1 = iy * step
                    x2 = min(x1 + tile_size, img_w)
                    y2 = min(y1 + tile_size, img_h)

                    # Adjust start if we hit right/bottom edge
                    x1 = max(0, x2 - tile_size)
                    y1 = max(0, y2 - tile_size)

                    window = Window(x1, y1, x2 - x1, y2 - y1)
                    arr = ds.read([1], window=window)  # read first 3 bands
                    tile = np.transpose(arr, (1, 2, 0))  # HWC

                    # Convert to uint8 if needed
                    if tile.dtype != np.uint8:
                        tmin, tmax = tile.min(), tile.max()
                        if tmax > tmin:
                            tile = ((tile - tmin) / (tmax - tmin) * 255).astype(np.uint8)
                        else:
                            tile = tile.astype(np.uint8)

                    tile_path = os.path.join(out_dir, f"tile_{tile_count:04d}.png")
                    cv2.imwrite(tile_path, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
                    tile_count += 1

            print(f"[info] ✅ All {tile_count} tiles saved to {out_dir}") 






