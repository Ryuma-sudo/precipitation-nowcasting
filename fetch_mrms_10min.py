import os
import requests
import datetime
import tempfile
import numpy as np
import zipfile
import gzip
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- CONFIGURATION ---
START_TIME = datetime.datetime(2025, 6, 1, 2, 0)
END_TIME = datetime.datetime(2025, 6, 1, 6, 0)

OUTPUT_FILE = Path("data/preprocessed-data/cmax_256x256.npy")

# Region filter - change to select different coverage areas
# Options: "CONUS" (Continental US), "Alaska", "Hawaii", "Guam", "Carib" (Caribbean)
# Set to None or "" to include all regions
REGION_FILTER = "CONUS"

# Bounding Box (CONUS) - 2048x2048 area, will be downscaled to 256x256
Y_MIN, Y_MAX = 202, 2250  # 2048 pixels
X_MIN, X_MAX = 4500, 6548  # 2048 pixels
DOWNSCALE_FACTOR = 8  # Downsample from 2048x2048 to 256x256

def create_session_with_retries():
    session = requests.Session()
    retry_strategy = Retry(
        total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def get_mrms_10min():
    # Helper to round down to the nearest hour for the URL
    current_hour_check = START_TIME.replace(minute=0, second=0, microsecond=0)
    
    frames = []
    session = create_session_with_retries()

    print(f"Fetching MRMS data from {START_TIME} to {END_TIME}...")

    while current_hour_check < END_TIME:
        # URL for the hourly ZIP
        url = f"https://mrms.agron.iastate.edu/{current_hour_check:%Y/%m/%d/%Y%m%d%H}.zip"
        print(f"\nProcessing ZIP for Hour: {current_hour_check:%Y-%m-%d %H:00}...")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, "data.zip")
                
                # 1. Download with progress bar
                r = session.get(url, stream=True, timeout=30)
                if r.status_code != 200:
                    print(f"  [!] ZIP not found/Error (Code {r.status_code})")
                    current_hour_check += datetime.timedelta(hours=1)
                    continue

                # Get file size for progress bar
                total_size = int(r.headers.get('content-length', 0))
                
                with open(zip_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                              desc="  Downloading", leave=False, ncols=80) as pbar:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                # Validate ZIP file was downloaded correctly
                downloaded_size = os.path.getsize(zip_path)
                print(f"  Downloaded {downloaded_size / (1024*1024):.1f} MB")
                
                if not zipfile.is_zipfile(zip_path):
                    print(f"  [!] Downloaded file is not a valid ZIP")
                    current_hour_check += datetime.timedelta(hours=1)
                    continue
                # 2. Extract and Process
                target_minutes = ['00', '10', '20', '30', '40', '50']
                
                try:
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        all_files = z.namelist()
                        selected_files = []
                        
                        for f_name in all_files:
                            # Robust Logic:
                            # 1. Must contain "PrecipRate"
                            # 2. Must match REGION_FILTER (if set)
                            # 3. Must NOT be "PrecipFlag" or "Synthetic"
                            # 4. Must be a grib2.gz file
                            if not ("PrecipRate" in f_name and 
                                    "Flag" not in f_name and 
                                    "Synthetic" not in f_name and 
                                    f_name.endswith(".grib2.gz")):
                                continue
                            
                            # Apply region filter if set
                            if REGION_FILTER and REGION_FILTER not in f_name:
                                continue
                                
                            try:
                                # Parse filename: .../MRMS_PrecipRate_00.00_20220601-123000.grib2.gz
                                # We split by '_' or '-' to find the date part
                                filename_only = f_name.split("/")[-1] # Handle folder path
                                date_time_part = filename_only.split("_")[-1].replace(".grib2.gz", "")
                                
                                # Parse "20220601-123000"
                                file_dt = datetime.datetime.strptime(date_time_part, "%Y%m%d-%H%M%S")
                                
                                # Check strict time range
                                if START_TIME <= file_dt <= END_TIME:
                                    if file_dt.strftime('%M') in target_minutes:
                                        selected_files.append((f_name, file_dt))
                                        
                            except ValueError:
                                continue
                        
                        # Sort by time
                        selected_files.sort(key=lambda x: x[1])
                        region_name = REGION_FILTER if REGION_FILTER else "All Regions"
                        print(f"  Found {len(selected_files)} {region_name} PrecipRate files")
                        
                        for gz_file, file_dt in selected_files:
                            extracted_gz_path = None
                            decompressed_grib2_path = None
                            try:
                                # Extract .grib2.gz from ZIP
                                print(f"    Extracting: {os.path.basename(gz_file)}...")
                                z.extract(gz_file, temp_dir)
                                extracted_gz_path = os.path.join(temp_dir, gz_file)
                                
                                # Check extracted file size
                                gz_size = os.path.getsize(extracted_gz_path)
                                print(f"      Size: {gz_size} bytes")
                                
                                if gz_size < 100:
                                    print(f"      [!] Too small, skipping")
                                    continue
                                
                                # Debug: Check file header to see what type it actually is
                                with open(extracted_gz_path, 'rb') as f:
                                    header = f.read(10)
                                    print(f"      Header bytes: {header[:10].hex()}")
                                
                                # Manual GZIP decompression to .grib2
                                decompressed_grib2_path = extracted_gz_path.replace(".grib2.gz", ".grib2")
                                
                                try:
                                    with gzip.open(extracted_gz_path, 'rb') as f_in:
                                        with open(decompressed_grib2_path, 'wb') as f_out:
                                            f_out.write(f_in.read())
                                    print(f"      GZIP decompressed OK")
                                except (gzip.BadGzipFile, OSError) as gz_err:
                                    # Maybe it's not actually gzipped, try reading directly
                                    print(f"      [!] GZIP failed ({gz_err}), trying direct read...")
                                    decompressed_grib2_path = extracted_gz_path  # Use the .gz file directly
                                
                                # Now read the .grib2 file with rasterio
                                with rasterio.open(decompressed_grib2_path) as src:
                                    data = src.read(1)
                                    crop = data[Y_MIN:Y_MAX, X_MIN:X_MAX]  # 512x512
                                    
                                    # Fix bad values
                                    if src.nodata is not None:
                                        crop[crop == src.nodata] = 0
                                    crop = np.nan_to_num(crop, nan=0.0)
                                    crop[crop < 0] = 0
                                    
                                    # Downsample from 512x512 to 256x256 using block averaging
                                    h, w = crop.shape
                                    new_h, new_w = h // DOWNSCALE_FACTOR, w // DOWNSCALE_FACTOR
                                    crop_downscaled = crop.reshape(new_h, DOWNSCALE_FACTOR, 
                                                                   new_w, DOWNSCALE_FACTOR).mean(axis=(1, 3))
                                    
                                    frames.append((crop_downscaled, file_dt))
                                    print(f"    -> Processed: {file_dt} (2048x2048 -> 256x256)")
                                    
                            except Exception as e:
                                print(f"    [!] Error reading {gz_file}: {e}")
                            finally:
                                # Cleanup both extracted and decompressed files
                                for path in [extracted_gz_path, decompressed_grib2_path]:
                                    if path and os.path.exists(path):
                                        try: os.remove(path)
                                        except: pass
                                    
                except zipfile.BadZipFile:
                    print(f"  [!] Corrupted ZIP file")

        except Exception as e:
            print(f"  [!] Critical Error: {e}")

        current_hour_check += datetime.timedelta(hours=1)

    # 4. Save frames (individual + combined) and Generate Preview
    if frames:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract just the arrays
        frame_arrays = [f[0] for f in frames]
        dataset = np.array(frame_arrays, dtype=np.float32)
        
        # Save combined stacked array for training (main output)
        np.save(OUTPUT_FILE, dataset)
        print(f"\n✓ Saved combined dataset to {OUTPUT_FILE}")
        print(f"  Shape: {dataset.shape}")
        print(f"  Value range: {dataset.min():.2f} - {dataset.max():.2f} mm/hr")
        
        # Also save individual frames (for incremental downloads/debugging)
        for i, (frame_data, frame_time) in enumerate(frames):
            frame_path = OUTPUT_FILE.parent / f"frame_{i+1:03d}.npy"
            np.save(frame_path, frame_data.astype(np.float32))
        
        print(f"  Also saved {len(frames)} individual frame files")
        
        # Generate Preview Image
        preview_path = OUTPUT_FILE.parent / "preview.png"
        n_preview = min(6, len(frames))
        
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        
        vmax = np.percentile(dataset, 99)
        
        for i in range(6):
            if i < n_preview:
                im = axes[i].imshow(dataset[i], cmap='Blues', vmin=0, vmax=vmax)
                frame_time = frames[i][1]  # Use actual timestamp
                axes[i].set_title(f"Frame {i+1}: {frame_time:%H:%M}", fontsize=10)
            else:
                axes[i].axis('off')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        # Add colorbar on the right side 
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Precipitation Rate (mm/hr)')
        
        fig.suptitle(f'MRMS Precipitation Preview ({START_TIME:%Y-%m-%d})', fontsize=14, fontweight='bold')
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Preview: {preview_path}")
    else:
        print("\nNo data found.")

if __name__ == "__main__":
    get_mrms_10min()