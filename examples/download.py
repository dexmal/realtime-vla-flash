import fsspec
import time
import concurrent.futures
import pathlib
import tqdm

url = "gs://openpi-assets/checkpoints/pi0_droid"
save_path = "/root/code"

local_path = pathlib.Path(save_path)

fs, _ = fsspec.core.url_to_fs(url)
info = fs.info(url)
# Folders are represented by 0-byte objects with a trailing forward slash.
if is_dir := (info["type"] == "directory" or (info["size"] == 0 and info["name"].endswith("/"))):
    total_size = fs.du(url)
else:
    total_size = info["size"]

with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fs.get, url, local_path, recursive=is_dir)
    while not future.done():
        current_size = sum(f.stat().st_size for f in [*local_path.rglob("*"), local_path] if f.is_file())
        pbar.update(current_size - pbar.n)
        time.sleep(1)
    pbar.update(total_size - pbar.n)