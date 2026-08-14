from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(destination, "wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination


def extract_archive(archive_path: Path, target_dir: Path) -> Path:
    if not archive_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {archive_path}")

    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = archive_path.suffix.lower()
    if suffix != ".zip":
        raise ValueError(f"Formato de archivo no soportado: {archive_path.suffix}")

    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(target_dir)

    return target_dir


def _find_single_root(path: Path) -> Path:
    entries = [entry for entry in path.iterdir() if entry.is_dir()]
    if len(entries) == 1:
        return entries[0]
    return path


def resolve_dataset_root(
    dataset_dir: Path | None,
    dataset_archive: Path | None,
    download_url: str | None,
    work_dir: Path,
) -> Path:
    if dataset_dir:
        if not dataset_dir.exists():
            raise FileNotFoundError(f"No existe el directorio del dataset: {dataset_dir}")
        return dataset_dir

    if dataset_archive:
        extracted = extract_archive(dataset_archive, work_dir / "extracted")
        return _find_single_root(extracted)

    if download_url:
        parsed = urlparse(download_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("La URL del dataset debe ser http o https")
        archive_name = Path(parsed.path).name or "lsa64.zip"
        downloaded = download_file(download_url, work_dir / "downloads" / archive_name)
        extracted = extract_archive(downloaded, work_dir / "extracted")
        return _find_single_root(extracted)

    raise ValueError("Tenés que indicar --dataset-dir, --dataset-archive o --download-url")

