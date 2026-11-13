"""Download the raw Telco churn CSV. Run once, then track the file with `dvc add`."""
from __future__ import annotations

import urllib.request

from src.common import ROOT, file_md5, get_logger, load_params

log = get_logger("ingest")


def main() -> None:
    params = load_params()
    dest = ROOT / params["data"]["raw_path"]
    if dest.exists():
        log.info("Raw data already present at %s (md5 %s)", dest, file_md5(dest))
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", params["data"]["source_url"])
    urllib.request.urlretrieve(params["data"]["source_url"], dest)
    log.info("Saved %s (md5 %s)", dest, file_md5(dest))


if __name__ == "__main__":
    main()
