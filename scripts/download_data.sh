#!/usr/bin/env bash
#
# Fetch the SAR dataset: Zenodo record 13761290, Part III only (CLAUDE.md §0).
#
#   02_Test_images_and_ground_truth.7z   ~9.9 GB
#   three class folders - oil, look-alike, oil-free - with ground-truth masks
#   georeferenced VV+VH Sigma0 in dB, 2048 x 2048 x 2
#
# Part I and Part II are not used in this project and are not fetched here.
#
# Usage: scripts/download_data.sh [target_dir]   (default data/raw)

set -euo pipefail

ZENODO_RECORD="13761290"
TARGET_FILE="02_Test_images_and_ground_truth.7z"
TARGET_DIR="${1:-data/raw}"
EXTRACT_DIR="${TARGET_DIR}/part3"

log() { printf '[download_data] %s\n' "$*"; }
die() { printf '[download_data] ERROR: %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || die "curl not found."

if command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  die "python not found."
fi

# 7-Zip is not installed on every dev machine; py7zr is pinned in
# backend/requirements.txt precisely so extraction never blocks on a system
# package.
if command -v 7z >/dev/null 2>&1; then
  EXTRACTOR="7z"
elif "${PY}" -c "import py7zr" >/dev/null 2>&1; then
  EXTRACTOR="py7zr"
else
  die "No 7z extractor. Install 7-Zip, or: pip install py7zr==0.22.0"
fi

mkdir -p "${TARGET_DIR}"
ARCHIVE="${TARGET_DIR}/${TARGET_FILE}"

log "Resolving ${TARGET_FILE} from Zenodo record ${ZENODO_RECORD}"

META_JSON="$(curl -fsSL "https://zenodo.org/api/records/${ZENODO_RECORD}")" \
  || die "Could not reach the Zenodo API."

FILE_INFO="$(
  printf '%s' "${META_JSON}" | TARGET_FILE="${TARGET_FILE}" "${PY}" -c '
import json
import os
import sys

record = json.load(sys.stdin)
target = os.environ["TARGET_FILE"]
for entry in record.get("files", []):
    if entry.get("key") == target:
        print(entry["links"]["self"], entry.get("checksum", ""), entry.get("size", 0))
        break
else:
    sys.exit(1)
'
)" || die "${TARGET_FILE} not present in record ${ZENODO_RECORD}."

read -r FILE_URL FILE_CHECKSUM FILE_SIZE <<< "${FILE_INFO}"

log "URL      ${FILE_URL}"
log "Size     ${FILE_SIZE} bytes"
log "Checksum ${FILE_CHECKSUM}"

log "Downloading (resumable; safe to re-run)"
curl -fL --retry 5 --retry-delay 5 -C - -o "${ARCHIVE}" "${FILE_URL}"

EXPECTED_MD5="${FILE_CHECKSUM#md5:}"
if [ -n "${EXPECTED_MD5}" ] && command -v md5sum >/dev/null 2>&1; then
  log "Verifying MD5"
  ACTUAL_MD5="$(md5sum "${ARCHIVE}" | cut -d' ' -f1)"
  [ "${ACTUAL_MD5}" = "${EXPECTED_MD5}" ] \
    || die "Checksum mismatch: expected ${EXPECTED_MD5}, got ${ACTUAL_MD5}. Delete ${ARCHIVE} and re-run."
  log "MD5 OK"
fi

log "Extracting to ${EXTRACT_DIR}"
mkdir -p "${EXTRACT_DIR}"
if [ "${EXTRACTOR}" = "7z" ]; then
  7z x -y -o"${EXTRACT_DIR}" "${ARCHIVE}" >/dev/null
else
  "${PY}" -m py7zr x "${ARCHIVE}" "${EXTRACT_DIR}"
fi

log "Done. Part III is at ${EXTRACT_DIR}"
log "Split is enforced in scripts/data_split.py - indices 146-150 are demo holdout."
