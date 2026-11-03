# Dataset Directory

This directory is designated for the Breast Ultrasound Image (BUSI) dataset and local runtime caches.

## Policy & Notice
- **Do not commit raw or processed image files** to Git.
- Expected dataset structure:
  - `data/BUSI/benign/`
  - `data/BUSI/malignant/`
  - `data/BUSI/normal/`
  Each directory contains original ultrasound images along with corresponding ground-truth masks (e.g. `benign (1).png` and `benign (1)_mask.png`).
- Ground-truth masks must NEVER be JPEG compressed.
- Download the official BUSI dataset locally and link or configure its root path in the Sender GUI.

