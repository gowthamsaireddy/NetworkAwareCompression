# System Architecture & Technical Specifications

## 1. Overview
The **Network-Aware Adaptive Compression for Breast Ultrasound Image Transmission and Deep Learning-Based Lesion Segmentation** research framework connects two physically distinct machines over a direct Ethernet link:

```
+------------------------------------------+        Physical Ethernet        +------------------------------------------+
|          LAPTOP 1 (Ubuntu VM)            | <-----------------------------> |          LAPTOP 2 (Windows 11)           |
|                                          |                                 |                                          |
| - VirtualBox Bridged Network Adapter     |                                 | - Windows 11 Native Host                 |
| - Dataset Manager (BUSI Dataset)         |                                 | - TCP Receiver Server (Daemon)           |
| - JPEG Compression Engine (Q30-Q90)      |                                 | - SHA-256 Integrity Verifier             |
| - Linux tc / netem Traffic Shaping       |                                 | - Frozen U-Net Inference Engine          |
| - Length-prefixed TCP Client             |                                 | - Ground-truth Matching & Evaluation     |
| - Provisional Adaptive Controller        |                                 | - Dice & IoU Metric Logger               |
+------------------------------------------+                                 +------------------------------------------+
```

## 2. Shared Protocol Framing
Every TCP message sent over the wire uses binary length-prefixed framing defined in `shared/protocol.py`:
- 4 bytes magic prefix (`b"NAC1"`)
- 2 bytes protocol version (uint16)
- 4 bytes metadata length `N` (uint32)
- `N` bytes UTF-8 JSON metadata
- 4 bytes image payload length `M` (uint32)
- `M` bytes raw compressed image payload

## 3. Strict Research Integrity Guidelines
- **Zero-Dummy-Data Policy**: No synthetic metrics, fake image counts, or fabricated Dice/IoU scores are displayed. Uncomputed fields display explicitly as "No data available" or "Not calculated".
- **Frozen Deep Learning Evaluation**: The U-Net model loaded on the Receiver is strictly in inference mode (`torch.no_grad()`). No training or weight updates occur during testing.
- **Traceability**: All experimental parameters (channel bandwidth, delay, packet loss, JPEG quality, transmission duration, throughput, and segmentation Dice/IoU) are logged to reproducible SQLite and CSV logs.

