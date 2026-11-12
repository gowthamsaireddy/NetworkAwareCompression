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
| - Automated Batch Runner & Telemetry     |                                 | - Automatic Pipeline & Analytics Export  |
+------------------------------------------+                                 +------------------------------------------+
```

## 2. Shared Protocol Framing & Safety Bounds
Every message sent across the TCP wire uses deterministic binary length-prefixed framing defined in `shared/protocol.py`:

### Image Transmission Frame (`NAC1`)
```
[4 bytes: MAGIC_BYTES b"NAC1"]
[2 bytes: Protocol version (uint16)]
[4 bytes: Metadata JSON byte length N (uint32)]  <-- Capped at MAX_METADATA_SIZE (64 KB)
[4 bytes: Image payload byte length M (uint32)]   <-- Capped at MAX_IMAGE_PAYLOAD_SIZE (50 MB)
[N bytes: UTF-8 Metadata JSON]
[M bytes: Raw compressed JPEG binary payload]
```

### Acknowledgment Frame (`NACK`)
```
[4 bytes: ACK_MAGIC_BYTES b"NACK"]
[2 bytes: Protocol version (uint16)]
[4 bytes: ACK JSON byte length K (uint32)]       <-- Capped at MAX_ACK_SIZE (16 KB)
[K bytes: UTF-8 Acknowledgment JSON]
```

### Denial-of-Service (DoS) and Traversal Defenses
- Headers announcing metadata $> 64\text{ KB}$ or payloads $> 50\text{ MB}$ are immediately rejected with a `ValueError` prior to allocation.
- Incoming `image_id` and `experiment_id` undergo regex sanitization and directory path normalization to strictly eliminate directory traversal (`..`, slashes, backslashes).
- Receiver inspects free disk space prior to saving files, verifying availability exceeds payload size $+ 50\text{ MB}$ safety margin.

## 3. Receiver Automatic Processing Pipeline
When configured with `auto_process_enabled = True`:
```
[TCP Arrival] 
     │
     ▼
[Header & Magic Check] ──(fail)──► [Send Error ACK (code 2)]
     │
     ▼
[SHA-256 Digest Validation] ──(mismatch)──► [Send Checksum Error ACK (code 1)]
     │
     ▼
[Save to data/received/]
     │
     ▼
[Frozen U-Net Inference (256x256)]
     │
     ▼
[Ground-Truth Lookup & Alignment]
     │
     ▼
[Compute Dice & IoU] 
     │
     ▼
[Persist to SQLite & CSV Logs]
     │
     ▼
[Send Verified ACK (code 0)] & [Update Receiver UI Dashboard]
```

## 4. Strict Research Integrity Guidelines
- **Zero-Dummy-Data Policy**: No synthetic metrics, fake image counts, or fabricated Dice/IoU scores are displayed. Uncomputed fields display explicitly as "No data available" or "Not calculated".
- **Frozen Deep Learning Evaluation**: The U-Net model loaded on the Receiver is strictly in inference mode (`torch.no_grad()`). No training or weight updates occur during testing.
- **Traceability**: All experimental parameters (channel bandwidth, delay, packet loss, JPEG quality, transmission duration, throughput, and segmentation Dice/IoU) are logged to reproducible SQLite and CSV logs.
- **Scientific Rationale on Rate Adaptation**: The adaptive quality controller's bandwidth thresholds are explicit heuristics for testing and are benchmarked against fixed $Q$ baselines across varying network profiles.
