# Binary Framed TCP Communication Protocol Specification

**Document Version:** 1.0  
**Implementation:** `shared/protocol.py`  
**Protocol Version Field:** `1` (`0x0001`)

---

## 1. Overview & Framing Design

The communication between the Sender node (Ubuntu) and the Receiver node (Windows 11) utilizes a standardized binary length-prefixed framing format over a single persistent or reconnectable TCP stream.

To prevent buffer truncation, byte corruption, or framing misalignment over fragmented TCP streams:
1. All integer header fields are encoded in **network byte order (Big-Endian)**.
2. Every socket read loop employs strict exact-length accumulation (`recv_exact`) until the full frame payload is buffered.
3. Every payload is paired with a SHA-256 digest calculated strictly from raw compressed bytes.
4. Transmission success is strictly contingent upon receiving a verified ACK packet (`ACK_OK`) from the receiver.

---

## 2. Frame Formats

### 2.1 Image Transmission Frame (Sender $\rightarrow$ Receiver)

The image transmission frame consists of a 14-byte fixed binary header, followed by a UTF-8 JSON metadata payload, followed by raw compressed binary image bytes:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       Magic Bytes (b"NAC1")                   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Protocol Version       |      Metadata Length (N)      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       Metadata Length (N)     |     Image Payload Length (M)  |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                   Image Payload Length (M)                    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                   Metadata JSON Payload (N bytes)             +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                   Raw Compressed Image Data (M bytes)         +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

#### Header Fields:
- **Magic Bytes (4 bytes)**: Fixed ASCII `b"NAC1"` (`0x4E 0x41 0x43 0x31`).
- **Protocol Version (2 bytes, uint16)**: `0x0001`.
- **Metadata Length `N` (4 bytes, uint32)**: Byte count of the serialized UTF-8 JSON metadata block.
- **Image Length `M` (4 bytes, uint32)**: Byte count of the raw binary image payload.
- **Struct Format String**: `!4sHII` (Total size: 14 bytes).

#### Metadata JSON Schema:
```json
{
  "protocol_version": 1,
  "experiment_id": "TRIAL_1793733748",
  "image_id": "benign (1)",
  "compression_quality": 70,
  "image_width": 256,
  "image_height": 256,
  "image_format": "JPEG",
  "original_file_size": 100000,
  "compressed_file_size": 30438,
  "image_byte_length": 30438,
  "sha256_checksum": "e8045fdacc8dc53fd7d04c79198e107374466feb45202cd51b01948949180609"
}
```

---

### 2.2 Acknowledgment Frame (Receiver $\rightarrow$ Sender)

The acknowledgment frame is returned by the receiver immediately after receiving the image payload and completing SHA-256 integrity verification:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       Magic Bytes (b"NACK")                   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Protocol Version       |        ACK JSON Length (K)    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       ACK JSON Length (K)     |                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+                               +
|                                                               |
+                   Acknowledgment JSON Payload (K bytes)       +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

#### Header Fields:
- **Magic Bytes (4 bytes)**: Fixed ASCII `b"NACK"` (`0x4E 0x41 0x43 0x4B`).
- **Protocol Version (2 bytes, uint16)**: `0x0001`.
- **ACK JSON Length `K` (4 bytes, uint32)**: Byte count of serialized ACK metadata.
- **Struct Format String**: `!4sHI` (Total size: 10 bytes).

#### Status Codes:
| Status Code | Constant | Meaning |
| :--- | :--- | :--- |
| `200` | `ACK_OK` | Frame received, length matched, SHA-256 verified, saved to disk |
| `400` | `ACK_CHECKSUM_MISMATCH` | Image length matched, but computed SHA-256 does not match header |
| `401` | `ACK_CORRUPTED_FRAME` | Invalid magic bytes, JSON parse error, or unsupported protocol version |
| `500` | `ACK_SERVER_ERROR` | Local disk write failure or unexpected server exception |

#### ACK JSON Schema:
```json
{
  "protocol_version": 1,
  "experiment_id": "TRIAL_1793733748",
  "image_id": "benign (1)",
  "status_code": 200,
  "checksum_verified": true,
  "message": "Image received and verified successfully",
  "timestamp": "2026-11-04 12:00:00"
}
```

---

## 3. Streaming & Error Handling Rules

1. **Chunking**: Payloads are streamed in `65,536` byte (64 KB) chunks to prevent socket buffer exhaustion and maximize throughput.
2. **Partial Reads**: Any call to `sock.recv()` may return fewer bytes than requested. The receiver buffers chunks until exactly `N` or `M` bytes are collected.
3. **Timeouts**: Socket operations default to a 15.0-second timeout. If a peer stalls or disconnects midway, the socket is cleanly torn down.
4. **Zero Dummy-Data Rule**: Sender will never claim success unless `status_code == 200` and `checksum_verified == True`.
