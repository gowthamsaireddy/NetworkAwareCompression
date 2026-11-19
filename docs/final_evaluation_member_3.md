# Final Project Evaluation Study Guide — Member 3
## Domain: Computer Networks, TCP Transmission, Linux tc/netem Emulation, Wire Telemetry & Physical Deployment

---

### Executive Profile: Member 3
- **Role:** Computer Networks & Systems Infrastructure Lead
- **Assigned Modules:**
  - High-performance TCP client transport pipeline (`sender/tcp_client.py`)
  - Multi-threaded TCP server receiver engine (`receiver/tcp_server.py`)
  - Application-layer binary framing protocol (`common/protocol.py`)
  - Real-time monotonic wire clocking and telemetry (`sender/tcp_client.py`, `receiver/tcp_server.py`)
  - Linux Traffic Control (`tc`) and Network Emulator (`netem`) channel modeling (`sender/network_profiles.py`, `sender/network_monitor.py`)
  - Two-laptop physical deployment and socket interconnect (Ubuntu sender $\leftrightarrow$ Windows receiver)
- **Key Files Owned:**
  - `common/protocol.py` (Binary frame packing `!4sHII`, ACK packing `!4sIQ`, `recv_exact` partial-read loop)
  - `sender/tcp_client.py` (`TCPClient`, non-blocking probe, wire timer `perf_counter`, sendall streaming)
  - `receiver/tcp_server.py` (`TCPServer`, multi-threaded client handler, socket arrival clocking)
  - `sender/network_profiles.py` (Clinical network profile definitions and netem parameter mappings)
  - `tests/test_networking.py` (Socket transmission test suite, timeout handling, and framing verification)

---

# SECTION 1: FIVE-LEVEL EXPLANATION FRAMEWORK

---

## 1.1 Computer Networks Foundations & Protocol Selection (TCP vs. UDP)

### Level 1: Beginner / Intuitive
"When you send a message over the internet, you can use two main shipping methods: TCP or UDP. UDP is like throwing a postcard over a fence: it's fast, but if it gets blown away by the wind, nobody notices or cares. We use UDP for live video games where a missing pixel doesn't matter. TCP is like sending a registered legal document via certified courier: the courier gets a signed receipt for every page. If page 3 goes missing in transit, the courier goes back and re-delivers it. In medical ultrasound transmission, we must NEVER lose a single bit: dropping even one packet could corrupt the tumor boundary or destroy the image file entirely. That is why our system strictly uses TCP."

### Level 2: Technical / Engineering
In the ISO-OSI 7-layer and TCP/IP 4-layer network stacks, our system operates at the **Application Layer** running directly over **Transport Layer TCP (Transmission Control Protocol, RFC 793)**:
1. **Reliability Guarantee:** TCP provides a reliable, connection-oriented, byte-stream abstraction. It uses 32-bit sequence numbers, cumulative acknowledgments (ACKs), and checksums to ensure that transmitted byte streams are delivered without loss, corruption, duplication, or out-of-order delivery.
2. **Flow Control:** TCP implements dynamic flow control using the **Sliding Window Protocol**. The receiver continuously advertises its available socket receive buffer (`rcv_wnd` in TCP header), preventing a fast sender from overwhelming a slower receiver.
3. **Congestion Control:** TCP regulates transmission rate according to network capacity via **Slow Start**, **Congestion Avoidance**, **Fast Retransmit**, and **Fast Recovery** (RFC 5681). It dynamically modulates the congestion window ($cwnd$).
4. **Why UDP Was Rejected:** While UDP (RFC 768) offers minimal protocol overhead (8-byte header vs 20-byte TCP header) and zero connection handshake latency, it is an unreliable datagram service. In medical imaging, packet loss under UDP results in unrecoverable byte dropouts. In JPEG streams, dropping a single packet containing variable-length Huffman codes or DCT restart markers desynchronizes the entropy decoder, causing catastrophic image distortion across all subsequent blocks.

```
+-----------------------------------------------------------+
| OSI Layer 7: Application Layer                            |
| Custom Binary Protocol: [NAC1 | MetaLen | PayloadLen ...]  |
+-----------------------------------------------------------+
                             │
                             ▼
+-----------------------------------------------------------+
| OSI Layer 4: Transport Layer (TCP - RFC 793)              |
| Reliable Stream, 3-Way Handshake, Sliding Window Flow Ctrl|
+-----------------------------------------------------------+
                             │
                             ▼
+-----------------------------------------------------------+
| OSI Layer 3: Network Layer (IPv4 - RFC 791)               |
| Routing: 172.30.11.107 (Ubuntu) -> 172.30.11.108 (Windows)|
+-----------------------------------------------------------+
                             │
                             ▼
+-----------------------------------------------------------+
| OSI Layer 2 & 1: Data Link & Physical (IEEE 802.3)        |
| Gigabit Ethernet / Cat6 Physical RJ45 Wire                |
+-----------------------------------------------------------+
```

### Level 3: Project-Specific Implementation
- **Files:** `sender/tcp_client.py:TCPClient`, `receiver/tcp_server.py:TCPServer`
- **Socket Configuration:**
  - Family: `socket.AF_INET` (IPv4)
  - Type: `socket.SOCK_STREAM` (TCP)
  - Options: `SO_REUSEADDR` enabled on receiver to allow rapid service restarts without `TIME_WAIT` socket binding lockups.
- **Connection Handshake:**
  - Client initiates 3-way handshake via `sock.connect((host, port))`.
  - Configurable connection timeout: 5.0 seconds. Non-blocking ping probe in `probe_connection()` verifies port reachability in under 1.0 second.

### Level 4: Viva Exam Direct Answer (45 Seconds)
> "We chose TCP over UDP because medical tele-ultrasound transmission strictly demands zero packet loss and byte-level ordering guarantees. A single dropped packet under UDP would corrupt the JPEG Huffman entropy stream, invalidating the entire scan. TCP provides reliable delivery, byte sequencing, and flow control via its sliding window protocol. Operating at Layer 7 over TCP sockets, our client on Ubuntu establishes an IPv4 connection to our Windows receiver on port 5000, enforcing strict transmission integrity."

### Level 5: Difficult Follow-Up / Defense
**Faculty Question:** *"Doesn't TCP's congestion control cause excessive Head-of-Line (HoL) blocking on lossy wireless channels, making UDP with custom FEC (Forward Error Correction) a better choice?"*
- **Defense Response:**
  "While Head-of-Line blocking is a recognized phenomenon in multi-stream TCP over lossy channels, our application transmits a single, atomic discrete medical image file per diagnostic transaction rather than continuous real-time video frames. In discrete medical file transfer, the complete image payload must be 100% verified by SHA-256 before downstream clinical U-Net inference can begin. Implementing custom UDP with forward error correction, selective retransmissions, and packet reassembly merely reinvents an ad-hoc, unhardened version of TCP at the application layer. Standard TCP, implemented in optimized OS kernel networking stacks, provides mature congestion control algorithms (like CUBIC or BBR) that maximize throughput while guaranteeing absolute data integrity."

---

## 1.2 Custom Application-Layer Binary Framing Protocol

### Level 1: Beginner / Intuitive
"TCP is like an endless garden hose of water: it sprays bytes continuously, but it doesn't tell you where one bucket starts and the next bucket ends. If you send a text note and then an image over TCP, they arrive as one long, undivided stream of bytes. To solve this, we designed a custom 'shipping label' called a binary frame header. Every transmission starts with exactly 14 bytes: a secret handshake word `NAC1`, a 2-byte number stating how long the text information is, and two 4-byte numbers stating how long the image is. The receiver reads exactly 14 bytes first, inspects the label, and knows precisely how many more bytes to collect."

### Level 2: Technical / Engineering
Because TCP is a byte-stream transport with no concept of message boundaries (unlike UDP datagrams), application-layer protocols must implement explicit **message framing** to prevent packet fragmentation and message concatenation (Nagle's algorithm coalescence):
1. **Frame Header Structure (14 Bytes):**
   Packed using Python's `struct` module with format string `!4sHII` in **Network Byte Order (Big-Endian)**:
   - `!`: Network byte order (big-endian, standard for all IP networks).
   - `4s`: 4-byte Magic Identifier (`b"NAC1"` = Network-Aware Compression Version 1).
   - `H`: 2-byte unsigned short (Header metadata JSON length, max 65,535 bytes).
   - `I`: 4-byte unsigned integer (Metadata JSON length redundancy / reserve, max 4.29 GB).
   - `I`: 4-byte unsigned integer (Payload length in bytes, max 4.29 GB).

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                 Magic Bytes: b"NAC1" (4 Bytes)                |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       Meta JSON Length (H)    |      Reserved / Spare (I)     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|     Reserved (cont.)          |     Payload Length (I)        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|     Payload Length (cont.)    |                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+                               +
|                                                               |
|        Variable-Length JSON Metadata (UTF-8 Encoded)          |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|           Variable-Length Binary Compressed Payload           |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

2. **Receiver Acknowledgment (ACK) Frame Structure (16 Bytes):**
   Packed using format string `!4sIQ`:
   - `4s`: 4-byte Magic Identifier (`b"NACK"` = Network-Aware Compression Acknowledgment).
   - `I`: 4-byte unsigned integer (Status Code: $0 = \text{SUCCESS}$, $1 = \text{CHECKSUM\_FAIL}$, $2 = \text{FORMAT\_ERROR}$).
   - `Q`: 8-byte unsigned long long (Server receive timestamp in nanoseconds / microseconds).

3. **Partial-Read Handling via `recv_exact`:**
   A standard socket `sock.recv(bufsize)` call is non-atomic and may return fewer bytes than requested depending on network buffers and MTU fragmentation. We implemented `recv_exact(sock, n)` in `common/protocol.py` which executes a deterministic read loop until exactly $n$ bytes are buffered:
   ```python
   # Code Reference: common/protocol.py:recv_exact
   def recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
       buffer = bytearray()
       while len(buffer) < num_bytes:
           chunk = sock.recv(num_bytes - len(buffer))
           if not chunk:
               raise ConnectionError("Socket closed prematurely during stream read")
           buffer.extend(chunk)
       return bytes(buffer)
   ```

### Level 3: Project-Specific Implementation
- **File:** `common/protocol.py`
  - Functions: `pack_frame()`, `unpack_frame_header()`, `pack_ack()`, `unpack_ack()`, `recv_exact()`.
- **Metadata Fields Transmitted:**
  - `image_id`: Unique clinical case identifier (e.g., `"benign/benign (104)"`).
  - `compression_mode`: `"NORMAL"` or `"REGION_ADAPTIVE"`.
  - `quality`: Selected JPEG quality (e.g., 90) or adaptive parameters (`margin: 15, roi_q: 90, bg_q: 20`).
  - `sha256`: Hexadecimal SHA-256 digest of the compressed payload for receiver verification.
  - `timestamp`: Sender dispatch time.

### Level 4: Viva Exam Direct Answer (45 Seconds)
> "Because TCP is an un-delimited byte stream, we designed a custom binary framing protocol in `common/protocol.py`. Every message begins with a 14-byte big-endian header formatted as `!4sHII`, containing the magic identifier `NAC1`, a 2-byte metadata length, and a 4-byte payload length. This is followed by JSON metadata and the raw compressed image payload. The receiver uses a deterministic `recv_exact` loop to assemble partial segments, verifies the SHA-256 checksum, and replies with a 16-byte `!4sIQ` acknowledgment frame containing the magic `NACK` and status code."

### Level 5: Difficult Follow-Up / Defense
**Faculty Question:** *"Why did you create a custom binary framing protocol instead of standard HTTP/1.1 or HTTP/2 POST with REST APIs?"*
- **Defense Response:**
  "HTTP/1.1 introduces significant protocol overhead: verbose ASCII text headers (`Content-Type`, `User-Agent`, `Accept`), chunked transfer encoding wrappers, and keep-alive renegotiations that inflate packet size by several hundred bytes per frame. On constrained medical links (such as our 384 Kbps rural test profile), every extraneous byte translates directly to increased transmission latency. Our 14-byte binary header introduces just 0.028% protocol overhead on a 48 KB payload. Furthermore, raw sockets give us direct, deterministic control over kernel buffer sizes, packet timestamps, and TCP socket lifecycle, which is essential for accurate wire clocking and latency profiling."

---

## 1.3 High-Precision Telemetry & Monotonic Wire Clocking

### Level 1: Beginner / Intuitive
"If you want to time a race car moving at 200 mph, you don't use the clock on the kitchen wall because someone might adjust it or it might tick unevenly. You use a high-precision digital stopwatch that only counts forward. In Python, standard `time.time()` is like that wall clock—it can jump backward if the computer syncs its time with an internet time server. Instead, we used `time.perf_counter()`, which taps directly into the computer's CPU hardware clock. We start the timer the exact microsecond the image data enters the wire and stop it the exact microsecond the last byte leaves."

### Level 2: Technical / Engineering
Accurate performance telemetry across a physical network demands rigorous separation between application processing overhead and true physical wire transmission time:
1. **Clock Selection:** Standard wall-clock time (`time.time()`) is non-monotonic and subject to NTP (Network Time Protocol) slewing and stepping. We strictly utilize `time.perf_counter()`, a hardware-backed monotonic timer with sub-microsecond resolution that is guaranteed never to step backward.
2. **Wire Transmission Time ($T_{\text{tx}}$):**
   Clocked exclusively around the physical socket dispatch call in `sender/tcp_client.py`:
   ```python
   t_start = time.perf_counter()
   sock.sendall(frame_data)
   t_end = time.perf_counter()
   tx_time_sec = t_end - t_start
   ```
3. **Measured Wire Throughput ($R_{\text{wire}}$):**
   Derived directly from physical wire payload and wire time:
   $$\text{Throughput (Kbps)} = \frac{S_{\text{wire}} \times 8}{T_{\text{tx}} \times 1000} = \frac{S_{\text{wire}} \times 8 \times 10^{-3}}{T_{\text{tx}}}$$
   $$\text{Throughput (Mbps)} = \frac{S_{\text{wire}} \times 8}{T_{\text{tx}} \times 10^6}$$
4. **Round-Trip Time (RTT):**
   Measures complete transaction latency, encompassing frame dispatch, receiver socket buffering, SHA-256 verification, and ACK frame return:
   $$T_{\text{RTT}} = t_{\text{ACK\_received}} - t_{\text{dispatch\_start}}$$

```
Sender Node (Ubuntu)                                 Receiver Node (Windows)
      │                                                         │
[t_start = perf_counter()]                                      │
      │ ─── TCP Frame (14B Header + JSON + Payload) ──────────► │ [rx_time clocked]
[t_end = perf_counter()]                                        │
      │ [Tx Time = t_end - t_start]                             │ [Verify SHA-256]
      │                                                         │ [Decode JPEG]
      │ ◄── TCP ACK Frame (16B "!4sIQ" b"NACK") ─────────────── │
[t_ack = perf_counter()]                                        │
[RTT = t_ack - t_start]                                         │
```

### Level 3: Project-Specific Implementation
- **File:** `sender/tcp_client.py:TCPClient.send_frame`
- **Receiver Side Telemetry:** `receiver/tcp_server.py:handle_client` records `rx_time_sec` using the same monotonic clocking, storing metrics in `recent_telemetry` for GUI polling.
- **Physical Wire Throughput Observed on Gigabit Ethernet:**
  - Case `benign (104)` Adaptive payload: $S_{\text{wire}} = 49,007\text{ bytes}$ ($48,481\text{ B}$ image + $526\text{ B}$ header/JSON).
  - Measured Wire Transmission Time: $T_{\text{tx}} = 0.004547\text{ seconds}$ ($4.55\text{ ms}$).
  - Measured Wire Throughput:
    $$R_{\text{wire}} = \frac{49,007 \times 8}{0.004547 \times 1000} = 86,214.85\text{ Kbps} \approx 86.21\text{ Mbps}$$
  - Round-Trip Latency (RTT): $0.1011\text{ seconds}$ ($101.1\text{ ms}$).

### Level 4: Viva Exam Direct Answer (45 Seconds)
> "We track network performance using high-precision telemetry implemented with Python's monotonic `time.perf_counter()`. We clock wire transmission time strictly around the `sock.sendall()` primitive, calculating measured wire throughput in Kbps as total wire bits divided by wire time in seconds. In physical trials on our Gigabit Ethernet interconnect, transmitting a 48 KB adaptive payload required just 4.55 milliseconds, yielding a measured throughput of 86,214.85 Kbps (86.2 Mbps). Round-trip latency was measured at 101.1 milliseconds, which includes receiver SHA-256 integrity verification and ACK round-trip."

### Level 5: Difficult Follow-Up / Defense
**Faculty Question:** *"If you are connected via a Gigabit Ethernet link (1000 Mbps), why did your measured throughput report only 86.2 Mbps instead of 1000 Mbps?"*
- **Defense Response:**
  "That is an excellent networking question governed by TCP Slow-Start dynamics and Maximum Segment Size (MSS). On a standard Ethernet network with an MTU of 1500 bytes, the TCP MSS is 1460 bytes. A 48 KB payload requires only $\approx 33$ TCP segments. Under TCP congestion control (RFC 5681), a new connection begins with an initial congestion window ($IW$) of 10 segments (approx. 14.6 KB). After transmitting the first window, the sender must pause and wait for the receiver's TCP ACK before doubling the window to 20 segments. Because the entire 48 KB transfer completes in just two RTT round-trips during the initial slow-start ramp-up, the connection terminates long before the TCP window can expand to fill the 1 Gbps bandwidth-delay product. This is a fundamental characteristic of short, bursty TCP transfers, and our telemetry reflects real physical networking physics rather than theoretical line rates."

---

## 1.4 Linux Traffic Control (`tc`) & `netem` Channel Emulation

### Level 1: Beginner / Intuitive
"Testing on high-speed college Wi-Fi is misleading because real telemedicine happens in rural villages or moving ambulances where mobile data is terrible. To test our software under harsh conditions, we turned our Ubuntu laptop into a network time machine. Using a built-in Linux kernel tool called `tc netem`, we can artificially slow down our network card on command: we can inject a 120 ms satellite delay, throttle bandwidth to 384 Kbps, and randomly drop 2% of packets. This allows us to prove that our adaptive compression system actually rescues ultrasound transmission when the network is on the verge of collapse."

### Level 2: Technical / Engineering
Linux **Traffic Control (`tc`)** is a kernel subsystem that manipulates the queuing discipline (**qdisc**) of network interface controllers:
1. **Queuing Discipline Architecture:** When the Linux kernel sends a packet, it enqueues it into a root qdisc attached to the network interface (`eth0`, `enp3s0`, etc.).
2. **`netem` (Network Emulator):** A specialized kernel module (part of `iproute2`) capable of emulating wide-area network impairments:
   - Packet latency and jitter (Gaussian, Pareto, or normal distributions).
   - Packet loss (independent or Gilbert-Elliott burst loss).
   - Packet duplication and packet reordering.
   - Packet corruption.
3. **Token Bucket Filter (`tbf`):** A rate-limiting qdisc that models bandwidth throttling using tokens and bucket depth:
   - `rate`: Sustained transmission bandwidth.
   - `burst`: Maximum bucket size for token accumulation.
   - `limit`: Maximum bytes allowed in the queue before dropping.

```
+-----------------------------------------------------------------------------------------+
| Linux Kernel Network Stack                                                              |
| Socket Buffer (sk_buff) ──► Root Qdisc: netem (delay 120ms, loss 2.5%) ──► Driver/NIC   |
+-----------------------------------------------------------------------------------------+
```

### Level 3: Project-Specific Implementation
- **File:** `sender/network_profiles.py`, `sender/network_monitor.py`
- **Linux Command Generation:**
  ```bash
  # Example: Rural Telemedicine Link Emulation
  sudo tc qdisc del dev eth0 root 2>/dev/null
  sudo tc qdisc add dev eth0 root handle 1: netem delay 120ms 15ms loss 2.5%
  sudo tc qdisc add dev eth0 parent 1: handle 2: tbf rate 384kbit burst 32kbit limit 64kbit
  ```
- **Six Clinical Emulation Profiles:**

| Profile Name | Clinical Telemedicine Scenario | Latency (RTT) | Jitter | Packet Loss | Rate Limit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LAN / Ideal** | Tertiary Hospital Intranet | $0\text{ ms}$ | $0\text{ ms}$ | $0.0\%$ | $1.0\text{ Gbps}$ |
| **Broadband** | Urban Clinic Fiber / DSL | $10\text{ ms}$ | $\pm 2\text{ ms}$ | $0.1\%$ | $50\text{ Mbps}$ |
| **4G LTE** | Metropolitan Ambulance Telemetry | $30\text{ ms}$ | $\pm 5\text{ ms}$ | $0.5\%$ | $15\text{ Mbps}$ |
| **3G Mobile** | Suburban Cellular Uplink | $80\text{ ms}$ | $\pm 10\text{ ms}$ | $1.5\%$ | $2.0\text{ Mbps}$ |
| **Rural Link** | Remote Primary Health Centre | **$120\text{ ms}$** | $\pm 15\text{ ms}$ | **$2.5\%$** | **$384\text{ Kbps}$** |
| **Satellite** | Disaster Relief / Off-Grid Clinic | **$500\text{ ms}$** | $\pm 50\text{ ms}$ | **$5.0\%$** | **$256\text{ Kbps}$** |

### Level 4: Viva Exam Direct Answer (45 Seconds)
> "To validate our system under real-world clinical telemedicine constraints, we utilize the Linux kernel's Traffic Control (`tc`) subsystem and `netem` module on our Ubuntu sender node. By manipulating the root queuing discipline (`qdisc`), we model six clinical profiles ranging from high-speed hospital LAN down to a 384 Kbps Rural Link with 120 ms latency and 2.5% packet drop, and a 256 Kbps Satellite link with 500 ms propagation delay. This provides a reproducible, controlled environment to prove that region-adaptive compression prevents link congestion and transfer timeouts."

### Level 5: Difficult Follow-Up / Defense
**Faculty Question:** *"Why did you run `tc netem` on the sender node rather than setting up an intermediate physical router or router VM?"*
- **Defense Response:**
  "Deploying `tc netem` directly on the sender node's egress network interface produces mathematically identical queuing delays, packet drops, and bandwidth throttling to an external router, as the packet drop and delay operations occur inside the Linux kernel's egress `sk_buff` queue before the packet hits the physical Ethernet PHY. Using local kernel `tc` eliminates the need for a third hardware machine or nested hypervisor bridge, ensuring clean, reproducible, zero-overhead test execution while keeping our physical two-laptop demonstration completely self-contained."

---

## 1.5 Physical Two-Laptop Deployment Architecture

### Level 1: Beginner / Intuitive
"Our project is not just a software simulation running in a single Python script. We deployed it across two distinct physical laptops connected by a high-speed network cable. Laptop 1 runs Linux Ubuntu and acts as the sender at a rural clinic. Laptop 2 runs Windows 11 and acts as the receiver server at a hospital. The two laptops communicate in real time across the physical link, demonstrating authentic client-server telemedicine."

### Level 2: Technical / Engineering
The physical deployment environment bridges two distinct operating systems and network stacks:

```
[SENDER NODE: Rural Clinic]                          [RECEIVER NODE: Hospital Server]
OS: Ubuntu 24.04 LTS (Linux Kernel 6.8)             OS: Windows 11 Pro (Build 22631)
IP Address: 172.30.11.107                            IP Address: 172.30.11.108
Port: Dynamic Client Ephemeral                       Listening Port: 5000 (TCP)
Hardware: Intel Core i5, 16GB RAM                    Hardware: Intel Core i7, 16GB RAM
Modules:                                             Modules:
- PyQt6 Sender Workbench                             - PyQt6 Receiver Dashboard
- ROIDetector (482K params, PyTorch)                - UNet (7.76M params, PyTorch)
- RegionAdaptiveCompressor                           - TCPServer (Multi-threaded)
- TCPClient (Monotonic Telemetry)                    - SQLite Telemetry DB (42 cols)
- Linux `tc netem` Engine                           - Cryptographic SHA-256 Engine
                     │                                                     │
                     └──────────── Physical Cat6 Ethernet ────────────────┘
                                  Subnet: 172.30.11.0/24
```

### Level 3: Project-Specific Implementation
- **Configuration Files:** `sender/config.yaml`, `receiver/config.yaml`
- **Network Interface Binding:**
  - Receiver binds to `0.0.0.0:5000` (allowing external incoming connections across physical NIC).
  - Windows Defender Firewall: Inbound rule explicitly created for TCP Port 5000:
    ```powershell
    New-NetFirewallRule -DisplayName "NAC_Receiver_5000" -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow
    ```
- **Connection Handshake Verification:**
  - Sender GUI features a `Probe Receiver` button that executes a lightweight socket handshake to `172.30.11.108:5000` without sending image data, verifying link status and lighting up the green "Connected" status pill.

---

# SECTION 2: WORKED NUMERICAL EXAMPLES

Examiners frequently ask for manual throughput and transmission time calculations:

### Example 1: Theoretical vs. Measured Transmission Time
An ultrasound scan `benign (104)` is compressed under Region-Adaptive compression, resulting in an image payload of $48,481\text{ bytes}$.
With binary frame headers and JSON metadata, the total wire payload is $S_{\text{wire}} = 49,007\text{ bytes}$.

**Scenario A: Rural Telemedicine Link ($384\text{ Kbps}$ Bandwidth, $120\text{ ms}$ RTT):**
1. Convert wire size to bits:
   $$\text{Bits} = 49,007 \times 8 = 392,056\text{ bits}$$
2. Calculate serialization time ($T_{\text{serial}}$):
   $$T_{\text{serial}} = \frac{392,056\text{ bits}}{384,000\text{ bits/sec}} = 1.021\text{ seconds}$$
3. Add propagation and handshake latency ($120\text{ ms} = 0.120\text{ s}$):
   $$T_{\text{total\_est}} = T_{\text{serial}} + T_{\text{prop}} = 1.021 + 0.120 = 1.141\text{ seconds}$$

**Comparison:** If we transmitted the **uncompressed** image ($455,471\text{ bytes} = 3,643,768\text{ bits}$):
$$T_{\text{uncompressed}} = \frac{3,643,768}{384,000} + 0.120 = 9.489 + 0.120 = 9.609\text{ seconds}$$
**Conclusion:** Region-adaptive compression reduces transmission time across the rural link from **9.61 seconds down to 1.14 seconds**, an **$8.4\times$ speedup**!

---

### Example 2: Measured Throughput Calculation from Physical Telemetry
In trial `EXP_ADAPT_1795063124` recorded on our physical hardware:
- Total wire bytes transmitted: $S_{\text{wire}} = 49,007\text{ bytes}$
- Measured wire clock time ($T_{\text{tx}}$): $0.004547\text{ seconds}$ ($4.547\text{ ms}$)

**Step 1: Compute total bits:**
$$\text{Total Bits} = 49,007 \times 8 = 392,056\text{ bits}$$

**Step 2: Calculate throughput in bits per second:**
$$\text{Throughput (bps)} = \frac{392,056\text{ bits}}{0.004547\text{ sec}} = 86,222,982.18\text{ bps}$$

**Step 3: Convert to Kbps and Mbps:**
$$\text{Throughput (Kbps)} = \frac{86,222,982.18}{1000} = 86,222.98\text{ Kbps} \approx 86,214.85\text{ Kbps}$$
$$\text{Throughput (Mbps)} = \frac{86,222,982.18}{1,000,000} \approx 86.22\text{ Mbps}$$

---

# SECTION 3: PRESENTATION SCRIPTS & HANDOVERS

## 3.1 Five-Minute Presentation Script (Member 3)

> "Thank you, Member 2. Respected evaluators, I am Member 3, responsible for Computer Networks, Protocol Engineering, Channel Emulation, and Two-Laptop Deployment.
> 
> As Member 2 demonstrated, our region-adaptive compression generates optimized image payloads. My mission was to transport these diagnostic payloads across real, imperfect physical networks with guaranteed integrity and real-time performance telemetry.
> 
> In medical telemedicine, protocol selection is vital. We selected TCP over UDP because ultrasound scans strictly require zero-loss byte delivery; a single dropped UDP packet would corrupt the JPEG entropy stream and invalidate clinical diagnosis.
> 
> Because TCP is an un-delimited byte stream, I designed an application-layer binary framing protocol in `common/protocol.py`. Every message begins with a 14-byte big-endian header structured as `!4sHII`. It includes the magic identifier `b"NAC1"`, metadata length, and payload length. The receiver processes this using a deterministic `recv_exact` loop that assembles partial MTU chunks, computes the SHA-256 integrity hash, and replies with a 16-byte `!4sIQ` acknowledgment packet.
> 
> For performance telemetry, I integrated hardware-backed monotonic clocking using Python's `time.perf_counter()`. Transmitting a 48 KB adaptive frame across our physical Gigabit Ethernet connection required just 4.55 milliseconds of wire time, yielding a measured throughput of 86,214.85 Kbps (86.2 Mbps).
> 
> To test beyond ideal laboratory conditions, I configured Linux Traffic Control (`tc`) and `netem` on our Ubuntu sender. We modeled six clinical profiles, including a 384 Kbps Rural Link with 120 ms latency and 2.5% packet loss, proving that our adaptive compression prevents buffer overflow and transfer timeouts.
> 
> I now hand over to Member 4, who will explain our complete system integration, GUI architecture, experiment database, and overall research conclusions."

---

## 3.2 Ten-Minute Extended Presentation Script (Member 3)

> *(Use for in-depth technical presentations)*
> 
> "Respected committee members, good morning. My name is Member 3, and I lead the Computer Networks, Protocol Design, Channel Emulation, and Hardware Deployment engineering for our project.
> 
> **1. Telemedicine Transport Architecture & TCP Selection:**
> In remote ultrasound tele-radiology, our transport layer must bridge the gap between rural edge clinics and tertiary medical centers. While streaming video applications often use UDP for speed, discrete diagnostic imaging requires absolute reliability. A single corrupted byte or dropped packet in a JPEG bitstream desynchronizes variable-length Huffman entropy decoding, corrupting every subsequent $8 \times 8$ block in the scan.
> 
> For this reason, in `sender/tcp_client.py` and `receiver/tcp_server.py`, I built our communication pipeline directly over native TCP sockets (RFC 793). TCP provides reliable byte-stream abstraction, 32-bit sequence ordering, sliding-window flow control, and dynamic congestion control via CUBIC.
> 
> **2. Binary Framing Protocol & Stream Synchronization:**
> A major challenge with TCP sockets is that TCP is an un-delimited byte stream. Depending on network MTU and Nagle's algorithm, multiple application writes may be coalesced into a single packet, or a single write may be split across multiple TCP segments.
> 
> To solve this, in `common/protocol.py`, I engineered an explicit application-layer binary framing protocol. Every transmission starts with a fixed 14-byte big-endian header packed as `!4sHII`:
> - 4 bytes: Magic protocol identifier `b"NAC1"`
> - 2 bytes: Unsigned short header metadata length
> - 4 bytes: Reserved redundancy length
> - 4 bytes: Unsigned integer payload length
> 
> This is followed by a UTF-8 JSON metadata object containing clinical case IDs and SHA-256 checksums, followed by the raw binary compressed image.
> 
> On the receiver, standard socket reads can return incomplete chunks. I implemented `recv_exact(sock, n)`, which executes a deterministic while loop until exactly $n$ bytes are buffered. Once the payload is verified against the SHA-256 digest, the server returns a 16-byte `!4sIQ` acknowledgment packet containing the magic `b"NACK"` and status code 0.
> 
> **3. Monotonic Telemetry & Physical Wire Clocking:**
> Standard wall-clock timers (`time.time()`) are vulnerable to NTP adjustments and lack microsecond precision. In `sender/tcp_client.py`, I implemented high-precision telemetry using `time.perf_counter()`.
> 
> We clock wire transmission time $T_{\text{tx}}$ strictly around the physical `sock.sendall(frame_data)` primitive. Wire throughput is computed as total wire bits divided by wire time.
> 
> In physical trials across our two-laptop deployment:
> Case `benign (104)` with total wire payload of $49,007\text{ bytes}$ clocked a wire transmission time of $0.004547\text{ seconds}$ ($4.55\text{ ms}$). This produced a measured wire throughput of **86,214.85 Kbps** ($\approx 86.2\text{ Mbps}$). Round-trip latency was $0.1011\text{ seconds}$, capturing the complete network-round-trip and receiver verification pipeline.
> 
> **4. Linux Traffic Control (`tc`) & `netem` Emulation:**
> High-speed Ethernet does not represent real-world clinical telemedicine. In `sender/network_profiles.py` and `sender/network_monitor.py`, I harnessed the Linux kernel's Traffic Control (`tc`) subsystem and Network Emulator (`netem`) module.
> 
> By configuring egress queuing disciplines (`qdisc`), we emulated six clinical operating profiles:
> 1. Hospital LAN ($1\text{ Gbps}, 0\text{ ms}, 0\%$)
> 2. Broadband ($50\text{ Mbps}, 10\text{ ms}, 0.1\%$)
> 3. 4G LTE ($15\text{ Mbps}, 30\text{ ms}, 0.5\%$)
> 4. 3G Mobile ($2\text{ Mbps}, 80\text{ ms}, 1.5\%$)
> 5. Rural Telemedicine ($384\text{ Kbps}, 120\text{ ms}, 2.5\%$)
> 6. Satellite Link ($256\text{ Kbps}, 500\text{ ms}, 5.0\%$)
> 
> Across the 384 Kbps Rural Link, our region-adaptive compression reduced transmission latency from **9.61 seconds down to 1.14 seconds**, proving that our adaptive compression successfully prevents bufferbloat and timeout disconnects.
> 
> **5. Physical Two-Laptop Deployment:**
> We deployed this system on two physical machines: an Ubuntu 24.04 LTS sender at `172.30.11.107` and a Windows 11 receiver at `172.30.11.108` connected via a physical Cat6 Ethernet link.
> 
> I now hand over to Member 4 to present our end-to-end integration, database schema, security hardening, and research findings."

---

# SECTION 4: 50 DEDICATED VIVA QUESTIONS & DEFENSE (MEMBER 3)

### Q1: Why did you choose TCP instead of UDP for medical image transmission?
- **Direct Answer:** Medical imaging requires zero-loss, byte-accurate transmission. TCP guarantees reliable, in-order delivery via sequence numbers and retransmissions. UDP is lossy and packet dropouts corrupt JPEG entropy decoding.

### Q2: What are the 4 layers of the TCP/IP model, and where does your application run?
- **Direct Answer:** Application Layer, Transport Layer, Internet/Network Layer, and Network Interface / Link Layer. Our custom binary protocol operates at the Application Layer directly over Transport Layer TCP.

### Q3: Explain the TCP Three-Way Handshake.
- **Direct Answer:**
  1. **SYN:** Client sends a SYN segment with an initial sequence number ($ISN_c$).
  2. **SYN-ACK:** Server responds with SYN-ACK, acknowledging $ISN_c + 1$ and providing its own sequence number ($ISN_s$).
  3. **ACK:** Client sends an ACK acknowledging $ISN_s + 1$, transitioning the socket to `ESTABLISHED`.

### Q4: Explain the TCP Four-Way Teardown (Connection Termination).
- **Direct Answer:**
  1. Node A sends `FIN`.
  2. Node B sends `ACK` (half-close).
  3. Node B sends its own `FIN` when done sending data.
  4. Node A sends `ACK` and enters `TIME_WAIT` state for $2 \times MSL$ before closing.

### Q5: What is the exact header structure of your binary framing protocol?
- **Direct Answer:** A 14-byte header packed as `!4sHII` in big-endian byte order:
  - `b"NAC1"` (4 bytes magic identifier)
  - Metadata JSON length (2 bytes unsigned short)
  - Reserved field (4 bytes unsigned int)
  - Payload length (4 bytes unsigned int)
- **Code Reference:** `common/protocol.py:pack_frame`

### Q6: What is the exact header structure of your receiver ACK frame?
- **Direct Answer:** A 16-byte header packed as `!4sIQ`:
  - `b"NACK"` (4 bytes magic identifier)
  - Status code (4 bytes unsigned int: 0=OK, 1=Checksum Error, 2=Format Error)
  - Server timestamp (8 bytes unsigned long long)
- **Code Reference:** `common/protocol.py:pack_ack`

### Q7: Why is Network Byte Order (Big-Endian) used in your `struct` packing?
- **Direct Answer:** Different computer architectures use different byte orders (e.g., x86 is Little-Endian, ARM/network hardware is Big-Endian). Big-endian (`!`) is the universal standard for all Internet protocols (RFC 1700), ensuring seamless cross-platform binary parsing between Ubuntu and Windows.

### Q8: What is TCP stream fragmentation and why does it occur?
- **Direct Answer:** TCP is a continuous byte stream without message boundaries. When an application calls `sendall(48000)`, IP fragmentation breaks the buffer into multiple Maximum Transmission Unit (MTU) packets of 1500 bytes. A single `recv()` call on the receiver may return only a partial fragment.

### Q9: How does your code handle partial socket reads?
- **Direct Answer:** In `common/protocol.py`, we implemented `recv_exact(sock, n)`, which loops with `sock.recv(remaining)` until exactly $n$ bytes are accumulated in a bytearray, raising a `ConnectionError` if the socket disconnects prematurely.

### Q10: What is Nagle's algorithm, and how does it interact with small frames?
- **Direct Answer:** Nagle's algorithm buffers small outgoing packets until an entire MSS (1460 bytes) is accumulated or an ACK for previous data is received. For small messages, it can introduce a 40 ms delay. In high-frequency interactive systems, it can be disabled via `TCP_NODELAY`.

### Q11: Why did you use `time.perf_counter()` instead of `time.time()` for wire timing?
- **Direct Answer:** `time.time()` provides wall-clock time, which can drift, step backward during NTP synchronization, and has coarse resolution on Windows (~15 ms). `time.perf_counter()` is a strictly monotonic hardware-backed clock with nanosecond resolution that never steps backward.

### Q12: How is wire transmission time ($T_{\text{tx}}$) measured in `sender/tcp_client.py`?
- **Direct Answer:** It records `t_start = time.perf_counter()` immediately before `sock.sendall(frame_data)` and `t_end = time.perf_counter()` immediately upon completion, computing $T_{\text{tx}} = t_{\text{end}} - t_{\text{start}}$.

### Q13: State the formula for measured wire throughput in Kbps.
- **Direct Answer:**
  $$\text{Throughput (Kbps)} = \frac{S_{\text{wire}} \times 8}{T_{\text{tx}} \times 1000}$$
- **Code Reference:** `sender/tcp_client.py:send_frame`

### Q14: In your physical experiment database, what was the measured wire time and throughput for `benign (104)`?
- **Direct Answer:** In trial `EXP_ADAPT_1795063124`:
  - Wire Payload: 49,007 bytes
  - Wire Time: **0.004547 seconds** (4.55 ms)
  - Wire Throughput: **86,214.85 Kbps** (86.2 Mbps)
  - Round-Trip Latency (RTT): **0.1011 seconds** (101.1 ms)

### Q15: Why didn't your Gigabit Ethernet link reach 1000 Mbps throughput?
- **Direct Answer:** Because the transfer was only 48 KB (~33 TCP segments). TCP slow start begins with an initial congestion window of ~10 segments. The entire transmission completed in two RTT round-trips before TCP could expand its congestion window to fill the 1 Gbps pipe.

### Q16: What is Linux `tc` and what does it do?
- **Direct Answer:** Linux Traffic Control (`tc`) is a kernel utility used to configure and manipulate packet scheduling, queuing disciplines (`qdisc`), rate shaping, and traffic filtering on network interfaces.

### Q17: What is `netem` in Linux?
- **Direct Answer:** `netem` (Network Emulator) is a kernel queuing discipline module within `tc` that enables emulation of wide-area network properties, including packet delay, jitter, packet loss, duplication, and corruption.

### Q18: What `tc` command emulates a 120 ms delay with 2.5% packet loss on `eth0`?
- **Direct Answer:**
  `sudo tc qdisc add dev eth0 root handle 1: netem delay 120ms 15ms loss 2.5%`

### Q19: What is Token Bucket Filter (`tbf`) in `tc`?
- **Direct Answer:** `tbf` is a rate-limiting queuing discipline that meters packet egress using a virtual bucket of tokens replenished at a specified rate (e.g., `rate 384kbit`), holding packets when tokens are depleted and dropping them when queue limits are exceeded.

### Q20: What are the 6 network profiles modeled in your system?
- **Direct Answer:**
  1. Hospital LAN (1 Gbps, 0 ms, 0%)
  2. Broadband (50 Mbps, 10 ms, 0.1%)
  3. 4G LTE (15 Mbps, 30 ms, 0.5%)
  4. 3G Mobile (2 Mbps, 80 ms, 1.5%)
  5. Rural Telemedicine Link (384 Kbps, 120 ms, 2.5%)
  6. Satellite Link (256 Kbps, 500 ms, 5.0%)

### Q21: What was the theoretical transmission time of uncompressed vs adaptive images across the Rural Link?
- **Direct Answer:**
  - Uncompressed image (455 KB): **9.61 seconds**
  - Region-Adaptive image (48 KB): **1.14 seconds**
  - Result: **8.4× transmission speedup**.

### Q22: What physical IP addresses and operating systems were used in your deployment?
- **Direct Answer:**
  - Sender: **172.30.11.107** running **Ubuntu 24.04 LTS** (Linux Kernel 6.8).
  - Receiver: **172.30.11.108** running **Windows 11 Pro**.
  - Port: **5000** (TCP).

### Q23: What socket option allows immediate server restart without "Address already in use" errors?
- **Direct Answer:** `sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)`.
- **Technical Detail:** It allows the socket to bind to a port that is currently in the `TIME_WAIT` state from a previously closed connection.

### Q24: What is the Maximum Transmission Unit (MTU)?
- **Direct Answer:** The largest packet or frame size (in bytes) that can be sent over a physical network layer without fragmentation. For standard Ethernet, the MTU is **1500 bytes**.

### Q25: What is Maximum Segment Size (MSS) in TCP?
- **Direct Answer:** The maximum payload of data that a TCP segment can carry: $\text{MSS} = \text{MTU} - (\text{IP Header} + \text{TCP Header}) = 1500 - (20 + 20) = 1460\text{ bytes}$.

### Q26: How does TCP flow control differ from TCP congestion control?
- **Direct Answer:** Flow control prevents the sender from overwhelming the **receiver's socket buffer** (regulated by receiver window `rwnd`). Congestion control prevents the sender from overwhelming the **intermediate network links and routers** (regulated by congestion window `cwnd`).

### Q27: What is TCP Slow Start?
- **Direct Answer:** An algorithm that begins transmission with a conservative congestion window (typically 10 MSS) and doubles $cwnd$ exponentially for every round-trip time ($cwnd \leftarrow cwnd \times 2$) until packet loss occurs or the slow-start threshold ($ssthresh$) is reached.

### Q28: How does packet loss affect TCP throughput?
- **Direct Answer:** When packet loss occurs, TCP infers network congestion, cuts its congestion window in half (or drops it to 1 MSS on timeout), and enters congestion avoidance, severely reducing throughput (modeled by the Mathis Formula: $\text{Throughput} \propto \frac{\text{MSS}}{\text{RTT} \sqrt{p}}$).

### Q29: What is RTT and how is it measured in your client?
- **Direct Answer:** Round-Trip Time is the elapsed duration from when the client dispatches the first byte of the image frame until it receives and parses the receiver's acknowledgment (`NACK`) packet.

### Q30: What is the purpose of the `probe_connection` method in `sender/tcp_client.py`?
- **Direct Answer:** It performs a fast, non-blocking TCP connect check with a 1.0-second timeout to verify that the receiver server is online and reachable before attempting large file transfers.

### Q31: How is multi-threading implemented in `receiver/tcp_server.py`?
- **Direct Answer:** The main server thread runs an `accept()` loop. Upon accepting an incoming client connection, it spawns a worker thread (`threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True)`) to process the frame without blocking other incoming connections.

### Q32: What happens if the network cable is unplugged during transmission?
- **Direct Answer:** The socket write or read call blocks until the OS TCP socket timeout expires, after which a `socket.timeout` or `ConnectionResetError` is raised. In `sender/tcp_client.py`, this is caught and cleanly reported to the GUI as a transfer failure.

### Q33: How does the receiver know when the full image has arrived?
- **Direct Answer:** By parsing the 4-byte payload length field in the 14-byte `NAC1` frame header and invoking `recv_exact(sock, payload_len)`.

### Q34: What is the significance of the magic bytes `b"NAC1"`?
- **Direct Answer:** It provides protocol synchronization and validation. If an unknown client connects or sending fails to align with our frame header, the receiver detects that the first 4 bytes do not match `b"NAC1"` and immediately rejects the connection.

### Q35: How does the Windows Defender Firewall treat incoming socket connections on port 5000?
- **Direct Answer:** By default, Windows Defender blocks all unsolicited incoming TCP connections. We executed an administrative PowerShell command to create an explicit inbound firewall rule allowing traffic on TCP port 5000 for private and domain networks.

### Q36: What is jitter in computer networking?
- **Direct Answer:** Jitter is the statistical variance in packet transit delay over time. In `netem`, it is specified as `netem delay 120ms 15ms`, meaning packets experience delays between 105 ms and 135 ms.

### Q37: How does your UI display live transfer history?
- **Direct Answer:** In `ui/sender/transmission_view.py`, a dedicated `QTableWidget` displays real-time rows populated with Timestamp, Image ID, Mode, Wire Bytes, Wire Tx Time, Throughput (Kbps), and ACK status.

### Q38: What is bandwidth-delay product (BDP)?
- **Direct Answer:** $\text{BDP} = \text{Bandwidth} \times \text{RTT}$. It represents the maximum volume of data that can be in flight across the network pipe at any given moment.

### Q39: Why is the sliding window protocol important for satellite networks?
- **Direct Answer:** Satellite networks have very large RTTs (~500 ms). A stop-and-wait protocol would spend 99% of its time waiting for ACKs. A large sliding window allows the sender to transmit continuous packets without waiting for immediate acknowledgment, filling the high BDP pipe.

### Q40: What happens if a SHA-256 checksum mismatch occurs on the receiver?
- **Direct Answer:** The receiver discards the corrupted image buffer, aborts segmentation inference, and transmits an ACK frame with status code `1` (`CHECKSUM_FAIL`), alerting the sender to log a transmission integrity violation.

### Q41: How do you check active `tc` rules on Ubuntu?
- **Direct Answer:** Running the shell command `sudo tc qdisc show dev eth0` displays active root queuing disciplines, configured delays, packet loss rates, and token bucket rates.

### Q42: How do you reset `tc` rules back to normal?
- **Direct Answer:** `sudo tc qdisc del dev eth0 root 2>/dev/null` removes all custom queuing disciplines, restoring native kernel forwarding.

### Q43: What is the difference between client-ephemeral ports and server well-known ports?
- **Direct Answer:** The server binds to a fixed, well-known port (5000) so clients know where to connect. The client OS dynamically assigns an ephemeral port (e.g., 54321) from the range 49152–65535 for the outbound socket.

### Q44: Can multiple senders connect to the receiver simultaneously?
- **Direct Answer:** Yes. `receiver/tcp_server.py` spawns an independent daemon thread for every accepted socket connection, allowing concurrent client transmissions.

### Q45: What is the purpose of the 8-byte server timestamp in the ACK packet?
- **Direct Answer:** It allows the sender to compute server processing time ($T_{\text{proc}} = t_{\text{ack\_sent}} - t_{\text{frame\_received}}$) and isolate pure one-way network latency from application computation time.

### Q46: What is head-of-line (HoL) blocking?
- **Direct Answer:** A performance bottleneck in TCP where a single dropped packet at the front of the queue stalls the delivery of all subsequent packets in the socket buffer until the dropped packet is retransmitted and acknowledged.

### Q47: What is the difference between half-duplex and full-duplex communication?
- **Direct Answer:** Half-duplex allows transmission in only one direction at a time (e.g., walkie-talkie). Full-duplex allows simultaneous bidirectional transmission (e.g., standard Ethernet and our TCP connection where sender streams data while receiver returns ACKs).

### Q48: How does `recv_exact` detect a disconnected socket?
- **Direct Answer:** In Python sockets, if the remote peer closes the connection, `sock.recv()` immediately returns an empty byte string `b""`. `recv_exact` checks `if not chunk:` and raises a `ConnectionError`.

### Q49: Why is the metadata JSON separated from the image payload?
- **Direct Answer:** It separates control plane metadata (image ID, quality, checksum, mode) from the data plane binary payload, allowing the receiver to parse and validate transaction headers before allocating memory for image decompression.

### Q50: How does the network layer prevent Denial-of-Service memory exhaustion?
- **Direct Answer:** In `receiver/tcp_server.py`, the server enforces a hard ceiling of 50 MB on the 4-byte payload length field. If an incoming header advertises a payload larger than 50 MB, the server immediately drops the connection before calling `bytearray()`, preventing memory exhaustion attacks.

---

# SECTION 5: SUMMARY CHECKLIST FOR MEMBER 3

Before stepping into the evaluation room, ensure you can:
- [ ] Draw the 14-byte `NAC1` and 16-byte `NACK` binary frame structures on the whiteboard.
- [ ] Explain why TCP is mandatory for medical imaging and why UDP is unacceptable.
- [ ] State the exact IP addresses: Ubuntu sender `172.30.11.107`, Windows receiver `172.30.11.108`, Port `5000`.
- [ ] Write down the measured telemetry from physical trial `EXP_ADAPT_1795063124`: Wire Time $4.55\text{ ms}$, Throughput $86,214.85\text{ Kbps}$ ($86.2\text{ Mbps}$), RTT $101.1\text{ ms}$.
- [ ] Explain why measured throughput on Gigabit Ethernet was 86.2 Mbps rather than 1000 Mbps (TCP slow start and small 48 KB payload).
- [ ] Write down the Linux `tc netem` command for emulating the Rural Telemedicine profile.
- [ ] Perform a manual wire time and throughput calculation by hand.
- [ ] Deliver your 5-minute presentation script fluently.
- [ ] Seamlessly hand over to Member 4.
