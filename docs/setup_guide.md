# Dual-Laptop Physical Deployment & Setup Guide

This guide details the physical deployment, network configuration, and environment setup for executing the academic research project across two physical laptops connected directly via Ethernet.

---

## 1. Physical Hardware & Operating Systems

| Node | Physical Machine | Operating System | Role |
| :--- | :--- | :--- | :--- |
| **Sender** | Laptop 1 | Windows Host running **Ubuntu Linux VM** (or Native Ubuntu) | Dataset scanning, JPEG compression, Linux `tc`/`netem` traffic shaping, adaptive control, TCP client |
| **Receiver** | Laptop 2 | **Windows 10 / Windows 11 Native Host** | TCP server daemon, SHA-256 verification, automatic frozen U-Net inference ($256 \times 256$), Dice/IoU evaluation |

---

## 2. Direct Ethernet Link & IP Addressing

Connect Laptop 1 and Laptop 2 directly using a Category 5e/6 Ethernet cable (modern NICs feature Auto-MDIX, so crossover cables are unnecessary).

### Static IP Assignment

Configure static IPv4 addresses in the same `/24` subnet:

- **Laptop 1 (Ubuntu VM Sender)**: `192.168.10.1` (Netmask: `255.255.255.0` / `/24`)
- **Laptop 2 (Windows 11 Receiver)**: `192.168.10.2` (Netmask: `255.255.255.0` / `/24`)

### VirtualBox Bridged Adapter Configuration (Laptop 1)
To allow the Ubuntu VM to directly control and shape the physical Ethernet link:
1. Open Oracle VirtualBox settings for the Ubuntu VM.
2. Navigate to **Network** -> **Adapter 1**.
3. Set **Attached to:** `Bridged Adapter`.
4. Set **Name:** Select your physical Ethernet NIC (e.g., `Realtek PCIe GbE Family Controller` or `Intel Ethernet Connection`).
5. Under Advanced, set **Promiscuous Mode:** `Allow All`.
6. Boot into Ubuntu and configure static IP on the bridged interface:
   ```bash
   sudo ip addr add 192.168.10.1/24 dev eth0
   sudo ip link set eth0 up
   ```

### Link Verification
From Ubuntu VM, ping the Windows 11 host:
```bash
ping -c 4 192.168.10.2
```
From Windows 11 Command Prompt or PowerShell, ping the Ubuntu VM:
```cmd
ping 192.168.10.1
```

---

## 3. Ubuntu Sudoers Configuration for Linux `tc` (Laptop 1)

Linux kernel traffic control (`tc`) and network emulation (`netem`) require root privileges. To enable seamless execution from the GUI application without password stalls:

1. Open the sudoers configuration:
   ```bash
   sudo visudo
   ```
2. Append the following rule at the end of the file (replace `ubuntu` with your Linux username):
   ```text
   ubuntu ALL=(ALL) NOPASSWD: /sbin/tc, /usr/sbin/tc
   ```
3. Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

---

## 4. One-Click Setup & Launch Scripts

### Laptop 1 (Ubuntu Sender)

```bash
# 1. Run automated setup script
chmod +x setup_ubuntu.sh run_sender.sh
./setup_ubuntu.sh

# 2. Verify config.json has the Windows Receiver IP:
# "receiver_ip": "192.168.10.2"

# 3. Launch Sender GUI
./run_sender.sh
```

### Laptop 2 (Windows 11 Receiver)

```powershell
# 1. Run automated setup script in PowerShell
.\setup_windows.ps1

# 2. Configure Windows Firewall (Run once in Administrator PowerShell):
New-NetFirewallRule -DisplayName "NetworkAwareCompression-Receiver" -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow

# 3. Launch Receiver GUI
.\run_receiver.bat
```

---

## 5. Configuration via `config.json` and Environment Variables

Settings can be specified in the project-root `config.json` or overridden via environment variables without altering code:

```json
{
  "sender": {
    "receiver_ip": "192.168.10.2",
    "receiver_port": 5000,
    "connection_timeout": 15.0,
    "max_retries": 3,
    "retry_delay_sec": 1.5,
    "dataset_path": "data/BUSI",
    "default_quality": 70,
    "default_network_profile": "unshaped",
    "network_interface": "eth0",
    "log_level": "INFO"
  },
  "receiver": {
    "listen_ip": "0.0.0.0",
    "listen_port": 5000,
    "connection_timeout": 30.0,
    "save_directory": "data/received",
    "model_checkpoint_path": "models/best_unet.pt",
    "device": "auto",
    "auto_process_enabled": true,
    "log_level": "INFO"
  }
}
```

### Environment Variable Overrides
- `NAC_RECEIVER_IP`: Target receiver IP address
- `NAC_RECEIVER_PORT`: Target receiver TCP port
- `NAC_LISTEN_PORT`: Receiver bind port
- `NAC_AUTO_PROCESS`: `1` to enable automatic segmentation on arrival, `0` for manual
- `NAC_MODEL_PATH`: Custom path to U-Net checkpoint (`.pt` or `.pth`)
- `NAC_DEVICE`: Compute device (`cuda`, `cpu`, or `auto`)

---

## 6. End-to-End Execution Workflow

### Step 1: Start Receiver on Laptop 2
1. Run `.\run_receiver.bat`.
2. Ensure the **Dashboard** displays `Server Status: Running` on port 5000 and `Model: Loaded (CUDA/CPU)`.
3. Check `Auto-Process Received Images` is checked for zero-touch automatic inference.

### Step 2: Start Sender on Laptop 1
1. Run `./run_sender.sh`.
2. On the **Transmission** tab, click **Probe Receiver** to test network reachability and measure round-trip time (RTT).
3. Connect to the receiver.
4. Run individual transmissions or click **Run Batch Experiment...** to execute a full sweep over multiple images and network profiles.

### Step 3: Inspecting & Exporting Results
1. On both nodes, navigate to the **Experiment Results** tab.
2. Use the **Search bar**, **Quality filter**, and **Mode filter** to inspect specific trials.
3. Click **Export Summary Report** to generate aggregated academic performance tables ($\text{mean} \pm \text{std}$) grouped by Quality tier.
