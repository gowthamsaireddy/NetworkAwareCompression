# Dual-Laptop Physical Deployment & Setup Guide

This guide details the physical deployment, network configuration, and environment setup for executing the academic research project across two physical laptops connected directly via Ethernet.

---

## 1. Physical Hardware & Operating Systems

| Node | Physical Machine | Operating System | Role |
| :--- | :--- | :--- | :--- |
| **Sender** | Laptop 1 | Windows Host running **Ubuntu Linux VM** | Dataset scanning, JPEG compression, Linux `tc`/`netem` traffic shaping, adaptive control, TCP client |
| **Receiver** | Laptop 2 | **Windows 11 Native Host** | TCP server daemon, SHA-256 verification, frozen U-Net inference ($256 \times 256$), Dice/IoU evaluation |

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

### Verification
From Ubuntu VM, ping the Windows 11 host:
```bash
ping 192.168.10.2
```
From Windows 11 Command Prompt, ping the Ubuntu VM:
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
2. Append the following rule at the end of the file (replace `ubuntu` with your username):
   ```text
   ubuntu ALL=(ALL) NOPASSWD: /sbin/tc, /usr/sbin/tc
   ```
3. Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

---

## 4. Software Installation & Virtual Environments

### Laptop 1 (Ubuntu Sender VM)
```bash
# Update package index and install system packages
sudo apt update
sudo apt install -y python3 python3-pip python3-venv iproute2 libgl1-mesa-glx

# Clone repository
git clone https://github.com/gowthamsaireddy/NetworkAwareCompression.git
cd NetworkAwareCompression

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install sender dependencies
pip install --upgrade pip
pip install -r requirements/sender.txt
```

### Laptop 2 (Windows 11 Receiver)
```powershell
# In PowerShell (Run as Administrator if needed)
git clone https://github.com/gowthamsaireddy/NetworkAwareCompression.git
cd NetworkAwareCompression

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install receiver dependencies (including PyTorch)
python -m pip install --upgrade pip
pip install -r requirements/receiver.txt
```

---

## 5. Running the Applications

### Step 1: Start Receiver on Laptop 2 (Windows 11)
```powershell
python receiver/app.py
```
1. Navigate to the **TCP Server** tab.
2. Ensure Port is set to `5000` (or your configured port).
3. Click **Start TCP Server**. The status will display `Server Listening on 0.0.0.0:5000`.
4. (Optional) In **Model Management**, load your trained U-Net checkpoint (`.pt` or `.pth`).

### Step 2: Start Sender on Laptop 1 (Ubuntu VM)
```bash
python3 sender/app.py
```
1. In **Dataset Management**, browse to the local BUSI dataset folder and click **Scan Directory**.
2. Select any ultrasound scan to view dimensions, file size, and ground-truth masks.
3. In **Network Emulation**, select an interface (e.g., `eth0`), choose a research profile (e.g., `2 Mbps`), and click **Apply Shaping Profile**.
4. In **TCP Transmission**, set Receiver IP to `192.168.10.2` and Port to `5000`.
5. Connect and send images or batch transmission.
6. In **Adaptive Compression**, configure provisional thresholds and run adaptive trials.
