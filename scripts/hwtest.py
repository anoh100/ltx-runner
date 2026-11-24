"""Технический тест машины до установки моделей (~2 мин). Печатает JSON с результатами.
python3 hwtest.py
"""
import json, os, subprocess, threading, time
import torch

R = {}


def smi():
    q = "name,power.limit,power.draw,clocks.sm,clocks.max.sm,temperature.gpu,pcie.link.gen.current,pcie.link.width.current,memory.total"
    return subprocess.check_output(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"]).decode().strip().split(", ")


# 1. GPU под нагрузкой: bf16 matmul 25 с + замеры nvidia-smi
info = smi(); R["gpu"] = info[0]; R["power_limit_w"] = float(info[1]); R["max_sm_mhz"] = int(info[4])
R["pcie_link"] = f"gen{info[6]} x{info[7]}"; R["vram_mb"] = int(info[8])
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); b = torch.randn_like(a)
samples, stop = [], False


def sampler():
    while not stop:
        s = smi(); samples.append((float(s[2]), int(s[3]), int(s[5]))); time.sleep(1)


th = threading.Thread(target=sampler); th.start()
torch.cuda.synchronize(); t0 = time.time(); n = 0
while time.time() - t0 < 25:
    for _ in range(20):
        a @ b
    torch.cuda.synchronize(); n += 20
dt = time.time() - t0; stop = True; th.join()
R["bf16_tflops"] = round(2 * 8192 ** 3 * n / dt / 1e12, 1)
tail = samples[len(samples) // 3:]
R["load_power_w"] = round(sum(s[0] for s in tail) / len(tail)); R["load_sm_mhz"] = round(sum(s[1] for s in tail) / len(tail))
R["load_temp_c"] = max(s[2] for s in tail)
del a, b; torch.cuda.empty_cache()

# 2. PCIe: копия 4 ГБ pinned RAM → GPU
h = torch.empty(4 * 1024 ** 3, dtype=torch.uint8).pin_memory(); torch.cuda.synchronize()
t0 = time.time(); d = h.to("cuda", non_blocking=True); torch.cuda.synchronize()
R["h2d_gb_s"] = round(4 / (time.time() - t0), 1); del h, d; torch.cuda.empty_cache()

# 3. Диск: запись 4 ГБ (direct) и ОЗУ
out = subprocess.run("dd if=/dev/zero of=/root/ddtest bs=16M count=256 oflag=direct 2>&1 | tail -1", shell=True, capture_output=True, text=True).stdout
R["disk_write"] = out.strip().split(",")[-1].strip(); os.remove("/root/ddtest")
mem = open("/proc/meminfo").read().split("\n")
R["ram_avail_gb"] = round(int([l for l in mem if l.startswith("MemAvailable")][0].split()[1]) / 1024 ** 2)

# 4. Сеть: 1 ГБ с Hugging Face на машину
url = "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
t0 = time.time()
subprocess.run(["curl", "-sL", "-r", "0-1073741823", "-o", "/dev/null", url], check=False)
R["hf_down_mb_s"] = round(1024 / (time.time() - t0))
# файл 50 МБ для замера скорости отдачи к пользователю
subprocess.run("head -c 52428800 /dev/urandom > /root/logs/net50.bin", shell=True)
print("HWTEST " + json.dumps(R, ensure_ascii=False), flush=True)
