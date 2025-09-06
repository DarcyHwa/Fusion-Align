#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
check_torch_env.py

在当前项目环境中检测：
- 系统 / Python 信息
- （独立检查）GPU/驱动可用性：nvidia-smi + CUDA Driver API（不依赖 PyTorch）
- PyTorch / CUDA / cuDNN 信息
- nvidia-smi 驱动与运行时信息

适用：Win11 / Linux /（macOS 无 NVIDIA）
"""

import os
import sys
import platform
import subprocess
from textwrap import indent

# ===================== 通用工具 =====================

def print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def try_run(cmd):
    """运行外部命令，返回 (returncode, stdout, stderr)"""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", check=False
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError as e:
        return 127, "", str(e)
    except Exception as e:
        return 1, "", str(e)

# ===================== 系统信息 =====================

def show_system_info():
    print_header("系统与 Python 环境")
    print(f"OS         : {platform.platform()}")
    print(f"Python     : {sys.version.splitlines()[0]}")
    print(f"Executable : {sys.executable}")
    print(f"CUDA_VISIBLE_DEVICES = {os.getenv('CUDA_VISIBLE_DEVICES')}")

# ===================== 独立 GPU 可用性检查 =====================

def independent_gpu_check():
    """
    不依赖 PyTorch 的 GPU/驱动检查：
    A) 通过 nvidia-smi 粗略判断
    B) 直连 CUDA Driver API（ctypes）：初始化驱动、枚举 GPU、获取名称/显存/算力；
       尝试创建上下文并进行 1MB 显存分配/释放做烟雾测试
    """
    print_header("独立 GPU/驱动 可用性检查（不依赖 PyTorch）")

    # ---- A) nvidia-smi 探测 ----
    rc, out, err = try_run(["nvidia-smi", "-L"])
    if rc == 0 and out:
        lines = [ln for ln in out.splitlines() if ln.strip()]
        print("nvidia-smi -L 输出：")
        for ln in lines:
            print("  " + ln)
    else:
        print("nvidia-smi 不可用或未发现 NVIDIA GPU（这并不一定代表不可用，继续用 Driver API 深查）。")
        if err:
            print(indent(err, "  "))

    # ---- B) CUDA Driver API 深查 ----
    try:
        import ctypes
        import ctypes.util

        def load_cuda_driver():
            # Win: nvcuda.dll; Linux: libcuda.so / libcuda.so.1; macOS: 通常无
            candidates = []
            if os.name == "nt":
                candidates = ["nvcuda.dll"]
            else:
                # 先用 ctypes.util.find_library，再回退常见名
                found = ctypes.util.find_library("cuda")
                if found:
                    candidates.append(found)
                candidates += ["libcuda.so.1", "libcuda.so"]
            last_err = None
            for name in candidates:
                try:
                    return ctypes.WinDLL(name) if os.name == "nt" else ctypes.CDLL(name)
                except Exception as e:
                    last_err = e
                    continue
            raise OSError(f"无法加载 CUDA Driver 库（尝试：{candidates}）：{last_err}")

        cu = load_cuda_driver()

        # 绑定符号（不同版本可能带 _v2 后缀）
        def sym(names):
            for n in names:
                try:
                    return getattr(cu, n)
                except AttributeError:
                    pass
            raise AttributeError(f"未找到符号：{names}")

        # 原型设定
        # CUresult 是 int，0 表示 CUDA_SUCCESS
        c_int = ctypes.c_int
        c_uint = ctypes.c_uint
        c_char_p = ctypes.c_char_p
        c_size_t = ctypes.c_size_t
        c_void_p = ctypes.c_void_p
        c_ulonglong = ctypes.c_ulonglong  # CUdeviceptr

        cuInit = sym(["cuInit"])
        cuInit.argtypes = [c_uint]
        cuInit.restype = c_int

        cuDeviceGetCount = sym(["cuDeviceGetCount"])
        cuDeviceGetCount.argtypes = [ctypes.POINTER(c_int)]
        cuDeviceGetCount.restype = c_int

        cuDeviceGet = sym(["cuDeviceGet"])
        cuDeviceGet.argtypes = [ctypes.POINTER(c_int), c_int]
        cuDeviceGet.restype = c_int

        cuDeviceGetName = sym(["cuDeviceGetName"])
        cuDeviceGetName.argtypes = [ctypes.c_char_p, c_int, c_int]
        cuDeviceGetName.restype = c_int

        cuDeviceTotalMem = sym(["cuDeviceTotalMem_v2", "cuDeviceTotalMem"])
        cuDeviceTotalMem.argtypes = [ctypes.POINTER(c_ulonglong), c_int]
        cuDeviceTotalMem.restype = c_int

        cuDeviceGetAttribute = sym(["cuDeviceGetAttribute"])
        cuDeviceGetAttribute.argtypes = [ctypes.POINTER(c_int), c_int, c_int]
        cuDeviceGetAttribute.restype = c_int

        cuCtxCreate = sym(["cuCtxCreate_v2", "cuCtxCreate"])
        cuCtxCreate.argtypes = [ctypes.POINTER(c_void_p), c_uint, c_int]
        cuCtxCreate.restype = c_int

        cuCtxDestroy = sym(["cuCtxDestroy_v2", "cuCtxDestroy"])
        cuCtxDestroy.argtypes = [c_void_p]
        cuCtxDestroy.restype = c_int

        cuMemAlloc = sym(["cuMemAlloc_v2", "cuMemAlloc"])
        cuMemAlloc.argtypes = [ctypes.POINTER(c_ulonglong), c_size_t]
        cuMemAlloc.restype = c_int

        cuMemFree = sym(["cuMemFree_v2", "cuMemFree"])
        cuMemFree.argtypes = [c_ulonglong]
        cuMemFree.restype = c_int

        # 关键属性常量（参考 CUDA Driver API）
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76

        def check(ret, what):
            if ret != 0:
                raise RuntimeError(f"{what} 失败（CUresult={ret}）")

        # 1) 初始化驱动
        check(cuInit(0), "cuInit")
        # 2) 设备数量
        count = c_int(0)
        check(cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
        print(f"CUDA Driver API 已初始化，检测到设备数量：{count.value}")

        if count.value == 0:
            print("未发现可用的 NVIDIA GPU（Driver API 返回 0 块设备）。")
            return

        # 3) 遍历设备并打印信息
        for idx in range(count.value):
            dev = c_int(0)
            check(cuDeviceGet(ctypes.byref(dev), idx), f"cuDeviceGet({idx})")
            # 名称
            buf = ctypes.create_string_buffer(100)
            check(cuDeviceGetName(buf, 100, dev.value), "cuDeviceGetName")
            name = buf.value.decode("utf-8", errors="ignore")
            # 显存
            total = c_ulonglong(0)
            check(cuDeviceTotalMem(ctypes.byref(total), dev.value), "cuDeviceTotalMem")
            total_gb = total.value / (1024 ** 3)
            # 算力
            major = c_int(0)
            minor = c_int(0)
            check(cuDeviceGetAttribute(ctypes.byref(major), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev.value),
                  "cuDeviceGetAttribute(MAJOR)")
            check(cuDeviceGetAttribute(ctypes.byref(minor), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev.value),
                  "cuDeviceGetAttribute(MINOR)")
            print(f"  - GPU {idx}: {name} | Compute Capability {major.value}.{minor.value} | {total_gb:.2f} GB")

        # 4) 在 0 号设备上创建上下文并分配 1MB 显存做烟雾测试
        dev0 = c_int(0)
        check(cuDeviceGet(ctypes.byref(dev0), 0), "cuDeviceGet(0)")
        ctx = c_void_p()
        # flags=0 -> 自动
        check(cuCtxCreate(ctypes.byref(ctx), 0, dev0.value), "cuCtxCreate")
        try:
            ptr = c_ulonglong(0)
            one_mb = 1 << 20
            check(cuMemAlloc(ctypes.byref(ptr), one_mb), "cuMemAlloc(1MB)")
            check(cuMemFree(ptr), "cuMemFree")
            print("CUDA Driver API 烟雾测试：上下文创建 + 1MB 显存分配/释放 ✅")
        finally:
            cuCtxDestroy(ctx)

        print("结论：驱动与 GPU 正常，基础 CUDA Driver API 操作可用。")

    except OSError as e:
        print("无法加载 CUDA Driver 库（nvcuda.dll / libcuda.so）。通常表示未安装 NVIDIA 驱动。")
        print(indent(str(e), "  "))
    except RuntimeError as e:
        print("CUDA Driver API 调用失败：")
        print(indent(str(e), "  "))
    except Exception as e:
        print("独立 GPU 检测发生异常：")
        print(indent(repr(e), "  "))

# ===================== PyTorch 信息 =====================

def show_torch_info():
    print_header("PyTorch / CUDA / cuDNN 信息")
    try:
        import torch

        print(f"PyTorch 版本           : {torch.__version__}")
        torch_cuda = getattr(torch.version, "cuda", None)
        print(f"PyTorch 编译用 CUDA    : {torch_cuda}")
        try:
            cudnn_v = torch.backends.cudnn.version()
        except Exception:
            cudnn_v = None
        print(f"cuDNN 版本             : {cudnn_v}")

        is_cuda_available = torch.cuda.is_available()
        print(f"CUDA 是否可用          : {is_cuda_available}")

        if is_cuda_available:
            device_count = torch.cuda.device_count()
            print(f"CUDA 设备数量          : {device_count}")

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                name = props.name
                cc = f"{props.major}.{props.minor}"
                total_gb = props.total_memory / (1024 ** 3)
                print(f"  - GPU {i}: {name} | Compute Capability {cc} | {total_gb:.2f} GB")

            try:
                free_b, total_b = torch.cuda.mem_get_info()
                print(
                    f"当前设备显存（mem_get_info）: free {free_b / (1024**3):.2f} GB / total {total_b / (1024**3):.2f} GB"
                )
            except Exception:
                pass

            print("\nCUDA 运算自检：")
            for i in range(device_count):
                try:
                    with torch.cuda.device(i):
                        x = torch.rand(1024, 1024, device=f"cuda:{i}")
                        y = x @ x.t()
                        s = y.sum().item()
                    print(f"  - 在 cuda:{i} 上矩阵乘法成功，sum={s:.4f}")
                except Exception as e:
                    print(f"  - 在 cuda:{i} 上计算失败：{e}")
        else:
            x = torch.rand(3, 3)
            y = x @ x.t()
            print("CPU 运算自检成功。")
    except ImportError as e:
        print("未检测到 PyTorch，或导入失败。请确认已在当前项目环境中安装 pytorch。")
        print(f"ImportError: {e}")
    except Exception as e:
        print(f"获取 PyTorch 信息时发生异常：{e}")

# ===================== nvidia-smi 信息 =====================

def show_nvidia_smi():
    print_header("nvidia-smi（驱动 / 运行时）")
    rc, out, err = try_run(["nvidia-smi"])
    if rc != 0:
        print("未检测到 nvidia-smi 或执行失败。通常表示未安装 NVIDIA 驱动或非 NVIDIA GPU。")
        if err:
            print(indent(err, prefix="  "))
        return

    print("nvidia-smi 可用，简要信息：")
    print(indent("\n".join(out.splitlines()[:10]), prefix="  "))

    rc, out, err = try_run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,cuda_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    if rc == 0 and out:
        print("\n结构化 GPU 信息：")
        for line in out.splitlines():
            print("  " + line)
    elif err:
        print("\n无法获取结构化 GPU 信息：")
        print(indent(err, prefix="  "))

# ===================== 主入口 =====================

def main():
    show_system_info()
    independent_gpu_check()   # ← 新增的独立检查
    show_torch_info()
    show_nvidia_smi()
    print("\n检查完成。")

if __name__ == "__main__":
    main()
