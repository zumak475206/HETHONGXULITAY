"""
src/utils/process_optimizer.py
==============================
Tiện ích tối ưu hóa tiến trình hệ điều hành Windows cho thị giác máy tính và điều khiển chuột thời gian thực.
- Nâng độ ưu tiên lên ABOVE_NORMAL_PRIORITY_CLASS (0x00008000).
- Vô hiệu hóa Windows Power Throttling / EcoQoS (chống drop FPS khi chạy nền hoặc che khuất cửa sổ).
"""

import sys
import ctypes
from typing import Dict

if sys.platform == "win32":
    import ctypes.wintypes


def optimize_windows_process() -> Dict[str, bool]:
    """
    Nâng độ ưu tiên tiến trình và tắt Power Throttling / EcoQoS trên Windows.
    Đảm bảo khi ứng dụng chạy ngầm / camera bị che, Windows không hạ xung nhịp CPU,
    giữ vững 30-60 FPS cho camera và chuyển động chuột.

    Trả về
    ------
    Dict[str, bool]
        Trạng thái {"priority_set": bool, "throttling_disabled": bool}
    """
    results = {
        "priority_set": False,
        "throttling_disabled": False,
    }
    if sys.platform != "win32":
        return results

    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        handle = kernel32.GetCurrentProcess()

        # 1. Nâng priority lên ABOVE_NORMAL_PRIORITY_CLASS (0x00008000)
        kernel32.SetPriorityClass.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.DWORD]
        kernel32.SetPriorityClass.restype = ctypes.wintypes.BOOL
        ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
        if kernel32.SetPriorityClass(handle, ABOVE_NORMAL_PRIORITY_CLASS):
            results["priority_set"] = True

        # 2. Tắt Windows Power Throttling / EcoQoS (Windows 10 1709+ / Windows 11)
        class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
            _fields_ = [
                ("Version", ctypes.wintypes.ULONG),
                ("ControlMask", ctypes.wintypes.ULONG),
                ("StateMask", ctypes.wintypes.ULONG),
            ]

        power_state = PROCESS_POWER_THROTTLING_STATE()
        power_state.Version = 1
        power_state.ControlMask = 0x1  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED
        power_state.StateMask = 0      # 0 = Disable throttling (chạy tối đa hiệu năng)

        ProcessPowerThrottling = 4
        kernel32.SetProcessInformation.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.wintypes.DWORD,
        ]
        kernel32.SetProcessInformation.restype = ctypes.wintypes.BOOL

        if kernel32.SetProcessInformation(
            handle,
            ProcessPowerThrottling,
            ctypes.byref(power_state),
            ctypes.sizeof(power_state),
        ):
            results["throttling_disabled"] = True

    except Exception as e:
        print(f"[WARN] Toi uu tien trinh Windows that bai: {e}")

    return results
