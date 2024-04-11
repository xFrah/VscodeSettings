import micropython
from wifi_manager import WMS

micropython.alloc_emergency_exception_buf(100)
print("[BOOT] Emergency exception buffer allocated.")