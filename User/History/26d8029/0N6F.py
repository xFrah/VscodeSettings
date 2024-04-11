import micropython
from wifi_manager import WMS

micropython.alloc_emergency_exception_buf(100)
print("[BOOT] Emergency exception buffer allocated.")

WMS.connect_to_wifi()  # TODO we need credentials here, change wifi manager's code to get them by himself
