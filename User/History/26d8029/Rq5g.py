import micropython

micropython.alloc_emergency_exception_buf(100)
print("[BOOT] Emergency exception buffer allocated.")