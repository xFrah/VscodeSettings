import sys
import time
import pyudev
import subprocess
import usb.core


class USB_Dispatcher:
    def __init__(self, context):
        self.ports = {"Lidar": "1-2.1", "TOF1": "1-2.3.3", "TOF2": "1-2.3.2"}
        self.context = context
        self.resetting = False

    def get_tty_by_devicename(self, name):
        print("Getting tty for device:", name, end=" ")
        name = self.ports[name]
        ports = []
        for device in self.context.list_devices(subsystem="tty", ID_BUS="usb"):
            if device.parent.parent.parent.sys_name == name:
                ports.append(device["DEVNAME"])
        print(ports)
        return ports

    def get_ids_by_devicename(self, name):
        name = self.ports[name]
        for device in self.context.list_devices(subsystem="tty", ID_BUS="usb"):
            if device.parent.parent.parent.sys_name == name:
                return device.get("ID_VENDOR_ID"), device.get("ID_MODEL_ID")

    def reset(self, name):
        res = self.get_ids_by_devicename(name)
        if res is None:
            return
        vid, pid = res
        print("Resetting", name, vid, pid)
        self.resetting = True
        try:
            dev = usb.core.find(idVendor=int(vid, 16), idProduct=int(pid, 16), find_all=True)
        except Exception as e:
            print(e)
            return
        for cfg in dev:
            try:
                # sys.stdout.write("Hexadecimal VendorID=" + hex(cfg.idVendor) + " & ProductID=" + hex(cfg.idProduct) + "\n\n")
                cfg.detach_kernel_driver(0)
                time.sleep(1)
                # bind interface 0
                cfg.attach_kernel_driver(0)
                time.sleep(1)
                print("Resetting", name, "done")
            except Exception as e:
                print(e)
        self.resetting = False

    def restart_tof(self):
        to_reset = set()
        for name, _ in self.ports.items():
            if name.startswith("TOF"):
                to_reset.add(self.get_ids_by_devicename(name))

        for vid, pid in to_reset:
            dev = usb.core.find(idVendor=int(vid, 16), idProduct=int(pid, 16), find_all=True)
            for cfg in dev:
                try:
                    cfg.detach_kernel_driver(0)
                    time.sleep(1)
                    cfg.attach_kernel_driver(0)
                    time.sleep(1)
                except Exception as e:
                    print(e)


if __name__ == "__main__":
    # Create a Pyudev context object
    context = pyudev.Context()

    # Find the USB device with the specified vendor and product ID
    for device in context.list_devices(subsystem="tty", ID_BUS="usb"):
        # port = device.parent.get("ID_PATH_TAG")
        print(device["DEVNAME"], device.parent.parent.parent.sys_name, device.get("ID_VENDOR_ID"), device.get("ID_MODEL_ID"))

        try:
            dev = usb.core.find(idVendor=int(device.get("ID_VENDOR_ID"), 16), idProduct=int(device.get("ID_MODEL_ID"), 16), find_all=True)
        except Exception as e:
            print(e)
            continue
        for cfg in dev:
            sys.stdout.write("Hexadecimal VendorID=" + hex(cfg.idVendor) + " & ProductID=" + hex(cfg.idProduct) + "\n\n")
            cfg.detach_kernel_driver(0)
            time.sleep(1)
            # bind interface 0
            cfg.attach_kernel_driver(0)
            time.sleep(1)
    # Find the USB device with the matching vendor ID and product ID
    # device = usb.core.find(idVendor=int(vendor_id, 16), idProduct=int(product_id, 16))

    # If the device is not found, print an error message and exit the script
    # if device is None:
    #    print("Error: Device not found")
    #    exit()
