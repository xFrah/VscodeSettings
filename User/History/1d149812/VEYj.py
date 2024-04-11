import sys
import time
import pyudev
import subprocess
import usb.core


class USB_Dispatcher:
    def __init__(self, context):
        self.ports = {"Lidar": "1-2.1", "TOF1": "1-2.3.3", "TOF2": "1-2.3.4"}
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

    def bind_usb(self, port):
        subprocess.call(["echo", port, ">/sys/bus/usb/drivers/usb/bind"])

    def unbind_usb(self, port):
        subprocess.call(["echo", port, ">/sys/bus/usb/drivers/usb/unbind"])

    def bind(self, device):
        self.bind_usb(self.ports[device])

    def unbind(self, device):
        self.unbind_usb(self.ports[device])

    def bind_all(self):
        for device in self.ports:
            self.bind(device)
            time.sleep(2)

    def unbind_all(self):
        for device in self.ports:
            self.unbind(device)

    def reset(self):



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
