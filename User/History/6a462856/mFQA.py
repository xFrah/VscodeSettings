import bleak
import asyncio


async def main():
    characteristic_write = "0000fff1-0000-1000-8000-00805f9b34fb"
    characteristic_notify = "0000fff2-0000-1000-8000-00805f9b34fb"
    # get device with mac address
    device = await bleak.BleakScanner.find_device_by_address(
        args.address, cb=dict(use_bdaddr=args.macos_use_bdaddr)
    )


if __name__ == "__main__":
    asyncio.run(main())
