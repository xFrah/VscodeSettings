import bleak
import asyncio


async def main():
    characteristic_write = "0000fff1-0000-1000-8000-00805f9b34fb"
    characteristic_notify = "0000fff2-0000-1000-8000-00805f9b34fb"
    address = "00:00:00:00:00:00"
    # get device with mac address
    device = await bleak.BleakScanner.find_device_by_address(address)
    if device is None:
        print("Device not found")
        return

    async with bleak.BleakClient(device) as client:
        print("connected")

        for service in client.services:
            print("[Service] %s", service)

            for char in service.characteristics:
                if "read" in char.properties:
                    try:
                        value = await client.read_gatt_char(char.uuid)
                        print(
                            "  [Characteristic] %s (%s), Value: %r",
                            char,
                            ",".join(char.properties),
                            value,
                        )
                    except Exception as e:
                        print(
                            "  [Characteristic] %s (%s), Error: %s",
                            char,
                            ",".join(char.properties),
                            e,
                        )

                else:
                    print("  [Characteristic] %s (%s)", char, ",".join(char.properties))

                for descriptor in char.descriptors:
                    try:
                        value = await client.read_gatt_descriptor(descriptor.handle)
                        print("    [Descriptor] %s, Value: %r", descriptor, value)
                    except Exception as e:
                        print("    [Descriptor] %s, Error: %s", descriptor, e)

        print("disconnecting...")

    print("disconnected")


if __name__ == "__main__":
    asyncio.run(main())
