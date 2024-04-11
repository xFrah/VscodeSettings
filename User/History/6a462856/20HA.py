import bleak
import asyncio

char_write = "0000fff1-0000-1000-8000-00805f9b34fb"
characteristic_notify = "0000fff2-0000-1000-8000-00805f9b34fb"
address = "00:00:00:00:00:00"


async def start_ppg_stream(client: bleak.BleakClient):
    await client.write_gatt_char(char_write, bytearray([0x3D, 0x01]))


async def notification_handler(sender, data):
    print("received: ", data.hex())


async def main():
    # get device with mac address
    device = await bleak.BleakScanner.find_device_by_address(address)
    if device is None:
        print("Device not found")
        return

    async with bleak.BleakClient(device) as client:
        print("connected")

        # subscribe to notifications
        await client.start_notify(characteristic_notify, notification_handler)
        print("disconnecting...")

    print("disconnected")


if __name__ == "__main__":
    asyncio.run(main())
