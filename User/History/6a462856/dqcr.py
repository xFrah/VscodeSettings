import bleak
import asyncio


async def main():
    devices = await bleak.discover()
    characteristic_write = "0000fff1-0000-1000-8000-00805f9b34fb"
    characteristic_notify = "0000fff2-0000-1000-8000-00805f9b34fb"
    for d in devices:
        print(characteristic_notify)


if __name__ == "__main__":
    asyncio.run(main())
