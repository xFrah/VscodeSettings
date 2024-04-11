import bleak
import asyncio


async def main():
    devices = await bleak.discover()
    for d in devices:
        print(d)


if __name__ == "__main__":
    asyncio.run(main())
