# Uses clean sessions to avoid backlog when OOR.
# Publishes connection statistics.

from gc import collect
from mqtt_as import MQTTClient

collect()
import uasyncio as asyncio
from ubinascii import hexlify
from machine import unique_id
from web import file_download_lock

collect()
import network

collect()

MQTTClient.DEBUG = True


# Default "do little" coro for optional user replacement
async def eliza(*_):  # e.g. via set_wifi_handler(coro): see test program
    await asyncio.sleep_ms(50)
    print("[MQTT] Eliza: I'm not pining, I'm not pinin'!")


config = {
    "client_id": hexlify(unique_id()),
    "gateway": False,
    "keepalive": 60,
    "ping_interval": 0,
    "ssl": False,
    "ssl_params": {},
    "response_time": 10,
    "clean_init": True,
    "clean": True,
    "max_repubs": 4,
    "subs_cb": lambda *_: None,
    "wifi_coro": eliza,
    "connect_coro": eliza,
    "ssid": "TIM-23398578",
    "wifi_pw": "axf6gH4aTcbrq3ksMeCTqngJ",
    "gwtopic": None,
    "queue_len": 1,  # Use event interface with default queue
}  # "gwtopic": called from gateway. See docs.


class mqtt_obj:
    """
    MQTT Interface object, used to communicate with the MQTT broker.

    Args:
        server (str): The IP address of the MQTT broker.
        port (int): The port of the MQTT broker.
        user (str): The username to use when connecting to the MQTT broker.
        password (str): The password to use when connecting to the MQTT broker.
        topic (str): The topic to publish to.
    """

    def __init__(self, server, port, user, password, topic="shed"):
        self.outages = 0
        self.TOPIC = topic  # For demo publication and last will use same topic
        config["will"] = (topic, "Goodbye cruel world!", False, 0)
        config["server"] = server
        config["port"] = port
        config["user"] = user
        config["password"] = password
        self.client = MQTTClient(config)

    async def messages(self):
        async for topic, msg, retained in self.client.queue:
            print(f'[MQTT] Topic: "{topic.decode()}" Message: "{msg.decode()}" Retained: {retained}')

    async def down(self):
        while True:
            await self.client.down.wait()  # Pause until connectivity changes
            self.client.down.clear()
            self.outages += 1
            print("[MQTT] WiFi or broker is down.")

    async def up(self):
        while True:
            await self.client.up.wait()
            self.client.up.clear()
            print("[MQTT] We are connected to broker.")
            await self.client.subscribe("foo_topic", 1)

    async def start(self):
        """Starts asyncio tasks for the MQTT client."""
        asyncio.create_task(self.main())

    async def main(self):
        tasks: asyncio.Task = []
        # get sta if
        sta_if = network.WLAN(network.STA_IF)

        async def loop_setup():
            await asyncio.sleep(1)
            for task in tasks:
                task.cancel()
            tasks.clear()
            print(f"[MQTT] Connecting to {config['server']}:{config['port']}")

        failed = False
        while True:

            await loop_setup()
            try:
                await self.client.connect()
            
            except OSError as e:
                print(f"[MQTT] Connection failed for following reason: {e}")
                async with file_download_lock:  # TODO this is fucking shit.
                    sta_if.disconnect()
                await asyncio.sleep(5)
                continue
            for task in (self.up, self.down, self.messages):
                tasks.append(asyncio.create_task(task()))
            n = 0
            while True:
                await asyncio.sleep(5)
                print("[MQTT] Publish", n)
                # If WiFi is down the following will pause for the duration.
                await self.client.publish(self.TOPIC, "{} repubs: {} outages: {}".format(n, self.client.REPUB_COUNT, self.outages), qos=1)
                n += 1
