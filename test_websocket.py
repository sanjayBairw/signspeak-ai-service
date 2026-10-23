import asyncio
import websockets

async def main():
    url = "wss://signspeak-ai-service.onrender.com/ws/sign"
    async with websockets.connect(url) as ws:
        print("Connected to SignSpeak AI!")
        print("Server response:")
        print(await ws.recv())

asyncio.run(main())
