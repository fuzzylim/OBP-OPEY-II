import asyncio
import uvicorn
from fastapi import FastAPI

app = FastAPI()
condition = asyncio.Condition()
data = {}

async def waiter(key: str):
    async with condition:
        print(f"[Waiter] Waiting for {key}")
        while key not in data:
            await condition.wait()
            print(f"[Waiter] Woke up for {key}, checking condition")
        return data.pop(key)

@app.post("/notify/{key}")
async def notify(key: str):
    async with condition:
        data[key] = "test"
        print(f"[Notifier] Setting {key}")
        condition.notify_all()
        print(f"[Notifier] Notified all for {key}")
    return {"status": "ok"}

@app.get("/test/{key}")
async def test(key: str):
    result = await waiter(key)
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)