import asyncio
import httpx

# Replace with your actual API keys from different Google accounts
API_KEYS = ["AIzaSyDVR0oKgE3VTibkZNm6cSsmKoDudlqbRwE", "AIzaSyBHioBFsKxQIwaPw7VOR4qMPl8kiT-pmrE"]
URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent"

async def call_gemini(api_key, prompt):
    headers = {"Authorization": f"Bearer {api_key}"}
    json_data = {"contents": [{"parts": [{"text": prompt}]}]}
    timeout = httpx.Timeout(10.0, read=30.0)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{URL}?key={api_key}", json=json_data, timeout=timeout)
            return response.json()
        except httpx.ReadTimeout:
            print(f"Request timed out for API Key {api_key}. Retrying...")
            response = await client.post(f"{URL}?key={api_key}", json=json_data, timeout=timeout)
            return response.json()

async def main():
    prompts = ["Tell me a joke.", "What is AI?"]
    
    # Create tasks for each API call
    tasks = [call_gemini(API_KEYS[i], prompts[i]) for i in range(2)]
    
    # Execute tasks and print results as they complete
    for task in asyncio.as_completed(tasks):
        result = await task
        print(f"\nResponse: {result}")

# Run the async main function
asyncio.run(main())
