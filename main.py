from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
import asyncio
import uvicorn

app = FastAPI()

# Use a global variable for the model
global_model = None

# Enable CORS for your entire FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You should configure this properly in a production environment
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model when the FastAPI app starts and reuse it for each request
async def initialize_and_get_model():
    global global_model
    if global_model is None:
        loop = asyncio.get_event_loop()
        global_model = await loop.run_in_executor(pool, initialize_model)
    return global_model

def initialize_model():
    return pipeline('fill-mask', model='nlpaueb/legal-bert-base-uncased', tokenizer='nlpaueb/legal-bert-base-uncased')

# Initialize the thread pool executor with an initializer function
pool = ThreadPoolExecutor(max_workers=1)

async def generate_response(prompt, model=Depends(initialize_and_get_model)):
    words = prompt.split()
    for i, k in enumerate(words):
        if k[0] == '<' and k[-1] == '>':
            words[i] = '[MASK]'
    encoded_text = ' '.join(map(str, words))
    pred = model(encoded_text)
    words = encoded_text.split()
    token_replace_count = 0
    for i, word in enumerate(words):
        if word == '[MASK]':
            words[i] = pred[token_replace_count][0]['token_str']
            token_replace_count += 1
    prompt_generated = ' '.join(map(str, words))
    return prompt_generated

@app.post('/prompt')
async def process(request: Request, model=Depends(initialize_and_get_model)):
    # Get the input data from the POST request
    data = await request.json()

    # Check if 'prompt' is in the JSON data
    if 'prompt' in data:
        prompt = data['prompt']
        # Use the global model for prediction
        prompt_generated = await generate_response(prompt, model)
        return {'response': prompt_generated}

    else:
        raise HTTPException(status_code=400, detail="Missing prompt parameter")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
