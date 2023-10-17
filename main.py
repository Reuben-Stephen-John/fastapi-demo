from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ProcessPoolExecutor
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

def initialize_model():
    return pipeline('fill-mask',model='nlpaueb/legal-bert-base-uncased',tokenizer='nlpaueb/legal-bert-base-uncased')

# Initialize the process pool executor with an initializer function
pool = ProcessPoolExecutor(max_workers=1)

async def generate_response(prompt):
    words = prompt.split()
    for i, k in enumerate(words):
        if k[0] == '<' and k[-1] == '>':
            words[i] = '[MASK]'
    encoded_text = ' '.join(map(str, words))
    pred = global_model(encoded_text)
    words = encoded_text.split()
    token_replace_count = 0
    for i, word in enumerate(words):
        if word == '[MASK]':
            words[i] = pred[token_replace_count][0]['token_str']
            token_replace_count += 1
    prompt_generated = ' '.join(map(str, words))
    return prompt_generated

@app.on_event("startup")
async def startup_event():
    global global_model
    loop = asyncio.get_event_loop()
    # Initialize the model when the FastAPI app starts
    global_model = await loop.run_in_executor(pool, initialize_model)


@app.post('/prompt')
async def process(request: Request):
    # Get the input data from the POST request
    data = await request.json()

    # Check if 'prompt' is in the JSON data
    if 'prompt' in data:
        prompt = data['prompt']

        if global_model is None:
            raise HTTPException(status_code=500, detail="Model is not initialized")
        # Use the global model for prediction
        prompt_generated = await generate_response(prompt)
        return {'response': prompt_generated}

    else:
        raise HTTPException(status_code=400, detail="Missing prompt parameter")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)