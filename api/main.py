from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
import torch
import torch.nn as nn
import json
import re

# 1. Define Request/Response Schemas
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    is_toxic: bool
    confidence: float

# 2. Replicate the Exact Same PyTorch Model Class
class ToxicLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[0], hidden[1]), dim=1)
        logits = self.fc(hidden_cat)
        return logits.squeeze(1)

# 3. Setup Global Variables
word2idx = {}
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_THRESHOLD = 0.8460  # The optimal threshold found during training
MAX_LEN = 100

# 4. Load Models on Startup using modern lifespan pattern
@asynccontextmanager
async def lifespan(app: FastAPI):
    global word2idx, model

    print("Loading word2idx dictionary...")
    with open("word2idx.json", "r", encoding="utf-8") as f:
        word2idx = json.load(f)

    print(f"Loading LSTM model (Vocab Size: {len(word2idx)})...")
    model = ToxicLSTM(vocab_size=len(word2idx)).to(device)

    # Load state dict safely handling CPU/GPU
    model.load_state_dict(torch.load("toxic_lstm.pt", map_location=device, weights_only=False))
    model.eval()  # Set to evaluation mode
    print("Models successfully loaded and ready for inference!")

    yield  # App runs here

    # Cleanup on shutdown (optional)
    print("Shutting down...")

# 5. Setup FastAPI App
app = FastAPI(
    title="Toxic Comment Detector",
    description="Inference API for the PyTorch LSTM model.",
    lifespan=lifespan,
)

# Allow all origins so the HTML frontend can call this API from any domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve the frontend UI
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("frontend/index.html")

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return text.strip()

def tokenize(text: str) -> list[str]:
    return text.lower().split()

def encode(text: str) -> list[int]:
    tokens = tokenize(text)
    return [word2idx.get(token, word2idx.get("<UNK>", 1)) for token in tokens]

def pad_sequence(seq: list[int]) -> list[int]:
    if len(seq) > MAX_LEN:
        return seq[:MAX_LEN]
    return seq + [0] * (MAX_LEN - len(seq))

# 6. Expose the API Endpoint
@app.post("/predict", response_model=PredictResponse)
async def predict_toxicity(request: PredictRequest):
    # Process text
    clean_text = preprocess(request.text)
    encoded = encode(clean_text)
    padded = pad_sequence(encoded)
    
    # Convert to Tensor (batch size of 1)
    tensor_input = torch.tensor([padded], dtype=torch.long).to(device)
    
    # Run Inference
    with torch.no_grad():
        logits = model(tensor_input)
        probability = torch.sigmoid(logits).item()
        
    is_toxic = bool(probability > BEST_THRESHOLD)
    
    return PredictResponse(
        text=request.text,
        is_toxic=is_toxic,
        confidence=probability
    )
