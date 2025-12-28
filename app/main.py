import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .schemas import SummarizeRequest
from .services import model_service

app = FastAPI(title="ViT5 Summarizer", version="0.1.0")
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    # Run model loading in a separate thread to avoid blocking event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, model_service.load_model)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/summarize")
async def summarize(payload: SummarizeRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    try:
        summary = model_service.generate_summary(
            payload.text,
            max_length=payload.max_length,
            min_length=payload.min_length,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"summary": summary}
