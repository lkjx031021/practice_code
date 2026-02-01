from time import time
import json
import os
from pathlib import Path
import uuid

from langchain_deepseek import ChatDeepSeek
from langchain.chat_models import init_chat_model
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from typing import Optional, List, Dict, Any
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

BASE_DIR = Path(__file__).parent

RESULTS_DIR = BASE_DIR / "file_saved"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

llm = init_chat_model(model="deepseek-chat")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store for file data (in-memory for demonstration)
file_store: Dict[str, Dict[str, Any]] = {}

class ChatRequest(BaseModel):
    file_id: Optional[str] = None
    question: str

class ChartData(BaseModel):
    chart_type: str
    data: Dict[str, Any]

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload and store data in memory.

    Improvements over previous implementation:
    - Accepts common CSV/XLS/XLSX content-types and falls back to filename extension
    - Enforces a 10 MB size limit
    - Parses CSV with a charset fallback and provides clear error codes/messages
    - Returns consistent metadata: fileId, columns, sample, rows
    """
    allowed_cts = {
        "text/csv",
        "application/csv",
        "application/vnd.ms-excel",  # old .xls
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
        "application/octet-stream",
    }
    max_bytes = 10 * 1024 * 1024  # 10 MB

    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()
    print(filename, content_type)

    # Read raw bytes so we can enforce size and try multiple parsers/encodings
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传了空文件")
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="文件过大，最大允许 10 MB")

    file_id = str(uuid.uuid4())
    temp_file = RESULTS_DIR / f"temp_{file_id}_{file.filename}"
    with open(temp_file, "wb") as f:
        f.write(content)


    # Determine whether this is CSV-like or Excel by content-type first, then by extension, then by heuristic
    is_csv = False
    if content_type in allowed_cts:
        if "csv" in content_type:
            is_csv = True
        elif "excel" in content_type or "spreadsheet" in content_type:
            is_csv = False
        else:
            is_csv = filename.endswith(".csv")
    else:
        if filename.endswith((".csv", ".txt")):
            is_csv = True
        elif filename.endswith((".xls", ".xlsx")):
            is_csv = False
        else:
            # simple heuristic: if the first chunk contains commas or tabs, treat as CSV
            try:
                sample = content[:2048].decode("utf-8", errors="ignore")
                is_csv = ("," in sample) or ("\t" in sample)
            except Exception:
                is_csv = True

    from io import BytesIO
    bio = BytesIO(content)

    try:
        if is_csv:
            # try utf-8 then fallback to latin1
            try:
                bio.seek(0)
                df = pd.read_csv(bio)
            except Exception:
                bio.seek(0)
                df = pd.read_csv(bio, encoding="latin1")
        else:
            bio.seek(0)
            df = pd.read_excel(bio)

        file_store[file_id] = {
            "name": file.filename,
            "data": df.to_dict(orient="records"),
            "columns": df.columns.tolist(),
            "sample": df.head().to_dict(),
            "rows": len(df),
        }

        return JSONResponse(content={
            "fileId": file_id,
            "columns": df.columns.tolist(),
            "sample": df.head().to_dict(),
            "rows": len(df),
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理文件时出错: {str(e)}")
    finally:
        try:
            await file.close()
        except Exception:
            pass

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Process chat request and return appropriate response"""
    try:
        # Check if file_id is provided and exists
        if request.file_id and request.file_id in file_store:
            file_data = file_store[request.file_id]
            df = pd.DataFrame(file_data["data"])
            
            # Process question (simple example)
            question = request.question.lower()

            temp_file = RESULTS_DIR / f"temp_{request.file_id}_{file_data['name']}"
            df = pd.read_csv(temp_file).to_html()

            prompts = PromptTemplate.from_template(
                "你是一个数据分析专家，基于以下数据集内容回答用户的问题。\n"\
                    "数据集内容:\n{data}\n用户问题: {question}\n请给出简明扼要的回答。",
            )

            
            # Basic query processing
            if "columns" in question:
                return JSONResponse(content={"response": f"The dataset contains the following columns: {', '.join(file_data['columns'])}"})
            
            elif "rows" in question or "count" in question:
                return JSONResponse(content={"response": f"The dataset contains {len(df)} rows of data."})
            
            elif "summary" in question:
                return JSONResponse(content={"response": df.describe().to_markdown()})
            
            # Simple chart generation
            elif "chart" in question:
                # Determine chart type from question
                chart_type = "bar"
                if "pie" in question:
                    chart_type = "pie"
                elif "line" in question:
                    chart_type = "line"
                
                # Use first two columns for demonstration
                columns = file_data["columns"]
                if len(columns) >= 2:
                    chart_data = {
                        "chart_type": chart_type,
                        "data": {
                            "categories": df[columns[0]].unique().tolist(),
                            "items": [
                                {"label": str(label), "value": df[df[columns[0]] == label][columns[1]].sum()}
                                for label in df[columns[0]].unique()
                            ]
                        }
                    }
                    return JSONResponse(content=chart_data)
                else:
                    return JSONResponse(content={"response": "Not enough columns for chart generation"})
            
            # Default response
            return JSONResponse(content={"response": "I can help you analyze this data. What would you like to know?"})
        

    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/test")
async def test():
    """Server-Sent Events (SSE) endpoint — streams LLM deltas as SSE "data:" events.

    - Frames each chunk as a JSON `data:` event so `EventSource` can consume it.
    - Emits an `event: done` when finished and `event: error` on exceptions.
    - Re-uses the module-level `llm` (avoids re-init on every request).
    """
    async def _aiter_or_iter(obj):
        # helper: support both async and sync generators returned by LLM
        if hasattr(obj, "__aiter__"):
            async for item in obj:
                yield item
        else:
            for item in obj:
                yield item

    async def event_generator():
        try:
            last_ping = time()
            stream = llm.stream("你好,请介绍一下高斯")
            async for chunk in _aiter_or_iter(stream):
                # each SSE event must end with a double newline
                payload = {"type": "delta", "text": getattr(chunk, "content", str(chunk))}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                # lightweight keep-alive comment (helps some proxies)
                if time() - last_ping > 15:
                    yield ": ping\n\n"
                    last_ping = time()

            # final event to signal completion
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:
            err = {"type": "error", "message": str(exc)}
            yield f"event: error\ndata: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

@app.post("/test2")
async def test2():
    """Server-Sent Events (SSE) endpoint — streams LLM deltas as SSE "data:" events.

    - Frames each chunk as a JSON `data:` event so `EventSource` can consume it.
    - Emits an `event: done` when finished and `event: error` on exceptions.
    - Re-uses the module-level `llm` (avoids re-init on every request).
    """

    async def event_generator():
        try:
            for chunk in llm.stream("你好,请介绍一下高斯"):
                # each SSE event must end with a double newline
                payload = {"type": "delta", "text": getattr(chunk, "content", str(chunk))}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            # final event to signal completion
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:
            err = {"type": "error", "message": str(exc)}
            yield f"event: error\ndata: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/test-ndjson")
async def test_ndjson():
    """Alternative streaming format: newline-delimited JSON (NDJSON).

    Useful for `fetch` + ReadableStream consumption (parses each line as a JSON object).
    """
    async def _aiter_or_iter(obj):
        if hasattr(obj, "__aiter__"):
            async for item in obj:
                yield item
        else:
            for item in obj:
                yield item

    async def gen():
        try:
            stream = llm.stream("你好,请介绍一下高斯")
            async for chunk in _aiter_or_iter(stream):
                out = {"delta": getattr(chunk, "content", str(chunk))}
                yield json.dumps(out, ensure_ascii=False) + "\n"
            yield json.dumps({"done": True}, ensure_ascii=False) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app",
                host="0.0.0.0",
                port=8001,
                reload=True,
                log_level="info")