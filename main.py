from fastapi import FastAPI
from pydantic import BaseModel
from orchestrator_runner import run_orchestrator
import uvicorn

app = FastAPI()

class GenerateRFPRequest(BaseModel):
    raw_input: dict
    include_sections: dict

@app.post("/generate_rfp")
async def generate_rfp(payload: GenerateRFPRequest):
    try:
        result = await run_orchestrator(
            payload.raw_input,
            payload.include_sections
        )
        return {"success": True, "decisions": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
