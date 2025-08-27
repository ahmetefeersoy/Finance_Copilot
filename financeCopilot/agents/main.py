# main.py
import os
import json
import time
import asyncio
from typing import Optional, List

import google.generativeai as genai
from dotenv import load_dotenv
import yfinance as yf

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from agents.baseAgent import Agent
from agents.expenseAnalyzerAgent import ExpenseAnalyzerAgent, expenseAnalyzerRole
from agents.normalChatAgent import NormalChatAgent, normalChatAgentRole
from agents.orcestratorAgent import Orcestrator, orcestratorAgentRole, agents
from agents.lifePlannerAgent import LifePlannerAgent, lifePlannerAgentRole
from agents.budgetPlannerAgent import BudgetPlannerAgent, budgetPlannerAgentRole
from agents.investmentAdvisorAgent import InvestmentAdvisorAgent, investmentAdvisorAgentRole
from agents.exportReportAgent import generate_transaction_pdf, generate_budget_pdf
from agents.job_tracking import job_status

app = FastAPI(title="Finance Copilot API (FastAPI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Authorization"],
)


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

orchestrator = Orcestrator("Orchestrator", orcestratorAgentRole)
budget_planner = BudgetPlannerAgent("BudgetPlannerAgent", budgetPlannerAgentRole)

class BudgetRequest(BaseModel):
    userId: str

class EmbeddingRequest(BaseModel):
    text: str

def _get_current_market_prices_fast(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        symbols = [line.strip() for line in f if line.strip()]

    data = yf.download(
        tickers=" ".join(symbols),
        period="1d",
        interval="1m",
        group_by="ticker",
        threads=True,
        progress=False
    )

    results = []
    for symbol in symbols:
        try:
            last_price = data[symbol]["Close"].dropna().iloc[-1]
            results.append(f"{symbol}: {last_price:.2f} $")
        except Exception:
            results.append(f"{symbol}: Price not available")

    return results



@app.post("/chat")
async def handle_user_input(
    message: Optional[str] = Form(default=None),
    user: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
):
    """
    Accepts multipart/form-data:
      - message: str
      - user: str (optional)
      - file: PDF (optional, only for expense analyzer agent)
    """
    if not message and not file:
        raise HTTPException(status_code=400, detail="Message or file is required")

    track_id = "static-track-id"
    if track_id in job_status:
        del job_status[track_id]
    job_status[track_id] = {
        "status": "processing",
        "step": "routing to agent",
        "user_input": message
    }

    try:
        orchestrator_decision = await asyncio.to_thread(orchestrator.model.generate_content, message)
        decision_text = orchestrator_decision.text
        decision_json = json.loads(decision_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestrator decision failed: {e}")

    job_type = decision_json.get("job")

    if job_type == "routing":
        selected_agent_key = decision_json.get("selected_agent")
        if selected_agent_key not in agents:
            raise HTTPException(status_code=400, detail=f"Unknown agent: {selected_agent_key}")

        agent = agents[selected_agent_key]

        if selected_agent_key == "expenseanalyzeragent":
            if not file or not file.filename:
                raise HTTPException(status_code=400, detail="Please upload a PDF file for analysis")
          
            result = await asyncio.to_thread(agent.categorize_pdf, file.file)

        elif selected_agent_key == "lifeplanneragent":
            result = await agent.get_life_plan(message, user)

        elif selected_agent_key == "investmentadvisoragent":
            result = await agent.get_financal_advise(message, user)

        else:
            result = await asyncio.to_thread(agent.generate_response, message)

        orchestrator.conversation_history.append({
            "user_input": message,
            "agent_key": selected_agent_key,
            "agent_response": result
        })

        try:
            final_response = await asyncio.to_thread(orchestrator.generate_final_response, message, result)
            if not final_response:
                raise ValueError("Empty final response")

            if isinstance(final_response, str):
                parsed_response = json.loads(final_response)
            else:
                parsed_response = final_response

            agent_resp = parsed_response.get("agent_response")
            if isinstance(agent_resp, str):
                extra_parsed = json.loads(agent_resp)
            else:
                extra_parsed = agent_resp

        except Exception as parse_err:
            raise HTTPException(status_code=500, detail=f"Final response could not be parsed: {parse_err}")

        return JSONResponse(content={"success": True, "response": extra_parsed})

    elif job_type == "transporting":
        return JSONResponse(content={"success": True, "response": decision_json.get("transporting")})

    else:
        raise HTTPException(status_code=400, detail="Invalid job type from orchestrator")


@app.post("/budget-analysis")
async def handle_budget_analysis(payload: BudgetRequest):
    """
    JSON body: { "userId": "..." }
    """
    user_id = payload.userId
    try:
        result = await asyncio.to_thread(budget_planner.run_budget_analysis, user_id)
        if not result:
            raise HTTPException(status_code=404, detail="No data available for analysis")
        return JSONResponse(content={"success": True, "response": result})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Budget analysis failed: {e}")


@app.post("/embeddings")
async def get_embeddings(payload: EmbeddingRequest):
    if not payload.text:
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        result = await asyncio.to_thread(
            genai.embed_content,
            model="models/embedding-001",
            content=payload.text,
            task_type="retrieval_document",
        )
        return JSONResponse(content={"success": True, "embeddings": result["embedding"]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")


@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    job = job_status.get(job_id)
    if not job:
        return JSONResponse(
            content={
                "success": True,
                "status": {"status": "idle", "step": "waiting for input"}
            }
        )
    return JSONResponse(content={"success": True, "status": job})


@app.post("/export-transaction")
async def transaction_export(data: dict):
    """
    Accepts arbitrary JSON to feed your PDF generator.
    Returns a file response.
    """
    try:
        # generator is sync → offload
        path = await asyncio.to_thread(generate_transaction_pdf, data)
        if not os.path.exists(path):
            raise HTTPException(status_code=500, detail="Generated file not found")
        return FileResponse(path, filename=os.path.basename(path), media_type="application/pdf")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export (transaction) failed: {e}")


@app.post("/export-budget")
async def budget_export(data: dict):
    try:
        path = await asyncio.to_thread(generate_budget_pdf, data)
        if not os.path.exists(path):
            raise HTTPException(status_code=500, detail="Generated file not found")
        return FileResponse(path, filename=os.path.basename(path), media_type="application/pdf")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export (budget) failed: {e}")


@app.get("/market-prices")
async def fetch_market_prices():
    """
    Reads ./agents/financeAgent/sp500_symbols.txt, fetches latest minute close using yfinance,
    writes results to ./agents/financeAgent/market_prices_output.txt, returns timing.
    """
    symbols_file = "./agents/financeAgent/sp500_symbols.txt"
    output_file = "./agents/financeAgent/market_prices_output.txt"

    start = time.time()
    try:
        prices = await asyncio.to_thread(_get_current_market_prices_fast, symbols_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"yfinance failed: {e}")

    elapsed = time.time() - start

    try:
        with open(output_file, "w") as f:
            for p in prices:
                f.write(p + "\n")
            f.write(f"\nExecution Time: {elapsed:.2f} seconds\n")
    except Exception as e:
        pass

    return JSONResponse(content={"status": "success", "method": "yfinance", "execution_time": elapsed})


