# server.py
import os
import json
import hashlib
import sqlite3
from datetime import datetime
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==========================================
#   FIX: PREVENT MODEL LOADING TWICE
# ==========================================
IS_WORKER = os.environ.get("SERVER_WORKER", "0") == "1"
generator = None  # Lazy-load model only once

def load_model():
    global generator
    if generator is None:
        print("🔥 Loading AI model (DistilGPT-2)...")
        from transformers import pipeline
        generator = pipeline("text-generation", model="distilgpt2")
    return generator


# ==========================================
#   FASTAPI APP CONFIG
# ==========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
#   DATABASE SETUP
# ==========================================
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    email TEXT PRIMARY KEY,
    password TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT,
    total REAL,
    breakdown TEXT,
    created_at TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS goals(
    email TEXT PRIMARY KEY,
    weekly_target REAL
)
""")

conn.commit()


def hash_password(p: str) -> str:
    return hashlib.sha256(p.encode()).hexdigest()


def row_to_history(row):
    return {
        "id": row[0],
        "email": row[1],
        "total": row[2],
        "breakdown": json.loads(row[3]),
        "created_at": row[4],
    }


# ==========================================
#   MODELS
# ==========================================
class User(BaseModel):
    email: str
    password: str


class FootprintData(BaseModel):
    total: float
    breakdown: Dict[str, float]
    email: Optional[str] = None


class Goal(BaseModel):
    email: str
    weekly_target: float


# ==========================================
#   AI ADVICE ENDPOINT
# ==========================================
@app.post("/advice")
async def get_advice(data: FootprintData):
    model = load_model()

    prompt = (
        f"Carbon footprint breakdown: {json.dumps(data.breakdown)}, Total: {data.total:.2f} kg CO2.\n"
        "Give 3 short bullet points with practical eco-friendly advice.\n"
        "- "
    )

    result = model(prompt, max_length=120, num_return_sequences=1, do_sample=True, top_p=0.92)
    text = result[0]["generated_text"]

    lines = [line.strip() for line in text.split("\n") if line.strip().startswith("-")]
    if len(lines) == 0:
        lines = ["- Reduce unnecessary travel.", "- Save electricity.", "- Reduce waste."]

    return {"advice": "\n".join(lines[:3])}


# ==========================================
#   AUTH ENDPOINTS
# ==========================================
@app.post("/signup")
def signup(user: User):
    try:
        cursor.execute("INSERT INTO users VALUES (?, ?)", (user.email, hash_password(user.password)))
        conn.commit()
        return {"message": "Account created"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already exists")


@app.post("/login")
def login(user: User):
    cursor.execute("SELECT password FROM users WHERE email=?", (user.email,))
    row = cursor.fetchone()
    if row and row[0] == hash_password(user.password):
        return {"message": "Login successful", "email": user.email}
    raise HTTPException(status_code=401, detail="Invalid email or password")


# ==========================================
#   HISTORY ENDPOINTS
# ==========================================
@app.post("/save_footprint")
def save_footprint(data: FootprintData):
    if not data.email:
        raise HTTPException(status_code=400, detail="Email required")

    timestamp = datetime.utcnow().isoformat()
    cursor.execute(
        "INSERT INTO history (email, total, breakdown, created_at) VALUES (?, ?, ?, ?)",
        (data.email, data.total, json.dumps(data.breakdown), timestamp),
    )
    conn.commit()

    return {"message": "Saved", "created_at": timestamp}


@app.get("/get_history")
def get_history(email: str = Query(...), limit: int = 50):
    cursor.execute(
        "SELECT * FROM history WHERE email=? ORDER BY created_at DESC LIMIT ?",
        (email, limit),
    )
    rows = cursor.fetchall()
    return {"history": [row_to_history(r) for r in rows]}


# ==========================================
#   GOALS
# ==========================================
@app.post("/set_goal")
def set_goal(goal: Goal):
    cursor.execute("INSERT OR REPLACE INTO goals VALUES (?, ?)", (goal.email, goal.weekly_target))
    conn.commit()
    return {"message": "Goal updated"}


@app.get("/get_goal")
def get_goal(email: str = Query(...)):
    cursor.execute("SELECT weekly_target FROM goals WHERE email=?", (email,))
    row = cursor.fetchone()
    return {"weekly_target": row[0] if row else None}


# ==========================================
#   LEADERBOARD
# ==========================================
@app.get("/leaderboard")
def leaderboard(limit: int = 10):
    cursor.execute("""
        SELECT email, AVG(total) AS avg_total, COUNT(*) AS cnt
        FROM history
        GROUP BY email
        HAVING cnt > 0
        ORDER BY avg_total ASC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    data = [{"email": r[0], "avg_total": r[1], "count": r[2]} for r in rows]
    return {"leaderboard": data}


# ==========================================
#   PASSWORD RESET
# ==========================================
@app.post("/reset_password")
def reset_password(user: User):
    cursor.execute("SELECT email FROM users WHERE email=?", (user.email,))
    if not cursor.fetchone():
        raise HTTPException(status_code=404, detail="Email not found")

    cursor.execute("UPDATE users SET password=? WHERE email=?", (hash_password(user.password), user.email))
    conn.commit()

    return {"message": "Password updated"}


# ==========================================
#   RUN SERVER SAFELY
# ==========================================
if __name__ == "__main__":
    os.environ["SERVER_WORKER"] = "1"
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
