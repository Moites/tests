from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
import sqlite3
import pandas as pd
from typing import Optional, List
import uvicorn

app = FastAPI(title="Edu Materials API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class MaterialReq(BaseModel):
    content: str
    topic: Optional[str] = ""

class TrajectoryReq(BaseModel):
    known_topics: List[str] = []
    daily_minutes: int = 60
    deadline_days: int = 30

class ModelAPI:
    def __init__(self):
        self.model_path = 'model_v.pkl'
        self.db_path = 'materials.db'
        self.models = None
        self.load_models()

    def load_models(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("model_v.pkl не найден. Запустите AgentV.py")
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']

    def get_db_data(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM mats", conn)
        conn.close()
        return df

    def check_material(self, content):
        length = len(content)
        words = len(content.split())
        if length < 200:
            return "НЕДОПУСТИМО: слишком короткий"
        if length > 50000:
            return "НЕДОПУСТИМО: слишком длинный"
        if words < 50:
            return "НЕДОПУСТИМО: мало слов"
        return "ДОПУСТИМО"

    def predict_parallel(self, content, has_prev=0, has_next=0):
        length = len(content)
        words = len(content.split())
        is_gen = 1 if 'generated://' in content else 0
        feats = pd.DataFrame([[length, words, has_prev, has_next, is_gen]],
                             columns=['len', 'words', 'has_prev', 'has_next', 'is_gen'])
        return int(self.models['parallel'].predict(feats)[0])

    def predict_sequential(self, content, has_prev=0, has_next=0):
        length = len(content)
        words = len(content.split())
        is_gen = 1 if 'generated://' in content else 0
        feats = pd.DataFrame([[length, words, has_prev, has_next, is_gen]],
                             columns=['len', 'words', 'has_prev', 'has_next', 'is_gen'])
        return int(self.models['sequential'].predict(feats)[0])

    def predict_difficulty(self, content):
        length = len(content)
        words = len(content.split())
        feats = pd.DataFrame([[length, words, 0, 0, 0]],
                             columns=['len', 'words', 'has_prev', 'has_next', 'is_gen'])
        return self.models['difficulty'].predict(feats)[0]

    def estimate_time(self, content):
        words = len(content.split())
        base = words / 200
        diff = self.predict_difficulty(content)
        coef = {'Низкая': 0.8, 'Средняя': 1.0, 'Высокая': 1.5}.get(diff, 1.0)
        return round(base * coef, 1)

api = ModelAPI()

@app.get("/")
def root():
    return {"service": "Edu Materials API", "status": "running"}

@app.post("/check")
def check_material(req: MaterialReq):
    result = api.check_material(req.content)
    return {
        "topic": req.topic,
        "status": result,
        "length": len(req.content),
        "words": len(req.content.split())
    }

@app.post("/analyze")
def analyze_material(req: MaterialReq):
    return {
        "topic": req.topic,
        "check": api.check_material(req.content),
        "parallel_cluster": api.predict_parallel(req.content),
        "sequential_cluster": api.predict_sequential(req.content),
        "difficulty": api.predict_difficulty(req.content),
        "est_time_min": api.estimate_time(req.content)
    }

@app.post("/trajectory")
def build_trajectory(req: TrajectoryReq):
    df = api.get_db_data()
    df['len'] = df['content'].str.len()
    df['words'] = df['content'].str.split().str.len()
    unknown = df[~df['topic'].isin(req.known_topics)].copy()
    if unknown.empty:
        return {"status": "Все темы изучены!", "plan": []}
    unknown['est_time'] = unknown['content'].apply(api.estimate_time)
    unknown = unknown.sort_values(['subject', 'topic'])
    total_available = req.daily_minutes * req.deadline_days
    unknown['cum_time'] = unknown['est_time'].cumsum()
    feasible = unknown[unknown['cum_time'] <= total_available]

    return {
        "status": "ok",
        "total_available_min": total_available,
        "planned_min": feasible['est_time'].sum(),
        "materials_count": len(feasible),
        "plan": feasible[['subject', 'topic', 'est_time']].to_dict(orient='records')
    }

@app.get("/subjects")
def get_subjects():
    df = api.get_db_data()
    return {"subjects": df['subject'].unique().tolist()}

@app.get("/topics")
def get_topics(subject: Optional[str] = None):
    df = api.get_db_data()
    if subject:
        df = df[df['subject'] == subject]
    return {"topics": df['topic'].unique().tolist()}

@app.get("/help")
def help_info():
    return {
        "endpoints": {
            "/": "GET - статус",
            "/check": "POST - проверка материала",
            "/analyze": "POST - полный анализ",
            "/trajectory": "POST - траектория обучения",
            "/subjects": "GET - список предметов",
            "/topics": "GET - список тем"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)