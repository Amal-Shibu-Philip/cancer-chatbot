import os, joblib, numpy as np, pandas as pd, shap, glob, pickle, faiss
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

app = FastAPI(title="LUAD vs LUSC Chatbot")
BASE = Path(__file__).parent
DATA = BASE / "data"

# 1) Clinical metadata
_pheno_frames = []
for p in DATA.glob("*clinicalMatrix*"):
    df = pd.read_csv(p, sep="\t", index_col=0)
    _pheno_frames.append(df)
if not _pheno_frames:
    print(" No clinicalMatrix files found under data/. Clinical fields will be 'Unknown'.")
    pheno = pd.DataFrame()
else:
    pheno = pd.concat(_pheno_frames, axis=0)
    pheno.index = pheno.index.str.slice(0, 12)
    pheno = pheno[~pheno.index.duplicated(keep="first")]

# 2) Expression + model + SHAP
expr_path = BASE / "expression.tsv"
if not expr_path.exists():
    # create a tiny placeholder so the app doesn't crash
    expr = pd.DataFrame()
    clf = None; top_genes = []; explainer = None
else:
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    clf = joblib.load(BASE / "xena_luad_lusc_rf.joblib")
    top_genes = joblib.load(BASE / "xena_selected_genes.pkl")
    Xbg = np.log1p(expr.T[top_genes])
    explainer = shap.Explainer(clf, Xbg.sample(min(100, len(Xbg)), random_state=42), feature_names=top_genes)

# 3) RAG index
_sbert = SentenceTransformer("all-MiniLM-L6-v2")
_txts = list((DATA / "pubmed").glob("*.txt"))
if _txts:
    META = DATA / "pubmed_meta.pkl"
    IDXF = DATA / "pubmed.index"
    if IDXF.exists() and META.exists():
        idx = faiss.read_index(str(IDXF))
        with open(META, "rb") as f:
            pubmed_docs = pickle.load(f)
    else:
        pubmed_docs = []
        for path in _txts:
            text = open(path, encoding="utf-8").read()
            pubmed_docs.append({"id": Path(path).name, "text": text})
        embs = _sbert.encode([d["text"] for d in pubmed_docs], convert_to_numpy=True)
        idx = faiss.IndexFlatL2(embs.shape[1]); idx.add(embs)
        faiss.write_index(idx, str(IDXF)); open(META, "wb").write(pickle.dumps(pubmed_docs))
else:
    idx = None; pubmed_docs = []

def retrieve_snippets(query: str, top_k: int = 3) -> str:
    if idx is None:
        return "No literature corpus available."
    q_emb = _sbert.encode([query]); _, I = idx.search(q_emb, top_k)
    snips = []
    for i in I[0]:
        txt = pubmed_docs[i]["text"].replace("\n", " ").strip()
        snips.append(txt[:200] + "")
    return " ".join(snips)

class Query(BaseModel):
    sample_id: str
    ask: str

class ChatResponse(BaseModel):
    sample_id: str
    prediction: str
    confidence: float
    raw_explanation: str
    top_features: List[str]
    phenotype: Dict[str, Any]

def make_explanation(sid, pred, conf, vals, genes):
    arr = vals.flatten()[: len(genes)]
    idxs = np.argsort(-np.abs(arr))[:5]
    lines = [f"I predict sample **{sid}** is **{pred}** ({conf:.1%}). Top drivers:"]
    top5 = []
    for i in idxs:
        g, v = genes[i], arr[i]
        s = "+" if v > 0 else "-"
        lines.append(f"{g} ({s}{abs(v):.3f})")
        top5.append(g)
    return " ".join(lines), top5

@app.get("/", response_class=FileResponse)
def serve_index():
    html = BASE / "index.html"
    if html.exists(): return html
    # fallback minimal page
    tmp = BASE / "_index_tmp.html"
    tmp.write_text("<html><body><h3>LUAD vs LUSC Chatbot API</h3><p>POST /chat with JSON {sample_id, ask}</p></body></html>", encoding="utf-8")
    return tmp

@app.post("/chat", response_model=ChatResponse)
def chat(q: Query):
    sid = q.sample_id.strip()
    ask = q.ask.strip().lower()

    # classification intent
    if any(kw in ask for kw in ["type of cancer", "what type", "driver"]):
        if expr.empty or clf is None:
            raise HTTPException(503, "Model/data not available in this snapshot.")
        if sid not in expr.columns:
            raise HTTPException(404, detail=f"Sample '{sid}' not found")
        x = np.log1p(expr[sid].loc[top_genes]).to_frame().T
        pred = clf.predict(x)[0]
        proba = clf.predict_proba(x)[0]
        conf = float(proba[list(clf.classes_).index(pred)])
        shap_vals = explainer(x)[0].values
        raw, top_feats = make_explanation(sid, pred, conf, shap_vals, top_genes)

        # clinical metadata
        if not pheno.empty and sid in pheno.index:
            r = pheno.loc[sid]
            clinical = {
                "stage": r.get("ajcc_pathologic_stage", "Unknown"),
                "smoking_history": r.get("tobacco_smoking_history_indicator", "Unknown"),
            }
        else:
            clinical = {"stage": "Unknown", "smoking_history": "Unknown"}

        return ChatResponse(sample_id=sid, prediction=str(pred), confidence=conf,
                            raw_explanation=raw, top_features=top_feats, phenotype=clinical)

    # RAG / free-text
    snippets = retrieve_snippets(q.ask)
    return ChatResponse(sample_id=sid, prediction="N/A", confidence=0.0,
                        raw_explanation=f"I found these relevant passages: {snippets}",
                        top_features=[], phenotype={})
