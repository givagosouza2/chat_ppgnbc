import os
import io
import re
import json
import fitz
import faiss
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from openai import OpenAI
from sentence_transformers import SentenceTransformer


# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Sensora Insight - Chat Científico",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Sensora Insight")
st.caption("Chat com RAG sobre sua produção científica + análise estruturada do CSV")




OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# =========================
# FUNÇÕES AUXILIARES
# =========================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text_parts = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text_parts.append(page.get_text("text"))
    except Exception:
        return ""
    return normalize_text("\n".join(text_parts))


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    text = normalize_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        start += chunk_size - overlap

    return chunks


def safe_int_series(series):
    return pd.to_numeric(series, errors="coerce")


def infer_reference_from_filename(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    base = base.replace("_", " ").replace("-", " ")
    return normalize_text(base)


def build_documents_from_uploaded_pdfs(uploaded_pdfs):
    docs = []
    for pdf in uploaded_pdfs:
        pdf_bytes = pdf.read()
        text = extract_text_from_pdf_bytes(pdf_bytes)
        docs.append({
            "source": pdf.name,
            "reference_guess": infer_reference_from_filename(pdf.name),
            "text": text,
        })
    return docs


def build_chunks(documents):
    chunked_docs = []
    for doc in documents:
        chunks = chunk_text(doc["text"])
        for i, ch in enumerate(chunks):
            chunked_docs.append({
                "source": doc["source"],
                "reference_guess": doc["reference_guess"],
                "chunk_id": i,
                "text": ch,
            })
    return chunked_docs


@st.cache_data(show_spinner=False)
def compute_embeddings(texts, _model):
    emb = _model.encode(texts, show_progress_bar=False)
    return np.array(emb).astype("float32")


def build_faiss_index(embeddings: np.ndarray):
    if len(embeddings) == 0:
        return None
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def retrieve_chunks(query, model, index, chunked_docs, k=5, allowed_sources=None):
    if index is None or not chunked_docs:
        return []

    query_emb = model.encode([query]).astype("float32")
    distances, indices = index.search(query_emb, min(k * 3, len(chunked_docs)))

    results = []
    seen = set()

    for idx in indices[0]:
        if idx < 0 or idx >= len(chunked_docs):
            continue
        item = chunked_docs[idx]

        if allowed_sources is not None and item["source"] not in allowed_sources:
            continue

        key = (item["source"], item["chunk_id"])
        if key in seen:
            continue
        seen.add(key)
        results.append(item)

        if len(results) >= k:
            break

    return results


def classify_query_heuristic(query: str) -> str:
    q = query.lower()

    numeric_terms = [
        "quantos", "quantidade", "número", "numero", "proporção", "proporcao",
        "percentual", "frequência", "frequencia", "distribuição", "distribuicao",
        "por ano", "por periódico", "por periodico", "gráfico", "grafico",
        "tabela", "estatística", "estatistica"
    ]

    semantic_terms = [
        "resultado", "resultados", "método", "metodo", "conclusão", "conclusao",
        "discussão", "discussao", "achados", "artigo", "artigos", "estudo",
        "estudos", "o que", "como", "quais"
    ]

    has_numeric = any(term in q for term in numeric_terms)
    has_semantic = any(term in q for term in semantic_terms)

    if has_numeric and has_semantic:
        return "hybrid"
    if has_numeric:
        return "structured"
    return "semantic"


def build_filtered_dataframe(df, year_range, selected_types, selected_journals):
    dff = df.copy()

    if "Ano" in dff.columns:
        dff["Ano_num"] = safe_int_series(dff["Ano"])
        dff = dff[
            (dff["Ano_num"].isna()) |
            ((dff["Ano_num"] >= year_range[0]) & (dff["Ano_num"] <= year_range[1]))
        ]

    if selected_types and "Tipo de artigo" in dff.columns:
        dff = dff[dff["Tipo de artigo"].isin(selected_types)]

    if selected_journals and "Periódico" in dff.columns:
        dff = dff[dff["Periódico"].isin(selected_journals)]

    return dff


def df_summary_for_prompt(dff: pd.DataFrame) -> str:
    if dff.empty:
        return "A base filtrada está vazia."

    summary = {}

    summary["total_artigos"] = int(len(dff))

    if "Ano" in dff.columns:
        anos = pd.to_numeric(dff["Ano"], errors="coerce").dropna()
        if not anos.empty:
            summary["periodo"] = f"{int(anos.min())}–{int(anos.max())}"
            summary["artigos_por_ano"] = dff.groupby("Ano").size().to_dict()

    if "Tipo de artigo" in dff.columns:
        summary["tipos_artigo"] = dff["Tipo de artigo"].value_counts().to_dict()

    if "Periódico" in dff.columns:
        summary["periodicos"] = dff["Periódico"].value_counts().head(10).to_dict()

    binary_cols = [c for c in dff.columns if "Presença" in c or "Você está" in c or "discente" in c or "tecnológica" in c]
    bin_summary = {}
    for col in binary_cols:
        vc = dff[col].astype(str).str.strip().value_counts().to_dict()
        bin_summary[col] = vc
    if bin_summary:
        summary["variaveis_binarias"] = bin_summary

    return json.dumps(summary, ensure_ascii=False, indent=2)


def get_structured_answer(query: str, dff: pd.DataFrame):
    q = query.lower()

    if dff.empty:
        return {
            "answer": "Não há artigos no recorte filtrado.",
            "table": None,
            "chart_df": None,
            "chart_type": None,
            "insights": ["Ajuste os filtros para ampliar a base analisada."]
        }

    if "tipo" in q and "artigo" in q and "Tipo de artigo" in dff.columns:
        table = dff["Tipo de artigo"].value_counts().reset_index()
        table.columns = ["Tipo de artigo", "Quantidade"]
        return {
            "answer": "Aqui está a distribuição dos tipos de artigo no recorte atual.",
            "table": table,
            "chart_df": table,
            "chart_type": "bar",
            "insights": [
                f"Total de artigos no recorte: {len(dff)}",
                "A distribuição pode ser refinada por ano ou periódico."
            ]
        }

    if ("ano" in q or "temporal" in q or "evolução" in q or "evolucao" in q) and "Ano" in dff.columns:
        table = dff.groupby("Ano").size().reset_index(name="Quantidade")
        table["Ano_num"] = pd.to_numeric(table["Ano"], errors="coerce")
        table = table.sort_values("Ano_num")
        table = table[["Ano", "Quantidade"]]
        return {
            "answer": "Aqui está a evolução temporal dos artigos no recorte atual.",
            "table": table,
            "chart_df": table,
            "chart_type": "line",
            "insights": [
                f"Foram encontrados {len(dff)} artigos no recorte atual.",
                "Você pode cruzar a evolução temporal com tipo de artigo ou periódico."
            ]
        }

    if ("periódico" in q or "periodico" in q or "revista" in q) and "Periódico" in dff.columns:
        table = dff["Periódico"].value_counts().reset_index()
        table.columns = ["Periódico", "Quantidade"]
        return {
            "answer": "Aqui está a distribuição por periódico no recorte atual.",
            "table": table,
            "chart_df": table.head(10),
            "chart_type": "bar",
            "insights": [
                "A tabela completa mostra todos os periódicos do recorte.",
                "O gráfico exibe os 10 mais frequentes."
            ]
        }

    # fallback geral
    resumo = []
    resumo.append(f"Total de artigos no recorte: **{len(dff)}**.")

    if "Ano" in dff.columns:
        anos = pd.to_numeric(dff["Ano"], errors="coerce").dropna()
        if not anos.empty:
            resumo.append(f"Período coberto: **{int(anos.min())} a {int(anos.max())}**.")

    if "Tipo de artigo" in dff.columns:
        top_tipo = dff["Tipo de artigo"].value_counts().head(3)
        resumo.append("Tipos mais frequentes: " + ", ".join([f"**{idx}** ({val})" for idx, val in top_tipo.items()]))

    return {
        "answer": "\n\n".join(resumo),
        "table": dff.head(20),
        "chart_df": None,
        "chart_type": None,
        "insights": [
            "Perguntas numéricas mais específicas tendem a produzir gráficos mais úteis.",
            "Exemplo: 'Mostre a distribuição por ano' ou 'Quantos artigos têm parceria internacional?'."
        ]
    }


def find_relevant_csv_rows(query: str, dff: pd.DataFrame, top_n=5):
    if dff.empty:
        return dff.head(0)

    text_cols = [c for c in dff.columns if dff[c].dtype == object]
    scores = []

    q_terms = set(re.findall(r"\w+", query.lower()))

    for idx, row in dff.iterrows():
        text = " ".join([str(row[c]) for c in text_cols])
        text_low = text.lower()
        row_terms = set(re.findall(r"\w+", text_low))
        score = len(q_terms.intersection(row_terms))
        scores.append((idx, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_idx = [idx for idx, score in scores[:top_n] if score > 0]

    if not top_idx:
        return dff.head(min(top_n, len(dff)))

    return dff.loc[top_idx]


def build_context_from_chunks(chunks):
    if not chunks:
        return "Nenhum trecho relevante foi recuperado."
    context_parts = []
    for i, ch in enumerate(chunks, 1):
        context_parts.append(
            f"[Fonte {i}] PDF: {ch['source']} | Chunk: {ch['chunk_id']}\n{ch['text']}"
        )
    return "\n\n".join(context_parts)


def build_context_from_rows(rows: pd.DataFrame):
    if rows is None or rows.empty:
        return "Nenhuma linha estruturada relevante foi recuperada."
    return rows.to_csv(index=False)


def call_llm(user_query: str, semantic_context: str, structured_context: str, mode: str):
    if client is None:
        return (
            "A API da OpenAI não está configurada. Defina a variável OPENAI_API_KEY "
            "para habilitar respostas com LLM."
        )

    system_prompt = """
Você é um assistente científico especializado em analisar a produção bibliográfica do usuário.
Responda em português do Brasil.
Use apenas as informações fornecidas no contexto.
Se não houver base suficiente, diga explicitamente que não há evidência suficiente no material recuperado.
Quando apropriado, organize a resposta em:
1. Resposta objetiva
2. Síntese científica
3. Limitações ou cautelas
Não invente DOI, resultados, amostras ou conclusões.
"""

    user_prompt = f"""
Modo da consulta: {mode}

Pergunta do usuário:
{user_query}

Contexto semântico recuperado dos PDFs:
{semantic_context}

Contexto estruturado recuperado do CSV:
{structured_context}

Tarefa:
- Responda à pergunta do usuário.
- Integre os dois contextos quando for útil.
- Cite nominalmente os PDFs ou referências quando isso ajudar.
- Seja claro, técnico e conciso.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


def generate_insights_with_llm(dff: pd.DataFrame):
    if dff.empty:
        return ["A base filtrada está vazia."]

    if client is None:
        insights = [f"Total de artigos no recorte: {len(dff)}."]
        if "Ano" in dff.columns:
            anos = pd.to_numeric(dff["Ano"], errors="coerce").dropna()
            if not anos.empty:
                insights.append(f"Período coberto: {int(anos.min())}–{int(anos.max())}.")
        if "Tipo de artigo" in dff.columns:
            top_tipo = dff["Tipo de artigo"].value_counts().head(1)
            if not top_tipo.empty:
                insights.append(f"Tipo mais frequente: {top_tipo.index[0]} ({int(top_tipo.iloc[0])}).")
        return insights

    prompt = f"""
Com base neste resumo JSON da produção científica filtrada, gere 3 insights curtos em português do Brasil.
Cada insight deve ter no máximo 20 palavras.

Resumo:
{df_summary_for_prompt(dff)}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "Você gera insights científicos curtos e fiéis aos dados."},
            {"role": "user", "content": prompt},
        ],
    )
    text = resp.choices[0].message.content.strip()

    lines = [re.sub(r"^[-•\d\.\)\s]+", "", line).strip() for line in text.split("\n") if line.strip()]
    return lines[:3] if lines else ["Não foi possível gerar insights."]


def suggest_questions_with_llm(dff: pd.DataFrame):
    if dff.empty:
        return ["Quais filtros devo ajustar para recuperar mais artigos?"]

    if client is None:
        return [
            "Quais temas aparecem com mais frequência?",
            "Como os artigos se distribuem ao longo dos anos?",
            "Quais periódicos concentram mais publicações?",
        ]

    prompt = f"""
Com base neste resumo da produção científica, gere 3 perguntas úteis que um usuário poderia fazer ao chatbot.
Escreva em português do Brasil.
Cada pergunta deve ser curta e clara.

Resumo:
{df_summary_for_prompt(dff)}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        messages=[
            {"role": "system", "content": "Você sugere perguntas úteis para exploração científica."},
            {"role": "user", "content": prompt},
        ],
    )
    text = resp.choices[0].message.content.strip()
    lines = [re.sub(r"^[-•\d\.\)\s]+", "", line).strip() for line in text.split("\n") if line.strip()]
    return lines[:3] if lines else ["Quais são os principais temas da base?"]


# =========================
# SIDEBAR - ENTRADA DE DADOS
# =========================
st.sidebar.header("Dados de entrada")

csv_file = st.sidebar.file_uploader("Carregue o CSV da base", type=["csv"])
pdf_files = st.sidebar.file_uploader("Carregue os PDFs dos artigos", type=["pdf"], accept_multiple_files=True)

if not OPENAI_API_KEY:
    st.sidebar.warning("OPENAI_API_KEY não detectada. O app funciona parcialmente, mas sem respostas LLM.")

# =========================
# LEITURA E PREPARAÇÃO
# =========================
if csv_file is not None:
    try:
        df = pd.read_csv(csv_file)
        df = normalize_columns(df)
    except Exception as e:
        st.error(f"Erro ao ler o CSV: {e}")
        st.stop()
else:
    st.info("Envie o CSV e os PDFs para começar.")
    st.stop()

if pdf_files:
    documents = build_documents_from_uploaded_pdfs(pdf_files)
    chunked_docs = build_chunks(documents)
else:
    documents = []
    chunked_docs = []

model = load_embedding_model()

if chunked_docs:
    texts = [x["text"] for x in chunked_docs]
    embeddings = compute_embeddings(texts, model)
    index = build_faiss_index(embeddings)
else:
    index = None


# =========================
# FILTROS
# =========================
st.sidebar.header("Filtros")

if "Ano" in df.columns:
    year_num = pd.to_numeric(df["Ano"], errors="coerce").dropna()
    if not year_num.empty:
        min_year = int(year_num.min())
        max_year = int(year_num.max())
    else:
        min_year, max_year = 2000, 2030
else:
    min_year, max_year = 2000, 2030

year_range = st.sidebar.slider("Período", min_year, max_year, (min_year, max_year))

type_options = sorted(df["Tipo de artigo"].dropna().astype(str).unique()) if "Tipo de artigo" in df.columns else []
selected_types = st.sidebar.multiselect("Tipo de artigo", type_options)

journal_options = sorted(df["Periódico"].dropna().astype(str).unique()) if "Periódico" in df.columns else []
selected_journals = st.sidebar.multiselect("Periódico", journal_options)

mode_choice = st.sidebar.radio(
    "Modo de busca",
    ["Ambos", "Texto completo", "Metadados"],
    index=0
)

response_style = st.sidebar.radio(
    "Estilo de resposta",
    ["Equilibrada", "Técnica", "Resumida", "Quantitativa"],
    index=0
)

dff = build_filtered_dataframe(df, year_range, selected_types, selected_journals)


# =========================
# LAYOUT PRINCIPAL
# =========================
col_left, col_center, col_right = st.columns([1.1, 2.3, 1.2], gap="large")


# =========================
# COLUNA ESQUERDA - DASHBOARD RÁPIDO
# =========================
with col_left:
    st.subheader("Panorama")

    st.metric("Artigos filtrados", len(dff))

    if "Ano" in dff.columns:
        anos = pd.to_numeric(dff["Ano"], errors="coerce").dropna()
        if not anos.empty:
            st.metric("Período", f"{int(anos.min())}–{int(anos.max())}")

    if "Tipo de artigo" in dff.columns and not dff.empty:
        top_tipo = dff["Tipo de artigo"].value_counts().head(1)
        if not top_tipo.empty:
            st.metric("Tipo dominante", top_tipo.index[0])

    st.markdown("---")
    st.subheader("Resumo estruturado")

    if "Ano" in dff.columns and not dff.empty:
        chart_year = dff.groupby("Ano").size().reset_index(name="Quantidade")
        chart_year["Ano_num"] = pd.to_numeric(chart_year["Ano"], errors="coerce")
        chart_year = chart_year.sort_values("Ano_num")
        fig_year = px.bar(chart_year, x="Ano", y="Quantidade", title="Artigos por ano")
        st.plotly_chart(fig_year, use_container_width=True)

    if "Tipo de artigo" in dff.columns and not dff.empty:
        chart_type = dff["Tipo de artigo"].value_counts().reset_index()
        chart_type.columns = ["Tipo de artigo", "Quantidade"]
        fig_type = px.pie(chart_type, names="Tipo de artigo", values="Quantidade", title="Tipos de artigo")
        st.plotly_chart(fig_type, use_container_width=True)


# =========================
# ESTADO DA CONVERSA
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_insights" not in st.session_state:
    st.session_state.last_insights = []

if "last_suggestions" not in st.session_state:
    st.session_state.last_suggestions = []


# =========================
# COLUNA CENTRAL - CHAT
# =========================
with col_center:
    st.subheader("Chat com a produção científica")

    if not pdf_files:
        st.warning("Você carregou o CSV, mas ainda não carregou os PDFs. O modo RAG textual ficará indisponível.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "table" in msg and msg["table"] is not None:
                st.dataframe(msg["table"], use_container_width=True)
            if "chart_df" in msg and msg["chart_df"] is not None and msg.get("chart_type") is not None:
                if msg["chart_type"] == "bar":
                    cols = msg["chart_df"].columns.tolist()
                    if len(cols) >= 2:
                        fig = px.bar(msg["chart_df"], x=cols[0], y=cols[1])
                        st.plotly_chart(fig, use_container_width=True)
                elif msg["chart_type"] == "line":
                    cols = msg["chart_df"].columns.tolist()
                    if len(cols) >= 2:
                        fig = px.line(msg["chart_df"], x=cols[0], y=cols[1], markers=True)
                        st.plotly_chart(fig, use_container_width=True)

    user_query = st.chat_input("Pergunte algo sobre os seus artigos...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        query_mode = classify_query_heuristic(user_query)

        # Restringe o modo conforme a escolha manual
        if mode_choice == "Texto completo":
            query_mode = "semantic"
        elif mode_choice == "Metadados":
            query_mode = "structured"

        with st.chat_message("assistant"):
            with st.spinner("Analisando sua base..."):

                retrieved_chunks = []
                relevant_rows = pd.DataFrame()

                if query_mode in ["semantic", "hybrid"] and index is not None:
                    allowed_sources = None
                    retrieved_chunks = retrieve_chunks(
                        user_query,
                        model,
                        index,
                        chunked_docs,
                        k=5,
                        allowed_sources=allowed_sources
                    )

                if query_mode in ["structured", "hybrid"]:
                    relevant_rows = find_relevant_csv_rows(user_query, dff, top_n=5)

                # structured puro sem LLM
                if query_mode == "structured" and mode_choice == "Metadados":
                    result = get_structured_answer(user_query, dff)
                    st.markdown(result["answer"])

                    if result["table"] is not None:
                        st.dataframe(result["table"], use_container_width=True)

                    if result["chart_df"] is not None and result["chart_type"] is not None:
                        if result["chart_type"] == "bar":
                            cols = result["chart_df"].columns.tolist()
                            fig = px.bar(result["chart_df"], x=cols[0], y=cols[1])
                            st.plotly_chart(fig, use_container_width=True)
                        elif result["chart_type"] == "line":
                            cols = result["chart_df"].columns.tolist()
                            fig = px.line(result["chart_df"], x=cols[0], y=cols[1], markers=True)
                            st.plotly_chart(fig, use_container_width=True)

                    assistant_msg = {
                        "role": "assistant",
                        "content": result["answer"],
                        "table": result["table"],
                        "chart_df": result["chart_df"],
                        "chart_type": result["chart_type"],
                    }
                    st.session_state.messages.append(assistant_msg)
                    st.session_state.last_sources = []
                    st.session_state.last_insights = result["insights"]
                    st.session_state.last_suggestions = suggest_questions_with_llm(dff)

                else:
                    semantic_context = build_context_from_chunks(retrieved_chunks)
                    structured_context = build_context_from_rows(relevant_rows)

                    llm_answer = call_llm(
                        user_query=user_query,
                        semantic_context=semantic_context,
                        structured_context=structured_context,
                        mode=query_mode
                    )

                    # ajuste simples por estilo
                    if response_style == "Resumida":
                        llm_answer = llm_answer[:1200]
                    elif response_style == "Quantitativa":
                        llm_answer = "Foque principalmente nos padrões quantitativos.\n\n" + llm_answer

                    st.markdown(llm_answer)

                    if not relevant_rows.empty:
                        with st.expander("Linhas estruturadas relevantes do CSV"):
                            st.dataframe(relevant_rows, use_container_width=True)

                    if retrieved_chunks:
                        with st.expander("Trechos recuperados dos PDFs"):
                            for ch in retrieved_chunks:
                                st.markdown(f"**{ch['source']}** — chunk {ch['chunk_id']}")
                                st.write(ch["text"][:1000] + ("..." if len(ch["text"]) > 1000 else ""))
                                st.markdown("---")

                    assistant_msg = {
                        "role": "assistant",
                        "content": llm_answer,
                        "table": None,
                        "chart_df": None,
                        "chart_type": None,
                    }
                    st.session_state.messages.append(assistant_msg)

                    st.session_state.last_sources = [ch["source"] for ch in retrieved_chunks]
                    st.session_state.last_insights = generate_insights_with_llm(dff)
                    st.session_state.last_suggestions = suggest_questions_with_llm(dff)


# =========================
# COLUNA DIREITA - FONTES E INSIGHTS
# =========================
with col_right:
    st.subheader("Fontes")

    if st.session_state.last_sources:
        unique_sources = []
        seen_sources = set()
        for src in st.session_state.last_sources:
            if src not in seen_sources:
                unique_sources.append(src)
                seen_sources.add(src)

        for src in unique_sources:
            st.markdown(f"- {src}")
    else:
        st.write("As fontes aparecerão aqui após a primeira consulta.")

    st.markdown("---")
    st.subheader("Insights")

    if st.session_state.last_insights:
        for insight in st.session_state.last_insights:
            st.markdown(f"- {insight}")
    else:
        st.write("Os insights serão gerados após a primeira consulta.")

    st.markdown("---")
    st.subheader("Perguntas sugeridas")

    if st.session_state.last_suggestions:
        for sug in st.session_state.last_suggestions:
            if st.button(sug, key=f"sug_{sug}"):
                st.session_state.messages.append({"role": "user", "content": sug})
                st.rerun()
    else:
        st.write("As sugestões aparecerão após a primeira análise.")

    st.markdown("---")
    st.subheader("Base filtrada")
    st.dataframe(dff.head(10), use_container_width=True)
