import json
import uuid
from langchain_core.prompts import ChatPromptTemplate
from src.evalution.evaluation_engine import evaluate_answer
from src.vector import create_embedding, create_pincone_database
from src.constants import CACHE_THRESHOLD, CACHE_VERSION, INDEX_NAME_CACHE_MEMORY,SIMILARITY_THRESHOLD_FIASS,SIMILARITY_THRESHOLD_PINECONE
from src.exception import CustomException
from src.logger import logging
from src.mcp import tavily_mcp_search
import src.mcp as mcp
from langchain_core.messages import ToolMessage

from src.chat_model import get_llm
from src.state import *

llm = get_llm()


# 01. Define the output structure using Pydantic[cite: 1, 4]
# ========================================================
# OptimizedQuery Decision
# ========================================================
query_optimizer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a query optimization module for a RAG retrieval system.
Your task is to optimize the user's question for document retrieval.

Goals:
1. Remove unnecessary conversational words.
2. Preserve the original meaning exactly.
3. Extract important keywords.
4. Extract important named entities.
5. Create a concise retrieval-friendly query.

Rules:
- Do NOT answer the question.
- Do NOT add information that is not present in the question.
""",
        ),
        (
            "human",
            "Question:\n{question}",
        ),
    ]
)

optimizer_llm = llm.with_structured_output(OptimizedQueryOutput)



def optimizer_retrieval_node(state: State) -> State:
    """
    Optimize the user question for retrieval.

    Args:
        state (State): Current LangGraph state.

    Returns:
        State: Updated state containing optimized query,
        keywords, and entities.
    """
    try:
        question = state.get("question", "").strip()

        if not question:
            return {
                "optimized_query": "",
                "keywords": [],
                "entities": []
            }

        result: OptimizedQueryOutput = optimizer_llm.invoke(
            query_optimizer_prompt.format_messages(
                question=question
            )
        )

        optimized_query = result.optimized_query.strip()

        if not optimized_query:
            optimized_query = question

        logging.info("\n========== QUERY OPTIMIZER ==========")
        logging.info("Original:", question)
        logging.info("Optimized:", optimized_query)
        logging.info("Keywords:", result.keywords)
        logging.info("Entities:", result.entities)

        return {
            "optimized_query": optimized_query,
            "keywords": result.keywords,
            "entities": result.entities
        }

    except Exception as e:
        logging.info(f"Query optimization failed: {e}")

        return {
            "optimized_query": state.get("question", ""),
            "keywords": [],
            "entities": []
        }

# -----------------------------
# 1) Decide retrieval
# -----------------------------

decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether retrieval is needed.\n"
            "Return JSON with key: should_retrieve (boolean).\n\n"
            "Guidelines:\n"
            "- should_retrieve=True if answering requires specific facts from company documents.\n"
            "- should_retrieve=False for general explanations/definitions.\n"
            "- If unsure, choose True."
        ),
        ("human", "Question: {question}"),
    ]
)

should_retrieve_llm = llm.with_structured_output(RetrieveDecision)


# -----------------------------
# 2) Direct answer (no retrieval)
# -----------------------------
direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Answer the user's question directly.

You have access to a web search tool called Tavily.

Use Tavily when:
- The question requires current or up-to-date information.
- The user asks about recent events.
- The user asks for current prices, news, weather, companies,
  products, regulations, or other information that may have changed.
- You are not confident that your internal knowledge is sufficient.

Do NOT use Tavily for:
- Simple general knowledge.
- Basic explanations.
- Mathematics.
- Programming concepts that do not require current information.

If you use Tavily, use the retrieved information to produce
the final answer.
""",
        ),
        (
            "human",
            "Question: {question}",
        ),
    ]
)


async def get_tool_llm():
    await mcp.initialize_mcp()

    return llm.bind_tools(
        [mcp.search_tool]
    )

async def generate_with_tools(state: State):
    try:
        tool_llm = await get_tool_llm()

        messages = direct_generation_prompt.format_messages(
            question=state["question"]
        )

        response = await tool_llm.ainvoke(messages)

        if not response.tool_calls:
            return {
                "answer": response.content
            }

        messages.append(response)

        for tool_call in response.tool_calls:

            if tool_call["name"] == "tavily_search":

                result = await mcp.search_tool.ainvoke(
                    tool_call["args"]
                )

                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    )
                )

        final_response = await tool_llm.ainvoke(messages)

        return {
            "answer": final_response.content
        }

    except Exception as e:
        logging.error(
            f"generate_with_tools failed: {e}"
        )
        raise


def decide_retrieval(state: State):
    """
    Decide whether the user question requires document retrieval.

    Args:
        state (State): Current LangGraph state.

    Returns:
        dict: Retrieval decision.

    Raises:
        CustomException: If the LLM decision fails.
    """
    try:
        logging.info("Starting retrieval decision.")

        question = state["question"]

        decision: RetrieveDecision = should_retrieve_llm.invoke(decide_retrieval_prompt.format_messages(question=question))

        logging.info(f"Retrieval decision: {decision.should_retrieve}")

        return {"need_retrieval": decision.should_retrieve}

    except Exception as e:
        logging.error(f"Error in decide_retrieval: {str(e)}")
        raise CustomException(e)

def generate_direct(state: State):
    """
    Generate an answer without document retrieval.

    Args:
        state (State): Current LangGraph state.

    Returns:
        dict: Generated answer.

    Raises:
        CustomException: If answer generation fails.
    """
    try:
        logging.info("Generating direct answer.")

        question = state["question"]

        out = llm.invoke(direct_generation_prompt.format_messages(question=question))

        logging.info("Direct answer generated successfully.")

        return {"answer": out.content}

    except Exception as e:
        logging.error(f"Error in generate_direct: {str(e)}")
        raise CustomException(e)

def route_after_decide(state: State) -> Literal["generate_with_tools", "retrieve"]:
    return "retrieve" if state["need_retrieval"] else "generate_with_tools"


# -----------------------------
# 3) Retrieve
# -----------------------------
# def retrieve(state: State):
#     q = state.get("retrieval_query") or state["question"]
#     return {"docs": retriever.invoke(q)}


# -----------------------------
# 4) Relevance filter (strict)
# -----------------------------

is_relevant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging document relevance at a TOPIC level.\n"
            "Return JSON matching the schema.\n\n"
            "A document is relevant if it discusses the same entity or topic area as the question.\n"
            "It does NOT need to contain the exact answer.\n\n"
            "Examples:\n"
            "- HR policies are relevant to questions about notice period, probation, termination, benefits.\n"
            "- Pricing documents are relevant to questions about refunds, trials, billing terms.\n"
            "- Company profile is relevant to questions about leadership, culture, size, or strategy.\n\n"
            "Do NOT decide whether the document fully answers the question.\n"
            "That will be checked later by IsSUP.\n"
            "When unsure, return is_relevant=true."
        ),
        ("human", "Question:\n{question}\n\nDocument:\n{document}"),
    ]
)


relevance_llm = llm.with_structured_output(RelevanceDecision)


def is_relevant(state: State):
    """
    Filter retrieved documents based on topic-level relevance.

    Args:
        state (State): Current LangGraph state.

    Returns:
        dict: Relevant documents.

    Raises:
        CustomException: If relevance evaluation fails.
    """
    try:
        logging.info("Starting document relevance evaluation.")

        relevant_docs: List[Document] = []

        docs = state.get("docs", [])

        logging.info(f"Evaluating {len(docs)} retrieved documents.")

        for doc in docs:
            decision: RelevanceDecision = relevance_llm.invoke(is_relevant_prompt.format_messages(question=state["question"], document=doc.page_content))

            if decision.is_relevant:
                relevant_docs.append(doc)

        logging.info(f"Relevant documents: {len(relevant_docs)}/{len(docs)}")

        return {"relevant_docs": relevant_docs}

    except Exception as e:
        logging.error(f"Error in is_relevant: {str(e)}")
        raise CustomException(e)

# def route_after_relevance(state: State) -> Literal["generate_from_context", "no_answer_found"]:
#     if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
#         return "generate_from_context"
#     return "no_answer_found"


# -----------------------------
# 5) Generate from context
# -----------------------------
rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a business rag chatbot.\n\n"
            "You will receive a CONTEXT block from internal company documents.\n"
            "Task:\n"
            "Answer the question based on the context"
            "Dont mention that you are getting a context in your answer"
        ),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ]
)

def generate_from_context(state: State):
    """
    Generate an answer using retrieved relevant documents.

    Args:
        state (State): Current LangGraph state.

    Returns:
        dict: Generated answer and context.

    Raises:
        CustomException: If context generation fails.
    """
    try:
        logging.info("Generating answer from retrieved context.")

        context = "\n\n---\n\n".join([d.page_content for d in state.get("relevant_docs", [])]).strip()

        if not context:
            logging.warning("No relevant context available.")

            return {"answer": "No answer found.", "context": ""}

        out = llm.invoke(rag_generation_prompt.format_messages(question=state["question"], context=context))

        logging.info("RAG answer generated successfully.")

        return {"answer": out.content, "context": context}

    except Exception as e:
        logging.error(f"Error in generate_from_context: {str(e)}")
        raise CustomException(e)
    

def no_answer_found(state: State):
    return {"answer": "No answer found.", "context": ""}


def check_retrieval_score(state: State) -> State:
    """
    Determine whether retrieved documents are sufficiently similar
    to the query based on their retrieval scores.

    Args:
        state (State): Current LangGraph state.

    Returns:
        State: Updated state containing relevant documents.
    """
    try:
        docs = state.get("docs", [])
        scores = state.get("retrieval_scores", [])

        relevant_docs = []

        for doc, score in zip(docs, scores):
            if score <= SIMILARITY_THRESHOLD_FIASS: ## FIASS
                relevant_docs.append(doc)
            # if score >= SIMILARITY_THRESHOLD_PINECONE: ## Pincone
            #     relevant_docs.append(doc)

        logging.info(
            f"Relevant documents: {len(relevant_docs)}/{len(docs)}"
        )

        return {
            "relevant_docs": relevant_docs
        }

    except Exception as e:
        logging.error(f"Retrieval score check failed: {str(e)}")
        raise CustomException(e)


def route_after_relevance(state: State) -> Literal["generate_from_context", "no_answer_found"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "no_answer_found"
    # return "query_optimizer"


# -----------------------------
# 6) IsSUP Decision
# -----------------------------

issup_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict RAG evidence verifier.

Your ONLY task is to determine whether the ANSWER is supported
by the provided CONTEXT.

Return JSON:

{{
    "issup": "fully_supported" | "partially_supported" | "no_support",
    "evidence": [...]
}}

Definitions:

1. fully_supported
- Every meaningful claim in the ANSWER is explicitly supported by CONTEXT.
- The answer does not introduce unsupported interpretations,
  abstractions, or qualitative claims.
- The exact relationship expressed in the answer must be supported.

2. partially_supported
- The core answer is supported, but one or more meaningful claims
  are only partially supported, inferred, or expressed with
  unsupported interpretation.

3. no_support
- The main claim of the ANSWER is not supported by CONTEXT.

STRICT RULES:

- Use ONLY the provided CONTEXT.
- Do NOT use outside knowledge.
- Do NOT infer missing relationships.
- Do NOT transform one relationship into a different relationship.
- A claim must be supported by the same entity/attribute/relationship
  expressed in the ANSWER.
- Matching words, names, or concepts alone are NOT sufficient evidence.
- Evidence must directly support the claim it is being used for.
- If a claim requires an unstated inference, it is not fully supported.
- If the answer contains multiple claims, evaluate each meaningful
  claim separately.
- If all meaningful claims are explicitly supported, use fully_supported.
- If some claims are supported but others require inference or are
  only partially supported, use partially_supported.
- If the main claim is unsupported, use no_support.
- Do NOT mark an answer as fully_supported merely because the context
  contains related information.
- Evidence should contain up to 3 short direct quotes from CONTEXT.
- Do not use outside knowledge.

Question:
{question}

Answer:
{answer}

Context:
{context}
""",
        ),
    ]
)


issup_llm = llm.with_structured_output(IsSUPDecision)


def is_sup(state: State) -> State:

    logging.info("\n========== IsSUP ==========")

    relevant_docs = state.get("relevant_docs", [])

    context_chunks = []

    for doc in relevant_docs:
        context_chunks.append(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_chunks)

    logging.info("Question:", state["question"])
    logging.info("Answer:", state.get("answer", ""))

    decision: IsSUPDecision = issup_llm.invoke(
        issup_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=context,
        )
    )

    logging.info("IsSUP:", decision.issup)
    logging.info("Evidence:")

    for evidence in decision.evidence:
        logging.info(" -", evidence)

    return {
        "issup": decision.issup,
        "evidence": decision.evidence,
    }


# -----------------------------
# 7) Revise Decision
# -----------------------------

revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a STRICT reviser.\n\n"
            "You must output based on the following format:\n\n"
            "FORMAT (quote-only answer):\n"
            "- <direct quote from the CONTEXT>\n"
            "- <direct quote from the CONTEXT>\n\n"
            "Rules:\n"
            "- Use ONLY the CONTEXT.\n"
            "- Do NOT add any new words besides bullet dashes and the quotes themselves.\n"
            "- Do NOT explain anything.\n"
            "- Do NOT say 'context', 'not mentioned', 'does not mention', 'not provided', etc.\n"
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Current Answer:\n{answer}\n\n"
            "CONTEXT:\n{context}"
        ),
    ]
)

def revise_answer(state: State):
    out = llm.invoke(
        revise_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {
        "answer": out.content,
        "retries": state.get("retries", 0) + 1,  # ✅ increment
    }


MAX_RETRIES = 3

def accept_answer(state: State):
    return {}  # keep answer as-is

def route_after_issup(state: State,) -> Literal["accept_answer","revise_answer","no_answer_found"]:

    logging.info("\n========== ROUTE AFTER ISSUP ==========")

    issup = state.get("issup")
    retries = state.get("retries", 0)

    logging.info("IsSUP:", issup)
    logging.info("Retries:", retries)

    if issup == "fully_supported":
        logging.info("➡️ ACCEPT ANSWER")
        return "accept_answer"

    # if issup == "partially_supported":
    #     logging.info("➡️ PARTIALLY SUPPORTED → REVISE ANSWER")
    #     return "revise_answer"
    # if issup == "no_support":
    #     logging.info("➡️ NO SUPPORT → REVISE ANSWER")
    #     return "web_search_answer"

    if retries >= MAX_RETRIES:
        logging.info("➡️ MAX RETRIES → NO ANSWER")
        return "no_answer_found"

    logging.info("➡️ REVISE ANSWER")
    return "revise_answer"


# -----------------------------
# 8) IsUSE Decision
# -----------------------------

isuse_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an evaluator that judges the USEFULNESS of an AI answer
for the user's question.

Your ONLY task is to determine whether the answer actually addresses
the user's question.

Do NOT evaluate whether the answer is factually correct or supported
by the context. That is handled by other evaluation nodes.

Return:
- isuse: "useful" or "not_useful"
- reason: one short sentence

Rules:

1. useful:
   - The answer directly answers the user's question.
   - OR it provides the specific information requested by the user.
   - The answer can be concise and still be useful.

2. not_useful:
   - The answer does not answer the question.
   - The answer is generic or vague.
   - The answer is off-topic.
   - The answer only provides related background without addressing
     what the user actually asked.

3. Do NOT use outside knowledge.

4. Do NOT evaluate:
   - hallucination
   - factual correctness
   - evidence grounding
   - context support

5. Only ask:
   "Does this answer actually satisfy the user's request?"

6. A short answer can still be useful if it directly answers
   the question.

7. If the question is ambiguous and the answer appropriately
   asks for clarification, consider it useful.

Keep the reason to ONE short sentence.
""",
        ),
        (
            "human",
            """
Question:
{question}

Answer:
{answer}
""",
        ),
    ]
)
isuse_llm = llm.with_structured_output(IsUSEDecision)

def is_use(state: State) -> State:

    logging.info("\n========== IsUSE ==========")

    question = state.get("question", "")
    answer = state.get("answer", "")

    decision: IsUSEDecision = isuse_llm.invoke(
        isuse_prompt.format_messages(
            question=question,
            answer=answer,
        )
    )

    logging.info("Question:", question)
    logging.info("Answer:", answer)
    logging.info("IsUSE:", decision.isuse)
    logging.info("Reason:", decision.use_reason)

    return {
        "isuse": decision.isuse,
        "use_reason": decision.use_reason,
    }

# -----------------------------
# 9) Rewrite Decision
# -----------------------------

rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's QUESTION into a query optimized for vector retrieval over INTERNAL company PDFs.\n\n"
            "Rules:\n"
            "- Keep it short (6–16 words).\n"
            "- Preserve key entities (e.g., NexaAI, plan names).\n"
            "- Add 2–5 high-signal keywords that likely appear in policy/pricing docs.\n"
            "- Remove filler words.\n"
            "- Do NOT answer the question.\n"
            "- Output JSON with key: retrieval_query\n\n"
            "Examples:\n"
            "Q: 'Do NexaAI plans include a free trial?'\n"
            "-> {{'retrieval_query': 'NexaAI free trial duration trial period plans'}}\n\n"
            "Q: 'What is NexaAI refund policy?'\n"
            "-> {{'retrieval_query': 'NexaAI refund policy cancellation refund timeline charges'}}"
        ),
        (
            "human",
            "QUESTION:\n{question}\n\n"
            "Previous retrieval query:\n{retrieval_query}\n\n"
            "Answer (if any):\n{answer}"
        ),
    ]
)



rewrite_llm = llm.with_structured_output(RewriteDecision)

def rewrite_question(state: State):
    decision: RewriteDecision = rewrite_llm.invoke(
        rewrite_for_retrieval_prompt.format_messages(
            question=state["question"],
            retrieval_query=state.get("retrieval_query", ""),
            answer=state.get("answer", ""),
        )
    )

    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        # ✅ optional: reset these so next pass is clean
        "docs": [],
        "relevant_docs": [],
        "context": "",
    }

# -----------------------------
# 10) Evaluation Decision
# -----------------------------

MAX_REWRITE_TRIES = 3

def route_after_isuse(state: State,) -> Literal["evaluate_answer","rewrite_question","no_answer_found",]:

    isuse = state.get("isuse")
    rewrite_tries = state.get("rewrite_tries", MAX_REWRITE_TRIES)

    logging.info("\n========== ROUTE AFTER ISUSE ==========")
    logging.info("IsUSE:", isuse)
    logging.info("Rewrite tries:", rewrite_tries)

    if isuse == "useful":
        logging.info("➡️ EVALUATE ANSWER")
        # return "END"
        return "evaluate_answer"

    if rewrite_tries >= MAX_REWRITE_TRIES:
        logging.info("➡️ NO ANSWER FOUND")
        return "no_answer_found"

    logging.info("➡️ REWRITE QUESTION")
    return "rewrite_question"



def evaluate_answer_node(state: State) -> State:
    """
    Evaluate the generated answer using the evaluation engine.

    Args:
        state (State): Current LangGraph state.

    Returns:
        State: Updated state containing the evaluation result.

    Raises:
        CustomException: If evaluation fails.
    """
    try:
        logging.info("========== EVALUATE ANSWER ==========")

        relevant_docs = state.get("relevant_docs", [])

        logging.info(f"Question: {state.get('question')}")
        logging.info(f"Answer: {state.get('answer')}")
        logging.info(f"Relevant docs: {len(relevant_docs)}")

        context_chunks = []

        for doc in relevant_docs:
            context_chunks.append({
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "content": doc.page_content,
            })

        logging.info("========== EVALUATION CONTEXT ==========")

        for i, chunk in enumerate(context_chunks, 1):
            logging.info(f"--- DOC {i} ---")
            logging.info(f"Source: {chunk['source']}")
            logging.info(f"Page: {chunk['page']}")
            logging.info(f"Chunk ID: {chunk['chunk_id']}")

        evaluation = evaluate_answer(
            question=state["question"],
            answer=state.get("answer", ""),
            context_chunks=context_chunks,
        )

        logging.info("========== FINAL EVALUATION ==========")
        logging.info("Hallucination Evaluation...")
        logging.info(f"Hallucination Score: {evaluation.get('hallucination', {}).get('score')}")
        logging.info(f"Groundedness Score: {evaluation.get('groundedness', {}).get('score')}")
        logging.info(f"Sycophancy Score: {evaluation.get('sycophancy', {}).get('score')}")
        logging.info(f"Overall Quality: {evaluation.get('overall_quality')}")
        logging.info(f"Reliable: {evaluation.get('is_reliable')}")
        logging.info(f"Status: {evaluation.get('evaluation_status')}")
        logging.info("========== EVALUATION ==========")

        return {
            "evaluation": evaluation
        }

    except Exception as e:
        logging.error(f"Error during answer evaluation: {str(e)}")
        raise CustomException(e)

# -----------------------------
# 01) Memory Cache Decision
# -----------------------------

def search_semantic_cache(question: str):
    """
    Search the semantic cache for a similar previous question.

    Args:
        question (str): User question.

    Returns:
        dict | None: Cached answer information if a valid match exists.

    Raises:
        CustomException: If cache search fails.
    """
    try:
        logging.info("Searching semantic cache.")

        query_embedding = create_embedding(question)

        cache_index = create_pincone_database(
            index_name=INDEX_NAME_CACHE_MEMORY
        )

        result = cache_index.query(
            vector=query_embedding,
            top_k=1,
            include_metadata=True,
        )
        logging.info(f"Semantic cache matching records: {result}")

        if not result.matches:
            logging.info("Semantic cache miss: no matching records.")
            return None

        match = result.matches[0]
        similarity = match.score

        logging.info(f"Cache similarity: {similarity:.4f}")

        if similarity < CACHE_THRESHOLD:
            logging.info(
                "Semantic cache miss: similarity below threshold."
            )
            return None

        metadata: CacheMetadata = match.metadata

        if metadata.get("cache_version") != CACHE_VERSION:
            logging.info(
                "Semantic cache miss: cache version mismatch."
            )
            return None

        logging.info("Semantic cache hit.")

        return {
            "answer": metadata.get("answer", ""),
            "question": metadata.get("question", ""),
            "evaluation": json.loads(
                metadata.get("evaluation", "{}")
            ),
            "similarity": similarity,
        }

    except Exception as e:
        logging.error(
            f"Error searching semantic cache: {str(e)}"
        )
        raise CustomException(e)

# -----------------------------
# 11) Memory Cache Decision
# -----------------------------
def save_semantic_cache(
    question: str,
    answer: str,
    evaluation: EvaluationResult | None = None,
):
    """
    Save a question-answer pair to the semantic cache.

    Args:
        question (str): User question.
        answer (str): Generated answer.
        evaluation (EvaluationResult | None): Final evaluation result.

    Raises:
        CustomException: If cache saving fails.
    """
    try:
        logging.info("Saving answer to semantic cache.")

        embedding = create_embedding(question)

        cache_index = create_pincone_database(
            index_name=INDEX_NAME_CACHE_MEMORY
        )

        metadata: CacheMetadata = {
            "question": question,
            "answer": answer,
            "evaluation": json.dumps(evaluation or {}),
            "cache_version": CACHE_VERSION,
        }

        cache_index.upsert(
            vectors=[
                {
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": metadata,
                }
            ]
        )

        logging.info("Answer saved to semantic cache successfully.")

    except Exception as e:
        logging.error(f"Error saving semantic cache: {str(e)}")
        raise CustomException(e)



def check_semantic_cache_node(state: State):
    """
    Check whether a semantically similar answer exists in cache.
    """

    try:
        question = state["question"]

        logging.info("=" * 50)
        logging.info("SEMANTIC CACHE CHECK")
        logging.info("=" * 50)
        logging.info(f"Question: {question}")

        cached = search_semantic_cache(question)

        if cached is None:
            logging.info("CACHE MISS")
            logging.info("Similarity: below threshold")

            return {
                "cache_hit": False,
                "cache_similarity": 0.0,
            }

        logging.info(
            f"Semantic cache HIT | "
            f"similarity={cached['similarity']:.3f} | "
            f"cached_question={cached['question']}"
        )

        return {
            "cache_hit": True,
            "cache_similarity": cached["similarity"],
            "answer": cached["answer"],
            "evaluation": cached["evaluation"],
        }

    except Exception as e:
        logging.error(
            f"Error checking semantic cache: {str(e)}"
        )
        raise CustomException(e)

# -----------------------------
# 11) Memory Cache Decision Node
# -----------------------------

def save_semantic_cache_node(state: State) -> State:
    """
    Save the final answer and evaluation to semantic cache.
    """
    save_semantic_cache(
        question=state["question"],
        answer=state.get("answer", ""),
        evaluation=state.get("evaluation"),
    )

    return {}


def route_after_cache(state: State):

    if state.get("cache_hit"):

        return "cached_answer"

    return "continue_rag"


async def web_search_node(state: State):

    query = (
        state.get("retrieval_query")
        or state["question"]
    )

    result = await tavily_mcp_search(query)

    return {
        "web_results": result
    }
