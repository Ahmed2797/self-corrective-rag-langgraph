# Self-Corrective Agentic RAG langgraph

An enterprise-grade, evaluation-driven **Self-Corrective RAG (Retrieval-Augmented Generation)** engine built with **LangGraph**, **Pinecone**, and **OpenAI**[cite: 1].

The system goes beyond standard single-pass RAG by dynamically evaluating its own outputs, self-correcting hallucinations, rewriting queries when necessary, and running an automated evaluation suite before returning answers[cite: 1].

---

## 🌟 Key Features

* **Adaptive Retrieval (`decide_retrieval`):** Smartly routes queries to determine whether vector search is required or direct LLM generation is sufficient[cite: 1].
* **Sentence-Level Chunk Refining:** Deconstructs documents into granular sentence units to filter out noise and retain only relevant context.
* **Hallucination Mitigation (`IsSUP` & `revise_answer`):** Checks if generated answers are strictly grounded in context and automatically revises hallucinated claims[cite: 1].
* **Query Reformulation (`rewrite_question`):** Rewrites search queries and re-executes retrieval loops if initial answers fail usefulness checks[cite: 1].
* **Evaluation Suite (`evaluate_answer`):** Scores final answers across **Groundedness**, **Hallucination**, and **Sycophancy** metrics to guarantee a reliability threshold[cite: 1].

---

## 🏗️ Architecture Flow

                         ┌──────────────────┐
                         │    User Query    │
                         └────────┬─────────┘
                                  │
                                  ▼
                  ┌──────────────────────────────┐
                  │      Semantic Cache Search   │
                  │       OpenAI Embeddings      │
                  │          + Pinecone          │
                  └──────────────┬───────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
             similarity ≥ 0.90          similarity < 0.90
                    │                         │
                    ▼                         ▼
          ┌─────────────────┐       ┌────────────────────┐
          │   Cache HIT     │       │ decide_retrieval   │
          │                 │       └─────────┬──────────┘
          │ Cached Answer   │                 │
          │ + Evidence      │                 ▼
          │ + Evaluation    │            Need Retrieval?
          └────────┬────────┘              /          \
                   │                    NO              YES
                   │                     │                │
                   │                     ▼                ▼
                   │              generate_direct      Retrieve
                   │                     │                │
                   │                     │                ▼
                   │                     │          is_relevant
                   │                     │             /      \
                   │                     │          YES        NO
                   │                     │           │           │
                   │                     │           ▼           ▼
                   │                     │  generate_from_    rewrite_question
                   │                     │     context           │
                   │                     │                       │
                   │                     │                       ▼
                   │                     │                    Retrieve
                   │                     │                       │
                   │                     │                       ▼
                   │                     │                  is_relevant
                   │                     │
                   │                     └──────────┐
                   │                                │
                   │                                ▼
                   │                         ┌─────────────┐
                   │                         │    is_sup   │
                   │                         └──────┬──────┘
                   │                                │
                   │                    ┌───────────┴───────────┐
                   │                    │                       │
                   │               fully_supported        not supported
                   │                    │                       │
                   │                    ▼                       ▼
                   │                 is_use              revise_answer
                   │                    │                       │
                   │              ┌─────┴─────┐                 │
                   │              │           │                 │
                   │           useful     not_useful            │
                   │              │           │                 │
                   │              ▼           ▼                 │
                   │       evaluate_answer  rewrite_question ◄──┘
                   │              │
                   │              ▼
                   │       ┌─────────────────┐
                   │       │ Quality Checks  │
                   │       │                 │
                   │       │ Hallucination   │
                   │       │ Groundedness    │
                   │       │ Sycophancy      │
                   │       │ Overall Quality │
                   │       └────────┬────────┘
                   │                │
                   │                ▼
                   │       save_semantic_cache
                   │                │
                   └────────────────┘
                                    │
                                    ▼
                                  END


             ┌──────────────────────┐
             │   no_answer_found    │
             │                      │
             │ Max retries reached │
             │ / no relevant docs  │
             └──────────┬───────────┘
                        │
                        ▼
                       END

## 🛠️ Tech Stack

* **Orchestration:** LangGraph / LangChain[cite: 1]
* **LLM & Embeddings:** OpenAI (`text-embedding-3-small`, `gpt-4o-mini` / `gpt-4o`)
* **Vector Database:** Pinecone / FAISS
* **Language:** Python 3.10+

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup

```bash
## Clone the repository and install the dependencies
git clone https://github.com/Ahmed2797/self-corrective-rag-langgraph.git
cd self-corrective-rag-langgraph

pip install -r requirements.txt

## .env

PINECONE_API_KEY = ""
OPENAI_API_KEY = "sk-proj-----"
GROQ_API_KEY = ""
TAVILY_API_KEY = "tvly-de"
SERPER_API_KEY = ""

AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_ECR_LOGIN_URI = "***********.us-east-1.amazonaws.com"
ECR_REPOSITORY_NAME = ''
AWS_REGION = "us-east-1"
AWS_DEFAULT_REGION = "us-east-1"
BUCKET_NAME = ""
```


## .env

``` bash

| Secret Name           | Example Value            |
| --------------------- | ------------------------ |
| AWS_ACCESS_KEY_ID     | AKIA...                  |
| AWS_SECRET_ACCESS_KEY | xxxxxxxxx                |
| AWS_DEFAULT_REGION    | eu-central-1             |
| ECR_REPO              | your-ecr-repository-name |
| PINECONE_API_KEY      | your-pinecone-key        |
| OPENAI_API_KEY        | your-openai-key          |


export AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_ACCESS_KEY"

```

## AWS-CICD-Deployment-with-Github-Actions

### 1. Login to AWS console

### 2. Create IAM user for deployment

``` bash
# with specific access
1. EC2 access : It is virtual machine

2. ECR: Elastic Container registry to save your docker image in aws

#Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2 

4. Pull Your image from ECR in EC2

5. Lauch your docker image in EC2

#Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess
```

### 3. Create ECR repo to store/save docker image

- Save the URI: 520551197421.dkr.ecr.us-east-1.amazonaws.com/self-rag

### 4. Create EC2 machine (Ubuntu)

### 5. Open EC2 and Install docker in EC2 Machine

``` bash

#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

```

### 6. Setup github secrets

```bash

AWS_ACCESS_KEY_ID =

AWS_SECRET_ACCESS_KEY =

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = ''

ECR_REPOSITORY_NAME = Rag-AI
```

