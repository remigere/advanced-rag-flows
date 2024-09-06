FROM langchain/langgraph-api:3.12

RUN apt-get update && apt-get install -y g++ build-essential

ADD . /deps/advanced-rag-flows

RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*

ENV LANGSERVE_GRAPHS='{"agent": "/deps/advanced-rag-flows/graph/graph.py:app"}'

WORKDIR /deps/advanced-rag-flows