-- Runs once, on first start of the Postgres volume: a separate database for pytest.
CREATE DATABASE rag_test OWNER rag;
