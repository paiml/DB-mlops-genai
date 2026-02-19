# Lab: Delta Tables

Create and manage Delta Lake tables: INSERT, UPDATE, MERGE operations, time travel queries, and schema enforcement.

## Objectives

- Create Delta tables from DataFrames
- Perform INSERT, UPDATE, and MERGE (upsert) operations
- Use time travel to query historical versions
- Understand schema enforcement and evolution

## Lab Exercise

See [`labs/course1/week3/lab_delta.py`](https://github.com/paiml/DB-mlops-genai/tree/main/labs/course1/week3)

## Key Tasks

1. **Create table** — Build an inventory Delta table with 6+ products
2. **INSERT** — Append new products
3. **UPDATE** — Modify prices for a category
4. **MERGE** — Upsert with matched updates and unmatched inserts
5. **Time travel** — View history, query version 0, compare price changes
6. **Schema enforcement** — Verify that mismatched schemas are rejected

## Key SQL Patterns

```sql
-- MERGE pattern
MERGE INTO target USING source ON target.key = source.key
WHEN MATCHED THEN UPDATE SET ...
WHEN NOT MATCHED THEN INSERT ...

-- Time travel
SELECT * FROM table VERSION AS OF 0

-- History
DESCRIBE HISTORY table
```

## Validation

The lab includes a `validate_lab()` function that checks:
- Delta table exists
- Table has at least 6 rows
- Multiple versions exist (DML operations were performed)
