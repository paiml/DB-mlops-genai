# Lab: Spark Operations

Practice core Spark DataFrame operations: select, filter, groupBy, aggregations, and joins using sales data.

## Objectives

- Use select() to choose and transform columns
- Use filter() to select rows by condition
- Use groupBy() with aggregation functions (sum, avg, count, max)
- Perform inner and left joins between DataFrames
- Write equivalent SQL queries

## Lab Exercise

See [`labs/course1/week2/lab_spark.py`](https://github.com/paiml/DB-mlops-genai/tree/main/labs/course1/week2)

## Key Tasks

1. **Select** — Create derived columns (total_revenue, discounted_price)
2. **Filter** — Find rows by price, category, region, and date range
3. **GroupBy** — Compute revenue by category, average price by region, max price per category
4. **Join** — Combine sales with region lookup, then aggregate by territory
5. **SQL** — Register DataFrames as views and write equivalent SQL queries

## Validation

The lab includes a `validate_lab()` function that checks:
- Sales data loaded (10 rows)
- Select returns correct number of columns
- Filter returns non-empty results
- GroupBy produces correct number of groups
- Join produces correct row count
