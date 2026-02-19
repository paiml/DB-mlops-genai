# CLAUDE.md

This repository contains course materials for the **Databricks Specialization on Coursera** (Courses 1, 3 & 4). Course 1 covers Lakehouse Fundamentals (Databricks-only). Courses 3 & 4 cover MLOps and GenAI Engineering with a dual-layer pedagogy (Databricks + Sovereign AI Stack).

## Build Commands

```bash
make check      # Run all quality checks (lint + test)
make lint       # Lint Python and Rust code
make format     # Auto-format code
make test       # Run test suite
make test-fast  # Quick syntax validation
make demos      # Run demo validation
```

## Repository Structure

```
demos/
  course1/          # Lakehouse Fundamentals demos (weeks 1-3, Databricks-only)
  course3/          # MLOps Engineering demos (weeks 1-3)
  course4/          # GenAI Engineering demos (weeks 1-3)
labs/
  course1/          # Lakehouse hands-on labs
  course3/          # MLOps hands-on labs
  course4/          # GenAI hands-on labs
examples/
  databricks/       # Databricks notebook examples
  sovereign/        # Sovereign AI Stack (Rust) examples
docs/
  outline.md        # Course outline
  courses-3-4-sovereign-ai-stack.md  # Detailed design
```

## Course Architecture

**Dual-layer pedagogy:**
1. **Databricks layer** (~80%) — Hands-on with MLflow, Feature Store, Model Serving, Vector Search, Foundation Models
2. **Sovereign AI Stack layer** (~20%) — Build equivalent systems in Rust to understand internals

## Environment Setup

### Databricks
- Use Databricks Free Edition: https://www.databricks.com/
- No paid features required

### Sovereign AI Stack (Rust demos)
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Key crates used
cargo install batuta    # Orchestration
cargo install realizar  # Inference server
cargo install pmat      # Quality gates
```

### Python (Databricks notebooks)
```bash
uv sync --all-extras
```

## Sovereign AI Stack Components

| Component | Purpose | Course |
|-----------|---------|--------|
| `trueno` | SIMD tensor operations | 3, 4 |
| `aprender` | ML algorithms | 3 |
| `realizar` | Inference serving | 3, 4 |
| `entrenar` | LoRA/QLoRA training | 4 |
| `pacha` | Model registry | 3, 4 |
| `batuta` | Orchestration | 3, 4 |
| `trueno-rag` | RAG pipeline | 4 |
| `alimentar` | Data loading | 3, 4 |
| `pmat` | Quality gates | 3, 4 |

## Code Search

NEVER use grep or find for code search. Use `pmat query` instead:

```bash
# Search for functions by intent
pmat query "chunking overlap" --limit 10

# Include fault pattern detection (batuta integration)
pmat query "error handling" --faults

# Full analysis with churn and faults
pmat query "validation" --faults --churn
```

The `--faults` flag runs batuta bug-hunter to detect mutation targets and boundary conditions inline with results.

## Code Quality

- Python: `ruff` for linting/formatting, `ty` for type checking, `uv` for package management
- Rust: `cargo clippy`, `cargo fmt`
- Quality gates enforced via `pmat`
- All Python tools run via `uvx` (uv toolchain)

## Demo Conventions

Each demo follows the pattern:
```
demos/course1/week{1-3}/          # Databricks-only (no Cargo.toml)
  databricks-{topic}/
    {notebook}.py

demos/course{3,4}/week{1-3}/     # Dual-layer (Databricks + Rust)
  {demo-name}/
    Cargo.toml          # Rust manifest
    src/main.rs         # Rust entry point
  databricks-{topic}/
    {notebook}.py       # Databricks notebook export
```

## Testing

```bash
make test           # Full test suite
make test-course1   # Course 1 tests only (syntax)
make test-course3   # Course 3 tests only
make test-course4   # Course 4 tests only
```
