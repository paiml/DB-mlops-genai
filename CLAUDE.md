# CLAUDE.md

This repository contains course materials for **MLOps & GenAI Engineering on Databricks** (Courses 3 & 4 of the Databricks Specialization on Coursera).

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
  course3/          # MLOps Engineering demos (weeks 1-6)
  course4/          # GenAI Engineering demos (weeks 1-7)
labs/
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
1. **Databricks layer** — Hands-on with MLflow, Feature Store, Model Serving, Vector Search, Foundation Models
2. **Sovereign AI Stack layer** — Build equivalent systems in Rust to understand internals

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

## Code Quality

- Python: `ruff` for linting, `uv` for package management
- Rust: `cargo clippy`, `cargo fmt`
- Quality gates enforced via `pmat`

## Demo Conventions

Each demo follows the pattern:
```
demos/course{3,4}/week{N}/
  Cargo.toml          # Rust manifest (if Rust demo)
  src/main.rs         # Rust entry point
  notebook.py         # Databricks notebook export
  README.md           # Demo instructions
```

## Testing

```bash
make test           # Full test suite
make test-course3   # Course 3 tests only
make test-course4   # Course 4 tests only
```
