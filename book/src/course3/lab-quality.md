# Lab: Quality Gates

Implement production quality enforcement with pmat.

## Objectives

- Configure quality thresholds
- Implement pre-commit hooks
- Enforce TDG scoring

## Demo Code

See [`demos/course3/week5/quality-gates/`](https://github.com/noahgift/DB-mlops-genai/tree/main/demos/course3/week5/quality-gates)

## Lab Exercise

See [`labs/course3/week5/lab_5_5_quality_gates.py`](https://github.com/noahgift/DB-mlops-genai/tree/main/labs/course3/week5)

## Configuration

```toml
# .pmat-gates.toml
[gates]
min_tdg_score = "B"
max_cyclomatic = 30
max_cognitive = 25
min_line_coverage = 80
min_branch_coverage = 70

[pre_commit_checks]
checks = ["complexity", "dead-code", "security", "duplicates"]
```

## Commands

```bash
# Repository health score
pmat repo-score

# Quality gate check
pmat quality-gate

# Rust project score
pmat rust-project-score

# Analyze complexity
pmat analyze complexity --path .
```
