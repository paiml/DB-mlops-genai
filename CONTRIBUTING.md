# Contributing to MLOps & GenAI Engineering on Databricks

## Development Setup

1. Fork and clone the repository
2. Install dependencies:
   ```bash
   make setup
   ```
3. Create a feature branch

## Code Standards

### Python
- Format with `ruff format`
- Lint with `ruff check`
- Follow PEP 8 style guidelines

### Rust
- Format with `cargo fmt`
- Lint with `cargo clippy -- -D warnings`
- Follow Rust API guidelines

### Documentation
- Each demo must include a README.md
- Code comments for non-obvious logic
- Update docs/outline.md for curriculum changes

## Testing

```bash
# Run all tests
make test

# Run specific course tests
make test-course3
make test-course4

# Quick syntax check
make test-fast
```

## Pull Request Process

1. Ensure `make check` passes
2. Update documentation as needed
3. Add tests for new functionality
4. Keep commits atomic and well-described
5. Request review from maintainers

## Demo Structure

Each demo should follow this structure:

```
demos/course{N}/week{N}/
├── Cargo.toml          # Rust manifest (if Rust demo)
├── src/
│   └── main.rs         # Rust entry point
├── notebook.py         # Databricks notebook export
├── tests/              # Test files
└── README.md           # Demo instructions
```

## Lab Structure

Each lab should include:
- Clear learning objectives
- Step-by-step instructions
- Both Databricks and Sovereign AI Stack versions
- Validation criteria

## Questions?

Open an issue or discussion on GitHub.
