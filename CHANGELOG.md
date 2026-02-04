# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Course 3: MLOps Engineering demos (Weeks 1-6)
  - Week 1: MLflow client in Rust
  - Week 2: Feature pipeline with SIMD
  - Week 3: Model training with gradient descent
  - Week 4: Inference server implementation
  - Week 5: Quality gates with pmat
  - Week 6: Fraud detection capstone
- Course 4: GenAI Engineering demos (Weeks 1-7)
  - Week 1: LLM serving and tokenization
  - Week 2: Prompt engineering templates
  - Week 3: Vector search with SIMD
  - Week 4: RAG pipeline implementation
  - Week 5: LoRA/QLoRA fine-tuning
  - Week 6: Production deployment patterns
  - Week 7: Enterprise knowledge assistant capstone
- Lab exercises for both courses
- mdBook documentation with GitHub Pages deployment
- Architecture diagram (SVG)
- pmat quality gates configuration
- GitHub Actions CI/CD workflows
- Falsification checklist for quality assurance

### Changed
- Updated CI workflow to use `uvx ruff` instead of `uv run ruff`
- Fixed repository URLs from noahgift to paiml

### Security
- Added cargo-audit to CI workflow
- Added cargo-deny configuration for license compliance
- Added supply chain security checks

## [0.1.0] - 2026-02-04

### Added
- Initial course outline
- Repository structure
- Basic README and CLAUDE.md

[Unreleased]: https://github.com/paiml/DB-mlops-genai/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/paiml/DB-mlops-genai/releases/tag/v0.1.0
