# Project Todos

## Active

### 🔥 Critical Priority (Week 1)
- [ ] Replace pickle with safer serialization (parquet/feather) in simple_pipeline/utils/cache.py | Due: 11/19/2024
- [ ] Add input validation wrapper for file paths in salony_pipeline.py | Due: 11/19/2024
- [ ] Fix dynamic module execution in scripts/run_pipeline.py | Due: 11/19/2024

### 🚨 High Priority (Week 2-4)
- [ ] Expand test coverage to >80% - Add unit tests for each step class | Due: 11/26/2024
- [ ] Add integration tests for LLM steps with mocked Ollama API calls | Due: 11/26/2024
- [ ] Add comprehensive API documentation with docstrings to all public methods | Due: 12/03/2024
- [ ] Generate API docs with Sphinx | Due: 12/03/2024
- [ ] Implement security scanning (bandit, safety) | Due: 12/10/2024

### 📊 Medium Priority
- [ ] Add type checking with mypy for static analysis
- [ ] Break down long functions in salony_pipeline.py:136-305
- [ ] Replace magic numbers with constants in judge_step.py:115,184
- [ ] Add performance monitoring with execution time logging
- [ ] Add pre-commit hooks to enforce code quality
- [ ] Add async processing option for improved LLM API throughput

### 📝 Low Priority
- [ ] Add contribution guidelines (CONTRIBUTING.md)
- [ ] Set up GitHub Actions for CI/CD
- [ ] Add configuration validation with Pydantic models
- [ ] Implement structured JSON logging option
- [ ] Add streaming support for large datasets
- [ ] Add parallelization options

## Completed
- [x] Fix the mixed language comments (should be only english) | Done: 11/12/2024