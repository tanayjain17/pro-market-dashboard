# In your GitHub Actions workflow, add a debug step before tests:
- name: 🔍 Debug directory structure
  run: |
    pwd
    ls -la
    ls -la utils/
    cat utils/__init__.py || echo "utils/__init__.py not found"
