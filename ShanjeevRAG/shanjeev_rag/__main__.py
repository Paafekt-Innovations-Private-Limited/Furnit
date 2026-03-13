"""Allow running as: python -m shanjeev_rag index | query [question]"""
from shanjeev_rag.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
