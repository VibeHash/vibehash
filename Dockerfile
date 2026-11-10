# A clean machine for the tests and the reproduction gate, on CPU: what .github/workflows/tests.yml's full job installs
# (Python 3.12, the finetune extra, CPU PyTorch), so a clone can be checked without a GPU or anyone's PC.
#
#   docker build -t vibehash-check .
#   docker run --rm vibehash-check                      # every test, then scripts/reproduce.py --check
#   docker run --rm vibehash-check python -m unittest tests.test_core   # or anything else
#
# Tests that need a CUDA GPU, OpenVINO, transformers/peft or VIBEHASH_REAL_MODELS=1 skip here. tests.test_deepseek
# fetches DeepSeek's reference code from the Hub (~100 KB), so the run needs network access, as CI's does.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src

RUN python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN python -m pip install -e ".[finetune]"

CMD ["sh", "-c", "python -m unittest discover -s tests -t . && python scripts/reproduce.py --check"]
