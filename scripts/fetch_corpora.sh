#!/bin/sh
# Downloads the three doc sets into corpora/<name>/. Corpora are not committed; rerun to refresh.
set -eu
cd "$(dirname "$0")/.."
PYVER="${PYVER:-3.15}"

mkdir -p corpora/python corpora/mcp corpora/claude

curl -fsSL "https://docs.python.org/3/archives/python-${PYVER}-docs-text.tar.bz2" -o corpora/python.tar.bz2
tar -xjf corpora/python.tar.bz2 -C corpora/python --strip-components=1
rm corpora/python.tar.bz2

curl -fsSL https://modelcontextprotocol.io/llms-full.txt -o corpora/mcp/llms-full.txt
curl -fsSL https://platform.claude.com/llms-full.txt -o corpora/claude/llms-full.txt

du -sh corpora/*
