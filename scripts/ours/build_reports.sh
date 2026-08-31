#!/bin/bash
# Builds the pdfs/<Folder>/src/*.tex reports and leaves the PDF one level up.
#
# Layout this enforces, so the folders stay readable:
#   pdfs/<Folder>/<name>.pdf     <- the only thing at the top level
#   pdfs/<Folder>/src/<name>.tex + .aux .log .out  <- everything else
#
# pdflatex insists on writing its aux files next to its output, so it is run
# with -output-directory=src and the PDF is moved up afterwards.
#
# Usage:
#   python scripts/ours/make_reports.py && bash scripts/ours/build_reports.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}/pdfs"

status=0
for tex in */src/*.tex */*/src/*.tex; do
  [ -e "${tex}" ] || continue
  dir=$(dirname "${tex}")            # e.g. QED/src
  top=$(dirname "${dir}")            # e.g. QED
  name=$(basename "${tex}" .tex)
  # src/ also holds table fragments meant to be \input by a document
  # (sweep_table.tex, table4_*.tex). Only compile actual documents.
  grep -q '\\documentclass' "${tex}" || continue
  echo "== ${top}/${name} =="
  # Twice, so \ref and the table-of-contents-style forward references resolve.
  for pass in 1 2; do
    if ! pdflatex -interaction=nonstopmode -halt-on-error \
        -output-directory="${dir}" "${tex}" > "${dir}/${name}.build.log" 2>&1; then
      echo "   FAILED on pass ${pass} -- see ${dir}/${name}.build.log"
      grep -m3 -A2 '^!' "${dir}/${name}.build.log" | sed 's/^/   /'
      status=1
      continue 2
    fi
  done
  mv -f "${dir}/${name}.pdf" "${top}/${name}.pdf"
  echo "   -> ${top}/${name}.pdf  ($(du -h "${top}/${name}.pdf" | cut -f1))"
done

echo
echo "== layout =="
find . -maxdepth 3 -name '*.pdf' | sort | sed 's/^/   /'
exit ${status}
