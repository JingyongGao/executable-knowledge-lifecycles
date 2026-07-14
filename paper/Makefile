.PHONY: metrics pdf audit clean

metrics:
	../.venv/bin/python ../scripts/generate_paper_metrics.py

pdf:
	../scripts/reproduce_and_compile.sh

audit:
	../.venv_clean/bin/python ../scripts/audit_paper.py --paper-dir . --render-dir ../tmp/pdfs/main

clean:
	rm -f main.aux main.log main.out main.pdf report.html
