all: build/header.pdf

build/ds.pdf: ds.py einzelspalt1.csv doppelspalt.csv | build
	python ds.py

build/vergleich.pdf: ds.py einzelspalt1.csv doppelspalt.csv | build
	python ds.py

build/header.pdf: header.tex build/vergleich.pdf build/ds.pdf | build
	lualatex --output-directory=build header.tex

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean
