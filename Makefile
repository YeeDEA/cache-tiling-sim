CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++14 -Wall -Wextra

cachesim: src/cachesim.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test: cachesim
	./cachesim --selftest
	./cachesim --csv --size 256 --assoc 1 --block 64 --trace tests/basic.trace | diff - tests/basic.expected
	./cachesim --csv --size 256 --assoc 1 --block 64 --trace tests/wb.trace | diff - tests/wb.expected
	! ./cachesim --policy bogus 2>/dev/null
	@echo "trace tests passed"

policy: cachesim
	bash scripts/run_policy.sh

heatmap: cachesim
	bash scripts/sweep_heatmap.sh > docs/results/heatmap.csv
	python3 scripts/plot_heatmap.py docs/results/heatmap.csv docs/heatmap.png

experiments: cachesim
	bash scripts/run_experiments.sh
	bash scripts/run_elem.sh

clean:
	rm -f cachesim

.PHONY: test experiments policy heatmap clean
