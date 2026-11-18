CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++14 -Wall -Wextra

cachesim: src/cachesim.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test: cachesim
	./cachesim --selftest

experiments: cachesim
	bash scripts/run_experiments.sh
	bash scripts/run_elem.sh

clean:
	rm -f cachesim

.PHONY: test experiments clean
