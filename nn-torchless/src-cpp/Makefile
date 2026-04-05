#   make && ./nn-cpp
# Default CSV: ../data/data.csv (cwd should be src-cpp). Override: ./nn-cpp /path/to/data.csv
#
# RAPIDCSV_INCLUDE=/path/to/rapidcsv/src make   — skip downloading .deps/rapidcsv.h

RAPIDCSV_VERSION := v8.92
RAPIDCSV_URL     := https://raw.githubusercontent.com/d99kris/rapidcsv/$(RAPIDCSV_VERSION)/src/rapidcsv.h

DEPS_DIR   := .deps
RAPIDCSV_H := $(DEPS_DIR)/rapidcsv.h

CXX      ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra

ifneq ($(strip $(RAPIDCSV_INCLUDE)),)
  CPPFLAGS := -I$(RAPIDCSV_INCLUDE)
  RAPIDCSV_PREREQ :=
else
  CPPFLAGS := -I$(DEPS_DIR)
  RAPIDCSV_PREREQ := $(RAPIDCSV_H)
endif

.PHONY: all clean

all: nn-cpp

nn-cpp: main.cpp $(RAPIDCSV_PREREQ)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ main.cpp

$(RAPIDCSV_H):
	@mkdir -p $(dir $@)
	curl -fsSL '$(RAPIDCSV_URL)' -o $@

clean:
	rm -f nn-cpp
	rm -rf $(DEPS_DIR)
