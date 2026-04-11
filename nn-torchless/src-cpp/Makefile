#   make && ./nn-cpp
# Default CSV: ../data/data.csv (cwd should be src-cpp).
#
# Eigen: fetched into .deps/eigen-3.4.0/ if not found via pkg-config.
# Override include path: make EIGEN_INCLUDE=/path/to/parent/of/Eigen

RAPIDCSV_VERSION := v8.92
RAPIDCSV_URL     := https://raw.githubusercontent.com/d99kris/rapidcsv/$(RAPIDCSV_VERSION)/src/rapidcsv.h

EIGEN_TAG        := 3.4.0
EIGEN_TARBALL    := https://gitlab.com/libeigen/eigen/-/archive/$(EIGEN_TAG)/eigen-$(EIGEN_TAG).tar.bz2

DEPS_DIR         := .deps
RAPIDCSV_H       := $(DEPS_DIR)/rapidcsv.h
EIGEN_ROOT       := $(DEPS_DIR)/eigen-$(EIGEN_TAG)
EIGEN_CORE       := $(EIGEN_ROOT)/Eigen/Core

CXX      ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra

ifneq ($(strip $(RAPIDCSV_INCLUDE)),)
  CPPFLAGS := -I$(RAPIDCSV_INCLUDE)
  RAPIDCSV_PREREQ :=
else
  CPPFLAGS := -I$(DEPS_DIR)
  RAPIDCSV_PREREQ := $(RAPIDCSV_H)
endif

# Eigen: explicit path > pkg-config > vendored tarball in .deps
ifneq ($(strip $(EIGEN_INCLUDE)),)
  EIGEN_CPPFLAGS := -I$(EIGEN_INCLUDE)
  EIGEN_PREREQ :=
else
  EIGEN_PKG := $(shell pkg-config --silence-errors --cflags eigen3 2>/dev/null)
  ifneq ($(strip $(EIGEN_PKG)),)
    EIGEN_CPPFLAGS := $(EIGEN_PKG)
    EIGEN_PREREQ :=
  else
    EIGEN_CPPFLAGS := -I$(EIGEN_ROOT)
    EIGEN_PREREQ := $(EIGEN_CORE)
  endif
endif

.PHONY: all clean

all: nn-cpp

nn-cpp: main.cpp $(RAPIDCSV_PREREQ) $(EIGEN_PREREQ)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(EIGEN_CPPFLAGS) -o $@ main.cpp

$(RAPIDCSV_H):
	@mkdir -p $(dir $@)
	curl -fsSL '$(RAPIDCSV_URL)' -o $@

$(EIGEN_CORE):
	@mkdir -p $(DEPS_DIR)
	curl -fsSL '$(EIGEN_TARBALL)' -o $(DEPS_DIR)/eigen-$(EIGEN_TAG).tar.bz2
	tar -xjf $(DEPS_DIR)/eigen-$(EIGEN_TAG).tar.bz2 -C $(DEPS_DIR)

clean:
	rm -f nn-cpp
	rm -rf $(DEPS_DIR)
