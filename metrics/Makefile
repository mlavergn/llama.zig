# metrics/: performance of the stock C against the port, side by side.
# See README.md for what each metric measures and what it cannot.
#
#   make              libs, drivers, then RUNS rounds on Metal and the report
#   make cpu          the same on the CPU device alone -- the ported CPU kernels
#   make report       summarise the newest results file again
#   make libs         both library sets at --release=fast, into build/
#   make drivers      link perf_driver.c against each
#   make PORT_REV=v   the port built from git revision v, not the working tree
#
# The reference is llama.cpp.zmake, the stock C built by the same Zig
# toolchain at the same optimisation level, so the port is the only variable.
# One C driver is linked against both, as scripts/parity-port does.
#
# PORT_REV builds the port from `git archive <rev>` under build/, so edits in
# flight in the working tree -- this one is shared with other sessions -- cannot
# leak into a measurement, and a baseline can be re-measured later at the same
# revision. llama.cpp and ../zigjinja are symlinked in; both are pinned
# elsewhere. Without it the port is whatever the working tree builds.
#
# Both sides install into build/, never into zig-out/ or
# llama.cpp.zmake/zig-out/: those are shared with the gates, and other
# sessions rebuild them at other CPU targets (`-Dcpu=apple_m3` for the i8mm
# kernels). Measured both ways: a perf rebuild here clobbered a gate's
# reference, and a perf run could have linked a library built for a different
# CPU. CPU names the target for both sides explicitly, baseline by default.
#
# Power needs root: unprivileged, IOReport's CPU, DRAM and ANE energy channels
# do not move on macOS 27, only the GPU's. The driver runs under $(SUDO) and
# everything else as you. SUDO= runs without, and those columns read `na`.

SHELL := /bin/bash
.SHELLFLAGS := -eo pipefail -c

ROOT     := $(abspath $(CURDIR)/..)
B        := $(CURDIR)/build

MODEL    ?= $(ROOT)/Qwen3.5-2B-Q4_K_M.gguf
PROMPT   ?= How was the transistor discovered?
NPRED    ?= 512
RUNS     ?= 5
COOLDOWN ?= 5
IDLE_MS  ?= 2000
TAIL_MS  ?= 1500
FLAGS    ?=
SUDO     ?= sudo
PORT_REV ?=
FORCE    ?=
CPU      ?= baseline

REF_OUT   := $(B)/ref-$(CPU)
REF_LIBS  := $(REF_OUT)/lib/libllama.a $(REF_OUT)/lib/libggml.a
ifeq ($(PORT_REV),)
PORT_DIR  := $(ROOT)
PORT_OUT  := $(B)/port-worktree-$(CPU)
PORT_DESC := working tree
else
PORT_SHA  := $(shell git -C $(ROOT) rev-parse --short=9 $(PORT_REV)^{commit})
PORT_DIR  := $(B)/port-$(PORT_SHA)
PORT_OUT  := $(B)/port-$(PORT_SHA)-$(CPU)
PORT_DESC := $(PORT_REV) ($(PORT_SHA))
endif
PORT_LIBS := $(PORT_OUT)/lib/libllama.a $(PORT_OUT)/lib/libggml.a

SDK      := $(shell xcrun --sdk macosx --show-sdk-path)
LINK     := zig cc -O2 -mcpu=baseline $(CURDIR)/perf_driver.c -std=c11 \
            -I $(ROOT)/llama.cpp/include -I $(ROOT)/llama.cpp/ggml/include
SYSLIBS  := -lc++ -lIOReport -framework IOKit -framework CoreFoundation \
            -framework Foundation -framework Metal -framework MetalKit -framework Accelerate \
            -F $(SDK)/System/Library/Frameworks

export MODEL PROMPT NPRED RUNS COOLDOWN IDLE_MS TAIL_MS FLAGS SUDO FORCE PORT_DESC CPU

.PHONY: all run cpu report libs drivers clean

all: run

run: drivers
	./perf-run

cpu: drivers
	$(MAKE) --no-print-directory run-only FLAGS=cpu

.PHONY: run-only
run-only:
	./perf-run

report:
	./perf-report

# Always rebuilt: Zig's cache makes an unchanged tree cheap, and a stale
# library is the one way this comparison can measure the wrong thing.
libs:
	cd $(ROOT)/llama.cpp.zmake && zig build lib --release=fast -Dcpu=$(CPU) -p $(REF_OUT)
ifneq ($(PORT_REV),)
	@test -n "$(PORT_SHA)" || { echo "PORT_REV=$(PORT_REV) is not a commit" >&2; exit 2; }
	@if [ ! -f $(PORT_DIR)/build.zig ]; then \
	    mkdir -p $(PORT_DIR) && git -C $(ROOT) archive $(PORT_SHA) | tar -x -C $(PORT_DIR) && \
	    rm -rf $(PORT_DIR)/llama.cpp && ln -s $(ROOT)/llama.cpp $(PORT_DIR)/llama.cpp && \
	    ln -sfn $(abspath $(ROOT)/../zigjinja) $(B)/zigjinja; fi
endif
	cd $(PORT_DIR) && zig build reference --release=fast -Dcpu=$(CPU) -p $(PORT_OUT)

drivers: libs
	mkdir -p $(B)
	$(LINK) -o $(B)/perf-ref  $(REF_LIBS)  $(SYSLIBS)
	$(LINK) -o $(B)/perf-port $(PORT_LIBS) $(SYSLIBS)

clean:
	rm -rf $(B)
