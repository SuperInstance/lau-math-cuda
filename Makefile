# lau-math-cuda/Makefile
# Builds for SM 8.7 (Jetson Orin) and SM 8.9 (RTX 4050)
# Falls back gracefully when CUDA is unavailable.

NVCC      ?= nvcc
NVCC_FLAGS = -std=c++17 -O2 -Xcompiler -fPIC
NVCC_ARCHS = -gencode arch=compute_87,code=sm_87 \
             -gencode arch=compute_89,code=sm_89

# Libraries
CUDA_LIBS = -lcublas -lcusolver -lcusparse -lcurand
LIBS      = $(CUDA_LIBS) -lm

# Sources
DISPATCH_SRC  = lau_dispatch.cu
MATRIX_SRC    = lau_matrix_cuda.cu
LAPLACIAN_SRC = lau_laplacian_cuda.cu
HEAT_SRC      = lau_heat_kernel_cuda.cu
FLEET_SRC     = lau_agent_fleet_cuda.cu
CONSERV_SRC   = lau_conservation_cuda.cu

ALL_SRC = $(DISPATCH_SRC) $(MATRIX_SRC) $(LAPLACIAN_SRC) \
          $(HEAT_SRC) $(FLEET_SRC) $(CONSERV_SRC)

# Outputs
STATIC_LIB = liblau_math_cuda.a
SHARED_LIB = liblau_math_cuda.so
TEST_BIN   = test_lau_math_cuda

.PHONY: all clean test static shared check_nvcc

# Check if nvcc is available
HAS_NVCC := $(shell which nvcc 2>/dev/null)

all: check_nvcc static shared

check_nvcc:
ifndef HAS_NVCC
	@echo "⚠️  nvcc not found. Install CUDA toolkit to build GPU kernels."
	@echo "   CPU-only stubs will be used."
	@echo "   Continuing with limited build..."
endif

# Compile object files from .cu sources
%.o: %.cu lau_dispatch.h
ifdef HAS_NVCC
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCHS) -c $< -o $@
else
	@echo "Skipping $< (no nvcc)"
endif

# Object files
DISPATCH_OBJ  = lau_dispatch.o
MATRIX_OBJ    = lau_matrix_cuda.o
LAPLACIAN_OBJ = lau_laplacian_cuda.o
HEAT_OBJ      = lau_heat_kernel_cuda.o
FLEET_OBJ     = lau_agent_fleet_cuda.o
CONSERV_OBJ   = lau_conservation_cuda.o

ALL_OBJ = $(DISPATCH_OBJ) $(MATRIX_OBJ) $(LAPLACIAN_OBJ) \
          $(HEAT_OBJ) $(FLEET_OBJ) $(CONSERV_OBJ)

# Static library
static: $(STATIC_LIB)

$(STATIC_LIB): $(ALL_OBJ)
ifdef HAS_NVCC
	ar rcs $@ $^
	@echo "✅ Built $(STATIC_LIB)"
else
	@echo "⚠️  Cannot build static lib without nvcc"
endif

# Shared library
shared: $(SHARED_LIB)

$(SHARED_LIB): $(ALL_OBJ)
ifdef HAS_NVCC
	$(NVCC) -shared $(ALL_OBJ) -o $@ $(LIBS)
	@echo "✅ Built $(SHARED_LIB)"
else
	@echo "⚠️  Cannot build shared lib without nvcc"
endif

# Test binary
test: $(TEST_BIN)

$(TEST_BIN): test_main.cu $(ALL_SRC) lau_dispatch.h
ifdef HAS_NVCC
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCHS) -o $@ test_main.cu $(ALL_SRC) $(LIBS)
	@echo "✅ Built $(TEST_BIN)"
	@echo "   Run with: ./$(TEST_BIN)"
else
	@echo "⚠️  Cannot build test binary without nvcc"
	@echo "   To test on CPU-only, compile test_main.cu with g++ and stubs."
endif

# Run tests
run_test: $(TEST_BIN)
	./$(TEST_BIN)

# Clean
clean:
	rm -f *.o $(STATIC_LIB) $(SHARED_LIB) $(TEST_BIN)

# Help
help:
	@echo "lau-math-cuda build targets:"
	@echo "  all       - Build static and shared libraries"
	@echo "  static    - Build liblau_math_cuda.a"
	@echo "  shared    - Build liblau_math_cuda.so"
	@echo "  test      - Build test binary"
	@echo "  run_test  - Build and run tests"
	@echo "  clean     - Remove build artifacts"
	@echo ""
	@echo "Requirements:"
	@echo "  CUDA Toolkit 12+ with nvcc"
	@echo "  cuBLAS, cuSOLVER, cuSPARSE"
	@echo "  Targets: SM 8.7 (Jetson Orin), SM 8.9 (RTX 4050)"
