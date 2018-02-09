CXX := nvcc
## maximum compatibility for CUDA 9.0
## (googled, from http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
# CXXFL := -arch=sm_30 \
# -gencode=arch=compute_30,code=sm_30 \
# -gencode=arch=compute_50,code=sm_50 \
# -gencode=arch=compute_52,code=sm_52 \
# -gencode=arch=compute_60,code=sm_60 \
# -gencode=arch=compute_61,code=sm_61 \
# -gencode=arch=compute_62,code=sm_62 \
# -gencode=arch=compute_70,code=sm_70 \
# -gencode=arch=compute_70,code=compute_70
## compile targeting specifically my GPU card Nvidia GeForce GTX 1050
CXXFL := -arch=compute_61
CXXFL += -g -O2 --compiler-options='-Wall'

# $(targ)

$(targ): $(targ).cu
	$(CXX) $(CXXFL) $^ -o $@

clean:
#	@$(RM) slurm*.out
#	@$(RM) test
	@find . -type f -executable -delete

.PHONY: clean
