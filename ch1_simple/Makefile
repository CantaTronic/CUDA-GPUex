CXX := nvcc
CXXFL := -arch=compute_35 --gpu-code=sm_35,sm_37

# $(targ)

$(targ): $(targ).cu
	$(CXX) $(CXXFL) $^ -o $@

clean:
	@$(RM) slurm*.out
	@$(RM) test
