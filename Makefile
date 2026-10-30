.PHONY: lib clean
lib: csrc/libbitnet.so
csrc/libbitnet.so: csrc/bitnet_kernels.cpp
	g++ -O3 -march=native -std=c++17 -shared -fPIC -o $@ $<
clean:
	rm -f csrc/libbitnet.so
