all: bin

bin:
	rm -rf build
	mkdir -p build/net
	cd build && cmake ..
	cd build && make -j$(nproc)
	cp build/bev .

clean:
	rm -rf build
	rm -rf test

