all: bin

lib:
	cd ../control/lib && $(MAKE) all

bin:
	rm -rf build
	mkdir -p build
	cd build && cmake ..
	cd build && make
	mv build/remote_control .
	rm -rf build

clean:
	rm -rf build
	rm remote_control

