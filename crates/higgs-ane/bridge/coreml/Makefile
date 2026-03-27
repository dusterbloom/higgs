CC      = xcrun clang
CFLAGS  = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc -fPIC
FRAMEWORKS = -framework Foundation -framework CoreML -framework Accelerate
TARGET  = libcoreml_bridge.dylib

all: $(TARGET)

$(TARGET): coreml_bridge.m coreml_bridge.h
	$(CC) $(CFLAGS) -dynamiclib -o $@ coreml_bridge.m $(FRAMEWORKS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
