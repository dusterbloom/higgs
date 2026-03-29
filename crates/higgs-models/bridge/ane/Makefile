CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc -fPIC
FRAMEWORKS = -framework Foundation -framework IOSurface -framework Accelerate -ldl
TARGET = libane_bridge.dylib

all: $(TARGET)

$(TARGET): ane_bridge.m ane_bridge.h
	$(CC) $(CFLAGS) -dynamiclib -o $@ ane_bridge.m $(FRAMEWORKS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
