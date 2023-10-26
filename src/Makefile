TARGET = estimate
CC     = gcc
CFLAGS = -g -std=c99 -Wall -Wvla -Werror -fsanitize=address,undefined
LIBS   = -lm

$(TARGET): $(TARGET).c
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

clean:
	rm -rf $(TARGET) *.o *.a *.dylib *.dSYM
