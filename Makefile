.RECIPEPREFIX := >

CC := gcc
CFLAGS := -Wall -Wextra -O2 -fopenmp -Iinclude

TARGET_TP8 := image_processing_tp8
TARGET_TP10 := image_processing_tp10

SRC_COMMON := src/lib_bmp.c

.PHONY: all tp8 tp10 clean

all: tp8

tp8: $(TARGET_TP8)

tp10: $(TARGET_TP10)

$(TARGET_TP8): $(SRC_COMMON) src/tp8_etudiants.c include/lib_bmp.h
>$(CC) $(CFLAGS) $(SRC_COMMON) src/tp8_etudiants.c -o $(TARGET_TP8)

$(TARGET_TP10): $(SRC_COMMON) src/main_tp10.c include/lib_bmp.h
>$(CC) $(CFLAGS) $(SRC_COMMON) src/main_tp10.c -o $(TARGET_TP10)

clean:
>rm -f $(TARGET_TP8) $(TARGET_TP10)
