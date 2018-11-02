
CC=$(TARGET_OUTPUT_DIR)/../../toolchain/gcc/linux-x86/aarch64/gcc-linaro-6.3.1-2017.02-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc
CROSS_COMPILE=$(TARGET_OUTPUT_DIR)/../../toolchain/gcc/linux-x86/aarch64/gcc-linaro-6.3.1-2017.02-x86_64_aarch64-linux-gnu/aarch64-linux-gnu-


SDK_LIB=$(TARGET_OUTPUT_DIR)/build/npu/
OVX_DIR=$(TARGET_OUTPUT_DIR)
LIB_DIR=$(TARGET_OUTPUT_DIR)/../../toolchain/gcc/linux-x86/aarch64/gcc-linaro-6.3.1-2017.02-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc/lib

INCLUDE += -I$(SDK_LIB)/sdk/inc -I$(SDK_LIB)/applib/ovxinc/include -I$(OVX_DIR)/build/libjpeg-9b
CFLAGS += $(INCLUDE)

################################################################################
# Supply necessary libraries.
LIBS += -L$(SDK_LIB)/sharelib -l OpenVX -l OpenVXU -l CLC -l VSC -l m -lGAL -l LLVM_viv
LIBS += -L$(SDK_LIB)/sharelib -l ovxlib
LIBS += -L$(OVX_DIR)/target/usr/lib64 -l jpeg
LIBS +=-L$(LIB_DIR) -l stdc++

CFLAGS +=-I. -g -fPIE -Wall -pthread $(LIBS)
ODIR=obj
OFILE=dncnn

_OBJ=main.o
_OBJ += vnn_post_process.o
_OBJ += vnn_dncnn.o
_OBJ += vnn_pre_process.o
OBJ=$(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

$(OFILE): $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) -pie

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o $(OFILE)
