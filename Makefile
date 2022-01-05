############################################################
# project configuration

BUILD    := ./build
OBJ_DIR  := $(BUILD)/objects
APP_DIR  := $(BUILD)/apps
TARGET   := RGBD_CV

CXX      := g++
#CXXFLAGS := -Wall -Wextra -DGL_GLEXT_PROTOTYPES
CXXFLAGS := -Wall -Wextra

INCLUDE  := -I./include \
            -I/usr/include/ \
			-I/usr/local/include \
			-I/usr/local/include/opencv4 \
			-I/usr/local/include/eigen3 \
			-I../../common/inc \
			-I/usr/local/cuda-11.4/samples/common/inc \
			-I/usr/local/cuda-11.4/include

LIBDIRS  := -L/usr/lib  \
			-L/usr/local/lib \
			-L/usr/local/cuda-11.4/lib64			

SRC      :=                      \
	$(wildcard src/*.cpp)        \

CVLIBS	:= -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_hdf -lboost_filesystem -lboost_system
MISCLIBS:= -lm -lstdc++ -lm -lglfw -lGL -lX11 -lpthread -lXrandr -lXi -lGLEW -lrealsense2
LDFLAGS	:=	$(LIBDIRS) $(MISCLIBS) $(CVLIBS) -lcuda -lcublas -lcurand -lcudart

OBJECTS	:= $(SRC:%.cpp=$(OBJ_DIR)/%.o) 

DEPENDENCIES := $(OBJECTS:.o=.d)


############################################################
# some CUDA stuff
CUDA_PATH ?= /usr/local/cuda-11.4
TARGET_SIZE := 64

NVCC	:= $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     :=

CUDA_INCLUDES:= -I./include -I/usr/local/include/opencv4 -I../../common/inc 


ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS    += --threads 0

CUDA_SRC      :=                \
	$(wildcard src/*.cu)        \ 


# Gencode arguments
SMS ?= 35 37 50 52 60 61 70 75 80 86

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif	


############################################################
# make commands for building and and other functionality

# note: can add @ to the beggining of all make commands to suppress the terminal output
# note: Makefile very finicky with tabs or spaces or something. If it breaks, good place to look

all: build $(APP_DIR)/$(TARGET)
	
$(OBJ_DIR)/src/cuda_RGBD_kernel.o: src/cuda_RGBD_kernel.cu
	$(NVCC) $(CUDA_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -MMD -o $@

$(APP_DIR)/$(TARGET): $(OBJECTS) $(OBJ_DIR)/src/cuda_RGBD_kernel.o
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $(APP_DIR)/$(TARGET) $^ $(LDFLAGS)

-include $(DEPENDENCIES)

.PHONY: all build clean debug release info

build:
	@mkdir -p $(APP_DIR)
	@mkdir -p $(OBJ_DIR)

debug: 
	CXXFLAGS += -DDEBUG -g
	NVCCFLAGS += -g -G
debug: all

release: CXXFLAGS += -O2
release: all

clean:
	rm -rvf $(OBJ_DIR)/*
	rm -rvf $(APP_DIR)/*

info:
	@echo "[*] Application dir: ${APP_DIR}     "
	@echo "[*] Object dir:      ${OBJ_DIR}     "
	@echo "[*] Sources:         ${SRC}         "
	@echo "[*] Objects:         ${OBJECTS}     "
	@echo "[*] Dependencies:    ${DEPENDENCIES}"

run:
	./build/apps/$(TARGET)


