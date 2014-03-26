EXE=hdr
SRCDIR=../src
OBJDIR=obj

SUFFIXES += .d

SDLIMGDIR = /usr/lib/x86_64-linux-gnu/
SDLIMGINC = -I/usr/include/SDL/
SDLIMGLIBd = -L$(SDLIMGDIR) -lSDL_image

JPGDIR = /usr/include/x86_64-linux-gnu/
JPGINC = -I/usr/include/
JPGLIBd = -L$(JPGDIR) -ljpeg

INCS = $(SDLIMGINC) $(JPGINC)
LIBS = $(SDLIMGLIBd) $(JPGLIBd)


CXX      = g++
CXXFLAGS = -I$(SRCDIR) -O2 -fopenmp -DCL_USE_DEPRECATED_OPENCL_1_1_APIS
LDFLAGS  = -lOpenCL -lpthread -lglut -lGL -lGLEW -lGLU
MODULES  = Filter Stitching HistEq GammaAdj Reinhard
OBJECTS  = $(MODULES:%=$(OBJDIR)/%.o)
SOURCES  = $(MODULES:%=$(SRCDIR)/%.cpp)
DEPFILES = $(MODULES:%=$(OBJDIR)/%.d)


ifneq ($(wildcard .halide),)
	HALIDE = 1
endif
ifeq ($(HALIDE),1)
	CXXFLAGS += -DENABLE_HALIDE
	HALIDE_FILES = halide/*.s
endif

all: prebuild $(OBJDIR) $(EXE)

halide:
	$(MAKE) all HALIDE=1

$(EXE): $(OBJECTS) $(HALIDE_FILES) hdr.cpp
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) $(INCS) $(LIBS) -o $@

prebuild:
	$(MAKE) -C ../src/opencl -f $(shell pwd)/Makefile prebuild_opencl
ifeq ($(HALIDE),1)
	$(MAKE) -C ../src/halide -f $(shell pwd)/Makefile prebuild_halide
endif

prebuild_opencl:
	./stringify_kernels.sh

prebuild_halide:
	HL_TARGET=x86-64-opencl ./gen_filters.sh ../../linux/halide

$(OBJDIR)/%.d: $(SRCDIR)/%.cpp $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCS) $(LIBS) -MM -MT $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$<) $< -MF $@ 2>/dev/null

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCS) $(LIBS) -o $@ -c $<

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(EXE) ../src/opencl/*.h halide

.PHONY: clean

ifeq (0, $(words $(findstring $(MAKECMDGOALS), clean opencl halide)))
-include $(DEPFILES)
endif
