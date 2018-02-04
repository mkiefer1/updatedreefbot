MEX32_FILES = resize.mexglx dt.mexglx features.mexglx getdetections.mexglx fconv.mexglx
MEX64_FILES = resize.mexa64 dt.mexa64 features.mexa64 getdetections.mexa64 fconv.mexa64

LBITS := ${shell getconf LONG_BIT}

ifeq ($(LBITS), 64)
	MEX_FILES = $(MEX64_FILES)
else
	MEX_FILES = $(MEX32_FILES)
endif

all: learn mex

mex: ${MEX_FILES}
	make -C star-cascade

learn: learn.cc
	g++ -O3 -D_FILE_OFFSET_BITS=64 -o learn learn.cc

%.mexa64: %.cc
	mex -glnxa64 -O $< -o $@

%.mexglx: %.cc
	mex -glnx86 -O $< -o $@

clean:
	/bin/rm learn
	rm *.mex*
	make clean -C star-cascade