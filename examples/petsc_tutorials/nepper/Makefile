####### Compiler, tools and options

build = 2d

OBJDIR = .obj_2d
TARGET = executable


#boost
INCPATH += -I/home/pouria/Libraries/boost_1_74_0/boost 
LIBS    += -L/home/pouria/Libraries/boost_1_74_0/libs  



# p4est
P4EST_DIR  = /home/pouria/Libraries/P4EST_openmpi
INCPATH   += -I$(P4EST_DIR)/include/
LIBS      += -Wl,-rpath,$(P4EST_DIR)/lib/ -L$(P4EST_DIR)/lib -lp4est -lsc


# petsc
PETSC_DIR  = /home/pouria/Libraries/petsc-3.15.0-openmpi
PETSC_ARCH = arch-linux-c-opt
INCPATH   += -I${PETSC_DIR}/include -I/usr/local/cuda-10.1/include -I${PETSC_DIR}/${PETSC_ARCH}/include
LIBS      += -L/home/pouria/Libraries/PETSC/libs -lpetsc  -L/usr/local/cuda-10.1/lib64 -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc


# MPICH
CC   = @echo compiling $< && mpicc
CXX  = @echo compiling $< && mpicxx
LINK = @echo linking $@ && mpicxx
#MPI_DIR = /home/pouria/Libraries/PETSC/bin/
#CC      = $(MPI_DIR)/mpicc
#CXX     = $(MPI_DIR)/mpicxx
#LINK    = $(MPI_DIR)/mpicxx

CFLAGS += 
CXXFLAGS += 
LFLAGS += #-lpetsc -lboost_timer -lpthread -lX11 -lm -lcufft -lcublas -lcudart -lcusparse -lcusolver -lstdc++ -ldl -lmpi -lgfortran -lm -lgfortran -lm -lgcc_s -lquadmath -lstdc++ -ldl


# casl p4est
OPERA = /home/pouria/Projects/OPERA
INCPATH += -I$(OPERA)

vpath %.c   $(OPERA)/src
vpath %.cpp $(OPERA)/src
vpath %.cpp $(OPERA)/examples/petsc_tutorials/nepper

#
# ####### Files
ifeq ($(build),2d)
SOURCES = \
	$(OPERA)/examples/petsc_tutorials/nepper/main.cpp \
	$(OPERA)/src/test.cpp 
else

HEADERS += 

endif

STRIP_SRC = $(notdir $(SOURCES))
OBJECTS = $(patsubst %.cpp, $(OBJDIR)/%.o, $(filter %.cpp, $(STRIP_SRC)))       $(patsubst %.c, $(OBJDIR)/%.o, $(filter %.c, $(STRIP_SRC)))

####### Implicit rules

$(OBJDIR)/%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

$(OBJDIR)/%.o: %.c
	$(CC) -c $(CFLAGS) $(INCPATH) -o "$@" "$<"

####### Build rules

#all: $(SOURCES) $(TARGET)
all: $(TARGET)

$(TARGET):  $(OBJECTS)
	$(LINK) -o $(TARGET) $(OBJECTS) $(LIBS) $(LFLAGS)

clean: 
	rm -rf $(OBJDIR)/*.o $(TARGET)


