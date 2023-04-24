

all: build run


build: 
	@nvcc -m64 -o biology .\Genome.cpp .\Specimen.cpp .\Species.cpp .\Taxonomy.cpp .\BiologyMain.cpp 

buildcc:
	@nvcc -o biology .\Genome.cpp .\Species.cpp .\Specimen.cpp .\Taxonomy.cpp .\BiologyMain.cpp -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe"

run:
	@.\biology
