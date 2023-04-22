

all: build run

build: 
	@nvcc -o biology .\Genome.cpp .\Specimen.cpp .\Species.cpp .\Taxonomy.cpp .\BiologyMain.cpp -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe"

run:
	.\biology


spec:
	nvcc -o biology .\Genome.cpp .\Specimen.cpp .\Species.cpp .\SpecimenMain.cpp -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe"
