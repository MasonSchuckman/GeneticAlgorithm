

all: build run

build:
	@nvcc -rdc=true -lineinfo -o runner .\biology\Genome.cpp .\Simulator.cu .\Runner.cu .\Kernels.cu .\BasicSimulation.cu

buildcc:
	@nvcc -rdc=true -lineinfo -o runner .\Runner.cu .\TargetSimulation.cu .\BasicSimulation.cu .\Simulator.cu .\Kernels.cu .\biology\Genome.cpp -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe"

run:
	.\runner