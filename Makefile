

all: build run

build:
	@nvcc -rdc=true -lineinfo -o runner .\biology\Genome.cpp .\Simulator.cu .\Runner.cu .\Kernels.cu .\BasicSimulation.cu

buildcc:
	@nvcc -rdc=true -O3 -lineinfo -o runner biology/Species.cpp biology/Specimen.cpp biology/Taxonomy.cpp biology/Genome.cpp Simulator.cu Runner.cu Kernels.cu simulations\BasicSimulation.cu simulations\TargetSimulation.cu simulations\MultibotSimulation.cu simulations\PongSimulation.cu simulations\PongSimulation2.cu simulations\AirHockeySimulation.cu -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe"

buildforsteve:
	@nvcc -rdc=true -lineinfo -o runner biology/Genome.cpp Simulator.cu Runner.cu Kernels.cu simulations\BasicSimulation.cu simulations\TargetSimulation.cu simulations\MultibotSimulation.cu simulations/PongSimulation.cu simulations\AirHockeySimulation.cu -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe"

run:
	.\runner
