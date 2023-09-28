INC_DIR = sample_generation
CXXFLAGS = -Wall -std=c++14 -I$(INC_DIR)/..
SAMPLE_GENERATION_SRCS = $(wildcard sample_generation/*.cc)

all: generate_samples

release: CXXFLAGS += -O3
release: generate_samples

DEBUG: debug

debug: CXXFLAGS += -DDEBUG -g
debug: generate_samples

generate_samples: generate_samples.cc $(SAMPLE_GENERATION_SRCS)
	g++ $(CXXFLAGS) -o generate_samples generate_samples.cc $(SAMPLE_GENERATION_SRCS)

clean:
	rm -Rf generate_samples generate_samples.dSYM