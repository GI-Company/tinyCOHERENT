CC = cc
CFLAGS = -O2 -Wall -Wextra -std=c11 -Isrc
LDLIBS = -lm
SRC = src/tcmodel.c src/optim.c src/tokenizer.c src/tc_ops.c

ifeq ($(shell uname),Darwin)
CFLAGS += -DACCELERATE_NEW_LAPACK
LDLIBS += -framework Accelerate -framework Metal -framework Foundation -framework MetalPerformanceShaders
SRC += src/tc_metal.m
endif

BUILD = build

.PHONY: all gradcheck embed_gradcheck train train_scale train_rung6 train_rung6_continue embed_train dag embed rag faithcheck faithcheck_rung4 faithcheck_rung6 known_answer chat chat_rung6 clean

all: $(BUILD)/gradcheck $(BUILD)/embed_gradcheck $(BUILD)/train $(BUILD)/train_scale $(BUILD)/embed_train $(BUILD)/dag_demo $(BUILD)/embed_demo $(BUILD)/rag_demo $(BUILD)/faithcheck $(BUILD)/known_answer $(BUILD)/chat $(BUILD)/server $(BUILD)/eval_frozen $(BUILD)/official_samples

$(BUILD)/gradcheck: $(SRC) src/gradcheck.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/gradcheck.c $(LDLIBS)

$(BUILD)/embed_gradcheck: $(SRC) src/embed_gradcheck.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/embed_gradcheck.c $(LDLIBS)

$(BUILD)/train: $(SRC) src/glassbox.c src/train.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/glassbox.c src/train.c $(LDLIBS)

$(BUILD)/train_scale: $(SRC) src/glassbox.c src/bpe.c src/train_scale.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/glassbox.c src/bpe.c src/train_scale.c $(LDLIBS)

$(BUILD)/train_sft: $(SRC) src/glassbox.c src/bpe.c src/train_sft.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/glassbox.c src/bpe.c src/train_sft.c $(LDLIBS)

$(BUILD)/embed_train: $(SRC) src/embed_train.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/embed_train.c $(LDLIBS)

$(BUILD)/faithcheck: $(SRC) src/glassbox.c src/bpe.c src/faithcheck.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/glassbox.c src/bpe.c src/faithcheck.c $(LDLIBS)

$(BUILD)/known_answer: $(SRC) src/glassbox.c src/known_answer.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/glassbox.c src/known_answer.c $(LDLIBS)

$(BUILD)/degrade_test: $(SRC) src/glassbox.c src/degrade_test.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/glassbox.c src/degrade_test.c $(LDLIBS)

$(BUILD)/eval_frozen: $(SRC) src/glassbox.c src/bpe.c src/eval_frozen.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/glassbox.c src/bpe.c src/eval_frozen.c $(LDLIBS)

$(BUILD)/official_samples: $(SRC) src/glassbox.c src/bpe.c src/official_samples.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/glassbox.c src/bpe.c src/official_samples.c $(LDLIBS)

$(BUILD)/chat: $(SRC) src/glassbox.c src/bpe.c src/vectorstore.c src/steer.c src/chat.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/glassbox.c src/bpe.c src/vectorstore.c src/steer.c src/chat.c $(LDLIBS)

$(BUILD)/server: $(SRC) src/glassbox.c src/bpe.c src/vectorstore.c src/steer.c src/server.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/glassbox.c src/bpe.c src/vectorstore.c src/steer.c src/server.c $(LDLIBS)

$(BUILD)/steer_extract_bank: $(SRC) src/bpe.c src/steer.c src/steer_extract_bank.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/bpe.c src/steer.c src/steer_extract_bank.c $(LDLIBS)

$(BUILD)/test_steer: $(SRC) src/bpe.c src/steer.c src/test_steer.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/bpe.c src/steer.c src/test_steer.c $(LDLIBS)

$(BUILD)/dag_demo: $(SRC) src/glassbox.c src/registry.c src/dag.c src/vectorstore.c src/dag_demo.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/glassbox.c src/registry.c src/dag.c src/vectorstore.c src/dag_demo.c $(LDLIBS)

$(BUILD)/embed_demo: $(SRC) src/glassbox.c src/embed_demo.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/glassbox.c src/embed_demo.c $(LDLIBS)

$(BUILD)/rag_demo: $(SRC) src/glassbox.c src/registry.c src/dag.c src/vectorstore.c src/rag_demo.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/glassbox.c src/registry.c src/dag.c src/vectorstore.c src/rag_demo.c $(LDLIBS)

gradcheck: $(BUILD)/gradcheck $(BUILD)/embed_gradcheck
	./$(BUILD)/gradcheck
	./$(BUILD)/embed_gradcheck

train: $(BUILD)/train
	./$(BUILD)/train

train_scale: $(BUILD)/train_scale
	./$(BUILD)/train_scale

train_rung6: $(BUILD)/train_scale
	./$(BUILD)/train_scale --rung6

train_rung6_continue: $(BUILD)/train_scale
	./$(BUILD)/train_scale --rung6 --resume build/model_rung6.bin --steps 600 --lr 0.0008

train_sft: $(BUILD)/train_sft
	./$(BUILD)/train_sft --base build/model_rung4.bin --data data/sft_dialogue.txt --out build/model_sft.bin

embed_train: $(BUILD)/embed_train
	./$(BUILD)/embed_train

faithcheck: $(BUILD)/faithcheck
	./$(BUILD)/faithcheck build/model.bin build/embedder.bin

faithcheck_rung4: $(BUILD)/faithcheck
	./$(BUILD)/faithcheck build/model_rung4.bin -

faithcheck_rung6: $(BUILD)/faithcheck
	./$(BUILD)/faithcheck build/model_rung6.bin -

known_answer: $(BUILD)/known_answer
	./$(BUILD)/known_answer

chat: $(BUILD)/chat
	./$(BUILD)/chat build/model_rung3.bin

chat_rung4: $(BUILD)/chat
	./$(BUILD)/chat build/model_rung4.bin

chat_sft: $(BUILD)/chat
	./$(BUILD)/chat build/model_sft.bin

chat_rung6: $(BUILD)/chat
	./$(BUILD)/chat build/model_rung6.bin

ui: $(BUILD)/server
	./$(BUILD)/server 8080 build/model_rung4.bin

dag: $(BUILD)/dag_demo
	./$(BUILD)/dag_demo

embed: $(BUILD)/embed_demo
	./$(BUILD)/embed_demo

rag: $(BUILD)/rag_demo
	./$(BUILD)/rag_demo

extract_steer: $(BUILD)/steer_extract_bank
	./$(BUILD)/steer_extract_bank build/model_sft.bin data/bpe_merges.txt data/steering_vectors.bin

test_steer: $(BUILD)/test_steer
	./$(BUILD)/test_steer build/model_sft.bin data/bpe_merges.txt

clean:
	rm -f $(BUILD)/gradcheck $(BUILD)/embed_gradcheck $(BUILD)/train $(BUILD)/embed_train $(BUILD)/dag_demo $(BUILD)/embed_demo $(BUILD)/rag_demo $(BUILD)/faithcheck $(BUILD)/known_answer $(BUILD)/chat $(BUILD)/degrade_test $(BUILD)/train_scale $(BUILD)/train_sft $(BUILD)/server $(BUILD)/steer_extract_bank $(BUILD)/test_steer $(BUILD)/metalcheck

METAL ?= 0
ifeq ($(METAL),1)
CFLAGS += -DTC_METAL
LDLIBS += -framework Metal -framework Foundation -framework MetalPerformanceShaders
METAL_SRC = src/tc_metal.m
endif

$(BUILD)/metalcheck: $(SRC) src/tc_metal.m src/tc_metalcheck.c
	$(CC) $(CFLAGS) -DTC_METAL -fobjc-arc -o $@ $^ $(LDLIBS) -framework Metal -framework Foundation -framework MetalPerformanceShaders

metalcheck: $(BUILD)/metalcheck
	./$(BUILD)/metalcheck
