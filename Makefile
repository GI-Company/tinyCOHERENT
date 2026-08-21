CC = cc
CFLAGS = -O2 -Wall -Wextra -std=c11 -Isrc
LDLIBS = -lm

SRC = src/tcmodel.c src/optim.c src/tokenizer.c
BUILD = build

.PHONY: all gradcheck embed_gradcheck train embed_train dag embed rag faithcheck known_answer clean

all: $(BUILD)/gradcheck $(BUILD)/embed_gradcheck $(BUILD)/train $(BUILD)/embed_train $(BUILD)/dag_demo $(BUILD)/embed_demo $(BUILD)/rag_demo $(BUILD)/faithcheck $(BUILD)/known_answer

$(BUILD)/gradcheck: $(SRC) src/gradcheck.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/gradcheck.c $(LDLIBS)

$(BUILD)/embed_gradcheck: $(SRC) src/embed_gradcheck.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/embed_gradcheck.c $(LDLIBS)

$(BUILD)/train: $(SRC) src/glassbox.c src/train.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/glassbox.c src/train.c $(LDLIBS)

$(BUILD)/embed_train: $(SRC) src/embed_train.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/embed_train.c $(LDLIBS)

$(BUILD)/faithcheck: $(SRC) src/glassbox.c src/faithcheck.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/glassbox.c src/faithcheck.c $(LDLIBS)

$(BUILD)/known_answer: $(SRC) src/glassbox.c src/known_answer.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/glassbox.c src/known_answer.c $(LDLIBS)

$(BUILD)/degrade_test: $(SRC) src/glassbox.c src/degrade_test.c
	$(CC) $(CFLAGS) -o $@ $(SRC) src/glassbox.c src/degrade_test.c $(LDLIBS)

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

embed_train: $(BUILD)/embed_train
	./$(BUILD)/embed_train

faithcheck: $(BUILD)/faithcheck
	./$(BUILD)/faithcheck

known_answer: $(BUILD)/known_answer
	./$(BUILD)/known_answer

dag: $(BUILD)/dag_demo
	./$(BUILD)/dag_demo

embed: $(BUILD)/embed_demo
	./$(BUILD)/embed_demo

rag: $(BUILD)/rag_demo
	./$(BUILD)/rag_demo

clean:
	rm -f $(BUILD)/gradcheck $(BUILD)/embed_gradcheck $(BUILD)/train $(BUILD)/embed_train $(BUILD)/dag_demo $(BUILD)/embed_demo $(BUILD)/rag_demo $(BUILD)/faithcheck $(BUILD)/known_answer
