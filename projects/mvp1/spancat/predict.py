import spacy

nlp = spacy.load("./output/model-best")

test = nlp("I wonder if Commonwealth Care Alliance will ever fire Chris.")


spans = test.spans["sc"]
scores = spans.attrs["scores"]
for span, score in zip(spans, scores):
    print(f"span: {span}, score: {score}")
