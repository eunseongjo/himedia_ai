
#STEP1
from transformers import pipeline
# from transformers import AutoTokenizer
# from transformers import AutoModelForSequenceClassification


#STEP2
classifier = pipeline("sentiment-analysis", model="WhitePeak/bert-base-cased-Korean-sentiment")
# tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
# model = AutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")


#STEP3
text = "맛없어"

#STEP4
result = classifier(text)
# with torch.no_grad():
#     logits = model(**inputs).logits

# predicted_class_id = logits.argmax().itme()
# result = model.config.id2label[predicted_class_id]

#STEP5
print(result)