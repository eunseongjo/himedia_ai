#STEP1
from transformers import pipeline
from fastapi import FastAPI, Form

#STEP2
classifier = pipeline("sentiment-analysis", model="WhitePeak/bert-base-cased-Korean-sentiment")
summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
question_answerer = pipeline("question-answering", model="stevhliu/my_awesome_qa_model")


app = FastAPI()


@app.post("/classification/")
async def classification(text: str = Form()):

    #STEP3

    #STEP4
    result = classifier(text)

    #STEP5
    print(result)

    return {"result": result}

@app.post("/summarization/")
async def summarization(text: str = Form()):

    #STEP3

    #STEP4
    result = summarization(text)

    return {"result": result}

@app.post("/qna/")
async def qna(question: str = Form(), context: str = Form()):

    # STEP 3
    # TEXT

    # STEP 4
    result = question_answerer(question=question, context=context)

    return {"result": result}