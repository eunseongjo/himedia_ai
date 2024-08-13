#STEP1
from transformers import pipeline
from fastapi import FastAPI, Form

#STEP2
summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")

app = FastAPI()


@app.post("/summerizer/")
async def summerizer(text: str = Form()):

    #STEP3

    #STEP4
    result = summerizer(text)

    return {"result": result}