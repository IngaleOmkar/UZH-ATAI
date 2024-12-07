from openai import OpenAI

class AnswerWrapper:
    def __init__(self):
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    def wrap_answer(self, question, answer):
        completion = self.client.chat.completions.create(
            model="mistral-7b-instruct-v0.3",
            messages=[
                {"role": "user", "content": "Here is a question and its answer. I want you to take the answer and make it sound more human-like. Do not hallucinate, make up new information or change the meaning of the answer. Just refine it to sound more human-like. You may use parts of the question in the answer if necessary."},
                {"role": "user", "content": f"Question: {question}; Answer: {answer}"},
            ],
            temperature=0.7,
        )

        return completion.choices[0].message