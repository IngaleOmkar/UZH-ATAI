from openai import OpenAI

class AnswerWrapper:
    def __init__(self):
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    def wrap_answer(self, question, answer):
        try:
            completion = self.client.chat.completions.create(
                model="mistral-7b-instruct-v0.3",
                messages=[
                    {"role": "user", "content": f'''Here is a question and its answer. I want you to take the answer and make it sound more human-like. 
                     Do not hallucinate, do not make up new information and do not change the meaning of the answer. 
                     Just refine it to sound more human-like. You may use parts of the question in the answer if necessary. Some examples are:
                     
                     Question: "Who is the director of the movie Inception?" Answer: "Christopher Nolan"
                     Refinement: "The director of Inception is Christopher Nolan."

                     Question: "Who is the executive producer of X-Men: First Class? Answer: "Sheryl Lee Ralph"
                     Refinement: "Sheryl Lee Ralph is the executive producer of X-Men: First Class."

                     Question: {question}; Answer: {answer}
                    Refinement: 

                    Remember, only the refined answer is to be returned and not the entire thing. Good luck, I believe in you! 
                    '''},

                ],
                temperature=0.7,
                # place an upper bound on output token length
                max_tokens=100
            )

            return (True, completion.choices[0].message)
        except Exception as e:
            return (False, answer)