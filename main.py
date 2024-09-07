from dotenv import load_dotenv

load_dotenv()

import graph
from graph.graph import app

if __name__ == "__main__":
    print("Hello, cRAG!")
    # print(app.invoke(input={"question": "what is agent memory?"}))
