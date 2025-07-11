# Contents of /hello-world-rag/hello-world-rag/src/index.py

from retriever import Retriever
from generator import Generator

def main():
    retriever = Retriever()
    generator = Generator()

    # Retrieve context
    context = retriever.retrieve()

    # Generate response
    response = generator.generate(context)
    print(response)

if __name__ == "__main__":
    main()