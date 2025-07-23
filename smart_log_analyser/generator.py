import yaml
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

class Generator:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        llm_cfg = config['llm']
        self.llm = ChatGoogleGenerativeAI(
            model=llm_cfg['model_name'],  # Required, no default
            temperature=llm_cfg.get('temperature', 0.1),
            max_output_tokens=llm_cfg.get('max_output_tokens', 2048),
            top_p=llm_cfg.get('top_p', 0.8),
            top_k=llm_cfg.get('top_k', 40)
        )
        self.prompt = PromptTemplate(
            input_variables=["context"],
            template="""Given the following context, generate a helpful response.\n\nContext:\n{context}\n\nResponse:"""
        )

    def generate(self, context: str) -> str:
        prompt_text = self.prompt.format(context=context)
        response = self.llm.invoke(prompt_text)
        return response.content