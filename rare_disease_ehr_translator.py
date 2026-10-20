import os
import logging
from typing import Optional
from openai import OpenAI, OpenAIError

# Configure logging for professional tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class RareDiseaseEHRTranslator:
    """
    A professional clinical text translation pipeline utilizing Large Language Models (LLMs).
    Specialized for translating de-identified Chinese Electronic Health Records (EHRs) 
    of rare diseases into standard, interoperable English representations.
    """

    DEFAULT_SYSTEM_PROMPT = """
    You are a professional medical language model specialized in rare disease diagnosis and clinical text understanding. Your task is to translate de-identified Chinese electronic health records (EHRs) into accurate, fluent, and medically precise English while strictly preserving clinical semantics.
    
    Requirements:
    1. Maintain all clinical entities (e.g., symptoms, laboratory findings, imaging results, gene variants, treatments) with accurate medical terminology.
    2. Translate disease names, phenotypes, and diagnostic terms into their standard English medical equivalents (e.g., use HPO or Orphanet naming conventions when applicable).
    3. Retain the original structure and context of the clinical record, including sections such as chief complaint, history of present illness, past medical history, examination findings and test results.
    4. Do not omit or interpret information — translate faithfully without adding inferred content.
    5. Ensure the output is suitable for downstream phenotype extraction and clinical reasoning tasks.
    """

    def __init__(
        self, 
        api_key: Optional[str] = None, 
        base_url: str = "https://api.deepseek.com",
        model_name: str = "deepseek-reasoner",
        system_prompt: Optional[str] = None
    ) -> None:
        """
        Initialize the translator client.

        Args:
            api_key (Optional[str]): API authentication key. If None, fetches from env var 'DEEPSEEK_API_KEY'.
            base_url (str): The base URL for the LLM API endpoint.
            model_name (str): The specific model version to utilize.
            system_prompt (Optional[str]): Custom system instructions. If None, uses DEFAULT_SYSTEM_PROMPT.
        """
        self.api_key = api_key or os.environ.get('DEEPSEEK_API_KEY')
        if not self.api_key:
            logger.error("API key is missing. Please set the DEEPSEEK_API_KEY environment variable.")
            raise ValueError("API Key is required for initialization.")

        self.base_url = base_url
        self.model_name = model_name
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Initialize the LLM client
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logger.info(f"Successfully initialized LLM client with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    def translate_clinical_text(self, source_text: str) -> Optional[str]:
        """
        Execute the translation of the provided clinical text.

        Args:
            source_text (str): The original de-identified Chinese EHR text.

        Returns:
            Optional[str]: The translated medical English text, or None if the API call fails.
        """
        if not source_text or not source_text.strip():
            logger.warning("Empty source text provided. Skipping translation.")
            return ""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Input:\n{source_text}"}
        ]

        logger.info("Initiating translation request to the LLM API...")

        try:
            # Initiate API request with advanced reasoning configurations
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False,
                extra_body={
                    "reasoning_effort": "high",
                    "thinking": {"type": "enabled"}
                }
            )
            
            translated_text = response.choices[0].message.content
            logger.info("Translation completed successfully.")
            return translated_text

        except OpenAIError as api_err:
            logger.error(f"LLM API Error encountered during translation: {str(api_err)}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during translation: {str(e)}")
            return None

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    import argparse
    from rareseek.io import path, read_jsonl, write_jsonl, by_id
    p = argparse.ArgumentParser(description="Optional DeepSeek translation; explicitly sends non-English context to that provider")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--language", help="Input language for records without a language field")
    a = p.parse_args()
    rows = read_jsonl(path(a.input))
    by_id(rows)
    translator = None
    for row in rows:
        if "language" not in row:
            if not a.language:
                raise ValueError("Supply a language field or --language")
            row["language"] = a.language
        if row["language"] != "en":
            translator = translator or RareDiseaseEHRTranslator()
            translated = translator.translate_clinical_text(row["context"])
            if not translated:
                raise RuntimeError("Translation failed")
            row["context"] = translated
            row["language"] = "en"
    write_jsonl(path(a.output), rows)
