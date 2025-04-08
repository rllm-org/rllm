import random
import socket
import time
from typing import List

import torch
import vertexai
import openai

from google.cloud.aiplatform_v1beta1.types.content import SafetySetting
from sentence_transformers import SentenceTransformer, util
from vertexai.generative_models import GenerationConfig, GenerativeModel, HarmBlockThreshold, HarmCategory


from rllm.globals import GCP_PROJECT_ID, GCP_LOCATION, GEMINI_MODEL, OAI_RM_MODEL

def call_oai_rm_llm(
    prompt: str,
    system_prompt: str,
    n: int = 1,
    temperature: float = 1.0,
    model_id: str = OAI_RM_MODEL,

    retry_count: int = 1e9,
) -> List[str]:
    client = openai.OpenAI()

    backoff = 1
    retry_count = int(retry_count)
    
    for attempt in range(retry_count):
        try: 
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=temperature,
                n=n,
            )
            break
        except Exception as e:
            if "429" in str(e):
                print("Retry due to rate limit: ", e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)  # Exponential backoff up to 64s
                continue
            else:
                print("Exception: ", e)
                return []

    if n == 1:
        return response.choices[0].message.content
    return [choice.message.content for choice in response.choices]


def call_gemini_llm(
    prompt: str,
    system_prompt: str,
    n: int = 1,
    temperature: float = 1.0,
    project_id: str = GCP_PROJECT_ID,
    location: str = GCP_LOCATION,
    model_id: str = GEMINI_MODEL,
    retry_count: int = 1e9,
) -> List[str]:
    """
    Calls a Gemini LLM on Vertex AI to generate n responses at a given temperature.
    
    Args:
        prompt (str): The text prompt to send to the LLM.
        system_prompt (str): System instruction or system prompt to send to the model.
        n (int): Number of responses to generate.
        temperature (float): Sampling temperature.
        project_id (str): Your GCP project ID.
        location (str): The region to use (e.g., us-central1).
        model_id (str): The specific Gemini model resource name.
        retry_count (int): Number of times to retry on rate-limit errors.
    
    Returns:
        List[str]: A list of response texts from the Gemini model.
    """

    # Initialize the Vertex AI environment
    vertexai.init(project=project_id, location=location)

    # Define which harm categories to allow (or set thresholds).
    HARM_CATEGORIES = [
        HarmCategory.HARM_CATEGORY_UNSPECIFIED,
        HarmCategory.HARM_CATEGORY_HARASSMENT,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    ]

    # Instantiate the GenerativeModel
    model = GenerativeModel(
        model_name=model_id,
        system_instruction=[system_prompt],
    )

    # Add an exponential backoff for rate limit errors
    backoff = 1
    retry_count = int(retry_count)
    generation_config = GenerationConfig(
        temperature=temperature,
        candidate_count=n,
    )

    for attempt in range(retry_count):
        try:
            # Request multiple candidates by specifying n (candidate_count)
            response = model.generate_content(
                [prompt],
                generation_config=generation_config,
                safety_settings=[
                    SafetySetting(category=h, threshold=HarmBlockThreshold.BLOCK_NONE)
                    for h in HARM_CATEGORIES
                ]
            )
            # Once successful, break out of the retry loop
            break
        except Exception as e:
            # Retry if there's a rate-limit error (HTTP 429)
            if "429" in str(e):
                print("Retry due to rate limit: ", e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)  # Exponential backoff up to 64s
                continue
            elif "403" in str(e):
                print("NO ACCESS TO ENDPOINT", e)
                raise NotImplementedError
            else:
                print("Exception: ", e)
                return []  # or raise an exception if desired

    # Collect the texts from all returned candidates
    # Depending on the library version, this might need to be adjusted 
    # if the `response` shape is different

    try:
        # Keep this to check for errors in indexing.
        [candidate.text for candidate in response.candidates]
        if len(response.candidates) == 1:
            return response.candidates[0].text
        return [candidate.text for candidate in response.candidates]
    except Exception as e:
        print("Error extracting text from response:", e)
        return []

class RAG:

    def __init__(self, docs: List[str], model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            docs (List[str]): A list of documents to encode.
            model (str): The SentenceTransformer model to use.
        """
        # Load the SentenceTransformer model
        self.model = SentenceTransformer(model)
        self.docs = docs
        # Compute embeddings
        self.embeddings = self.model.encode(docs, convert_to_tensor=True)
    
    def top_k(self, query, k=1):
        # Create embedding for the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute cosine similarity [1 x N]
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]

        # Extract top_k indices
        top_results = torch.topk(cos_scores, k=k)

        # Prepare a list of (score, problem_text)
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append({
                'score': score,
                'text': self.docs[int(idx)],
                'idx': int(idx),
            })
        return results

def find_available_ports(base_port: int, count: int) -> List[int]:
    """Find consecutive available ports starting from base_port."""
    available_ports = []
    current_port = base_port

    while len(available_ports) < count:
        if is_port_available(current_port):
            available_ports.append(current_port)
        current_port += random.randint(100, 1000)

    return available_ports


def is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except socket.error:
            return False
        except OverflowError:
            return False

if __name__ == '__main__':
    print(is_port_available(8000))