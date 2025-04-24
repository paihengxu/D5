import pickle as pkl
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict
from tqdm import trange
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
sm = torch.nn.Softmax(dim=-1)
MAX_TARGET_LENGTH = 2
YES_NO_TOK_IDX = [150, 4273]
MAX_SOURCE_LENGTH = 1024
TEMPERATURE = 0.001


# if you have multiple GPUs, you can parallelize the T5 model across them
# future packages version will probably have more efficient ways to do this
def parallelize_across_device(model):
    num_heads = len(model.encoder.block)
    num_device = torch.cuda.device_count()
    other_device_alloc = num_heads // num_device + 1
    first_device = num_heads - (num_device - 1) * other_device_alloc
    device_map = {}
    cur = 0
    end = max(cur + first_device, 1)
    device_map[0] = list(range(cur, end))
    cur = end
    for i in range(1, num_device):
        end = min(cur + other_device_alloc, num_heads)
        device_map[i] = list(range(cur, end))
        cur += other_device_alloc
    print('device_map', device_map)
    model.parallelize(device_map)


DEFAULT_VALIDATOR_TEMPLATE = open('templates/t5_validator.txt', 'r').read()


class Validator:

    # model_path is the path to the T5 model weights used for validation
    # can also any other model name
    # the default is the best model we have trained
    def __init__(self, model_path: str = 'ruiqi-zhong/d5_t5_validator', batch_size: int = BATCH_SIZE,
                 verbose: bool = False, template: str = DEFAULT_VALIDATOR_TEMPLATE):
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        print('loading model weights')
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        print('done')
        parallelize_across_device(self.model)
        self.validator_template = template
        self.batch_size = batch_size
        self.verbose = verbose

    # input_dicts is a list of dictionaries, each dictionary has two keys: "hypothesis" (h) and "text" (x), mapping to the hypothesis and text to be validated
    # returns a list of scores, each score is a float between 0 and 1, corresponding to the probability that the hypothesis is true given the text for each input dictionary
    # note that it is an iterator, so you can use it in a for loop and save the results whenever some input dictionaries are processed
    def validate_w_scores(self, input_dicts: List[Dict[str, str]]) -> List[float]:
        prompts = []
        for i, input_dict in enumerate(input_dicts):
            hypothesis, text = input_dict['hypothesis'], input_dict['text']
            prompts.append(self.validator_template.format(hypothesis=hypothesis, text=text))

        with torch.no_grad():
            self.model.eval()
            num_batches = (len(prompts) - 1) // self.batch_size + 1
            if self.verbose:
                pbar = trange(num_batches)
                pbar.set_description('inference')
            else:
                pbar = range(num_batches)

            for batch_idx in pbar:
                input_prompts = prompts[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
                inputs = self.tokenizer(input_prompts,
                                        return_tensors="pt",
                                        padding="longest",
                                        max_length=MAX_SOURCE_LENGTH,
                                        truncation=True,
                                        ).to(device)
                generation_result = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    do_sample=True,
                    temperature=TEMPERATURE,
                    max_new_tokens=MAX_TARGET_LENGTH,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                scores = sm(generation_result.scores[0][:, YES_NO_TOK_IDX])[:, 1].detach().cpu().numpy().tolist()
                for s in scores:
                    yield s


class DummyValidator:

    def __init__(self):
        print('!!!!! WARNING: You are using a dummy verifier that returns random results!!!!!!!')
        pass

    def validate_w_scores(self, ind_dicts):
        for _ in range(len(ind_dicts)):
            yield 0.01


class GPTValidator:
    """
    A validator that uses OpenAI's GPT-4o mini model to validate hypotheses against texts.
    """
    
    def __init__(self, temperature=0, batch_size=100, verbose=False):
        """
        Initialize the GPT-4o mini validator.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from OPENAI_API_KEY environment variable.
            temperature: Temperature for the model generation. Lower means more deterministic.
            batch_size: Number of examples to process in a single API call.
            verbose: Whether to print progress information.
        """
        from openai import OpenAI
        from dotenv import load_dotenv

        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided or set as OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=self.api_key)
        self.temperature = temperature
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Template for the prompt
        self.validator_template = """
Consider the following hypothesis and text. Determine if the hypothesis is supported by the text.

Hypothesis: {hypothesis}

Text: {text}

Is the hypothesis supported by the text? Answer with only 'Yes' or 'No'.
"""

    def validate_w_scores(self, input_dicts: List[Dict[str, str]]) -> List[float]:
        """
        Validate a list of input dictionaries, each containing a hypothesis and text.
        
        Args:
            input_dicts: List of dictionaries, each with keys 'hypothesis' and 'text'.
            
        Returns:
            An iterator that yields scores between 0 and 1 for each input dictionary,
            representing the probability that the hypothesis is supported by the text.
        """
        
        batch = []
        for i, input_dict in enumerate(input_dicts):
            hypothesis, text = input_dict['hypothesis'], input_dict['text']
            prompt = self.validator_template.format(hypothesis=hypothesis, text=text)
            batch.append(prompt)
            
            # Process the batch if it reaches the batch size or this is the last item
            if len(batch) == self.batch_size or i == len(input_dicts) - 1:
                if self.verbose:
                    print(f"Processing batch {i // self.batch_size + 1} of {(len(input_dicts) - 1) // self.batch_size + 1}")
                
                results = self._process_batch(batch)
                
                for result in results:
                    yield result
                
                # Reset batch
                batch = []
    
    def _process_batch(self, batch):
        """
        Process a batch of prompts using the OpenAI API.
        
        Args:
            batch: List of prompts to process.
            
        Returns:
            List of scores between 0 and 1.
        """
        results = []
        
        # Process each prompt individually to get yes/no answers
        for prompt in batch:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=10
                )
                
                answer = response.choices[0].message.content.strip().lower()
                
                # Convert yes/no to a score
                if "yes" in answer:
                    score = 1.0
                elif "no" in answer:
                    score = 0.0
                else:
                    # Handle ambiguous responses - treat as uncertain
                    score = 0.5
                    
                results.append(score)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error processing prompt: {e}")
                # On error, return a neutral score
                results.append(0.5)
        
        return results


if __name__ == '__main__':
    input_dicts = [
        {'hypothesis': 'is a positive review', 'text': 'I like this movie.'},
        {'hypothesis': 'is a positive review', 'text': 'I hate this movie.'}
    ]
    input_dicts = input_dicts * 5
    
    # Make sure your API key is set in the environment variable OPENAI_API_KEY
    # or pass it directly to the constructor
    validator = GPTValidator(verbose=True)
    
    all_results = []
    for s in validator.validate_w_scores(input_dicts):
        all_results.append(s)
    
    import numpy as np
    all_results = np.array(all_results)
    pred = (all_results > 0.5).astype(int)
    gold = np.array([1, 0] * 5)
    print('acc', (pred == gold).mean())