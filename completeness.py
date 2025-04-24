import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import List, Dict, Any, Optional, Union
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import pickle as pkl


from validator import GPTValidator, Validator, DummyValidator


def load_prediction_results(file_path: str) -> List[float]:
    """
    Load prediction results from a file saved by get_representative.py's evaluate_and_save_predictions function.

    Args:
        file_path: Path to the saved prediction results file

    Returns:
        List of probabilities for all validation samples (A samples followed by B samples)
    """
    with open(file_path, 'rb') as f:
        results = pkl.load(f)

    # Combine A and B probabilities in the same order as validation samples (A followed by B)
    all_probabilities = np.concatenate([results['A_probabilities'], results['B_probabilities']]).tolist()

    return all_probabilities


class RoBERTaPredictor:
    """
    A predictor that uses a RoBERTa model to predict group membership.
    This uses the same approach as in get_representative.py but simplified for prediction.
    """

    def __init__(self, A_samples: List[str], B_samples: List[str], verbose: bool = False):
        """
        Initialize the RoBERTa predictor by training a model on the provided samples.

        Args:
            A_samples: List of samples from group A (treatment)
            B_samples: List of samples from group B (control)
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pretrain_model = "roberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrain_model)
        self.max_length = 128
        self.batch_size = 16

        # Train the model
        if self.verbose:
            print("Training RoBERTa model for non-parametric prediction...")

        self.model = self._train_model(A_samples, B_samples)

    def _train_model(self, A_samples: List[str], B_samples: List[str]) -> AutoModelForSequenceClassification:
        """
        Train a RoBERTa model on the provided samples.

        Args:
            A_samples: List of samples from group A (treatment)
            B_samples: List of samples from group B (control)

        Returns:
            Trained RoBERTa model
        """
        from itertools import chain
        import random
        from transformers import get_linear_schedule_with_warmup
        import tqdm

        # Prepare training data
        train_data_dicts = list(
            chain(
                [{"input": x, "label": 1} for x in A_samples],
                [{"input": x, "label": 0} for x in B_samples],
            )
        )

        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(self.pretrain_model).to(self.device)

        # Set up optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=5e-5)

        # Set up scheduler
        num_steps = 2000
        scheduler = get_linear_schedule_with_warmup(optimizer, 400, num_steps)

        # Train the model
        model.train()

        if self.verbose:
            iterator = tqdm.trange(num_steps)
        else:
            iterator = range(num_steps)

        for step in iterator:
            random.shuffle(train_data_dicts)
            input_texts = [d["input"] for d in train_data_dicts[:self.batch_size]]
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self.device)
            labels = torch.tensor([d["label"] for d in train_data_dicts[:self.batch_size]]).to(self.device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            loss.backward()
            if step % 2 == 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        return model

    def predict(self, texts: List[str]) -> List[float]:
        """
        Predict the probability that each text belongs to group A.

        Args:
            texts: List of texts to predict

        Returns:
            List of probabilities (between 0 and 1) that each text belongs to group A
        """
        self.model.eval()
        lsm = torch.nn.LogSoftmax(dim=-1)
        all_logits = []

        with torch.no_grad():
            cur_start = 0
            while cur_start < len(texts):
                texts_ = texts[cur_start : cur_start + self.batch_size]
                inputs = self.tokenizer(
                    texts_,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                ).to(self.device)
                logits = self.model(**inputs).logits
                logits = lsm(logits.detach().cpu()).numpy().tolist()
                all_logits.extend(logits)
                cur_start += self.batch_size

        # Extract probabilities for group A (index 1)
        probabilities = [np.exp(logit[1]) for logit in all_logits]

        return probabilities


class GPTPredictor:
    """
    A predictor that uses an LLM with an improved prompt to predict group membership.
    """

    def __init__(self, problem: Dict[str, Any], verifier_name: str = "gpt", verbose: bool = False):
        """
        Initialize the LLM predictor.

        Args:
            problem: Problem definition dictionary
            verifier_name: Name of the verifier to use ("gpt", "dummy", or a model path)
            verbose: Whether to print progress information
        """
        self.problem = problem
        self.verbose = verbose

        # Create the validator
        if verifier_name == "dummy":
            self.verifier = DummyValidator()
        elif verifier_name == "gpt":
            self.verifier = GPTValidator(verbose=verbose)
            # Override the template with an improved prompt
            self.verifier.validator_template = """
You are an expert at analyzing teaching strategies in educational settings.

I'm going to show you a transcript from a math classroom. Your task is to determine if this transcript is from a treatment group where teachers were coached to use metacognitive modeling strategies.

Metacognitive modeling is defined as thinking aloud about thinking in order to make a strategy, task, or process more accessible to students. Teachers in the treatment group were specifically trained to use this approach.

Classroom Transcript:
{text}

Based on the transcript, is this from the treatment group where teachers were coached to use metacognitive modeling? Look for evidence of the teacher thinking aloud about their thinking process, making their reasoning explicit, or modeling how to approach problems.

Answer with only 'Yes' or 'No'.
"""
        else:
            raise ValueError(f"Unknown verifier name: {verifier_name}")

    def predict(self, texts: List[str]) -> List[float]:
        """
        Predict the probability that each text belongs to the treatment group.

        Args:
            texts: List of texts to predict

        Returns:
            List of probabilities (between 0 and 1) that each text belongs to the treatment group
        """
        # Create input dictionaries for the validator
        validator_dicts = []
        for text in texts:
            # We don't need a hypothesis for the improved prompt
            validator_dict = {'hypothesis': "", 'text': text}
            validator_dicts.append(validator_dict)

        # Get scores from the validator
        scores = list(self.verifier.validate_w_scores(validator_dicts))

        # Store the samples and scores to a csv file
        df = pd.DataFrame({'text': texts, 'score': scores})
        df.to_csv("results/llm_scores.csv", index=False)

        return scores


def validate_samples_for_themes(h2h_dicts: Dict[str, Dict[str, Any]], 
                              samples: List[str], 
                              problem: Dict[str, Any],
                              verifier_name: str = "gpt",
                              verbose: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Validate samples for themes that don't have scores for these samples.

    Args:
        h2h_dicts: Dictionary from D5 containing hypothesis evaluation results
        samples: List of samples to validate
        problem: Problem definition dictionary containing validation samples
        verifier_name: Name of the verifier to use ("gpt", "dummy", or a model path)
        verbose: Whether to print progress information

    Returns:
        Updated h2h_dicts with scores for the samples
    """
    import random

    # Create a copy of h2h_dicts to avoid modifying the original
    h2h_dicts_copy = {h: {k: v.copy() if isinstance(v, dict) else v for k, v in h_dict.items()} 
                      for h, h_dict in h2h_dicts.items()}

    # For each theme, check which samples need validation
    validator_dicts = []
    theme_sample_map = {}  # Maps (theme, sample) to index in validator_dicts

    if verbose:
        print("Checking which samples need validation...")

    total_missing = 0
    for h, h_dict in h2h_dicts_copy.items():
        for sample in samples:
            if sample not in h_dict['sample2score']:
                validator_dict = {'hypothesis': h, 'text': sample}
                validator_dicts.append(validator_dict)
                theme_sample_map[(h, sample)] = len(validator_dicts) - 1
                total_missing += 1

    if total_missing == 0:
        if verbose:
            print("All samples already have scores for all themes.")
        return h2h_dicts_copy

    if verbose:
        print(f"Validating {total_missing} missing sample-theme pairs...")

    # Create the validator
    if verifier_name == "dummy":
        from validator import DummyValidator
        verifier = DummyValidator()
    elif verifier_name == "gpt":
        from validator import GPTValidator
        verifier = GPTValidator(verbose=verbose)
    else:
        from validator import Validator
        verifier = Validator(verifier_name, batch_size=32)

    # Get scores from the validator
    eps = 1e-5  # Small perturbation to avoid ties, same as in D5.py
    all_scores = list(verifier.validate_w_scores(validator_dicts))

    # Update h2h_dicts with the scores
    for (h, sample), idx in theme_sample_map.items():
        score = all_scores[idx] + eps * random.random()
        h2h_dicts_copy[h]['sample2score'][sample] = score

        # Also update sample2corpus if it exists
        if 'sample2corpus' in h2h_dicts_copy[h]:
            # Determine if this sample is from corpus A or B
            A_validation = problem['split']['validation']['A_samples']
            B_validation = problem['split']['validation']['B_samples']

            if sample in A_validation:
                corpus = 'A'
            elif sample in B_validation:
                corpus = 'B'
            else:
                # If not found in validation samples, check research samples
                A_research = problem['split']['research']['A_samples']
                B_research = problem['split']['research']['B_samples']

                if sample in A_research:
                    corpus = 'A'
                elif sample in B_research:
                    corpus = 'B'
                else:
                    # If still not found, use a default value
                    corpus = 'unknown'

            h2h_dicts_copy[h]['sample2corpus'][sample] = corpus

    if verbose:
        print(f"Validated {total_missing} missing sample-theme pairs.")

    return h2h_dicts_copy


def calculate_completeness(h2h_dicts: Dict[str, Dict[str, Any]], 
                          problem: Dict[str, Any], 
                          nonparam_method: str = "llm",
                          verifier_name: str = "gpt",
                          verbose: bool = False,
                          nonparam_scores: Optional[List[float]] = None) -> (Dict[str, float], Dict):
    """
    Calculate the completeness of themes in describing group differences.

    Args:
        h2h_dicts: Dictionary from D5 containing hypothesis evaluation results
        problem: Problem definition dictionary
        nonparam_method: Method to use for non-parametric prediction ("llm" or "roberta")
        verifier_name: Name of the verifier to use for LLM prediction ("gpt", "dummy", or a model path)
        verbose: Whether to print progress information
        nonparam_scores: Optional pre-computed non-parametric prediction scores for validation samples.
                         If provided, these scores will be used instead of computing new predictions.

    Returns:
        Dictionary containing completeness measure and related metrics
    """
    # Extract validation samples
    A_validation = problem['split']['validation']['A_samples']
    B_validation = problem['split']['validation']['B_samples']

    # Combine validation samples and create labels
    validation_samples = A_validation + B_validation
    validation_labels = [1] * len(A_validation) + [0] * len(B_validation)

    # Validate validation samples for all themes
    if verbose:
        print("Pre-validating validation samples for all themes...")
    h2h_dicts = validate_samples_for_themes(h2h_dicts, validation_samples, problem, verifier_name, verbose)

    # 1. Trivial predictor (constant prediction)
    # Calculate the majority class
    majority_class = 1 if sum(validation_labels) > len(validation_labels) / 2 else 0
    trivial_predictions = [majority_class] * len(validation_labels)
    trivial_loss = 1 - accuracy_score(validation_labels, trivial_predictions)

    # 2. Theme-based predictor (logistic regression on theme scores)
    # Get the top hypotheses (themes)
    h_sorted = sorted(h2h_dicts, key=lambda h: h2h_dicts[h]['diff_w_significance']['mu'], reverse=True)
    top_themes = h_sorted[:20]  # Use top 20 themes

    # Create feature matrix for validation samples
    X_theme = []
    missing_samples_count = 0
    for sample in validation_samples:
        sample_features = []
        for theme in top_themes:
            # If the sample has a score for this theme, use it
            # Otherwise, use 0.5 as a neutral score (should not happen after pre-validation)
            if sample in h2h_dicts[theme]['sample2score']:
                score = h2h_dicts[theme]['sample2score'][sample]
            else:
                missing_samples_count += 1
                score = 0.5
            sample_features.append(score)
        X_theme.append(sample_features)

    if missing_samples_count > 0 and verbose:
        print(f"Note: {missing_samples_count} validation sample-theme pairs were not found in the sample2score dictionary after pre-validation. Using default score of 0.5.")

    # Train logistic regression on research data
    X_train = []
    y_train = []

    # Get research samples
    A_research = problem['split']['research']['A_samples']
    B_research = problem['split']['research']['B_samples']
    research_samples = A_research + B_research
    research_labels = [1] * len(A_research) + [0] * len(B_research)

    # Create feature matrix for research samples
    research_missing_count = 0
    for i, sample in enumerate(research_samples):
        sample_features = []
        for theme in top_themes:
            # If the sample has a score for this theme, use it
            # Otherwise, use 0.5 as a neutral score
            if sample in h2h_dicts[theme]['sample2score']:
                score = h2h_dicts[theme]['sample2score'][sample]
            else:
                research_missing_count += 1
                score = 0.5
            sample_features.append(score)
        X_train.append(sample_features)
        y_train.append(research_labels[i])

    if research_missing_count > 0 and verbose:
        print(f"Note: {research_missing_count} research sample-theme pairs were not found in the sample2score dictionary. Using default score of 0.5.")

    # Train logistic regression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)

    # Print accuracy and coefficients for each theme on training data
    if verbose:
        print("Logistic Regression Coefficients:")
        for i, theme in enumerate(top_themes):
            print(f"Theme: {theme}, Coefficient: {lr.coef_[0][i]}")
        print("Accuracy on training data:", accuracy_score(y_train, lr.predict(X_train)))


    # Predict on validation data
    theme_predictions = lr.predict(X_theme)
    theme_loss = 1 - accuracy_score(validation_labels, theme_predictions)

    # 3. Non-parametric benchmark
    if nonparam_scores is not None:
        if verbose:
            print("Using provided non-parametric prediction scores...")

        # Ensure we have the correct number of scores
        if len(nonparam_scores) != len(validation_samples):
            raise ValueError(f"Number of provided scores ({len(nonparam_scores)}) does not match number of validation samples ({len(validation_samples)})")

    else:
        if nonparam_method == "llm":
            if verbose:
                print("Using LLM with improved prompt for non-parametric prediction...")

            # Use LLM with improved prompt
            llm_predictor = GPTPredictor(problem, verifier_name=verifier_name, verbose=verbose)
            nonparam_scores = llm_predictor.predict(validation_samples)

        elif nonparam_method == "roberta":
            if verbose:
                print("Using RoBERTa model for non-parametric prediction...")

            # Use RoBERTa model
            roberta_predictor = RoBERTaPredictor(A_research, B_research, verbose=verbose)
            nonparam_scores = roberta_predictor.predict(validation_samples)

        else:
            raise ValueError(f"Unknown non-parametric method: {nonparam_method}")

    # Convert scores to predictions
    nonparam_predictions = [1 if score > 0.5 else 0 for score in nonparam_scores]
    nonparam_loss = 1 - accuracy_score(validation_labels, nonparam_predictions)

    # Calculate completeness
    if trivial_loss == nonparam_loss:
        # If there's no predictive information in the text, return 0
        completeness = 0
    else:
        completeness = (trivial_loss - theme_loss) / (trivial_loss - nonparam_loss)

    return {
        'completeness': completeness,
        'trivial_loss': trivial_loss,
        'theme_loss': theme_loss,
        'nonparam_loss': nonparam_loss,
        'theme_accuracy': 1 - theme_loss,
        'nonparam_accuracy': 1 - nonparam_loss,
        'trivial_accuracy': 1 - trivial_loss
    }, h2h_dicts
