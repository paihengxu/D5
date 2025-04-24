from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
import torch
from collections import OrderedDict, defaultdict
import os
import random
from itertools import chain
import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import time
import pickle as pkl
from typing import List, Dict, Any


# set up the directory
tmp_model_dir = 'tmp_models'
if not os.path.exists(tmp_model_dir):
    os.mkdir(tmp_model_dir)

NUM_FOLD = 4 # split the data into 4 folds. train on three to predict on the fourth
bsize = 16 # batch size to fine-tune the Roberta model
NUM_STEPS = 2000 # number of fine-tuning steps
max_length = 512 # max length of the input text
DEBUG = False

# hyperparameters for debugging
if DEBUG:
    NUM_STEPS = 300
    NUM_FOLD = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
pretrain_model = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
lsm = torch.nn.LogSoftmax(dim=-1)

# create cross validation folds
# where each fold is represented by a set of training and test A and B samples
# Usually A_samples are the research split of Corpus A and B_samples are the research split of Corpus B
# K is the number of folds, usually set to 4
def cv(A_samples: List[str], B_samples: List[str], K: int) -> List[Dict[str, List[str]]]:
    return [
        {
            "train_A": [p for i, p in enumerate(A_samples) if i % K != k],
            "train_B": [n for i, n in enumerate(B_samples) if i % K != k],
            "test_A": [p for i, p in enumerate(A_samples) if i % K == k],
            "test_B": [n for i, n in enumerate(B_samples) if i % K == k],
        }
        for k in range(K)
    ]

# fine-tune a Roberta model on the training samples from the cross validation fold
# return the model
def train(cv_dict: Dict[str, List[str]]) -> AutoModelForSequenceClassification:
    train_data_dicts = list(
        chain(
            [{"input": x, "label": 1} for x in cv_dict["train_A"]],
            [{"input": x, "label": 0} for x in cv_dict["train_B"]],
        )
    )

    model = AutoModelForSequenceClassification.from_pretrained(pretrain_model).to(device)
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
    scheduler = get_linear_schedule_with_warmup(optimizer, 400, NUM_STEPS)
    model.train()

    for step in tqdm.trange(NUM_STEPS):
        random.shuffle(train_data_dicts)
        input_texts = [d["input"] for d in train_data_dicts[:bsize]]
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        labels = torch.tensor([d["label"] for d in train_data_dicts[:bsize]]).to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        loss.backward()
        if step % 2 == 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return model


# evaluate the model on the test sample
# return the logits for each sample
# the shape of the logits is (num_samples, 2)
# where the first column is the logit for the B class and the second column is the logit for the A class
def evaluate(texts: List[str], model: AutoModelForSequenceClassification) -> np.ndarray:
    model.eval()
    all_logits, all_highlights = [], []
    cur_start = 0
    while cur_start < len(texts):
        texts_ = texts[cur_start : cur_start + bsize]
        inputs = tokenizer(
            texts_,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        logits = model(**inputs).logits
        logits = lsm(logits.detach().cpu()).numpy().tolist()
        all_logits.extend(logits)
        cur_start += bsize
    assert len(all_logits) == len(texts)

    return np.array(all_logits)


def evaluate_and_save_predictions(model_path: str, validation_samples: Dict[str, List[str]], output_file: str = None) -> Dict[str, Any]:
    """
    Evaluate the model on validation samples and save the predictions to a file.

    Args:
        model_path: Path to the saved model
        validation_samples: Dictionary containing validation samples for groups A and B
        output_file: Path to save the predictions (if None, will not save)

    Returns:
        Dictionary containing prediction results
    """
    print(f"Loading model from {model_path} for validation...")
    model = load_model(model_path)

    # Evaluate on validation samples
    A_validation = validation_samples['A_samples']
    B_validation = validation_samples['B_samples']

    print(f"Evaluating model on {len(A_validation)} group A validation samples...")
    A_val_logits = evaluate(A_validation, model)

    print(f"Evaluating model on {len(B_validation)} group B validation samples...")
    B_val_logits = evaluate(B_validation, model)

    # Extract probabilities for group A (index 1)
    A_val_probs = np.exp(A_val_logits[:, 1])
    B_val_probs = np.exp(B_val_logits[:, 1])

    # Calculate accuracy
    A_val_preds = (A_val_probs > 0.5).astype(int)
    B_val_preds = (B_val_probs > 0.5).astype(int)

    A_accuracy = np.mean(A_val_preds)
    B_accuracy = np.mean(1 - B_val_preds)
    overall_accuracy = (np.sum(A_val_preds) + np.sum(1 - B_val_preds)) / (len(A_val_preds) + len(B_val_preds))

    print(f"Validation accuracy - Group A: {A_accuracy:.4f}, Group B: {B_accuracy:.4f}, Overall: {overall_accuracy:.4f}")

    # Create results dictionary
    results = {
        'A_samples': A_validation,
        'B_samples': B_validation,
        'A_probabilities': A_val_probs,
        'B_probabilities': B_val_probs,
        'A_predictions': A_val_preds,
        'B_predictions': B_val_preds,
        'A_accuracy': A_accuracy,
        'B_accuracy': B_accuracy,
        'overall_accuracy': overall_accuracy,
        'model_path': model_path
    }

    # Save results to file
    if output_file:
        print(f"Saving prediction results to {output_file}")
        with open(output_file, 'wb') as f:
            pkl.dump(results, f)

    return results


# train the model on the training samples from the cross validation fold
# and then evaluate the model on the test samples from the cross validation fold
def save_model(model, model_path):
    """
    Save a trained model to disk.

    Args:
        model: The trained model to save
        model_path: Path where the model will be saved
    """
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path):
    """
    Load a trained model from disk.

    Args:
        model_path: Path to the saved model

    Returns:
        The loaded model
    """
    model = AutoModelForSequenceClassification.from_pretrained(pretrain_model).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def train_and_eval(cv_dict: Dict[str, List[str]], save_model_path=None) -> Dict[str, Any]:
    model = train(cv_dict)
    A_eval_logits = evaluate(cv_dict["test_A"], model)
    B_eval_logits = evaluate(cv_dict["test_B"], model)

    all_logits_A = np.concatenate((A_eval_logits, B_eval_logits), axis=0)[:,1]
    all_labels = np.concatenate((np.ones(len(A_eval_logits)), np.zeros(len(B_eval_logits))), axis=0)

    auc = roc_auc_score(all_labels, all_logits_A)

    # Save the model if a path is provided
    if save_model_path:
        save_model(model, save_model_path)

    return {
            "test_A_scores": A_eval_logits[:, 1],
            "test_B_scores": B_eval_logits[:, 0],
            "auc_roc": auc,
            'model': model,
            'model_path': save_model_path
        }

# A_samples are usually the research split of the Corpus A
# B_samples are usually the research split of the Corpus B
def return_extreme_values(A_samples: List[str], B_samples: List[str], save_best_model=True):
    """
    Train models on cross-validation folds and return extreme values.

    Args:
        A_samples: List of samples from group A
        B_samples: List of samples from group B
        save_best_model: Whether to save the best model based on AUC score

    Returns:
        Dictionary containing scores, sorted samples, and path to the best model
    """
    A_sample2score, B_sample2score = {}, {}
    text2model_path = {}
    clf_scores = {}
    models = {}
    best_model_path = None
    best_auc = -1

    for fold_idx, cv_dict in enumerate(cv(A_samples, B_samples, NUM_FOLD)):
        model_tmp_path = os.path.join(tmp_model_dir, f"model_{fold_idx}_{int(time.time())}.pt")
        train_and_eval_result = train_and_eval(cv_dict, save_model_path=model_tmp_path)
        model = train_and_eval_result['model']

        for A_sample, score in zip(cv_dict["test_A"], train_and_eval_result["test_A_scores"]):
            A_sample2score[A_sample] = score
            text2model_path[A_sample] = model_tmp_path
        for B_sample, score in zip(cv_dict["test_B"], train_and_eval_result["test_B_scores"]):
            B_sample2score[B_sample] = score
            text2model_path[B_sample] = model_tmp_path

        auc = train_and_eval_result["auc_roc"]
        clf_scores[model_tmp_path] = auc
        models[model_tmp_path] = model

        # Track the best model
        if auc > best_auc:
            best_auc = auc
            best_model_path = model_tmp_path

        print(f"fold {fold_idx} done, auc: {auc}")

    # Save the best model to a special path
    if save_best_model and best_model_path:
        best_model_final_path = os.path.join(tmp_model_dir, "best_model.pt")
        save_model(models[best_model_path], best_model_final_path)
        print(f"Best model (AUC: {best_auc:.4f}) saved to {best_model_final_path}")
        best_model_path = best_model_final_path

    return {
        'clf_scores': clf_scores, # a mapping from model path to the AUC score for each fold, useful to tell how easy it is to separate the two corpora
        'A_sample2score': A_sample2score, # a mapping from the A sample to how representative each sample is for the A corpus
        'B_sample2score': B_sample2score, # a mapping from the B sample to how representative each sample is for the B corpus
        'sorted_A_samples': sorted(A_sample2score.keys(), key=A_sample2score.get, reverse=True), # sorted A samples by their scores
        'sorted_B_samples': sorted(B_sample2score.keys(), key=B_sample2score.get, reverse=True), # sorted B samples by their scores
        'best_model_path': best_model_path # path to the best model based on AUC score
    }


def example_run():
    """
    Run the representative sample finder on the example problem.
    This demonstrates how to use the return_extreme_values function.
    """
    example_problem = pkl.load(open('example_problem.pkl', 'rb'))
    A_samples, B_samples = example_problem['split']['research']['A_samples'], example_problem['split']['research']['B_samples']
    extreme_values = return_extreme_values(A_samples, B_samples)

    print('======== Most representative A samples:')
    for sample in extreme_values['sorted_A_samples'][:5]:
        print(sample)

    print('======== Most representative B samples:')
    for sample in extreme_values['sorted_B_samples'][:5]:
        print(sample)

    print('Average AUC score for the 4 folds:', np.mean(list(extreme_values['clf_scores'].values())))

    if 'best_model_path' in extreme_values and extreme_values['best_model_path']:
        print(f"Best model saved to: {extreme_values['best_model_path']}")

        # Test loading the model
        try:
            loaded_model = load_model(extreme_values['best_model_path'])
            print("Successfully loaded the best model")

            # Test prediction on a few samples
            test_samples = A_samples[:2] + B_samples[:2]
            logits = evaluate(test_samples, loaded_model)
            print("Test prediction logits:", logits)
        except Exception as e:
            print(f"Error loading or testing the model: {e}")

    return extreme_values


def simse_run(problem=None, subsample_size=1000, evaluate_validation=True, save_predictions=True, output_file="results/simse_validation_predictions.pkl"):
    """
    Run the representative sample finder on the SimSE educational problem.

    Args:
        problem: Problem definition dictionary. If None, will load the SimSE dataset.
        subsample_size: Number of samples to use for faster execution.
        evaluate_validation: Whether to evaluate the best model on validation data.
        save_predictions: Whether to save prediction results to a file.
        output_file: Path to save prediction results if save_predictions is True.

    Returns:
        Dictionary containing the extreme values results and validation results if applicable.
    """
    if problem is None:
        # Load the SimSE dataset
        from textDiff.data_utils import SimSEDataLoader
        import random
        import os

        print("Loading SimSE dataset...")
        dataloader = SimSEDataLoader()
        split_datasets = dataloader.split_datasets

        # Create the problem definition
        problem = {
            'generation': 'teaching samples from the treatment group and control group, where the treatment group is '
                        'when the teachers are coached to use a metacognitive modeling strategy and metacognitive '
                        'modeling is defined as thinking aloud about thinking in order to make a strategy, task, or '
                        'process more accessible to students',
            'dataset_description': 'classroom transcripts for teaching math word problems',
            'target': 'what teaching strategy is more frequent in the treatment group than the control group',
            'user': 'an education researcher',
            'A_desc': 'teaching samples in the treatment group',
            'B_desc': 'teaching samples in the control group',
            'example_hypotheses': [],
            'split': {
                'research': {
                    'A_samples': [ele['text'] for ele in split_datasets['train'] if ele['condition'] == 'Treatment'],
                    'B_samples': [ele['text'] for ele in split_datasets['train'] if ele['condition'] == 'Non-Experimental']
                },
                'validation': {
                    'A_samples': [ele['text'] for ele in split_datasets['dev'] + split_datasets['test'] if
                                ele['condition'] == 'Treatment'],
                    'B_samples': [ele['text'] for ele in split_datasets['dev'] + split_datasets['test'] if
                                ele['condition'] == 'Control']
                }
            }
        }

    # Subsample for faster execution
    def subsample(samples, n=100):
        selected_idxes = list(range(len(samples)))
        random.shuffle(selected_idxes)
        selected_idxes = selected_idxes[:n]
        return [samples[i] for i in sorted(selected_idxes)]

    print(f"Original sample sizes: A={len(problem['split']['research']['A_samples'])}, B={len(problem['split']['research']['B_samples'])}")
    A_samples = subsample(problem['split']['research']['A_samples'], subsample_size)
    B_samples = subsample(problem['split']['research']['B_samples'], subsample_size)
    print(f"Subsampled sizes: A={len(A_samples)}, B={len(B_samples)}")

    # Run the representative sample finder
    print("Finding representative samples...")
    extreme_values = return_extreme_values(A_samples, B_samples)

    print('======== Most representative A samples (Treatment):')
    for sample in extreme_values['sorted_A_samples'][:5]:
        print(sample[:200] + "..." if len(sample) > 200 else sample)

    print('======== Most representative B samples (Control):')
    for sample in extreme_values['sorted_B_samples'][:5]:
        print(sample[:200] + "..." if len(sample) > 200 else sample)

    print('Average AUC score for the folds:', np.mean(list(extreme_values['clf_scores'].values())))

    validation_results = None
    if 'best_model_path' in extreme_values and extreme_values['best_model_path']:
        best_model_path = extreme_values['best_model_path']
        print(f"Best model saved to: {best_model_path}")

        # Evaluate on validation set if requested
        if evaluate_validation:
            print("\nEvaluating best model on validation set...")

            # Create results directory if it doesn't exist
            os.makedirs("results", exist_ok=True)

            # Define output file path
            output_path = output_file if save_predictions else None

            # Evaluate and save predictions
            validation_results = evaluate_and_save_predictions(
                best_model_path, 
                problem['split']['validation'],
                output_path
            )

            # Add validation results to extreme_values
            extreme_values['validation_results'] = validation_results

    return extreme_values


if __name__ == "__main__":
    simse_run()
    # example_run()
