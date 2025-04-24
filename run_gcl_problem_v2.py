import pickle as pkl
import random
from tqdm import tqdm
import os
import time
import json
import sys
import argparse
import pandas as pd
import numpy as np

from D5 import D5
from validator import DummyValidator, Validator, GPTValidator
from lm_proposer import GPT3_Proposer
from get_representative import return_extreme_values


def subsample(samples, n=1000):
    selected_idxes = list(range(len(samples)))
    random.shuffle(selected_idxes)
    selected_idxes = selected_idxes[:n]
    return [samples[i] for i in sorted(selected_idxes)]


def calculate_percentile_thresholds(df, column, top_percentile=10, bottom_percentile=10):
    """
    Calculate thresholds for selecting top and bottom percentiles of a column.

    Args:
        df: DataFrame containing the data
        column: Column name to calculate percentiles for
        top_percentile: Percentile for top values (default: 10)
        bottom_percentile: Percentile for bottom values (default: 10)

    Returns:
        Dictionary with top and bottom thresholds
    """
    # Calculate the percentile values
    top_threshold = np.percentile(df[df[column] > 0][column], 100 - top_percentile)

    # For bottom threshold, check if there are enough non-zero values
    zero_percentage = (df[column] == 0).mean() * 100

    if zero_percentage >= bottom_percentile:
        # If there are more zeros than bottom_percentile, use 0 as threshold
        bottom_threshold = 0
    else:
        # Otherwise, calculate the bottom percentile among non-zero values
        bottom_threshold = np.percentile(df[df[column] > 0][column], bottom_percentile)

    return {
        'top_threshold': top_threshold,
        'bottom_threshold': bottom_threshold,
        'zero_percentage': zero_percentage
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--find_representative', default=False, action='store_true',
        help='whether to find the representative samples from each corpus to prompt the proposer. If False, we will randomly select samples to prompt the proposer.'
    )
    parser.add_argument(
        '--subsample', type=int, default=None,
        help='only consider <subsample> samples from each corpus to run faster.'
    )
    parser.add_argument(
        '--verifier_name', type=str, default='ruiqi-zhong/d5_t5_validator_3B',
        choices=['dummy', 'ruiqi-zhong/d5_t5_validator', 'ruiqi-zhong/d5_t5_validator_700M', 'ruiqi-zhong/d5_t5_validator_3B', 'gpt4o-mini'],
        help='The name of the verifier to use. If dummy, use a dummy verifier that returns random results. If gpt4o-mini, use OpenAI\'s GPT-4o-mini model.'
    )
    parser.add_argument(
        '--verifier_batch_size', type=int, default=32,
        help='The batch size to use for the verifier. Decrease it if you are running out of memory.'
    )
    parser.add_argument(
        '--problem_path', type=str, default='example_problem.pkl',
        help='The path to the problem pickle file. You can also use your own problem pickle file.'
    )
    parser.add_argument(
        '--output_path', type=str, default='output.pkl',
        help='The path to save the output pickle file. You can also use your own output pickle file.'
    )
    parser.add_argument(
        '--setup', type=str,
        choices=['gcl', 'diff', 'non_gcl', 'gcl_cf', 'diff_cf', 'non_gcl_cf'],
        help='setup for gcl study. _cf suffix indicates context-free version.'
    )
    parser.add_argument(
        '--percentile', type=int, default=10, choices=[10, 20],
        help='Percentile to use for selecting top and bottom samples (10 or 20).'
    )
    parser.add_argument(
        '--new_target', default=False, action='store_true',
        help='Use the new target description focusing on linguistic patterns, topics, and psychological processes.'
    )

    args = parser.parse_args()

    # loading the problem from gcl dataset
    data_dir = '/fs/nexus-projects/PandEval/PandEval/Twitter_replies_location'
    phe_tweet_df = pd.read_pickle(f'{data_dir}/phe_tweetids_network_full_1000.pkl').set_index('tweetid')
    phe_tweet_df.index = phe_tweet_df.index.astype(str)

    subset_phe_tweet_df = pd.read_csv(f"{data_dir}/tweet_ids/all_subtweets.csv", index_col=0)
    subset_phe_tweet_df['conversation_id'] = subset_phe_tweet_df['conversation_id'].astype(str)
    subset_phe_tweet_df = subset_phe_tweet_df.merge(phe_tweet_df, how="left", left_on="conversation_id",
                                                  right_on="tweetid")

    subset_phe_tweet_df['Diff'] = subset_phe_tweet_df['co-location'] - subset_phe_tweet_df['non-colocation']
    print(subset_phe_tweet_df.head())

    # Calculate thresholds based on specified percentile
    gcl_thresholds = calculate_percentile_thresholds(subset_phe_tweet_df, 'co-location', args.percentile, args.percentile)
    diff_thresholds = calculate_percentile_thresholds(subset_phe_tweet_df, 'Diff', args.percentile, args.percentile)
    non_gcl_thresholds = calculate_percentile_thresholds(subset_phe_tweet_df, 'non-colocation', args.percentile, args.percentile)

    print(f"=== {args.percentile}% Percentile Thresholds ===")
    print(f"GCL: top={gcl_thresholds['top_threshold']}, bottom={gcl_thresholds['bottom_threshold']}")
    print(f"Diff: top={diff_thresholds['top_threshold']}, bottom={diff_thresholds['bottom_threshold']}")
    print(f"Non-GCL: top={non_gcl_thresholds['top_threshold']}, bottom={non_gcl_thresholds['bottom_threshold']}")

    # Define new target description if requested
    new_target = "what features related to linguistic patterns, topics of discussion, and psychological processes are more frequent in Group A compared to Group B"

    # Setup base problem structure
    is_context_free = '_cf' in args.setup if args.setup else False
    base_setup = args.setup.replace('_cf', '') if args.setup else None

    if base_setup == 'gcl':
        # randomly shuffle the list
        random_A_samples = subset_phe_tweet_df[subset_phe_tweet_df['co-location'] >= gcl_thresholds['top_threshold']]['text'].tolist()
        random_A_samples = random.sample(random_A_samples, len(random_A_samples))
        random_B_samples = subset_phe_tweet_df[subset_phe_tweet_df['co-location'] == gcl_thresholds['bottom_threshold']]['text'].tolist()
        random_B_samples = random.sample(random_B_samples, len(random_B_samples))

        if is_context_free:
            problem = {
                'generation': 'different tweet engagement metrics',
                'dataset_description': 'two sets of tweets with different engagement metrics',
                'target': new_target if args.new_target else 'what kind of tweets is more frequent in Group A compared to Group B',
                'user': 'a public health communication researcher',
                'A_desc': 'tweets in Group A',
                'B_desc': 'tweets in Group B',
                'example_hypotheses': [],
                'split': {
                    'research': {
                        'A_samples': random_A_samples[:len(random_A_samples)//2],
                        'B_samples': random_B_samples[:len(random_B_samples)//2]
                    },
                    'validation': {
                        'A_samples': random_A_samples[len(random_A_samples)//2:],
                        'B_samples': random_B_samples[len(random_B_samples)//2:]
                    }
                }
            }
        else:
            problem = {
                'generation': 'geo-co-located engagement rate',
                'dataset_description': 'tweets with high and low geo-co-located engagement rate',
                'target': new_target if args.new_target else 'what kind of tweets is more frequent in tweets with high geo-co-located engagement rate',
                'user': 'a public health communication researcher',
                'A_desc': 'tweets with high geo-co-located engagement rate',
                'B_desc': 'tweets with low geo-co-located engagement rate',
                'example_hypotheses': [],
                'split': {
                    'research': {
                        'A_samples': random_A_samples[:len(random_A_samples)//2],
                        'B_samples': random_B_samples[:len(random_B_samples)//2]
                    },
                    'validation': {
                        'A_samples': random_A_samples[len(random_A_samples)//2:],
                        'B_samples': random_B_samples[len(random_B_samples)//2:]
                    }
                }
            }
    elif base_setup == 'diff':
        random_A_samples = subset_phe_tweet_df[subset_phe_tweet_df['Diff'] >= diff_thresholds['top_threshold']]['text'].tolist()
        random_A_samples = random.sample(random_A_samples, len(random_A_samples))
        random_B_samples = subset_phe_tweet_df[subset_phe_tweet_df['Diff'] <= diff_thresholds['bottom_threshold']]['text'].tolist()
        random_B_samples = random.sample(random_B_samples, len(random_B_samples))

        if is_context_free:
            problem = {
                'generation': 'different tweet engagement metrics',
                'dataset_description': 'two sets of tweets with different engagement metrics',
                'target': new_target if args.new_target else 'what kind of tweets is more frequent in Group A compared to Group B',
                'user': 'a public health communication researcher',
                'A_desc': 'tweets in Group A',
                'B_desc': 'tweets in Group B',
                'example_hypotheses': [],
                'split': {
                    'research': {
                        'A_samples': random_A_samples[:len(random_A_samples) // 2],
                        'B_samples': random_B_samples[:len(random_B_samples) // 2]
                    },
                    'validation': {
                        'A_samples': random_A_samples[len(random_A_samples) // 2:],
                        'B_samples': random_B_samples[len(random_B_samples) // 2:]
                    }
                }
            }
        else:
            problem = {
                'generation': 'difference between geo-co-located and non-geo-co-located engagement rate',
                'dataset_description': 'tweets with various geo-co-located and non-geo-co-located engagement rate',
                'target': new_target if args.new_target else 'what kind of tweets is more frequent in tweets with higher geo-co-located engagement rate but lower non-geo-co-located engagement rate',
                'user': 'a public health communication researcher',
                'A_desc': 'tweets with higher geo-co-located engagement rate but lower non-geo-co-located engagement rate',
                'B_desc': 'tweets with higher non-geo-co-located engagement rate but lower geo-co-located engagement rate',
                'example_hypotheses': [],
                'split': {
                    'research': {
                        'A_samples': random_A_samples[:len(random_A_samples) // 2],
                        'B_samples': random_B_samples[:len(random_B_samples) // 2]
                    },
                    'validation': {
                        'A_samples': random_A_samples[len(random_A_samples) // 2:],
                        'B_samples': random_B_samples[len(random_B_samples) // 2:]
                    }
                }
            }
    elif base_setup == 'non_gcl':
        random_A_samples = subset_phe_tweet_df[subset_phe_tweet_df['non-colocation'] >= non_gcl_thresholds['top_threshold']]['text'].tolist()
        random_A_samples = random.sample(random_A_samples, len(random_A_samples))
        random_B_samples = subset_phe_tweet_df[subset_phe_tweet_df['non-colocation'] == non_gcl_thresholds['bottom_threshold']]['text'].tolist()
        random_B_samples = random.sample(random_B_samples, len(random_B_samples))

        if is_context_free:
            problem = {
                'generation': 'different tweet engagement metrics',
                'dataset_description': 'two sets of tweets with different engagement metrics',
                'target': new_target if args.new_target else 'what kind of tweets is more frequent in Group A compared to Group B',
                'user': 'a public health communication researcher',
                'A_desc': 'tweets in Group A',
                'B_desc': 'tweets in Group B',
                'example_hypotheses': [],
                'split': {
                    'research': {
                        'A_samples': random_A_samples[:len(random_A_samples) // 2],
                        'B_samples': random_B_samples[:len(random_B_samples) // 2]
                    },
                    'validation': {
                        'A_samples': random_A_samples[len(random_A_samples) // 2:],
                        'B_samples': random_B_samples[len(random_B_samples) // 2:]
                    }
                }
            }
        else:
            problem = {
                'generation': 'non-geo-co-located engagement rate',
                'dataset_description': 'tweets with high and low non-geo-co-located engagement rate',
                'target': new_target if args.new_target else 'what kind of tweets is more frequent in tweets with high non-geo-co-located engagement rate',
                'user': 'a public health communication researcher',
                'A_desc': 'tweets with high non-geo-co-located engagement rate',
                'B_desc': 'tweets with low non-geo-co-located engagement rate',
                'example_hypotheses': [],
                'split': {
                    'research': {
                        'A_samples': random_A_samples[:len(random_A_samples) // 2],
                        'B_samples': random_B_samples[:len(random_B_samples) // 2]
                    },
                    'validation': {
                        'A_samples': random_A_samples[len(random_A_samples) // 2:],
                        'B_samples': random_B_samples[len(random_B_samples) // 2:]
                    }
                }
            }
    else:
        raise NotImplementedError

    # finding the representative samples from each corpus in the problem
    if args.find_representative:
        extreme_vals = return_extreme_values(problem['split']['research']['A_samples'], problem['split']['research']['B_samples'])
        problem['split']['research']['A_samples'], problem['split']['research']['B_samples'] = extreme_vals['sorted_A_samples'], extreme_vals['sorted_B_samples']

    # subsampling the representative samples
    if args.subsample:
        problem['split']['research']['A_samples'], problem['split']['research']['B_samples'] = subsample(problem['split']['research']['A_samples'], args.subsample), subsample(problem['split']['research']['B_samples'], args.subsample)

    # creating the proposer and verifier
    proposer = GPT3_Proposer(problem)

    # for actual use, the verifier is a validator with 11B parameters
    # for debugging, the verifier is a dummy validator returns a random value
    if args.verifier_name == 'dummy':
        verifier = DummyValidator()
    elif args.verifier_name == 'gpt4o-mini':
        verifier = GPTValidator(batch_size=args.verifier_batch_size, verbose=True)
    else:
        verifier = Validator(args.verifier_name, batch_size=args.verifier_batch_size)

    # goal-driven discovery and description of corpus-level differences
    d5 = D5(
        problem['split']['research']['A_samples'],
        problem['split']['research']['B_samples'],
        verifier,
        proposer,
        total_hypotheses_count=20,
        early_stop=True
    )
    h2h_dicts = d5.run()

    h_sorted = sorted(h2h_dicts, key=lambda h: h2h_dicts[h]['diff_w_significance']['mu'], reverse=True)
    pkl.dump(h2h_dicts, open(args.output_path, 'wb'))
    results = {'hypothesis': [], 'V': []}
    for h in h_sorted:
        h_dict = h2h_dicts[h]
        # print out the example hypothesis along with their V' score
        print(h_dict['hypothesis'], 'V\'', h_dict['diff_w_significance']['mu'])
        results['hypothesis'].append(h_dict['hypothesis'])
        results['V'].append(h_dict['diff_w_significance']['mu'])

    df = pd.DataFrame(results)
    df.to_csv(args.output_path.replace('.pkl', '.csv'), index=False)
