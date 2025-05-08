import pickle as pkl
import random
from tqdm import tqdm
import os
import time
import json
import sys
import argparse
import pandas as pd

from D5 import D5
from validator import DummyValidator, Validator
from lm_proposer import GPT3_Proposer
from get_representative import return_extreme_values


def subsample(samples, n=1000):
    selected_idxes = list(range(len(samples)))
    random.shuffle(selected_idxes)
    selected_idxes = selected_idxes[:n]
    return [samples[i] for i in sorted(selected_idxes)]


# whether to run a proof-of-concept demo
# if not we will run the entire pipeline that takes a longer time to run but produces better results
demo = True if len(sys.argv) > 1 and sys.argv[1] == 'demo' else False
use_subsample = True
find_representative = False if demo else True
use_dummy_verifier = True if demo else False


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
        choices=['dummy', 'ruiqi-zhong/d5_t5_validator', 'ruiqi-zhong/d5_t5_validator_700M', 'ruiqi-zhong/d5_t5_validator_3B'],
        help='The name of the verifier to use. If dummy, use a dummy verifier that returns random results. ruiqi-zhong/d5_t5_validator is the best model we have trained, but it is large. ruiqi-zhong/d5_t5_validator_700M and ruiqi-zhong/d5_t5_validator_3B are smaller distilled models that are faster to run but produce slightly worse results; however, they should still be able to perform easier tasks like classifying topics.'
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
        choices=['gcl', 'diff', 'non_gcl', 'gcl_replies'],
        help='setup for gcl study'
    )

    args = parser.parse_args()

    # loading the problem from the pickle file
    # problem = pkl.load(open(args.problem_path, 'rb'))

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

    if args.setup == 'gcl':
        # randomly shuffle the list
        random_A_samples = subset_phe_tweet_df[subset_phe_tweet_df['co-location'] >= 0.034]['text'].tolist()
        random_A_samples = random.sample(random_A_samples, len(random_A_samples))
        random_B_samples = subset_phe_tweet_df[subset_phe_tweet_df['co-location'] == 0]['text'].tolist()
        random_B_samples = random.sample(random_B_samples, len(random_B_samples))
        problem = {
            'generation': 'geo-co-located engagement rate',
            'dataset_description': 'tweets with high and low geo-co-located engagement rate',
            'target': 'what kind of tweets is more frequent in tweets with high geo-co-located engagement rate',
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
    elif args.setup == 'diff':
        random_A_samples = subset_phe_tweet_df[subset_phe_tweet_df['Diff'] >= 0.017241]['text'].tolist()
        random_A_samples = random.sample(random_A_samples, len(random_A_samples))
        random_B_samples = subset_phe_tweet_df[subset_phe_tweet_df['Diff'] <= -0.022387]['text'].tolist()
        random_B_samples = random.sample(random_B_samples, len(random_B_samples))
        problem = {
            'generation': 'difference between geo-co-located and non-geo-co-located engagement rate',
            'dataset_description': 'tweets with various geo-co-located and non-geo-co-located engagement rate',
            'target': 'what kind of tweets is more frequent in tweets with higher geo-co-located engagement rate but lower non-geo-co-located engagement rate',
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
    elif args.setup == 'non_gcl':
        random_A_samples = subset_phe_tweet_df[subset_phe_tweet_df['non-colocation'] >= 0.035347]['text'].tolist()
        random_A_samples = random.sample(random_A_samples, len(random_A_samples))
        random_B_samples = subset_phe_tweet_df[subset_phe_tweet_df['non-colocation'] == 0]['text'].tolist()
        random_B_samples = random.sample(random_B_samples, len(random_B_samples))
        problem = {
            'generation': 'non-geo-co-located engagement rate',
            'dataset_description': 'tweets with high and low non-geo-co-located engagement rate',
            'target': 'what kind of tweets is more frequent in tweets with high non-geo-co-located engagement rate',
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
    elif args.setup == 'gcl_replies':
        # Load the data
        sent_df = pd.read_csv(os.path.join(data_dir, 'results/Top500NC_reply_text_explicit_place_sentiment.csv'), delimiter=',', dtype=str)

        # Filter out specific authors if needed (same as in the original code)
        to_remove = {'thedailybeast', 'texastribune', 'nytopinion', 'foreignaffairs', 'bostonreview', 'propublica',
                     'theatlantic', 'sfgate', 'intelligencer', 'squawkcnbc', 'toustontx'}
        sent_df = sent_df[~sent_df['author_name'].isin(to_remove)]

        # Create co-located and non-co-located dataframes
        co_located_df = sent_df[(sent_df['author_state'] == sent_df['reply_state']) &
                                (sent_df['author_name'] != sent_df['reply_user_name']) &
                                (sent_df['author_state'] != 'None')].copy()

        non_co_located_df = sent_df[(sent_df['author_state'] != sent_df['reply_state']) &
                                    (sent_df['author_state'] != 'None') &
                                    (sent_df['reply_state'] != 'None')].copy()

        # Add gcl indicator`
        co_located_df['gcl'] = 1
        non_co_located_df['gcl'] = 0

        # Get random samples from each group
        random_A_samples = co_located_df['clean_reply_text'].tolist()
        random_A_samples = random.sample(random_A_samples, len(random_A_samples))
        random_B_samples = non_co_located_df['clean_reply_text'].tolist()
        random_B_samples = random.sample(random_B_samples, len(random_B_samples))

        # Create context-free problem dictionary
        problem = {
            'generation': 'different tweet reply patterns',
            'dataset_description': 'two sets of tweet replies with different characteristics',
            'target': 'what kind of tweet replies is more frequent in Group A compared to Group B',
            'user': 'a social media communication researcher',
            'A_desc': 'tweet replies in Group A',
            'B_desc': 'tweet replies in Group B',
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
        raise NotImplementedError



    # dataloader = NCTEDatasetLoader()
    # full_datasets = dataloader.dataset
    # temp_dataset = full_datasets.train_test_split(test_size=0.3, seed=42)
    # problem = {
    #     'generation': 'teaching quality reflected in the classroom transcripts',
    #     'dataset_description': 'classroom transcripts for math teaching',
    #     'target': 'what teaching strategy is more frequent in different groups of transcripts',
    #     'user': 'an education researcher',
    #     'A_desc': 'high quality teaching samples',
    #     'B_desc': 'low quality teaching samples',
    #     'example_hypotheses': [],
    #     'split': {
    #         'research': {
    #             'A_samples': [ele['transcript'] for ele in temp_dataset['train'] if ele['MQI5'] == 5],
    #             'B_samples': [ele['transcript'] for ele in temp_dataset['train'] if ele['MQI5'] == 1]
    #         },
    #         'validation': {
    #             'A_samples': [ele['transcript'] for ele in temp_dataset['test'] if ele['MQI5'] == 5],
    #             'B_samples': [ele['transcript'] for ele in temp_dataset['test'] if ele['MQI5'] == 1]
    #         }
    #     }
    # }

    # print(json.dumps(problem, indent=4))

    # finding the representative samples from each corpus in the problem
    # you can comment it out if you want to save time
    if args.find_representative:
        extreme_vals = return_extreme_values(problem['split']['research']['A_samples'], problem['split']['research']['B_samples'])
        problem['split']['research']['A_samples'], problem['split']['research']['B_samples'] = extreme_vals['sorted_A_samples'], extreme_vals['sorted_B_samples']

    # subsampling the representative samples
    # since verifying the hypotheses is expensive, we only verify a smaller subset of the samples
    # you can comment it out if you want to save time
    if args.subsample:
        problem['split']['research']['A_samples'], problem['split']['research']['B_samples'] = subsample(problem['split']['research']['A_samples'], args.subsample), subsample(problem['split']['research']['B_samples'], args.subsample)

    # creating the proposer and verifier
    proposer = GPT3_Proposer(problem)

    # for actual use, the verifier is a validator with 11B parameters
    # for debugging, the verifier is a dummy validator returns a random value
    if args.verifier_name == 'dummy':
        verifier = DummyValidator()
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
