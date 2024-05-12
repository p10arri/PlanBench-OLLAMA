''' This program evaluates multiple models from OLLAMA in the task of plan generation and the specified domain'''

import random
import argparse
import os
from prompt_generation import PromptGenerator
from response_evaluation import ResponseEvaluator
from response_generation import ResponseGenerator

from tqdm import tqdm

if __name__=="__main__":
    random.seed(10)
    parser = argparse.ArgumentParser()
    # parser.add_argument('--task', type=str, required=True, help='Task to run \
    #     \n t1 = Plan Generation\
    #     \n t2 = Optimal Planning \
    #     \n t3 = Plan Verification \
    #     \n t4 = Plan Reuse\
    #     \n t5 = Plan Generalization\
    #     \n t6 = Replanning \
    #     \n t7 = Reasoning about Plan Execution \
    #     \n t8_1 = Goal Reformulation (Goal shuffling) \
    #     \n t8_2 = Goal Reformulation (Full -> Partial) \
    #     \n t8_3 = Goal Reformulation (Partial -> Full) \
    #     ')
    #config
    parser.add_argument('--config', type=str, required=True, help='Config file name (no need to add .yaml)')
    
    # parser.add_argument('--engine', type=str, required=True, help='Engine to use \
    #                     \n gpt-4_chat = GPT-4 \
    #                     \n bloom = Bloom \
    #                     \n gpt-3.5-turbo_chat = GPT-3.5 Turbo \
    #                     \n davinci = GPT-3 Davinci \
    #                     \n curie = GPT-3 Curie \
    #                     \n babbage = GPT-3 Babbage \
    #                     \n ada = GPT-3 Ada \
    #                     \n ollama:moodelName -> Ex: ollama:llama3 \
    #                     ')
    
    parser.add_argument('--run_till_completion', type=str, default="False", help='Run till completion')
    parser.add_argument('--verbose', type=str, default="False", help='Verbose')
    parser.add_argument('--ignore_existing', action='store_true', help='Ignore existing output')
    parser.add_argument('--specific_instances', nargs='+', type=int, default=[], help='List of instances to run')
    parser.add_argument('--random_example', type=str, default="False", help='Random example')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    #  Parameters
    args = parser.parse_args()
    task = 't1' # plan generation task
    config = args.config # domain
    engine = 'ollama'
    verbose = eval(args.verbose)
    specified_instances = args.specific_instances
    seed=args.seed
    ignore_existing = args.ignore_existing
    random_example = eval(args.random_example)
    run_till_completion = eval(args.run_till_completion)
    # print(task, config, verbose, specified_instances, random_example)
    config_file = f'./configs/{config}.yaml'

    # list of models to evaluate
    list_of_models = [#'llama3', 
                      'llama2', 'tinyllama','phind-codellama',
                  'command-r-plus',
                  'wizardlm2',
                  'gemma',
                  'mistral-openorca',
                  'vicuna']
    
    # ========================= Prompt Generation =========================
    prompt_generator = PromptGenerator(config_file, verbose, ignore_existing, seed, engine) # engine parameter only affects task t3
    prompt_generator.task_1_plan_generation(specified_instances, random_example)
    
    # ========================= Response Generation =========================
    pbar = tqdm(list_of_models)
    for model in pbar:
        pbar.set_description(f"Generating responses of {model}")
        
        engine = f"{engine}:{model}"
        response_generator = ResponseGenerator(config_file, engine, verbose, ignore_existing)
        task_name = 'task_1_plan_generation'

        response_generator.get_responses(task_name, run_till_completion=run_till_completion)
    print('RESPONSES GENERATED STOP COLAB RUNTIME')
    # ========================= Response Evaluation =========================
    for model in pbar:
        pbar.set_description(f"Evaluating responses of {model}")

        engine = f"{engine}:{model}"
        response_evaluator = ResponseEvaluator(config_file, engine, specified_instances, verbose, ignore_existing)
        task_name = 'task_1_plan_generation'
        response_evaluator.evaluate_plan(task_name)
        


