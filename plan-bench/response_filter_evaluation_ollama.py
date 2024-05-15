import os
import random

import yaml
from Executor import Executor
from utils import *
from pathlib import Path
from tarski.io import PDDLReader
from model_parser.writer_new import ModelWriter
import argparse
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import json
np.random.seed(42)
from tqdm import tqdm


class ResponseFilterEvaluatorOllama:
    def __init__(self, config_file, engine):
        self.engine = engine
        self.config_file =config_file
        self.data = self.read_config(config_file)

        # self.results_file = f"results/{domain}/{self.engine}/task_1_plan_generation.json"
        # self.data_results = self.read_config()
        # self.instance_dir = self.data['instance_dir']
        
    
    def read_config(self, config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
        
    
    def load_json(self, task_name):
        output_dir = f"results/{self.data['domain_name']}/{self.engine}/"        
        with open(output_dir+f"{task_name}.json", 'r') as file:
            structured_output = json.load(file)
        return structured_output
    
    def save_json(self, structured_output, task_name):
        output_dir = f"results_filtered/{self.data['domain_name']}/{self.engine}/"        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_dir+f"{task_name}.json", 'w') as file:
            json.dump(structured_output, file, indent=4)

    def plan_to_action(self, plan_text):
        # plan example: (unstack orange blue)\n (unstack orange blue)\n
        action_list = []
        for line in plan_text.split('\n')[:-1]:
            action = line[1:-1].split()[0]
            objects = line[1:-1].split()[1:]
            
            action_list += [[action, objects]]
        return action_list

    def filter_redundant_action(self, action_list):

        if 'blocksworld' in self.data['domain_name']:
            # redundant actions for blocksworld: 
            # - pick-up A > put-down A
            # - stack A B > unstack A B / unstack B A
            action_only_list = list(list(zip(*action_list))[0])
            redundant_action_counter = 0
            filtered_plan = f""
            for a_id, action in enumerate(action_only_list):
                if action in ['pick-up', 'stack'] and a_id != len(action_only_list)-1:
                    if (action_only_list[a_id] == 'pick-up' and action_only_list[a_id+1] == 'put-down') or (action_only_list[a_id] == 'stack' and action_only_list[a_id+1] == 'unstack'):
                        
                        obj_1 = action_list[a_id][1]
                        obj_2 = action_list[a_id+1][1]

                        # check if objects are the same in sequential actions
                        if obj_1 == obj_2 or obj_1 == obj_2[::-1]: # unstack A B == unstack B A
                            redundant_action_counter += 1
                            continue
                filtered_plan += f"({action} {' '.join(action_list[a_id][1])})\n"

            return filtered_plan, redundant_action_counter
        
        elif 'logistics' in self.data['domain_name']:
            # redundant actions for logistics: 
            #
            raise ValueError(f"Redundant actions not set for {self.data['domain_name']}")
        else:
            raise ValueError(f"Redundant actions not set for {self.data['domain_name']}")

    
    def filter_evaluate_plan(self, task_name):
        
        structured_output = self.load_json(task_name)
        for instance_dict in tqdm(structured_output["instances"]):
            id = instance_dict["instance_id"]
            ground_truth = instance_dict['ground_truth_plan']

            if "extracted_llm_plan" not in instance_dict:
                print("WARNING: Response not generated for plan{id}")
                continue

            llm_plan = instance_dict["extracted_llm_plan"]
            
            
            if llm_plan !="":
                try:
                    
                    action_list = self.plan_to_action(llm_plan)                   
                    filtered_plan, redundant_action_counter = self.filter_redundant_action(action_list)

                    instance_dict["filtered_llm_plan"] = filtered_plan
                    instance_dict["redundant_action_counter"] = redundant_action_counter
                    instance_dict['llm_correct_filtered'] = True if filtered_plan == ground_truth else False

                except:
                    print(f"Warning: Plan filtering failed for plan {id}")
            else:
                print(f"WARNING: Extracted LLM plan failed for plan {id}")
            
            
            self.save_json(structured_output, task_name)

        
            
    

