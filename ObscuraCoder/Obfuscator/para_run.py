from obfs.tree_sitter_content_obfuscator import TSConObfuscator
import json
import argparse
from datasets import load_dataset
import random
import logging

def jsonl_iterator(jsonl_file):
    for line in jsonl_file:
        yield json.loads(line)['content']

def setup_logger():
    logger = logging.getLogger('exception_logger')
    handler = logging.FileHandler('log/exception_log.txt')  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)
    return logger

filePaths = {
    'typescript': 'test_files/small_real_data_typescript.jsonl',
    'cpp': 'test_files/small_real_data_cpp.jsonl',
    'c': 'test_files/small_real_data_c.jsonl',
    'python': 'test_files/small_real_data_python.jsonl',
    'java': 'test_files/small_real_data_java.jsonl',
    'rust': 'test_files/small_real_data_rust.jsonl',
    'go': 'test_files/small_real_data_go.jsonl',
    }

rulePaths = {
    'cpp': 'match_rules/cpp.scm',
    'c': 'match_rules/c.scm',
    'python': 'match_rules/python.scm',
    'javascript': 'match_rules/javascript.scm',
    'swift': 'match_rules/swift.scm',
    'rust': 'match_rules/rust.scm',
    'go': 'match_rules/go.scm',
    'java': 'match_rules/java.scm',
    'typescript': 'match_rules/typescript.scm'
}



def main(asgs):
    logger = setup_logger()
    rulePath = rulePaths[args.language]
    rule = open(rulePath, 'r').read()
    language = args.language
    
    def ds_map_fn(sample):
        code = sample['content']
        obs = TSConObfuscator(
                    language,
                    rule,)
        ret = obs.obfuscate(code,
                                obfuscateImportedIDs=True,
                                obfuscateImportedModules=True,
                                obfuscatePropotion=True)
        sample["obf_code"] = ret['obfCode']
        sample['probability'] = ret['obsPropotion']
        sample['obf_dict'] = ret['obsDict']
        return sample
    
    try:
        ds = load_dataset("OBF/OBF_obfuscated_code", language)
        obf_ds = ds.map(ds_map_fn, num_proc=80, load_from_cache_file=True)
        obf_ds.save_to_disk(f"real_dataset/test_obf/{language}")
    except Exception as e:
        logger.error("Exception occurred", exc_info=True)
        raise e

    dsi = iter(obf_ds['train'])
    rand_index = random.randint(0, len(obf_ds['train']) - 1)
    for _ in range(rand_index):
        next(dsi)
    sample = next(dsi)
    print(sample['content'])
    print(sample['obf_code'])
    print(f"probability: {sample['probability']} ")
    print(sample['obf_dict'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='obfuscator')
    parser.add_argument('language', type=str, default="cpp", nargs="?")
    args = parser.parse_args()
    
    main(args)