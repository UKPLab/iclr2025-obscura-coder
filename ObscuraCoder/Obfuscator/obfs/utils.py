import json
import os
import logging
import tree_sitter

file_paths = {
    'cpp': 'test_files/contest_cpp_small_dataset.jsonl',
    # 'java': 'test_files/contest_java_small_dataset.jsonl',
    'javascript': 'test_files/contest_javascript_small_dataset.jsonl',
    'go': 'test_files/contest_go_small_dataset.jsonl',
    'python': 'test_files/contest_python_small_dataset.jsonl'
}

def jsonl_iterator(jsonl_file):
    for line in jsonl_file:
        yield json.loads(line)['content']

def get_sample_iter(language):
    file_path = file_paths[language]
    file = open(file_path, 'r') 
    iterator = jsonl_iterator(file)

    return iterator


def init_logging(logging_file,
                erase_exist=True):
    if os.path.exists(logging_file) and erase_exist:
        os.remove(logging_file)
    logger = logging.getLogger('prototypeLogger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(logging_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger