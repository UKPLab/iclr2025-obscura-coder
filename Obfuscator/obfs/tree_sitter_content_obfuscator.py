import argparse
from obfs.document import Document
import logging
from obfs.text_edit import apply_text_edits, OverLappingTextEditException, get_cut_out_text, Position, Range
from obfs.obfuscated_names_generator import ObfuscatedNamesGenerator, ObfuscatedNameType
from collections.abc import Iterable
import json
from tree_sitter import Language, Parser
from obfs.tree_sitter_utils import tree_to_string, find_identifier, get_node_text
from obfs.utils import init_logging
import random
import os
from datetime import datetime
import signal
import re

def find_nodes_with_strings(node, doc, matching_dict):
    if node.start_point[0] == node.end_point[0] and node.type != 'namespace_identifier': #name cannot split in two lines
        node_text = get_node_text(node, doc)
        if node_text in matching_dict.keys():
            matching_dict[node_text].append((node.start_point, node.end_point))

    for child in node.children:
        find_nodes_with_strings(child, doc, matching_dict)

class TimeoutException(Exception): pass

class PyTsLongCodeException(Exception): pass # code too long for tree-sitter and cause segmentation fault
PYTS_MAX_CODE_LENGTH = 2000
# Define your signal handler
def handler(signum, frame):
    raise TimeoutException()

# Set the signal handler
signal.signal(signal.SIGALRM, handler)

class TSConObfuscator:
    def __init__(self,
                language,
                match_rules,
                logging_file:str = None,
                rootExpDir: str = None,
                timeOut: int = 5,
                obfToken: str = '<OBF>',
                maxObfNamesPerType: int = 150,

                ):
        if logging_file is not None:
            self.logger = init_logging(logging_file)
        else:
            self.logger = None
        self.language = language
        if rootExpDir:
            self.expDir = self.create_example_dir(rootExpDir)
        else:
            self.expDir = None
        self.timeOut = timeOut

        lang = Language('build/obs_languages.so',language)
        self.parser = Parser()
        self.parser.set_language(lang)
        self.query = lang.query(match_rules)
        self.obfToken = obfToken
        self.maxObfNamesPerType = maxObfNamesPerType

    
    def obfuscate(self,
                code,
                obfuscateImportedModules = True,
                obfuscateImportedIDs = True,
                obfuscatePropotion = False,):
        signal.alarm(self.timeOut)
        try:
            obsPropotion = random.random()
            doc = Document(code)
            if self.language in ['python', 'typescript']:
                if len(doc.lines) > PYTS_MAX_CODE_LENGTH:
                    raise PyTsLongCodeException
            obfNameGenerator = ObfuscatedNamesGenerator()
            tree = self.parser.parse(bytes(code, "utf8"))
            captures = self.query.captures(tree.root_node)   
            rangeDict, defDict, refDict, allDict, capturesMsg = build_obfuscation_dict(doc, captures, logging=self.logging, language=self.language)
            ret = None
            realObsPropotion = obsPropotion * 2
            while ret is None:
                realObsPropotion = realObsPropotion / 2
                ret = self.obfuscate_code(rangeDict, refDict, defDict, allDict, obfNameGenerator, 
                                                    obfuscateImportedModules, obfuscateImportedIDs, obfuscatePropotion, realObsPropotion) 
            obfuscatePropotion = realObsPropotion
            edits, obsDict = ret
            obfCode = apply_text_edits(doc, edits)
            self.logging(obfCode)
            obsDict = {obsName: (oriName, tokType.value) for (oriName, tokType), obsName in obsDict.items()}
            TextObsDict = str({obsName: oriName for obsName, (oriName, tokVal) in obsDict.items()})
            # obsDictText = f'{self.obfToken} {TextObsDict} {self.obfToken}'
            if self.expDir:
                metadata = (obfuscateImportedModules, obfuscateImportedIDs, obfuscatePropotion)
                if obfuscatePropotion:
                    metadata =(obfuscateImportedModules, obfuscateImportedIDs, obfuscatePropotion, obsPropotion)
                self.write_example(code, obfCode, metadata, obsDict, parseTree=tree_to_string(tree.root_node, code),
                capturesMsg=capturesMsg)
        except:
            obfCode = code
            TextObsDict = ''
            obsPropotion = 0.0
        finally:
            signal.alarm(0)
        return {'originCode': code, 'obfCode': obfCode, 'obsDict': TextObsDict, 'obsPropotion': obsPropotion}

    def write_example(self, oriCode, obfCode, exampleIndex, metadata, obsDict, parseTree = None, capturesMsg = None):
        headerline = '*' * 50 + '\n' * 2
        header = headerline + f'   Imported Modules Obfuscated: {metadata[0]} \n'
        header += f'   Imported Identifiers Obfuscated: {metadata[1]} \n'
        header += f'   Obfuscated propotionally: {metadata[2]} \n'
        if metadata[2]:
            header += f'    Obfuscated propotion: {metadata[3]}\n'
        header += '\n' + headerline

        
        examplePath = os.path.join(self.expDir, str(exampleIndex))
        os.makedirs(examplePath)
        oriCodePath = os.path.join(examplePath, 'original_code')
        obfCodePath = os.path.join(examplePath, 'obfuscated_code')

        open(oriCodePath, 'w').write(header + oriCode)
        open(obfCodePath, 'w').write(header + obfCode)

        for key, value in obsDict.items():
            obsDictPath = os.path.join(examplePath, 'obsDict')
            open(obsDictPath, 'w').write(str(key) + ' : ' + str(value) + '\n')

        if parseTree:
            parseTreePath = os.path.join(examplePath, 'parseTree')
            open(parseTreePath, 'w').write(parseTree)

        if capturesMsg:
            capturesPath = os.path.join(examplePath, 'captures')
            open(capturesPath, 'w').write(capturesMsg)

    def create_example_dir(self, rootExpDir):
        current_time = datetime.now().strftime("%H-%M-%S_%Y-%m-%d")
        dir_name = f"{rootExpDir}{self.language}_{current_time}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        else:
            raise RuntimeError('Not able to create example dir....')
        
        return dir_name

    def logging(self, info:str):
        if self.logger is not None:
            self.logger.debug(info)
    

    def obfuscate_code(self, rangeDict, refDict, defDict, allDict, obfNameGenerator, obfuscateImportedModules, obfuscateImportedIDs, obfuscatePropotion, obsPropotion):
        edits = []
        obsDict = {}
        obfNameCounterDict = {obfType: 0 for obfType in ObfuscatedNameType}

        for name, nameRanges in rangeDict.items():
            
            if obfuscatePropotion and random.random() > obsPropotion:
                    continue
            elif not obfuscateImportedModules and (name not in allDict or allDict[name] == ObfuscatedNameType.IMPORT):
                continue
            for nameRange in nameRanges:
                if nameRange in refDict:
                    nameAndType = refDict[nameRange]
                    rangeName = nameAndType[0]
                    tp = nameAndType[1]
                    if not obfuscateImportedIDs and (rangeName not in defDict or defDict[rangeName] != tp):
                        continue
                else:
                    continue
                if nameAndType in obsDict:
                    obsName = obsDict[nameAndType]
                elif (rangeName, ObfuscatedNameType.IMPORT) in obsDict: #cannot know the type of imported ids, if a imported exist, then use
                    obsName = obsDict[(rangeName, ObfuscatedNameType.IMPORT)]
                else:
                    obsName = obfNameGenerator.get_new_name(*nameAndType)
                    obsDict[nameAndType] = obsName
                    obfNameCounterDict[tp] += 1
                    if obfNameCounterDict[tp] > self.maxObfNamesPerType:
                        return None
                start = Position(line = nameRange[0][0], character = nameRange[0][1]) 
                end = Position(line = nameRange[1][0], character = nameRange[1][1]) 
                editRange = Range(start=start, end=end)
                edits.append({'newText': obsName, "range": editRange.dict()}) 
        return edits, obsDict

def build_obfuscation_dict(doc, captures, logging = None, language = None):
    rangeDict = {} # {name: [ranges]} Identifiers to mangle
    defDict = {} # {name: ObfuscatedNameType} User defined symbols
    refDict = {} # {range: ObfuscatedNameType} Type in specific range
    allDict = {} # {name: [ObfuscatedNameType]} All appeared symbols
    capturesMsg = ''
    for node, nodeType in captures:
        try:
            nodeText = get_node_text(node, doc) 
            nodeRange = (node.start_point, node.end_point)
            if language in ['c', 'cpp']:
                pattern = r'<(.*?)>|"(.*?)"' 
                matches = re.findall(pattern, nodeText)
                if matches:
                    nodeText = matches[0][0] if matches[0][0] else matches[0][1]
                    nodeRange = ((nodeRange[0][0], nodeRange[0][1] + 1), (nodeRange[1][0], nodeRange[1][1] - 1))
            elif language == 'go':
                pattern = r'"(.*?)"'  
                matches = re.findall(pattern, nodeText)
                if matches:
                    nodeText = matches[0]
                    nodeRange = ((nodeRange[0][0], nodeRange[0][1] + 1), (nodeRange[1][0], nodeRange[1][1] - 1)) 
            if nodeType.startswith('local'):
                if nodeType.startswith('local.definition.'):
                    defType = nodeType.split('local.definition.')[1]
                    if nodeText in defDict:
                        defDict[nodeText] = ObfuscatedNameType.ID
                    elif defType in {'var', 'parameter', 'field'}:
                        defDict[nodeText] = ObfuscatedNameType.VARIABLE
                    elif defType in {'function', 'method'}:
                        defDict[nodeText] = ObfuscatedNameType.FUNCTION
                    elif defType in {'type'}:
                        defDict[nodeText] = ObfuscatedNameType.CLASS

                if nodeText not in rangeDict:
                    rangeDict[nodeText] = [nodeRange]
                elif nodeRange not in rangeDict[nodeText]:
                    rangeDict[nodeText].append(nodeRange)
                else:
                    continue
            else:
                tokType = None
                if nodeType.startswith('function'):
                    tokType = ObfuscatedNameType.FUNCTION
                elif nodeType.startswith('import'):
                    tokType = ObfuscatedNameType.IMPORT
                    
                elif nodeType.startswith('type'):
                    tokType = ObfuscatedNameType.CLASS
                elif nodeType.startswith('variable') or nodeType.startswith('property'):
                    tokType = ObfuscatedNameType.VARIABLE
                
                if tokType is not None:
                    if nodeText not in allDict:
                        allDict[nodeText] = tokType
                    elif allDict[nodeText] is ObfuscatedNameType.VARIABLE:
                        allDict[nodeText] = tokType
                    elif tokType is not ObfuscatedNameType.VARIABLE:
                        allDict[nodeText] = ObfuscatedNameType.ID

                    if nodeRange not in refDict:
                        refDict[nodeRange] = (nodeText, tokType)
                    elif refDict[nodeRange] == (nodeText, ObfuscatedNameType.VARIABLE):
                        refDict[nodeRange] = (nodeText, tokType)
                    elif refDict[nodeRange] == (nodeText, ObfuscatedNameType.IMPORT):
                        pass
                    elif tokType is not ObfuscatedNameType.VARIABLE:
                        refDict[nodeRange] = (nodeText, ObfuscatedNameType.ID)
                     

            capturesMsg += f"identifier: {nodeText} type: {nodeType} node type: {node.type} \n"   
            if logging:
                logging('identifier: ' + nodeText + ' type: ' + nodeType + ' node type: ' +
                            node.type)
        except ValueError:
            if logging:
                logging(f'!!!!!!!Value error {node} type: {nodeType}')

    return rangeDict, defDict, refDict, allDict, capturesMsg


def is_range_overlapping(range_to_check, range_list):
    for existing_range in range_list:
        start_existing, end_existing = existing_range
        start_to_check, end_to_check = range_to_check

        # Check if the end of the existing range is before the start of the range to check
        if end_existing[0] < start_to_check[0] or \
           (end_existing[0] == start_to_check[0] and end_existing[1] < start_to_check[1]):
            continue

        # Check if the start of the existing range is after the end of the range to check
        if start_existing[0] > end_to_check[0] or \
           (start_existing[0] == end_to_check[0] and start_existing[1] > end_to_check[1]):
            continue

        # If neither of the above conditions is met, the ranges overlap
        return True

    # If no overlap was found, return False
    return False
