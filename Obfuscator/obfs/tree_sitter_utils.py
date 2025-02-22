def byte_offset_to_position(text, byte_offset):
    current_byte = 0
    lines = text.splitlines(keepends=True)

    for line_number, line in enumerate(lines):
        next_byte = current_byte + len(line.encode('utf-8'))
        
        if next_byte > byte_offset:
            column = byte_offset - current_byte
            return (line_number, column)

        current_byte = next_byte

    if current_byte == byte_offset:
        return (len(lines), 0)

    raise ValueError("Byte offset is out of range")

def tree_to_string(node, source_code, indent="", last='updown'):
    amount = 3  # Amount of indentation
    name = node.type
    start_byte = node.start_byte
    end_byte = node.end_byte
    start_pos = byte_offset_to_position(source_code, start_byte)
    end_pos = byte_offset_to_position(source_code, end_byte)
    iD = source_code[node.start_byte:node.end_byte]
    if len(iD) > 10:
        iD = iD[:10] + '...'

    # Symbols for tree branches
    up = '\u2514'  # 'L' shaped elbow
    down = '\u251C'  # 'T' shaped elbow
    updown = '\u2502'  # Vertical line

    # Build the string for the current node
    first_prefix = f"{indent}{up if last == 'up' else down if last == 'down' else updown} "
    following_prefix = f"{indent}{' ' if last == 'up' else updown} {' ' * amount}"
    tree_str = f"{first_prefix}{name} {iD} (Start: {start_pos}, End: {end_pos})\n"

    if node.child_count > 0:
        *middle_children, last_child = node.children
        for child in middle_children:
            tree_str += tree_to_string(child, source_code, indent=following_prefix, last='down')
        tree_str += tree_to_string(last_child, source_code, indent=following_prefix, last='up')

    return tree_str

def find_identifier(node, source_code, identifier_list, logger=None):
    if node.type.endswith('identifier'):
        iD = source_code[node.start_byte:node.end_byte]
        if logger:
            logger.debug("source code:" + iD)   
        start_position = byte_offset_to_position(source_code, node.start_byte)
        end_position = byte_offset_to_position(source_code, node.end_byte)
        identifier_list.append((iD, node.start_point, node.end_point))
        # return node, start_position, end_position

    for child in node.children:
        find_identifier(child, source_code, identifier_list, logger=logger)
        # if result is not None:
            # return result ["python", "-m", "pylsp"]


def get_node_text(node, doc):
    start = node.start_point
    end = node.end_point
    lines = doc.lines

    if start[0] >= len(lines) or end[0]>= len(lines):
        raise ValueError("Start or end line out of document bounds")

    if start[1] > len(lines[start[0]]) or end[1]> len(lines[end[0]]):
        raise ValueError("Start or end character out of line bounds")

    if start[0] == end[0]:
        return lines[start[0]][start[1]:end[1]]
    else:
        nodeText = lines[start[0]][start[1]:]
        nodeText += ''.join(lines[start[0] + 1:end[0]])
        nodeText += lines[end[0]][:end[1]]

    return nodeText