import os
import re

from collections import deque

def scrape(directory):
    markdown_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(os.path.join(root, file))

    return markdown_files

def split_all(path):
    all_parts = []

    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            #all_parts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            all_parts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            all_parts.insert(0, parts[1])

    # Remove extension
    if all_parts[-1][-3:] == ".md":
        all_parts[-1] = all_parts[-1][:-3]

    return all_parts

def detect_references(text):
    references = re.findall(r"\[\[(.+?)\]\]", text)
    return references

def resolve_file(text, knowledge_base):
    reference_filename = text

    for markdown_file in knowledge_base.markdown_files:
        match_reference_parts = reference_filename.split("/")
        match_reference_part_count = len(match_reference_parts)
        # Cannot match because lengths are not the same
        if len(markdown_file.parts) < match_reference_part_count:
            continue

        # Match found!
        if markdown_file.parts[-match_reference_part_count:] == match_reference_parts:
            return markdown_file.index
    else:
        return None
    
def sort_by_count(references_list, indices_list):
    sorted_lists = sorted(references_list, key=lambda x: len(x), reverse=True)
    sorted_indices = sorted(indices_list, key=lambda i: len(references_list[i]), reverse=True)

    return list(zip(sorted_lists, sorted_indices))

def sort_by_size(incoming_count, n):
    indices = list(range(len(incoming_count)))

    sorted_indices = sorted(indices, key=lambda i: incoming_count[i], reverse=True)[:n]
    sorted_counts = [ incoming_count[index] for index in sorted_indices ]
    return list(zip(sorted_indices, sorted_counts))

def index_to_filename(index, knowledge_base):
    return knowledge_base.markdown_files[index].filename

def translate_table(top_n, knowledge_base):
    top_table = "index  filename   incoming_edges\n===\n"
    for index, count in top_n:
        top_table += f"{index}  {index_to_filename(index, knowledge_base)}  {count}\n"

    return top_table

def translate_neighbours(neighbour_indices, knowledge_base):
    table = "index  filename\n===\n"
    for index in neighbour_indices:
        table += f"{index}  {index_to_filename(index, knowledge_base)}\n"

    return table


def shortest_path(start_id, end_id, knowledge_base):
    if start_id < 0 or start_id >= len(knowledge_base.markdown_files):
        return IndexError("start_id out of bounds")
    
    if end_id < 0 or end_id >= len(knowledge_base.markdown_files):
        return IndexError("end_id out of bounds")
    
    # BFS initialization
    queue = deque()
    queue.append((start_id, [start_id]))  # (current_id, path)
    visited = set()
    visited.add(start_id)

    while queue:
        current_id, path = queue.popleft()
        current_file = knowledge_base.markdown_files[current_id]

        # If we reached the end file, return the path
        if current_id == end_id:
            return path

        # Explore outgoing neighbors
        for neighbor_id in current_file.neighbours:
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                queue.append((neighbor_id, path + [neighbor_id]))

    return None

def shortest_path_friendly_names(start_id, end_id, knowledge_base):
    return [ knowledge_base.markdown_files[index].filename for index in shortest_path(start_id, end_id, knowledge_base) ]

class KnowledgeBase:
    def __init__(self, base_path, category_filter=None):
        self.base_path = base_path
        self.markdown_files = []
        markdown_file_paths = scrape(self.base_path)

        for index, markdown_file_path in enumerate(markdown_file_paths):
            # Retain only the relative path so matching becomes easier
            relative_path = markdown_file_path.replace(self.base_path, "")
            markdown_file = MarkdownFile(markdown_file_path, relative_path, index)
            self.markdown_files.append(markdown_file)

        self.category_filter = category_filter

        self.compute_outgoing_neighbours()
        self.resolve_outgoing_neighbours()
        self.compute_incoming_neighbours()

    def get_last_index(self):
        return len(self.markdown_files)
    
    def compute_outgoing_neighbours(self):
        for markdown_file in self.markdown_files:
            relative_path = markdown_file.path
            if markdown_file.path[0] == "/":
                relative_path = markdown_file.path[1:]
            
            full_path = os.path.join(self.base_path, f"{relative_path}")

            with open(full_path, "rt") as reader:
                content = reader.read()
                references_text = detect_references(content)
                markdown_file.outgoing_neighbours = [ Reference.from_text(reference_text) for reference_text in references_text ]

    def resolve_outgoing_neighbours(self):
        for markdown_file in self.markdown_files:
            for reference in markdown_file.outgoing_neighbours:
                reference_filename = reference.filename

                resolved_id = resolve_file(reference_filename, self)
                if resolved_id is not None:
                    reference.filtered = markdown_file.category == self.category_filter
                    reference.refers_to_index = resolved_id
                else:
                    # print(f"Creating placeholder for reference: {reference_filename}.")
                    placeholder = Placeholder(reference_filename, self.get_last_index())
                    self.markdown_files.append(placeholder)

                    reference.refers_to_placeholder = True
                    reference.refers_to_index = placeholder.index
    
    def compute_incoming_neighbours(self):
        for markdown_file in self.markdown_files:
            outgoing_neighbours = markdown_file.outgoing_neighbours

            for outgoing_reference in outgoing_neighbours:
                # bv verwijst naar naslagwerk
                incoming_reference = Reference(markdown_file.filename, None, None, "incoming")
                incoming_reference.refers_to_index = markdown_file.index
                incoming_reference.filtered = self.markdown_files[outgoing_reference.refers_to_index].category == self.category_filter

                self.markdown_files[outgoing_reference.refers_to_index].incoming_neighbours.append(incoming_reference)

    def get_top_incoming(self, n=10):
        incoming_count = [ len(markdown_file.incoming_indices) for markdown_file in self.markdown_files ]
        return sort_by_size(incoming_count, n)

    def get_top_outgoing(self, n=10):
        outgoing_count = [ len(markdown_file.outgoing_indices) for markdown_file in self.markdown_files ]
        return sort_by_size(outgoing_count, n)

class MarkdownFile:
    def __init__(self, full_path, path, index):
        self.index = index
        self.full_path = full_path
        self.path = path
        self.parts = split_all(self.path)

        self.is_placeholder = False
        
        self.outgoing_neighbours = []
        self.incoming_neighbours = []

    @property
    def content(self):
        with open(self.full_path, "rt") as reader:
            content = reader.read()
            return content

    @property
    def filename(self):
        return self.parts[-1]
    
    @property
    def category(self):
        return self.parts[0]

    @property
    def outgoing_indices(self):
        return set([ reference.refers_to_index for reference in self.outgoing_neighbours if reference.filtered ])

    @property
    def incoming_indices(self):
        return set([ reference.refers_to_index for reference in self.incoming_neighbours if reference.filtered ])
    
    @property
    def neighbours(self):
        outgoing_indices = self.outgoing_indices
        incoming_indices = self.incoming_indices

        return outgoing_indices.union(incoming_indices)

class Placeholder(MarkdownFile):
    def __init__(self, path, index):
        super().__init__(None, path, index)
        self.is_placeholder = True

class Reference:
    def __init__(self, filename, section, friendly_name, ref_type="outgoing"):
        self.filename = filename
        self.section = section
        self.friendly_name = friendly_name
        
        self.ref_type = ref_type
        
        self.refers_to_placeholder = False
        self.refers_to_index = None
        self.filtered = True

    @classmethod
    def from_text(self, text):
        filename = []
        section = []
        friendly_name = []

        current_pointer = filename
        for character in text:
            if character == "#":
                current_pointer = section
                continue
            elif character == "|":
                current_pointer = friendly_name
                continue
            else:
                current_pointer.append(character)

        filename = "".join(filename)
        section = "".join(section)
        friendly_name = "".join(friendly_name)

        if len(filename) == 0:
            raise ValueError(f"Filename cannot be empty. Detected text: {text}")
        
        instance = self(filename, section, friendly_name)
        return instance