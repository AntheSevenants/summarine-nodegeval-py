from openai import OpenAI

from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from nodegeval.knowledge_base import translate_table, translate_neighbours, shortest_path

import json
import functools

class RAG:
    def __init__(self, api_base, api_key, model, knowledge_base):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model

        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
        )

        self.knowledge_base = knowledge_base
        self.embeddings = []

        self.setup_tools()
        self.build_document_embeddings()
    
    def setup_tools(self):
        with open("tools.json", "rt") as reader:
            self.tools = json.loads(reader.read())

        self.available_functions = {
            "lookup": self.lookup,
            "get_top_n_documents": self.get_top_n_documents,
            "get_neighbours": self.get_neighbours,
            "get_document_by_index": self.get_document_by_index,
            "shortest_path": self.shortest_path,
        }

    def query(self, user_input):
        system_prompt = {"role": "system", "content": f"You are a chatbot helping a PhD student query his document database, explain concepts and draw conclusions.\n\nThe user can ask for explaining concepts, making comparisons, drafting article structures, or just ask questions in general. In your answer, never refer to document indices or document titles. Just draft the answer."}
        
        history = [system_prompt]

        user_message = {
            "role": "user",
            "content": user_input
        }
        history.append(user_message)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=history,
            tools=self.tools,
        )
        completion_message = completion.choices[0].message
        history.append(completion_message)

        print(completion)

        while completion_message.tool_calls:
            for tool_call in completion_message.tool_calls:
                tool_call_name = tool_call.function.name
                function_call_fn = self.available_functions[tool_call_name]

                function_call_arguments_str = tool_call.function.arguments
                function_call_arguments_json = json.loads(function_call_arguments_str)

                tool_call_result = function_call_fn(**function_call_arguments_json)

                function_call_result_message = {
                    "role": "tool",
                    "content": json.dumps(tool_call_result),
                    "tool_call_id": tool_call.id
                }

                history.append(function_call_result_message)
                print("\n",function_call_result_message,"\n")

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=history,
                tools=self.tools,
            )
            completion_message = completion.choices[0].message
            history.append(completion_message)
            print(completion_message)

        print(completion.choices[0].message.content)

    def build_document_embeddings(self):
        titles = [  markdown_file.filename for markdown_file in self.knowledge_base.markdown_files if not markdown_file.is_placeholder ]
        documents = [ markdown_file.content for markdown_file in self.knowledge_base.markdown_files if not markdown_file.is_placeholder ]
            
        document_pairs = list(zip(titles, documents))

        # TF-IDF approach
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform([doc for title, doc in document_pairs])

    def lookup(self, query, k=3, cutoff=0.3):
        query_vec = self.vectorizer.transform([query])
        cosine_similarities = (query_vec * self.tfidf_matrix.T).toarray().flatten()
    
        # Filter indices based on the cutoff threshold
        filtered_indices = [i for i, score in enumerate(cosine_similarities)
                            if score >= cutoff]

        # Sort the filtered indices by similarity score in descending order
        top_indices = sorted(filtered_indices, key=lambda i: -cosine_similarities[i])

        # Return the top k results (or fewer if cutoff filters them out)
        return [(i, float(cosine_similarities[i])) for i in top_indices[:k]]
    
    def get_top_n_documents(self, n=10):
        top_n_incoming = self.knowledge_base.get_top_incoming(n)
        return translate_table(top_n_incoming, self.knowledge_base)
    
    def get_neighbours(self, index):
        neighbours = self.knowledge_base.markdown_files[index].neighbours

        return translate_neighbours(neighbours, self.knowledge_base)
    
    def get_document_by_index(self, index):
        return f"""Index: {index}
Filename: {self.knowledge_base.markdown_files[index].filename}
Content:
{self.knowledge_base.markdown_files[index].content}
========
"""
    
    def shortest_path(self, start_index, end_index):
        path = shortest_path(start_index, end_index, self.knowledge_base)
        response = "This is the shortest path. Each line is a hop:\n"
        response += translate_neighbours(path, self.knowledge_base)

        return response