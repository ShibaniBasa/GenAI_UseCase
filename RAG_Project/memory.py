from collections import defaultdict

class ConversationMemory:

    def __init__(self, max_history=5):
        self.histories = defaultdict(list)
        self.max_history = max_history

    def add_message(self, session_id, role, content):

        self.histories[session_id].append({
            "role": role,
            "content": content
        })

        if len(self.histories[session_id]) > self.max_history * 2:
            self.histories[session_id] = \
                self.histories[session_id][-self.max_history * 2:]

    def get_history(self, session_id):
        return self.histories.get(session_id, [])

    def clear(self, session_id):
        self.histories.pop(session_id, None)

memory = ConversationMemory()