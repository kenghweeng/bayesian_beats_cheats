class Node():
    def __init__(self, student_name, is_plagiarism=False, adjacent_node=[], year_of_study=None, midterm_result=None, exp=None, tutorial_participation=None):
        """
        Args:
            student_name: name of the student for the mission
            is_plagiarism: plagiarism or not
            year_of_study: level of the student
            mission: name of the mission
            midterm_result: midterm result of the student
            exp: experience of the student
            tutorial_participation: tutorial participation of the student
            adjacent_node: list of adjacent nodes
        """
        self.student_name = student_name
        self.year_of_study = year_of_study
        self.mission = mission
        self.midterm_result = midterm_result
        self.exp = exp
        self.tutorial_participation = tutorial_participation
        self.is_plagiarism = is_plagiarism
        self.adjacent_node = adjacent_node


class Edge():
    def __init__(self, node_1, node_2, similarity):
        """
        Args:
            node_1: first node 
            node_2: second node 
            similarity: similarity of the two nodes
        """
        self.node_1 = node_1
        self.node_2 = node_2
        self.similarity = similarity

class Graph():
    def __init__(self, mission, node_list=None, adjacency_list=None):
        """
        Args:
            mission: name of mission for this graph
            node_list: node list of the graph
            adjacency_list: adjacency list of the graph
        """
        self.mission = mission
        self.node_list = node_list
        self.adjacency_list = adjacency_list