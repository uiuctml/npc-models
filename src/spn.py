import abc
import logger
import os
import torch

class Node:
    def __init__(self):
        self.children = []
        self.depth = -1
        self.device = None
        self.id = -1
        self.parents = []
        self.value_backward = None
        self.value_forward = None

        return

    @abc.abstractmethod
    def backward(self):
        pass

    @abc.abstractmethod
    def forward(self):
        pass

class CategoricalLeafNode(Node):
    def __init__(self):
        super().__init__()

        self.attribute_index = -1
        self.category_index = -1

        return

    def backward(self):
        return

    def forward(self):
        return

    def set(self, data):
        if self.attribute_index < 0 or self.category_index < 0:
            logger.log_fatal("Invalid categorical leaf node. Quit.")
            exit(-1)

        variables = data[:, self.attribute_index]

        settings = (variables == self.category_index)
        settings_marginal = (variables < 0)

        self.value_forward = torch.logical_or(settings, settings_marginal).float()
        self.value_forward = self.value_forward.to(self.device)

        # Compute forward values in log space
        self.value_forward = torch.log(self.value_forward)

        return

class ProductNode(Node):
    def __init__(self):
        super().__init__()

        return

    def backward(self):
        if len(self.parents) == 0:
            logger.log_fatal("Product node " + str(self.id) + " has no parents. Quit.")
            exit(-1)

        value_backward_parents = []
        value_forward_parents = []

        for parent in self.parents:
            value_backward_parents.append(parent.value_backward)
            value_forward_parents.append(parent.value_forward)

        # Compute backward values in log space
        # Log-Sum-Exp trick: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        value_backward_parents = torch.stack(value_backward_parents)
        value_backward_parents += value_forward_parents - self.value_forward
        value_backward_parents_max = torch.max(value_backward_parents, 0)[0]
        value_backward_parents -= value_backward_parents_max
        value_backward_parents = torch.exp(value_backward_parents)
        self.value_backward = torch.sum(value_backward_parents, 0)
        self.value_backward = torch.log(self.value_backward) + value_backward_parents_max

        return

    def forward(self):
        if len(self.children) == 0:
            logger.log_fatal("Product node " + str(self.id) + " has no children. Quit.")
            exit(-1)

        value_forward_children = []

        for child in self.children:
            value_forward_children.append(child.value_forward)

        # Compute forward values in log space
        value_forward_children = torch.stack(value_forward_children)
        self.value_forward = torch.sum(value_forward_children, 0)

        return

class SumNode(Node):
    def __init__(self):
        super().__init__()

        self.leaf = False
        self.weights = []

        return

    def backward(self):
        if len(self.parents) == 0:
            logger.log_fatal("Sum node " + str(self.id) + " has no parents. Quit.")
            exit(-1)

        value_backward_parents = []

        for parent in self.parents:
            value_backward_parents.append(parent.value_backward)

        # Compute backward values in log space
        # Log-Sum-Exp trick: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        value_backward_parents = torch.stack(value_backward_parents)
        value_backward_parents += torch.log(self.weights)
        value_backward_parents_max = torch.max(value_backward_parents, 0)[0]
        value_backward_parents -= value_backward_parents_max
        value_backward_parents = torch.exp(value_backward_parents)
        self.value_backward = torch.sum(value_backward_parents, 0)
        self.value_backward = torch.log(self.value_backward) + value_backward_parents_max

        return

    def forward(self):
        if len(self.children) == 0:
            logger.log_fatal("Sum node " + str(self.id) + " has no children. Quit.")
            exit(-1)

        value_forward_children = []

        for child in self.children:
            value_forward_children.append(child.value_forward)

        # Compute forward values in log space
        # Log-Sum-Exp trick: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        value_forward_children = torch.stack(value_forward_children)
        value_forward_children_max = torch.max(value_forward_children, 0)[0]
        value_forward_children -= value_forward_children_max
        value_forward_children = torch.exp(value_forward_children)
        value_forward_children *= self.weights
        self.value_forward = torch.sum(value_forward_children, 0)
        self.value_forward = torch.log(self.value_forward) + value_forward_children_max

        return

class SPN:
    def __init__(self):
        self.depth = -1
        self.device = torch.device("cuda")
        self.layers = {}
        self.leaf_nodes = {}
        self.nodes = []
        self.product_nodes = []
        self.reevalute = True
        self.root_node = []
        self.sum_nodes = []

        return

    def backward(self):
        if self.reevalute:
            if len(self.layers) == 0:
                logger.log_fatal("Empty tree. Quit.")
                exit(-1)

            if len(self.layers[0]) > 1:
                logger.log_fatal("Multiple root nodes. Quit.")
                exit(-1)

            # Skip root node
            for i in range(1, self.depth):
                for node in self.layers[i]:
                    node.backward()

            self.reevalute = False

        return

    def forward(self):
        if self.reevalute:
            if len(self.layers) == 0:
                logger.log_fatal("Empty tree. Quit.")
                exit(-1)

            if len(self.layers[0]) > 1:
                logger.log_fatal("Multiple root nodes. Quit.")
                exit(-1)

            for i in range(self.depth - 1, -1, -1):
                for node in self.layers[i]:
                    node.forward()

            self.reevalute = False

        return self.root_node.value_forward

    def load(self, file_path_spn):
        if not os.path.exists(file_path_spn):
            logger.log_fatal("Invalid SPN file path. Quit.")
            exit(-1)

        with open(file_path_spn, "r") as file_spn:
            reading_nodes = True

            for line in file_spn.readlines():
                line = line.strip()

                if line[0] == "#":
                    line = line.replace("#", "")

                    if line == "NODES":
                        reading_nodes = True
                    elif line == "EDGES":
                        reading_nodes = False

                    continue

                line_list = line.split(",")

                if reading_nodes:
                    node_id = int(line_list[0])
                    node_type = line_list[1]

                    if node_type == "SUM":
                        sum_node = SumNode()
                        sum_node.device = self.device
                        sum_node.id = node_id
                        self.nodes.append(sum_node)
                        self.sum_nodes.append(sum_node)
                    elif node_type == "PRD":
                        product_node = ProductNode()
                        product_node.device = self.device
                        product_node.id = node_id
                        self.nodes.append(product_node)
                        self.product_nodes.append(product_node)
                    elif node_type == "CatNode" or node_type == "CATNODE":
                        categorical_leaf_node_list = []
                        node_attribute_index = int(line_list[2])
                        node_probabilities = line_list[3:]

                        for i in range(0, len(node_probabilities)):
                            node_probabilities[i] = float(node_probabilities[i])

                        if node_attribute_index in self.leaf_nodes.keys():
                            categorical_leaf_node_list = self.leaf_nodes[node_attribute_index]
                        else:
                            for category_index in range(0, len(node_probabilities)):
                                categorical_leaf_node = CategoricalLeafNode()
                                categorical_leaf_node.attribute_index = node_attribute_index
                                categorical_leaf_node.category_index = category_index
                                categorical_leaf_node.device = self.device
                                categorical_leaf_node.id = -1
                                categorical_leaf_node_list.append(categorical_leaf_node)

                            self.leaf_nodes[node_attribute_index] = categorical_leaf_node_list

                        sum_node = SumNode()
                        sum_node.children = categorical_leaf_node_list
                        sum_node.device = self.device
                        sum_node.id = node_id
                        sum_node.leaf = True
                        sum_node.weights = node_probabilities
                        self.nodes.append(sum_node)
                        self.sum_nodes.append(sum_node)

                        for categorical_leaf_node in categorical_leaf_node_list:
                            categorical_leaf_node.parents.append(sum_node)
                else:
                    nodes = []
                    node_id_first = int(line_list[0])
                    node_id_second = int(line_list[1])

                    for node in self.nodes:
                        if node.id == node_id_first or node.id == node_id_second:
                            nodes.append(node)

                    if len(nodes) != 2:
                        logger.log_fatal("Invalid edge.")
                        exit(-1)

                    if len(line_list) >= 3:
                        node_weight = float(line_list[2])

                        if isinstance(nodes[0], SumNode) and not nodes[0].leaf:
                            nodes[0].children.append(nodes[1])
                            nodes[0].weights.append(node_weight)
                            nodes[1].parents.append(nodes[0])
                        else:
                            nodes[1].children.append(nodes[0])
                            nodes[1].weights.append(node_weight)
                            nodes[0].parents.append(nodes[1])
                    else:
                        if isinstance(nodes[0], ProductNode):
                            nodes[0].children.append(nodes[1])
                            nodes[1].parents.append(nodes[0])
                        else:
                            nodes[1].children.append(nodes[0])
                            nodes[0].parents.append(nodes[1])

        for sum_node in self.sum_nodes:
            sum_node.weights = torch.Tensor(sum_node.weights).reshape(-1, 1)
            sum_node.weights = sum_node.weights.to(self.device)

        root_nodes = []

        for node in self.nodes:
            if len(node.parents) == 0:
                root_nodes.append(node)

        if len(root_nodes) != 1:
            logger.log_fatal("Invalid SPN.")
            exit(-1)

        self.root_node = root_nodes[0]

        def traverse(self, node, layer):
            depth = layer
            node.depth = depth

            for child in node.children:
                if layer not in self.layers.keys():
                    self.layers[layer] = []

                self.layers[layer].append(child)

                depth_child = traverse(self, child, layer + 1)

                if depth_child > depth:
                    depth = depth_child

            return depth

        self.layers[0] = [self.root_node]
        self.depth = traverse(self, self.root_node, 1)

        return

    def set(self, data):
        for attribute_index in self.leaf_nodes.keys():
            leaf_nodes = self.leaf_nodes[attribute_index]

            for leaf_node in leaf_nodes:
                leaf_node.set(data)

        self.reevalute = True

        return
