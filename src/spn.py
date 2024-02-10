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
        self.weights_children = []
        self.weights_parents = []

        return

    def backward(self):
        if len(self.parents) == 0:
            logger.log_fatal("Node " + str(self.id) + " has no parents. Quit.")
            exit(-1)

        value_backward_parents_product = []
        value_backward_parents_sum = []
        value_backward_product = None
        value_backward_sum = None
        value_forward_parents_product = []
        weights_parents_sum = []

        for (parent, weight_parents) in zip(self.parents, self.weights_parents):
            if isinstance(parent, ProductNode):
                value_backward_parents_product.append(parent.value_backward)
                value_forward_parents_product.append(parent.value_forward)
            elif isinstance(parent, SumNode):
                value_backward_parents_sum.append(parent.value_backward)
                weights_parents_sum.append(weight_parents)

        if len(value_backward_parents_product) > 0:
            value_backward_parents_product = torch.stack(value_backward_parents_product)

        if len(value_backward_parents_sum) > 0:
            value_backward_parents_sum = torch.stack(value_backward_parents_sum)

        if len(value_forward_parents_product) > 0:
            value_forward_parents_product = torch.stack(value_forward_parents_product)

        if len(weights_parents_sum) > 0:
            weights_parents_sum = torch.Tensor(weights_parents_sum).reshape(-1, 1)
            weights_parents_sum = weights_parents_sum.to(self.device)

        if len(value_backward_parents_product) > 0:
            value_backward_product = value_backward_parents_product + value_forward_parents_product - self.value_forward

        if len(value_backward_parents_sum) > 0:
            value_backward_sum = value_backward_parents_sum + torch.log(weights_parents_sum)

        if value_backward_product is not None and value_backward_sum is None:
            self.value_backward = value_backward_product
        elif value_backward_product is None and value_backward_sum is not None:
            self.value_backward = value_backward_sum
        else:
            self.value_backward = torch.stack([value_backward_product, value_backward_sum])

        # Compute backward values in log space
        # Log-Sum-Exp trick: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        value_backward_max = torch.max(self.value_backward, 0)[0]
        self.value_backward -= value_backward_max
        self.value_backward = torch.exp(self.value_backward)
        self.value_backward = torch.sum(self.value_backward, 0)
        self.value_backward = torch.log(self.value_backward) + value_backward_max

        return

    @abc.abstractmethod
    def forward(self):
        pass

class CategoricalLeafNode(Node):
    def __init__(self):
        super().__init__()

        self.attribute_index = -1
        self.category_index = -1

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
        value_forward_children *= self.weights_children
        self.value_forward = torch.sum(value_forward_children, 0)
        self.value_forward = torch.log(self.value_forward) + value_forward_children_max

        return

class SPN:
    def __init__(self, device = torch.device("cuda")):
        self.depth = -1
        self.device = device
        self.layers = {}
        self.leaf_nodes = {}
        self.nodes = []
        self.product_nodes = []
        self.reuse_backward = False
        self.reuse_forward = False
        self.root_node = None
        self.settings = None
        self.sum_nodes = []

        return

    def backward(self):
        if not self.reuse_backward:
            if len(self.layers) == 0:
                logger.log_fatal("Empty tree. Quit.")
                exit(-1)

            if len(self.layers[0]) > 1:
                logger.log_fatal("Multiple root nodes. Quit.")
                exit(-1)

            if self.root_node is None:
                logger.log_fatal("Missing root node. Quit.")
                exit(-1)

            if self.root_node.value_forward is None:
                logger.log_fatal("Missing root node forward value. Quit.")
                exit(-1)

            # Initialize root node backward value in log space
            self.root_node.value_backward = torch.log(torch.ones(self.settings.shape[0]))
            self.root_node.value_backward = self.root_node.value_backward.to(self.device)

            # Skip root node
            for i in range(1, self.depth):
                for node in self.layers[i]:
                    node.backward()

            self.reuse_backward = True

        return

    def forward(self):
        if not self.reuse_forward:
            if len(self.layers) == 0:
                logger.log_fatal("Empty tree. Quit.")
                exit(-1)

            if len(self.layers[0]) > 1:
                logger.log_fatal("Multiple root nodes. Quit.")
                exit(-1)

            if self.root_node is None:
                logger.log_fatal("Missing root node. Quit.")
                exit(-1)

            for i in range(self.depth - 1, -1, -1):
                for node in self.layers[i]:
                    node.forward()

            self.reuse_backward = False
            self.reuse_forward = True

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
                                self.nodes.append(categorical_leaf_node)

                            self.leaf_nodes[node_attribute_index] = categorical_leaf_node_list

                        sum_node = SumNode()
                        sum_node.children = categorical_leaf_node_list
                        sum_node.device = self.device
                        sum_node.id = node_id
                        sum_node.leaf = True
                        sum_node.weights_children = node_probabilities
                        self.nodes.append(sum_node)
                        self.sum_nodes.append(sum_node)

                        for (categorical_leaf_node, node_probability) in zip(categorical_leaf_node_list, node_probabilities):
                            categorical_leaf_node.parents.append(sum_node)
                            categorical_leaf_node.weights_parents.append(node_probability)
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
                            nodes[0].weights_children.append(node_weight)
                            nodes[1].parents.append(nodes[0])
                            nodes[1].weights_parents.append(node_weight)
                        else:
                            nodes[1].children.append(nodes[0])
                            nodes[1].weights_children.append(node_weight)
                            nodes[0].parents.append(nodes[1])
                            nodes[0].weights_parents.append(node_weight)
                    else:
                        if isinstance(nodes[0], ProductNode):
                            nodes[0].children.append(nodes[1])
                            nodes[1].parents.append(nodes[0])
                            nodes[1].weights_parents.append(1)
                        else:
                            nodes[1].children.append(nodes[0])
                            nodes[0].parents.append(nodes[1])
                            nodes[0].weights_parents.append(1)

        for node in self.nodes:
            node.weights_children = torch.Tensor(node.weights_children).reshape(-1, 1)
            node.weights_parents = torch.Tensor(node.weights_parents).reshape(-1, 1)

            node.weights_children = node.weights_children.to(self.device)
            node.weights_parents = node.weights_parents.to(self.device)

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

    def set(self, settings):
        self.settings = settings

        for attribute_index in self.leaf_nodes.keys():
            leaf_nodes = self.leaf_nodes[attribute_index]

            for leaf_node in leaf_nodes:
                leaf_node.set(self.settings)

        self.reuse_backward = False
        self.reuse_forward = False

        return
