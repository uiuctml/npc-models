import abc
import logger
import os
import torch

class SPNOptimizer:
    def __init__(self, spn):
        self.spn = spn

        return

    @abc.abstractmethod
    def step(self):
        pass

class CCCPOfflineSPNOptimizer:
    def __init__(self):
        super().__init__()

        return

    def step(self):


        return

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

    def set(self, settings):
        if self.attribute_index < 0 or self.category_index < 0:
            logger.log_fatal("Invalid categorical leaf node. Quit.")
            exit(-1)

        variables = settings[:, self.attribute_index]
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
        self.leaf_nodes = []
        self.leaf_nodes_dict = {}
        self.nodes = []
        self.product_nodes = []
        self.reuse_backward = False
        self.reuse_forward = False
        self.root_node = None
        self.settings = None
        self.sum_nodes = []
        self.traversal_order_backward = []
        self.traversal_order_forward = []

        return

    def backward(self):
        if not self.reuse_backward:
            if len(self.traversal_order_backward) == 0:
                logger.log_fatal("Empty tree. Quit.")
                exit(-1)

            if self.root_node is None or self.traversal_order_backward[0].id != self.root_node.id:
                logger.log_fatal("Missing root node. Quit.")
                exit(-1)

            if self.root_node.value_forward is None:
                logger.log_fatal("Missing root node forward value. Quit.")
                exit(-1)

            # Initialize root node backward value in log space
            self.root_node.value_backward = torch.log(torch.ones(self.settings.shape[0]))
            self.root_node.value_backward = self.root_node.value_backward.to(self.device)

            for node in self.traversal_order_backward:
                # Skip root node
                if node.id == self.root_node.id:
                    continue

                node.backward()

            self.reuse_backward = True

        return

    def forward(self):
        if not self.reuse_forward:
            if len(self.traversal_order_forward) == 0:
                logger.log_fatal("Empty tree. Quit.")
                exit(-1)

            if self.root_node is None or self.traversal_order_forward[-1].id != self.root_node.id:
                logger.log_fatal("Missing root node. Quit.")
                exit(-1)

            for node in self.traversal_order_forward:
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

                        if node_attribute_index in self.leaf_nodes_dict.keys():
                            categorical_leaf_node_list = self.leaf_nodes_dict[node_attribute_index]
                        else:
                            for category_index in range(0, len(node_probabilities)):
                                categorical_leaf_node = CategoricalLeafNode()
                                categorical_leaf_node.attribute_index = node_attribute_index
                                categorical_leaf_node.category_index = category_index
                                categorical_leaf_node.device = self.device
                                categorical_leaf_node.id = -1
                                categorical_leaf_node_list.append(categorical_leaf_node)
                                self.leaf_nodes.append(categorical_leaf_node)
                                self.nodes.append(categorical_leaf_node)

                            self.leaf_nodes_dict[node_attribute_index] = categorical_leaf_node_list

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
                        elif isinstance(nodes[1], SumNode) and not nodes[0].leaf:
                            nodes[1].children.append(nodes[0])
                            nodes[1].weights_children.append(node_weight)
                            nodes[0].parents.append(nodes[1])
                            nodes[0].weights_parents.append(node_weight)
                    else:
                        if isinstance(nodes[0], ProductNode):
                            nodes[0].children.append(nodes[1])
                            nodes[1].parents.append(nodes[0])
                            nodes[1].weights_parents.append(1)
                        elif isinstance(nodes[1], ProductNode):
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

        self.depth = self.traverse({0: root_nodes}, 0)
        self.root_node = root_nodes[0]
        self.traversal_order_backward = self.topologicalSort()
        self.traversal_order_forward = self.traversal_order_backward.copy()
        self.traversal_order_forward.reverse()

        if len(self.traversal_order_backward) != len(self.nodes):
            logger.log_fatal("Invalid SPN backward traversal")
            exit(-1)

        if len(self.traversal_order_forward) != len(self.nodes):
            logger.log_fatal("Invalid SPN forward traversal")
            exit(-1)

        return

    def setLeafNodes(self, settings):
        self.settings = settings

        for leaf_node in self.leaf_nodes:
            leaf_node.set(self.settings)

        self.reuse_backward = False
        self.reuse_forward = False

        return

    def topologicalSort(self):
        parent_count = {}
        queue = []
        topological_order = []

        for node in self.nodes:
            parent_count[node] = len(node.parents)

        queue.append(self.root_node)

        while len(queue) > 0:
            node = queue[0]
            topological_order.append(node)

            for child in node.children:
                parent_count[child] -= 1

                if parent_count[child] <= 0:
                    queue.append(child)

            queue.pop(0)

        return topological_order

    def traverse(self, layers, depth):
        if depth not in layers.keys():
            return depth

        for node in layers[depth]:
            node.depth = depth

            for child in node.children:
                if depth + 1 not in layers.keys():
                    layers[depth + 1] = []

                layers[depth + 1].append(child)

        depth_next = self.traverse(layers, depth + 1)

        if depth_next > depth:
            depth = depth_next

        return depth
