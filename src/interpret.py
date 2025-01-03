#!/usr/bin/env python3

import header
import json
import os
import PyQt5.QtCore
import PyQt5.QtGui
import PyQt5.QtWidgets

application = PyQt5.QtWidgets.QApplication([])
combo_box_application_control = PyQt5.QtWidgets.QComboBox()
explanations = {}
group_box_viewer = PyQt5.QtWidgets.QGroupBox()
label_viewer = PyQt5.QtWidgets.QLabel()
layout_explanation = PyQt5.QtWidgets.QHBoxLayout()
layout_ground_truth = PyQt5.QtWidgets.QHBoxLayout()
layout_prediction = PyQt5.QtWidgets.QHBoxLayout()
layout_summary = PyQt5.QtWidgets.QHBoxLayout()

def deleteLayoutChildren(layout):
     if layout is not None:
         while layout.count():
             item = layout.takeAt(0)
             widget = item.widget()

             if widget is not None:
                 widget.setParent(None)
             else:
                 deleteLayoutChildren(item.layout())

def generateSummaryLabel(attribute_labels):
    initialized = False
    summary = "The final prediction would have been correct if "

    for i in range(len(attribute_labels)):
        if not initialized:
            initialized = True
        elif i < 2 and i >= len(attribute_labels) - 1:
            summary += " and "
        elif i >= len(attribute_labels) - 1:
            summary += ", and "
        else:
            summary += ", "

        attribute_label_counterfactual = attribute_labels[i][1].split(header.interpret_delimiter_label)[1]
        attribute_label_prediction = attribute_labels[i][0].split(header.interpret_delimiter_label)[1]
        attribute_name = attribute_labels[i][0].split(header.interpret_delimiter_label)[0]

        summary += "<b>" + attribute_name + "</b> was predicted as <span style='color: blue;'><b>" + attribute_label_counterfactual + "</b></span> instead of <span style='color: red;'><b>" + attribute_label_prediction + "</b></span>"

    summary += "."

    return summary

def generateSummaryProbability(attribute_probabilities):
    initialized = False
    summary = "The final prediction would have been correct if "

    for i in range(len(attribute_probabilities)):
        if not initialized:
            initialized = True
        elif i < 2 and i >= len(attribute_probabilities) - 1:
            summary += " and "
        elif i >= len(attribute_probabilities) - 1:
            summary += ", and "
        else:
            summary += ", "

        attribute_probability_counterfactual = "{:.1f}%".format(attribute_probabilities[i][2] * 100)
        attribute_probability_prediction = "{:.1f}%".format(attribute_probabilities[i][1] * 100)
        attribute_label = attribute_probabilities[i][0].split(header.interpret_delimiter_label)[1]
        attribute_name = attribute_probabilities[i][0].split(header.interpret_delimiter_label)[0]

        summary += "<b>" + attribute_name + "</b> was predicted as <b>" + attribute_label + "</b> with a probability of <span style='color: blue;'><b>" + attribute_probability_counterfactual + "</b></span> instead of <span style='color: red;'><b>" + attribute_probability_prediction + "</b></span>"

    summary += "."

    return summary

def pushButtonLastSlot():
    if combo_box_application_control.currentIndex() <= 0:
        return

    combo_box_application_control.setCurrentIndex(combo_box_application_control.currentIndex() - 1)

    return

def pushButtonNextSlot():
    if combo_box_application_control.currentIndex() >= combo_box_application_control.count() - 1:
        return

    combo_box_application_control.setCurrentIndex(combo_box_application_control.currentIndex() + 1)

    return

def updateInterpretationWidget():
    attribute_labels = []
    attribute_probabilities = []
    file_name = combo_box_application_control.currentText()

    if header.interpret_mpe:
        for (i, attribute_name) in enumerate(explanations[file_name]["mpe"].keys()):
            if attribute_name != "correct":
                attribute_label = explanations[file_name]["mpe"][attribute_name]
                label = layout_explanation.itemAt(i).widget()
                label.setText(attribute_label)
            else:
                correct = explanations[file_name]["mpe"][attribute_name]
                label = layout_explanation.itemAt(i).widget()

                if correct:
                    label.setText("Correct")
                else:
                    label.setText("Incorrect")
    else:
        for (i, attribute_name) in enumerate(explanations[file_name]["counterfactual"].keys()):
            attribute_label = explanations[file_name]["counterfactual"][attribute_name][0]
            attribute_label_prediction = explanations[file_name]["prediction"][attribute_name][0]
            attribute_probability = "{:.1f}".format(explanations[file_name]["counterfactual"][attribute_name][1] * 100)
            label = layout_explanation.itemAt(i).widget()
            text = attribute_label + ": " + attribute_probability + "%"
            label.setText(text)

            if attribute_label_prediction != attribute_label:
                if attribute_name != "original":
                    attribute_labels.append((attribute_label_prediction, attribute_label))

                label.setStyleSheet("color: blue;")
            else:
                label.setStyleSheet("")

    for (i, attribute_name) in enumerate(explanations[file_name]["ground_truth"].keys()):
        layout_labels = PyQt5.QtWidgets.QVBoxLayout()
        spacer_bottom = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)
        spacer_top = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)

        deleteLayoutChildren(layout_ground_truth.itemAt(0))
        layout_ground_truth.removeItem(layout_ground_truth.itemAt(0))

        layout_ground_truth.addLayout(layout_labels)
        layout_labels.addItem(spacer_top)

        if attribute_name == "original":
            attribute_label = explanations[file_name]["ground_truth"][attribute_name]
            label = PyQt5.QtWidgets.QLabel()
            label.setText(attribute_label)
            label.setFixedWidth(header.interpret_label_width_original)
            layout_labels.addWidget(label)
        else:
            for attribute_label in explanations[file_name]["ground_truth"][attribute_name]:
                label = PyQt5.QtWidgets.QLabel()
                label.setText(attribute_label)
                label.setFixedWidth(header.interpret_label_width_attribute)
                layout_labels.addWidget(label)

        layout_labels.addItem(spacer_bottom)

    for (i, attribute_name) in enumerate(explanations[file_name]["prediction"].keys()):
        if header.interpret_mpe:
            layout_labels = PyQt5.QtWidgets.QVBoxLayout()
            spacer_bottom = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)
            spacer_top = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)

            deleteLayoutChildren(layout_prediction.itemAt(0))
            layout_prediction.removeItem(layout_prediction.itemAt(0))

            layout_prediction.addLayout(layout_labels)
            layout_labels.addItem(spacer_top)

            if attribute_name == "original":
                attribute_label = explanations[file_name]["prediction"][attribute_name][0]
                attribute_probability = "{:.1f}".format(explanations[file_name]["prediction"][attribute_name][1] * 100)
                text = attribute_label + ": " + attribute_probability + "%"
                label = PyQt5.QtWidgets.QLabel()
                label.setText(text)
                label.setFixedWidth(header.interpret_label_width_original)
                layout_labels.addWidget(label)
            else:
                for attribute_label in explanations[file_name]["prediction"][attribute_name].keys():
                    attribute_probability = "{:.1f}".format(explanations[file_name]["prediction"][attribute_name][attribute_label] * 100)
                    text = attribute_label + ": " + attribute_probability + "%"
                    label = PyQt5.QtWidgets.QLabel()
                    label.setText(text)
                    label.setFixedWidth(header.interpret_label_width_attribute)
                    layout_labels.addWidget(label)

            layout_labels.addItem(spacer_bottom)
        else:
            attribute_label = explanations[file_name]["prediction"][attribute_name][0]
            attribute_label_explanation = explanations[file_name]["counterfactual"][attribute_name][0]
            attribute_probability = "{:.1f}".format(explanations[file_name]["prediction"][attribute_name][1] * 100)
            label = layout_prediction.itemAt(i).widget()
            text = attribute_label + ": " + attribute_probability + "%"
            label.setText(text)

            if attribute_label != attribute_label_explanation:
                label.setStyleSheet("color: red;")
            else:
                label.setStyleSheet("")

    if not header.interpret_mpe:
        if len(attribute_labels) == 0:
            for (i, attribute_name) in enumerate(explanations[file_name]["counterfactual"].keys()):
                if attribute_name == "original":
                    continue

                attribute_label_prediction = explanations[file_name]["prediction"][attribute_name][0]
                attribute_probability = explanations[file_name]["counterfactual"][attribute_name][1]
                attribute_probability_prediction = explanations[file_name]["prediction"][attribute_name][1]
                label = layout_explanation.itemAt(i).widget()

                if abs(attribute_probability - attribute_probability_prediction) > header.interpret_threshold_probability:
                    attribute_probabilities.append((attribute_label_prediction, attribute_probability_prediction, attribute_probability))

                    label.setStyleSheet("color: blue;")
                else:
                    label.setStyleSheet("")

            for (i, attribute_name) in enumerate(explanations[file_name]["prediction"].keys()):
                if attribute_name == "original":
                    continue

                attribute_probability_counterfactual = explanations[file_name]["counterfactual"][attribute_name][1]
                attribute_probability = explanations[file_name]["prediction"][attribute_name][1]
                label = layout_prediction.itemAt(i).widget()

                if abs(attribute_probability_counterfactual - attribute_probability) > header.interpret_threshold_probability:
                    label.setStyleSheet("color: red;")
                else:
                    label.setStyleSheet("")

        label = layout_summary.itemAt(0).widget()

        if len(attribute_labels) == 0:
            label.setText(generateSummaryProbability(attribute_probabilities))
        else:
            label.setText(generateSummaryLabel(attribute_labels))

    return

def updateViewerWidget():
    file_name = combo_box_application_control.currentText()
    dir_name_label = explanations[file_name]["ground_truth"]["original"]
    file_path_image = os.path.join(header.interpret_dir_dataset, dir_name_label, file_name)

    pixmap = PyQt5.QtGui.QPixmap(file_path_image)
    label_viewer.setPixmap(pixmap.scaled(header.interpret_viewer_width, header.interpret_viewer_height, PyQt5.QtCore.Qt.IgnoreAspectRatio))

    return

def comboBoxApplicationControlSlot():
    updateInterpretationWidget()
    updateViewerWidget()

    return

def createApplicationControlWidget():
    group_box_application_control = PyQt5.QtWidgets.QGroupBox()
    layout_application_control = PyQt5.QtWidgets.QHBoxLayout()
    push_button_last_application_control = PyQt5.QtWidgets.QPushButton()
    push_button_next_application_control = PyQt5.QtWidgets.QPushButton()

    combo_box_application_control.view().setVerticalScrollBarPolicy(PyQt5.QtCore.Qt.ScrollBarAsNeeded)
    group_box_application_control.setAlignment(PyQt5.QtCore.Qt.AlignHCenter)
    group_box_application_control.setLayout(layout_application_control)
    group_box_application_control.setTitle("Application Control")
    layout_application_control.addWidget(push_button_last_application_control)
    layout_application_control.addWidget(combo_box_application_control)
    layout_application_control.addWidget(push_button_next_application_control)
    push_button_last_application_control.clicked.connect(pushButtonLastSlot)
    push_button_last_application_control.setText("Last")
    push_button_next_application_control.clicked.connect(pushButtonNextSlot)
    push_button_next_application_control.setText("Next")

    for file_name in explanations.keys():
        combo_box_application_control.addItem(file_name)

    combo_box_application_control.model().sort(0)
    combo_box_application_control.setCurrentIndex(0)
    combo_box_application_control.currentIndexChanged.connect(comboBoxApplicationControlSlot)

    return group_box_application_control

def createInterpretationWidget():
    group_box_explanation = PyQt5.QtWidgets.QGroupBox()
    group_box_ground_truth = PyQt5.QtWidgets.QGroupBox()
    group_box_interpretation = PyQt5.QtWidgets.QGroupBox()
    group_box_prediction = PyQt5.QtWidgets.QGroupBox()
    group_box_summary = PyQt5.QtWidgets.QGroupBox()
    layout_interpretation = PyQt5.QtWidgets.QVBoxLayout()

    group_box_explanation.setAlignment(PyQt5.QtCore.Qt.AlignLeft)
    group_box_explanation.setLayout(layout_explanation)

    if header.interpret_mpe:
        group_box_explanation.setTitle("Most Probable Explanation")
    else:
        group_box_explanation.setTitle("Counterfactual Explanation")

    group_box_ground_truth.setAlignment(PyQt5.QtCore.Qt.AlignLeft)
    group_box_ground_truth.setLayout(layout_ground_truth)
    group_box_ground_truth.setTitle("Ground Truth")
    group_box_interpretation.setAlignment(PyQt5.QtCore.Qt.AlignHCenter)
    group_box_interpretation.setLayout(layout_interpretation)
    group_box_interpretation.setTitle("Interpretation")
    group_box_prediction.setAlignment(PyQt5.QtCore.Qt.AlignLeft)
    group_box_prediction.setLayout(layout_prediction)
    group_box_prediction.setTitle("Prediction")

    layout_interpretation.addWidget(group_box_ground_truth)
    layout_interpretation.addWidget(group_box_prediction)
    layout_interpretation.addWidget(group_box_explanation)

    if not header.interpret_mpe:
        group_box_summary.setAlignment(PyQt5.QtCore.Qt.AlignLeft)
        group_box_summary.setLayout(layout_summary)
        group_box_summary.setTitle("Summary")
        layout_interpretation.addWidget(group_box_summary)

    attribute_labels = []
    attribute_probabilities = []
    file_name = combo_box_application_control.currentText()

    if header.interpret_mpe:
        for attribute_name in explanations[file_name]["mpe"].keys():
            if attribute_name == "correct":
                correct = explanations[file_name]["mpe"][attribute_name]
                label = PyQt5.QtWidgets.QLabel()

                if correct:
                    label.setText("Correct")
                else:
                    label.setText("Incorrect")

                label.setFixedWidth(header.interpret_label_width_original)
                layout_explanation.addWidget(label)
            else:
                attribute_label = explanations[file_name]["mpe"][attribute_name]
                label = PyQt5.QtWidgets.QLabel()
                label.setText(attribute_label)
                label.setFixedWidth(header.interpret_label_width_attribute)
                layout_explanation.addWidget(label)
    else:
        for attribute_name in explanations[file_name]["counterfactual"].keys():
            attribute_label = explanations[file_name]["counterfactual"][attribute_name][0]
            attribute_label_prediction = explanations[file_name]["prediction"][attribute_name][0]
            attribute_probability = "{:.1f}".format(explanations[file_name]["counterfactual"][attribute_name][1] * 100)
            text = attribute_label + ": " + attribute_probability + "%"
            label = PyQt5.QtWidgets.QLabel()
            label.setText(text)
            layout_explanation.addWidget(label)

            if attribute_name == "original":
                label.setFixedWidth(header.interpret_label_width_original)
            else:
                label.setFixedWidth(header.interpret_label_width_attribute)

            if attribute_label_prediction != attribute_label:
                if attribute_name != "original":
                    attribute_labels.append((attribute_label_prediction, attribute_label))

                label.setStyleSheet("color: blue;")

    for attribute_name in explanations[file_name]["ground_truth"].keys():
        layout_labels = PyQt5.QtWidgets.QVBoxLayout()
        spacer_bottom = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)
        spacer_top = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)

        layout_ground_truth.addLayout(layout_labels)
        layout_labels.addItem(spacer_top)

        if attribute_name == "original":
            attribute_label = explanations[file_name]["ground_truth"][attribute_name]
            label = PyQt5.QtWidgets.QLabel()
            label.setText(attribute_label)
            label.setFixedWidth(header.interpret_label_width_original)
            layout_labels.addWidget(label)
        else:
            for attribute_label in explanations[file_name]["ground_truth"][attribute_name]:
                label = PyQt5.QtWidgets.QLabel()
                label.setText(attribute_label)
                label.setFixedWidth(header.interpret_label_width_attribute)
                layout_labels.addWidget(label)

        layout_labels.addItem(spacer_bottom)

    for attribute_name in explanations[file_name]["prediction"].keys():
        if header.interpret_mpe:
            layout_labels = PyQt5.QtWidgets.QVBoxLayout()
            spacer_bottom = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)
            spacer_top = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)

            layout_prediction.addLayout(layout_labels)
            layout_labels.addItem(spacer_top)

            if attribute_name == "original":
                attribute_label = explanations[file_name]["prediction"][attribute_name][0]
                attribute_probability = "{:.1f}".format(explanations[file_name]["prediction"][attribute_name][1] * 100)
                text = attribute_label + ": " + attribute_probability + "%"
                label = PyQt5.QtWidgets.QLabel()
                label.setText(text)
                label.setFixedWidth(header.interpret_label_width_original)
                layout_labels.addWidget(label)
            else:
                for attribute_label in explanations[file_name]["prediction"][attribute_name].keys():
                    attribute_probability = "{:.1f}".format(explanations[file_name]["prediction"][attribute_name][attribute_label] * 100)
                    text = attribute_label + ": " + attribute_probability + "%"
                    label = PyQt5.QtWidgets.QLabel()
                    label.setText(text)
                    label.setFixedWidth(header.interpret_label_width_attribute)
                    layout_labels.addWidget(label)

            layout_labels.addItem(spacer_bottom)
        else:
            attribute_label = explanations[file_name]["prediction"][attribute_name][0]
            attribute_label_explanation = explanations[file_name]["counterfactual"][attribute_name][0]

            attribute_probability = "{:.1f}".format(explanations[file_name]["prediction"][attribute_name][1] * 100)
            text = attribute_label + ": " + attribute_probability + "%"
            label = PyQt5.QtWidgets.QLabel()
            label.setText(text)
            layout_prediction.addWidget(label)

            if attribute_name == "original":
                label.setFixedWidth(header.interpret_label_width_original)
            else:
                label.setFixedWidth(header.interpret_label_width_attribute)

            if attribute_label != attribute_label_explanation:
                label.setStyleSheet("color: red;")

    if not header.interpret_mpe:
        if len(attribute_labels) == 0:
            for (i, attribute_name) in enumerate(explanations[file_name]["counterfactual"].keys()):
                if attribute_name == "original":
                    continue

                attribute_label_prediction = explanations[file_name]["prediction"][attribute_name][0]
                attribute_probability = explanations[file_name]["counterfactual"][attribute_name][1]
                attribute_probability_prediction = explanations[file_name]["prediction"][attribute_name][1]
                label = layout_explanation.itemAt(i).widget()

                if abs(attribute_probability - attribute_probability_prediction) > header.interpret_threshold_probability:
                    attribute_probabilities.append((attribute_label_prediction, attribute_probability_prediction, attribute_probability))

                    label.setStyleSheet("color: blue;")
                else:
                    label.setStyleSheet("")

            for (i, attribute_name) in enumerate(explanations[file_name]["prediction"].keys()):
                if attribute_name == "original":
                    continue

                attribute_probability_counterfactual = explanations[file_name]["counterfactual"][attribute_name][1]
                attribute_probability = explanations[file_name]["prediction"][attribute_name][1]
                label = layout_prediction.itemAt(i).widget()

                if abs(attribute_probability_counterfactual - attribute_probability) > header.interpret_threshold_probability:
                    label.setStyleSheet("color: red;")
                else:
                    label.setStyleSheet("")

        label = PyQt5.QtWidgets.QLabel()
        label.setTextFormat(PyQt5.QtCore.Qt.RichText)

        if len(attribute_labels) == 0:
            label.setText(generateSummaryProbability(attribute_probabilities))
        else:
            label.setText(generateSummaryLabel(attribute_labels))

        layout_summary.addWidget(label)

    return group_box_interpretation

def createViewerWidget():
    layout_viewer = PyQt5.QtWidgets.QHBoxLayout()
    spacer_left = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Expanding, PyQt5.QtWidgets.QSizePolicy.Minimum)
    spacer_right = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Expanding, PyQt5.QtWidgets.QSizePolicy.Minimum)

    group_box_viewer.setAlignment(PyQt5.QtCore.Qt.AlignHCenter)
    group_box_viewer.setLayout(layout_viewer)
    group_box_viewer.setTitle("Viewer")

    layout_viewer.addItem(spacer_left)
    layout_viewer.addWidget(label_viewer)
    layout_viewer.addItem(spacer_right)

    updateViewerWidget()

    return group_box_viewer

def createWindowLayout():
    layout_window = PyQt5.QtWidgets.QVBoxLayout()
    spacer_top = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)
    splitter = PyQt5.QtWidgets.QSplitter()
    widget_application_control = createApplicationControlWidget()
    widget_interpretation = createInterpretationWidget()
    widget_viewer = createViewerWidget()

    layout_window.addItem(spacer_top)
    layout_window.addWidget(splitter)
    layout_window.addWidget(widget_application_control)

    splitter.addWidget(widget_viewer)
    splitter.addWidget(widget_interpretation)
    splitter.setOrientation(PyQt5.QtCore.Qt.Horizontal)

    return layout_window

def preprocessCounterfactuals():
    file_names_correct = []

    for file_name in explanations.keys():
        if explanations[file_name]["prediction"]["original"][0] == explanations[file_name]["ground_truth"]["original"]:
            file_names_correct.append(file_name)

    for file_name_correct in file_names_correct:
        del explanations[file_name_correct]

    return

def preprocessMPEs():
    file_names_incorrect = []

    for file_name in explanations.keys():
        if explanations[file_name]["prediction"]["original"][0] != explanations[file_name]["ground_truth"]["original"]:
            file_names_incorrect.append(file_name)

    for file_name_incorrect in file_names_incorrect:
        del explanations[file_name_incorrect]

    return

def main():
    global explanations

    if header.interpret_mpe:
        with open(os.path.join(header.dir_output_mpe, header.file_name_mpe), "r") as file_mpe:
            explanations = json.load(file_mpe)
    else:
        with open(os.path.join(header.dir_output_counterfactual, header.file_name_counterfactual), "r") as file_counterfactual:
            explanations = json.load(file_counterfactual)
            preprocessCounterfactuals()

    window = PyQt5.QtWidgets.QWidget()

    window.setLayout(createWindowLayout())
    window.setWindowTitle("VISAT Interpretation Tool")

    window.show()
    window.setFixedSize(window.size())

    exit(application.exec())

if __name__ == "__main__":
    main()
