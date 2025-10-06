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
            layout_labels = PyQt5.QtWidgets.QVBoxLayout()
            spacer_bottom = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)
            spacer_top = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)

            deleteLayoutChildren(layout_explanation.itemAt(0))
            layout_explanation.removeItem(layout_explanation.itemAt(0))

            layout_explanation.addLayout(layout_labels)
            layout_labels.addItem(spacer_top)

            for attribute_label in explanations[file_name]["counterfactual"][attribute_name].keys():
                attribute_probability = "{:.1f}".format(explanations[file_name]["counterfactual"][attribute_name][attribute_label] * 100)
                label = PyQt5.QtWidgets.QLabel()
                text = attribute_label + ": " + attribute_probability + "%"
                label.setText(text)

                if attribute_name == "original":
                    label.setFixedWidth(header.interpret_label_width_original)
                else:
                    label.setFixedWidth(header.interpret_label_width_attribute)

                layout_labels.addWidget(label)

            layout_labels.addItem(spacer_bottom)

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
            layout_labels = PyQt5.QtWidgets.QVBoxLayout()
            spacer_bottom = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)
            spacer_top = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)

            deleteLayoutChildren(layout_prediction.itemAt(0))
            layout_prediction.removeItem(layout_prediction.itemAt(0))

            layout_prediction.addLayout(layout_labels)
            layout_labels.addItem(spacer_top)

            for attribute_label in explanations[file_name]["prediction"][attribute_name].keys():
                attribute_probability = "{:.1f}".format(explanations[file_name]["prediction"][attribute_name][attribute_label] * 100)
                label = PyQt5.QtWidgets.QLabel()
                text = attribute_label + ": " + attribute_probability + "%"
                label.setText(text)

                if attribute_name == "original":
                    label.setFixedWidth(header.interpret_label_width_original)
                else:
                    label.setFixedWidth(header.interpret_label_width_attribute)

                layout_labels.addWidget(label)

            layout_labels.addItem(spacer_bottom)

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
            layout_labels = PyQt5.QtWidgets.QVBoxLayout()
            spacer_bottom = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)
            spacer_top = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)

            layout_explanation.addLayout(layout_labels)
            layout_labels.addItem(spacer_top)

            for attribute_label in explanations[file_name]["counterfactual"][attribute_name].keys():
                attribute_probability = "{:.1f}".format(explanations[file_name]["counterfactual"][attribute_name][attribute_label] * 100)
                text = attribute_label + ": " + attribute_probability + "%"
                label = PyQt5.QtWidgets.QLabel()
                label.setText(text)

                if attribute_name == "original":
                    label.setFixedWidth(header.interpret_label_width_original)
                else:
                    label.setFixedWidth(header.interpret_label_width_attribute)

                layout_labels.addWidget(label)

            layout_labels.addItem(spacer_bottom)

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
            layout_labels = PyQt5.QtWidgets.QVBoxLayout()
            spacer_bottom = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)
            spacer_top = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)

            layout_prediction.addLayout(layout_labels)
            layout_labels.addItem(spacer_top)

            for attribute_label in explanations[file_name]["prediction"][attribute_name].keys():
                attribute_probability = "{:.1f}".format(explanations[file_name]["prediction"][attribute_name][attribute_label] * 100)
                text = attribute_label + ": " + attribute_probability + "%"
                label = PyQt5.QtWidgets.QLabel()
                label.setText(text)

                if attribute_name == "original":
                    label.setFixedWidth(header.interpret_label_width_original)
                else:
                    label.setFixedWidth(header.interpret_label_width_attribute)

                layout_labels.addWidget(label)

            layout_labels.addItem(spacer_bottom)

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

def preprocessCounterfactual():
    file_names_correct = []

    for file_name in explanations.keys():
        if list(explanations[file_name]["prediction"]["original"].keys())[0] == explanations[file_name]["ground_truth"]["original"]:
            file_names_correct.append(file_name)

    for file_name_correct in file_names_correct:
        del explanations[file_name_correct]

    return

def preprocessMPE():
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
        with open(os.path.join(header.interpret_dir_outputs, header.npc_mpe_file_name), "r") as file_mpe:
            explanations = json.load(file_mpe)

            if header.interpret_preprocess:
                preprocessMPE()
    else:
        with open(os.path.join(header.interpret_dir_outputs, header.npc_ce_file_name), "r") as file_counterfactual:
            explanations = json.load(file_counterfactual)

            if header.interpret_preprocess:
                preprocessCounterfactual()

    window = PyQt5.QtWidgets.QWidget()

    window.setLayout(createWindowLayout())
    window.setWindowTitle("NPC Interpretation Utility")

    window.show()
    window.setFixedSize(window.size())

    exit(application.exec())

if __name__ == "__main__":
    main()
