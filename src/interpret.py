#!/usr/bin/env python3

import header
import json
import os
import PyQt5.QtCore
import PyQt5.QtGui
import PyQt5.QtWidgets

application = PyQt5.QtWidgets.QApplication([])
combo_box_application_control = PyQt5.QtWidgets.QComboBox()
counterfactuals = {}
group_box_viewer = PyQt5.QtWidgets.QGroupBox()
label_viewer = PyQt5.QtWidgets.QLabel()
layout_counterfactual = PyQt5.QtWidgets.QHBoxLayout()
layout_ground_truth = PyQt5.QtWidgets.QHBoxLayout()
layout_prediction = PyQt5.QtWidgets.QHBoxLayout()

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

    for (i, attribute_name) in enumerate(counterfactuals[file_name]["counterfactual"].keys()):
        attribute_label = counterfactuals[file_name]["counterfactual"][attribute_name][0]
        attribute_label_prediction = counterfactuals[file_name]["prediction"][attribute_name][0]
        attribute_probability = "{:.1f}".format(counterfactuals[file_name]["counterfactual"][attribute_name][1] * 100)
        label = layout_counterfactual.itemAt(i).widget()
        text = attribute_label + ": " + attribute_probability + "%"
        label.setText(text)

        if attribute_label_prediction != attribute_label:
            label.setStyleSheet("color: blue;")
        else:
            label.setStyleSheet("")

    for (i, attribute_name) in enumerate(counterfactuals[file_name]["ground_truth"].keys()):
        attribute_label = counterfactuals[file_name]["ground_truth"][attribute_name]
        attribute_label_prediction = counterfactuals[file_name]["prediction"][attribute_name][0]
        label = layout_ground_truth.itemAt(i).widget()
        label.setText(attribute_label)

    for (i, attribute_name) in enumerate(counterfactuals[file_name]["prediction"].keys()):
        attribute_label = counterfactuals[file_name]["prediction"][attribute_name][0]
        attribute_label_counterfactual = counterfactuals[file_name]["counterfactual"][attribute_name][0]
        attribute_probability = "{:.1f}".format(counterfactuals[file_name]["prediction"][attribute_name][1] * 100)
        label = layout_prediction.itemAt(i).widget()
        text = attribute_label + ": " + attribute_probability + "%"
        label.setText(text)

        if attribute_label != attribute_label_counterfactual:
            label.setStyleSheet("color: red;")
        else:
            label.setStyleSheet("")

    return

def updateViewerWidget():
    file_name = combo_box_application_control.currentText()
    dir_name_label = counterfactuals[file_name]["ground_truth"]["original"]
    file_path_image = os.path.join(header.config_decomposed["dir_dataset_test"], dir_name_label, file_name)

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

    for file_name in counterfactuals.keys():
        combo_box_application_control.addItem(file_name)

    combo_box_application_control.model().sort(0)
    combo_box_application_control.setCurrentIndex(0)
    combo_box_application_control.currentIndexChanged.connect(comboBoxApplicationControlSlot)

    return group_box_application_control

def createInterpretationWidget():
    group_box_counterfactual = PyQt5.QtWidgets.QGroupBox()
    group_box_ground_truth = PyQt5.QtWidgets.QGroupBox()
    group_box_interpretation = PyQt5.QtWidgets.QGroupBox()
    group_box_prediction = PyQt5.QtWidgets.QGroupBox()
    layout_interpretation = PyQt5.QtWidgets.QVBoxLayout()

    group_box_counterfactual.setAlignment(PyQt5.QtCore.Qt.AlignLeft)
    group_box_counterfactual.setLayout(layout_counterfactual)
    group_box_counterfactual.setTitle("Counterfactual")
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
    layout_interpretation.addWidget(group_box_counterfactual)

    file_name = combo_box_application_control.currentText()

    for attribute_name in counterfactuals[file_name]["counterfactual"].keys():
        attribute_label = counterfactuals[file_name]["counterfactual"][attribute_name][0]
        attribute_label_prediction = counterfactuals[file_name]["prediction"][attribute_name][0]
        attribute_probability = "{:.1f}".format(counterfactuals[file_name]["counterfactual"][attribute_name][1] * 100)
        text = attribute_label + ": " + attribute_probability + "%"
        label = PyQt5.QtWidgets.QLabel()
        label.setText(text)
        layout_counterfactual.addWidget(label)

        if attribute_name == "original":
            label.setFixedWidth(header.interpret_label_width_original)
        else:
            label.setFixedWidth(header.interpret_label_width_attribute)

        if attribute_label_prediction != attribute_label:
            label.setStyleSheet("color: blue;")

    for attribute_name in counterfactuals[file_name]["ground_truth"].keys():
        attribute_label = counterfactuals[file_name]["ground_truth"][attribute_name]
        label = PyQt5.QtWidgets.QLabel()
        label.setText(attribute_label)
        layout_ground_truth.addWidget(label)

        if attribute_name == "original":
            label.setFixedWidth(header.interpret_label_width_original)
        else:
            label.setFixedWidth(header.interpret_label_width_attribute)

    for attribute_name in counterfactuals[file_name]["prediction"].keys():
        attribute_label = counterfactuals[file_name]["prediction"][attribute_name][0]
        attribute_label_counterfactual = counterfactuals[file_name]["counterfactual"][attribute_name][0]
        attribute_probability = "{:.1f}".format(counterfactuals[file_name]["prediction"][attribute_name][1] * 100)
        text = attribute_label + ": " + attribute_probability + "%"
        label = PyQt5.QtWidgets.QLabel()
        label.setText(text)
        layout_prediction.addWidget(label)

        if attribute_name == "original":
            label.setFixedWidth(header.interpret_label_width_original)
        else:
            label.setFixedWidth(header.interpret_label_width_attribute)

        if attribute_label != attribute_label_counterfactual:
            label.setStyleSheet("color: red;")

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
    splitter = PyQt5.QtWidgets.QSplitter()
    widget_application_control = createApplicationControlWidget()
    widget_interpretation = createInterpretationWidget()
    widget_viewer = createViewerWidget()

    layout_window.addWidget(splitter)
    layout_window.addWidget(widget_application_control)

    splitter.addWidget(widget_viewer)
    splitter.addWidget(widget_interpretation)
    splitter.setOrientation(PyQt5.QtCore.Qt.Horizontal)

    return layout_window

def preprocessCounterfactuals():
    file_names_correct = []

    for file_name in counterfactuals.keys():
        correct = True

        for attribute_name in counterfactuals[file_name]["ground_truth"].keys():
            if counterfactuals[file_name]["prediction"][attribute_name][0] != counterfactuals[file_name]["ground_truth"][attribute_name]:
                correct = False
                break

        if correct:
            file_names_correct.append(file_name)

    for file_name_correct in file_names_correct:
        del counterfactuals[file_name_correct]

    return

def main():
    global counterfactuals

    with open(os.path.join(header.dir_output_counterfactual, header.file_name_counterfactual), "r") as file_counterfactual:
        counterfactuals = json.load(file_counterfactual)
        preprocessCounterfactuals()

    window = PyQt5.QtWidgets.QWidget()

    window.setLayout(createWindowLayout())
    window.setWindowTitle("VISAT Interpretation Tool")

    window.show()
    window.setFixedSize(window.size())

    exit(application.exec())

if __name__ == "__main__":
    main()
