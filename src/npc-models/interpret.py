#!/usr/bin/env python3

"""
@file   interpret.py
@author Simon Yu
@date   03/12/2024
@brief  NPC Interpretation Utility.
"""

import header
import json
import os

os.environ["QT_LOGGING_RULES"] = "qt.gui.icc=false"

import PyQt5.QtCore
import PyQt5.QtGui
import PyQt5.QtWidgets

application = PyQt5.QtWidgets.QApplication([])
combo_box_application_control = PyQt5.QtWidgets.QComboBox()
explanations = {}
group_box_viewer = PyQt5.QtWidgets.QGroupBox()
label_viewer = PyQt5.QtWidgets.QLabel()
layout_ce = PyQt5.QtWidgets.QHBoxLayout()
layout_ground_truth = PyQt5.QtWidgets.QHBoxLayout()
layout_mpe = PyQt5.QtWidgets.QHBoxLayout()
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

    for (i, attribute_name) in enumerate(explanations[file_name]["ce"].keys()):
        layout_labels = PyQt5.QtWidgets.QVBoxLayout()
        spacer_bottom = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)
        spacer_top = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)

        deleteLayoutChildren(layout_ce.itemAt(0))
        layout_ce.removeItem(layout_ce.itemAt(0))

        layout_ce.addLayout(layout_labels)
        layout_labels.addItem(spacer_top)

        for attribute_label in explanations[file_name]["ce"][attribute_name].keys():
            attribute_probability = "{:05.1f}".format(explanations[file_name]["ce"][attribute_name][attribute_label] * 100)
            font = PyQt5.QtGui.QFontDatabase.systemFont(PyQt5.QtGui.QFontDatabase.FixedFont)
            label = PyQt5.QtWidgets.QLabel()
            text = '[' + attribute_probability + "%] " + attribute_label

            font.setPointSize(header.interpret_label_font_size)
            label.setText(text)

            if attribute_name == "class":
                label.setFixedWidth(header.interpret_label_width_class)
            else:
                label.setFixedWidth(header.interpret_label_width_attribute)

            label.setFont(font)
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

        if attribute_name == "class":
            attribute_label = "         " + explanations[file_name]["ground_truth"][attribute_name]
            font = PyQt5.QtGui.QFontDatabase.systemFont(PyQt5.QtGui.QFontDatabase.FixedFont)
            label = PyQt5.QtWidgets.QLabel()

            font.setPointSize(header.interpret_label_font_size)
            label.setText(attribute_label)
            label.setFixedWidth(header.interpret_label_width_class)
            label.setFont(font)
            layout_labels.addWidget(label)
        else:
            for attribute_label in explanations[file_name]["ground_truth"][attribute_name]:
                font = PyQt5.QtGui.QFontDatabase.systemFont(PyQt5.QtGui.QFontDatabase.FixedFont)
                label = PyQt5.QtWidgets.QLabel()

                font.setPointSize(header.interpret_label_font_size)
                label.setText("         " + attribute_label)
                label.setFixedWidth(header.interpret_label_width_attribute)
                label.setFont(font)
                layout_labels.addWidget(label)

        layout_labels.addItem(spacer_bottom)

    for (i, attribute_name) in enumerate(explanations[file_name]["mpe"].keys()):
        if attribute_name != "aligned":
            attribute_label = "         " + explanations[file_name]["mpe"][attribute_name]
            font = PyQt5.QtGui.QFontDatabase.systemFont(PyQt5.QtGui.QFontDatabase.FixedFont)
            label = layout_mpe.itemAt(i).widget()

            font.setPointSize(header.interpret_label_font_size)
            label.setText(attribute_label)
            label.setFont(font)
        else:
            aligned = explanations[file_name]["mpe"][attribute_name]
            font = PyQt5.QtGui.QFontDatabase.systemFont(PyQt5.QtGui.QFontDatabase.FixedFont)
            label = layout_mpe.itemAt(i).widget()

            font.setPointSize(header.interpret_label_font_size)

            if aligned:
                label.setText("         Aligned")
            else:
                label.setText("         Misaligned")

            label.setFont(font)

    for (i, attribute_name) in enumerate(explanations[file_name]["prediction"].keys()):
        layout_labels = PyQt5.QtWidgets.QVBoxLayout()
        spacer_bottom = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)
        spacer_top = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)

        deleteLayoutChildren(layout_prediction.itemAt(0))
        layout_prediction.removeItem(layout_prediction.itemAt(0))

        layout_prediction.addLayout(layout_labels)
        layout_labels.addItem(spacer_top)

        for attribute_label in explanations[file_name]["prediction"][attribute_name].keys():
            attribute_probability = "{:05.1f}".format(explanations[file_name]["prediction"][attribute_name][attribute_label] * 100)
            font = PyQt5.QtGui.QFontDatabase.systemFont(PyQt5.QtGui.QFontDatabase.FixedFont)
            label = PyQt5.QtWidgets.QLabel()
            text = '[' + attribute_probability + "%] " + attribute_label

            font.setPointSize(header.interpret_label_font_size)
            label.setText(text)

            if attribute_name == "class":
                label.setFixedWidth(header.interpret_label_width_class)
            else:
                label.setFixedWidth(header.interpret_label_width_attribute)

            label.setFont(font)
            layout_labels.addWidget(label)

        layout_labels.addItem(spacer_bottom)

    return

def updateViewerWidget():
    file_name = combo_box_application_control.currentText()
    dir_name_label = explanations[file_name]["ground_truth"]["class"]
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
    group_box_ce = PyQt5.QtWidgets.QGroupBox()
    group_box_ground_truth = PyQt5.QtWidgets.QGroupBox()
    group_box_interpretation = PyQt5.QtWidgets.QGroupBox()
    group_box_mpe = PyQt5.QtWidgets.QGroupBox()
    group_box_prediction = PyQt5.QtWidgets.QGroupBox()
    layout_interpretation = PyQt5.QtWidgets.QVBoxLayout()

    group_box_ce.setAlignment(PyQt5.QtCore.Qt.AlignLeft)
    group_box_ce.setLayout(layout_ce)
    group_box_ce.setTitle("Counterfactual Explanation")
    group_box_ground_truth.setAlignment(PyQt5.QtCore.Qt.AlignLeft)
    group_box_ground_truth.setLayout(layout_ground_truth)
    group_box_ground_truth.setTitle("Ground Truth")
    group_box_interpretation.setAlignment(PyQt5.QtCore.Qt.AlignHCenter)
    group_box_interpretation.setLayout(layout_interpretation)
    group_box_interpretation.setTitle("Interpretation")
    group_box_mpe.setAlignment(PyQt5.QtCore.Qt.AlignLeft)
    group_box_mpe.setLayout(layout_mpe)
    group_box_mpe.setTitle("Most Probable Explanation")
    group_box_prediction.setAlignment(PyQt5.QtCore.Qt.AlignLeft)
    group_box_prediction.setLayout(layout_prediction)
    group_box_prediction.setTitle("Prediction")

    layout_interpretation.addWidget(group_box_ground_truth)
    layout_interpretation.addWidget(group_box_prediction)
    layout_interpretation.addWidget(group_box_ce)
    layout_interpretation.addWidget(group_box_mpe)

    file_name = combo_box_application_control.currentText()

    for attribute_name in explanations[file_name]["ce"].keys():
        layout_labels = PyQt5.QtWidgets.QVBoxLayout()
        spacer_bottom = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)
        spacer_top = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)

        layout_ce.addLayout(layout_labels)
        layout_labels.addItem(spacer_top)

        for attribute_label in explanations[file_name]["ce"][attribute_name].keys():
            attribute_probability = "{:05.1f}".format(explanations[file_name]["ce"][attribute_name][attribute_label] * 100)
            font = PyQt5.QtGui.QFontDatabase.systemFont(PyQt5.QtGui.QFontDatabase.FixedFont)
            label = PyQt5.QtWidgets.QLabel()
            text = '[' + attribute_probability + "%] " + attribute_label

            font.setPointSize(header.interpret_label_font_size)
            label.setText(text)
            label.setFont(font)

            if attribute_name == "class":
                label.setFixedWidth(header.interpret_label_width_class)
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

        if attribute_name == "class":
            attribute_label = "         " + explanations[file_name]["ground_truth"][attribute_name]
            font = PyQt5.QtGui.QFontDatabase.systemFont(PyQt5.QtGui.QFontDatabase.FixedFont)
            label = PyQt5.QtWidgets.QLabel()

            font.setPointSize(header.interpret_label_font_size)
            label.setText(attribute_label)
            label.setFixedWidth(header.interpret_label_width_class)
            label.setFont(font)
            layout_labels.addWidget(label)
        else:
            for attribute_label in explanations[file_name]["ground_truth"][attribute_name]:
                font = PyQt5.QtGui.QFontDatabase.systemFont(PyQt5.QtGui.QFontDatabase.FixedFont)
                label = PyQt5.QtWidgets.QLabel()

                font.setPointSize(header.interpret_label_font_size)
                label.setText("         " + attribute_label)
                label.setFixedWidth(header.interpret_label_width_attribute)
                label.setFont(font)
                layout_labels.addWidget(label)

        layout_labels.addItem(spacer_bottom)

    for attribute_name in explanations[file_name]["mpe"].keys():
        if attribute_name != "aligned":
            attribute_label = "         " + explanations[file_name]["mpe"][attribute_name]
            font = PyQt5.QtGui.QFontDatabase.systemFont(PyQt5.QtGui.QFontDatabase.FixedFont)
            label = PyQt5.QtWidgets.QLabel()

            font.setPointSize(header.interpret_label_font_size)
            label.setText(attribute_label)
            label.setFixedWidth(header.interpret_label_width_attribute)
            label.setFont(font)
            layout_mpe.addWidget(label)
        else:
            aligned = explanations[file_name]["mpe"][attribute_name]
            font = PyQt5.QtGui.QFontDatabase.systemFont(PyQt5.QtGui.QFontDatabase.FixedFont)
            label = PyQt5.QtWidgets.QLabel()

            if aligned:
                label.setText("         Aligned")
            else:
                label.setText("         Misaligned")

            font.setPointSize(header.interpret_label_font_size)
            label.setFixedWidth(header.interpret_label_width_class)
            label.setFont(font)
            layout_mpe.addWidget(label)

    for attribute_name in explanations[file_name]["prediction"].keys():
            layout_labels = PyQt5.QtWidgets.QVBoxLayout()
            spacer_bottom = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)
            spacer_top = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)

            layout_prediction.addLayout(layout_labels)
            layout_labels.addItem(spacer_top)

            for attribute_label in explanations[file_name]["prediction"][attribute_name].keys():
                attribute_probability = "{:05.1f}".format(explanations[file_name]["prediction"][attribute_name][attribute_label] * 100)
                font = PyQt5.QtGui.QFontDatabase.systemFont(PyQt5.QtGui.QFontDatabase.FixedFont)
                label = PyQt5.QtWidgets.QLabel()
                text = '[' + attribute_probability + "%] " + attribute_label

                font.setPointSize(header.interpret_label_font_size)
                label.setText(text)

                if attribute_name == "class":
                    label.setFixedWidth(header.interpret_label_width_class)
                else:
                    label.setFixedWidth(header.interpret_label_width_attribute)

                label.setFont(font)
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
    spacer = PyQt5.QtWidgets.QSpacerItem(10, 10, PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)
    splitter = PyQt5.QtWidgets.QSplitter()
    widget_application_control = createApplicationControlWidget()
    widget_interpretation = createInterpretationWidget()
    widget_viewer = createViewerWidget()

    layout_window.addWidget(splitter)
    layout_window.addItem(spacer)
    layout_window.addWidget(widget_application_control)

    splitter.addWidget(widget_viewer)
    splitter.addWidget(widget_interpretation)
    splitter.setOrientation(PyQt5.QtCore.Qt.Horizontal)

    return layout_window

def main():
    global explanations

    with open(os.path.join(header.project_dir_outputs_interpret, header.dataset_prefix + ".json"), "r") as file_interpret:
        explanations = json.load(file_interpret)

    window = PyQt5.QtWidgets.QWidget()

    window.setLayout(createWindowLayout())
    window.setWindowTitle("NPC Interpretation Utility")

    window.show()

    exit(application.exec())

if __name__ == "__main__":
    main()
