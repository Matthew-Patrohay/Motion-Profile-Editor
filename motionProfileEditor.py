import sys
import os
import numpy as np
import logging
import matplotlib
import matplotlib.pyplot as plt
from PyQt6.QtCore import Qt
from pyscurve import ScurvePlanner
from pyscurve.trajectory import PlanningError
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QDoubleSpinBox, QPushButton, QHBoxLayout, QSpacerItem, QSizePolicy, QGridLayout
)

matplotlib.use('Agg')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Detect system theme (macOS/Linux GNOME-based example)
is_dark_mode = os.popen("defaults read -g AppleInterfaceStyle 2>/dev/null").read().strip() == "Dark"

# Set colors based on theme
bg_color = "#1c1c1e" if is_dark_mode else "#FFFFFF"
text_color = "#d1d1d6" if is_dark_mode else "#000000"

# Scale large integers
jerk_scale = 1000

plt.rcParams.update({
    "figure.facecolor": bg_color,
    "axes.facecolor": bg_color,
    "axes.edgecolor": text_color,
    "axes.labelcolor": text_color,
    "xtick.color": text_color,
    "ytick.color": text_color,
    "text.color": text_color
})

class TrajectoryPlanner(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Motion Profile Editor")
        self.setGeometry(100, 100, 1400, 800)

        # Apply dark/light mode theme to the entire GUI
        self.setStyleSheet(f"background-color: {bg_color}; color: {text_color};")

        # Initialize S-curve planner
        self.p = ScurvePlanner()
        
        # Default Parameters
        self.default_q0, self.default_q1 = 0, 90
        self.default_v0, self.default_v1 = 0, 0
        self.default_v_max, self.default_a_max, self.default_j_max = 15750, 3675000, 2500000000
        self.backup_default_jerk = self.default_j_max

        # Initialize parameters
        self.q0, self.q1 = self.default_q0, self.default_q1
        self.v0, self.v1 = self.default_v0, self.default_v1
        self.v_max, self.a_max, self.j_max = self.default_v_max, self.default_a_max, self.default_j_max
        
        # Create the Matplotlib figure and axis
        self.fig, (self.ax_position, self.ax_speed, self.ax_accel, self.ax_jerk) = plt.subplots(4, 1, figsize=(6, 8), sharex=True)
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0)
                    
        # Add axis labels with units
        self.ax_position.set_ylabel("degrees")
        self.ax_speed.set_ylabel("degrees/sec")
        self.ax_accel.set_ylabel("degrees/sec²")
        self.ax_jerk.set_ylabel("degrees/sec³")
        self.ax_jerk.set_xlabel("seconds")

        # Match figure background to GUI
        self.fig.patch.set_facecolor(bg_color)
        for ax in [self.ax_position, self.ax_speed, self.ax_accel, self.ax_jerk]:
            ax.set_facecolor(bg_color)
            ax.spines['bottom'].set_color(text_color)
            ax.spines['top'].set_color(text_color)
            ax.spines['left'].set_color(text_color)
            ax.spines['right'].set_color(text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.tick_params(colors=text_color)

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet(f"background-color: {bg_color};")

        # Initial plot setup
        self.time = np.linspace(0, 1, 100)
        self.position_line, = self.ax_position.plot(self.time, np.zeros_like(self.time), label="Position", color='#bf5af2')
        self.speed_line, = self.ax_speed.plot(self.time, np.zeros_like(self.time), label="Speed", color='#0b84ff')
        self.accel_line, = self.ax_accel.plot(self.time, np.zeros_like(self.time), label="Acceleration", color='#ff453a')
        self.jerk_line, = self.ax_jerk.plot(self.time, np.zeros_like(self.time), label="Jerk", color='#ff9f0b')

        self.ax_position.legend(facecolor=bg_color, edgecolor=text_color)
        self.ax_speed.legend(facecolor=bg_color, edgecolor=text_color)
        self.ax_accel.legend(facecolor=bg_color, edgecolor=text_color)
        self.ax_jerk.legend(facecolor=bg_color, edgecolor=text_color)
        
        # Layout for GUI elements
        main_layout = QHBoxLayout()

        # Left side: Matplotlib plot
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.canvas)

        # Right side: Sliders section
        right_column = QVBoxLayout()

        # Create a blank top section
        top_spacer_section = QVBoxLayout()
        top_spacer = QLabel("")
        top_spacer.setFixedHeight(200)
        top_spacer_section.addWidget(top_spacer)

        # Sliders section
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(2)

        # Warning label (placed above the plot)
        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: red; font-weight: bold;")
        plot_layout.addWidget(self.warning_label)

        # Updated Controls List (with correct syntax)
        self.controls = [
            ("Velocity Limit", 1, 50000, self.v_max),
            ("Acceleration Limit", 1, 12000000, self.a_max),
            ("Jerk Limit", 1, 10000000000, self.j_max),
            ("Initial Position", 0, 180, self.q0),
            ("Target Position", 0, 180, self.q1),
            ("Initial Velocity", -50000, 50000, self.v0),
            ("Final Velocity", -50000, 50000, self.v1),
        ]

        self.sliders = []
        self.inputs = []

        for i, (label, min_val, max_val, init_val) in enumerate(self.controls):
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {text_color};")

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(min_val)
            scaled_max_val = max_val if i != 2 else int(max_val / jerk_scale)
            slider.setMaximum(scaled_max_val)
            scaled_init_val = init_val if i != 2 else int(init_val / jerk_scale)
            slider.setValue(scaled_init_val)
            slider.setTickInterval(1)
            slider.setSingleStep(1)
            slider.setFixedHeight(25)

            input_box = QDoubleSpinBox()
            input_box.setRange(min_val, max_val)
            input_box.setValue(init_val)
            input_box.setSingleStep(1)
            input_box.setDecimals(0)
            input_box.setFixedHeight(20)
            input_box.setFixedWidth(120)

            # Apply a dark grey style ONLY to the first three sliders (Velocity, Acceleration, Jerk)
            if i >= 3:
                slider.setStyleSheet("""
                    QSlider::groove:horizontal {
                        background: #333333;  /* Dark Grey */
                        height: 6px;
                        border-radius: 3px;
                    }
                    QSlider::handle:horizontal {
                        background: #555555;  /* Slightly lighter grey handle */
                        width: 14px;
                        height: 14px;
                        border-radius: 7px;
                    }
                """)

            # Sync slider & input box
            if i == 2:  # If this is the Jerk Limit (scaled)
                slider.valueChanged.connect(lambda value, box=input_box: box.setValue(value * jerk_scale))
                input_box.valueChanged.connect(lambda value, s=slider: s.setValue(int(value / jerk_scale)))
            else:  # Regular behavior for other sliders
                slider.valueChanged.connect(lambda value, box=input_box: box.setValue(value))
                input_box.valueChanged.connect(lambda value, s=slider: s.setValue(int(value)))
            input_box.valueChanged.connect(self.update_plot)

            # Store references
            self.sliders.append(slider)
            self.inputs.append(input_box)

            row.addWidget(lbl)
            row.addWidget(slider)
            row.addWidget(input_box)
            controls_layout.addLayout(row)

        # Spacer section (below sliders)
        spacer_section = QVBoxLayout()

        # Reset button
        reset_button = QPushButton("Reset Values")
        reset_button.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3c;  /* Dark Grey */
                color: #d1d1d6;
                padding: 8px 15px;  /* Adjust padding */
                border-radius: 5px;  /* Rounded corners */
                font-size: 14px;
                min-width: 120px; /* Set a minimum width */
                max-width: 160px; /* Set a maximum width */
            }
            QPushButton:hover {
                background-color: #48484a;  /* Slightly lighter on hover */
            }
        """)

        # Function to reset sliders and input boxes
        def reset_values():
            default_values = [self.default_v_max,  # Default max velocity
                              self.default_a_max,  # Default max acceleration
                              self.default_j_max,  # Default jerk
                              self.default_q0,  # Default inital position
                              self.default_q1, # Default final position
                              self.default_v0,  # Default initial velocity
                              self.default_v1]  # Default final velocity

            for i, default in enumerate(default_values):
                self.inputs[i].setValue(default)
                if i is 2:
                    self.sliders[i].setValue(int(default / jerk_scale))
                else:
                    self.sliders[i].setValue(default)

        reset_button.clicked.connect(reset_values)
        
        # Trapezoidal Mode button
        trapezoidal_button = QPushButton("Trapezoidal Mode")
        trapezoidal_button.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3c;  /* Dark Grey */
                color: #d1d1d6;
                padding: 8px 15px;
                border-radius: 5px;
                font-size: 14px;
                min-width: 120px;
                max-width: 160px;
            }
            QPushButton:hover {
                background-color: #48484a;
            }
        """)

        self.trapezoidal_mode = False  # Track mode state

        # Function to toggle between Trapezoidal and S-Curve Mode
        def toggle_mode():
            if not self.trapezoidal_mode:
                self.sliders[2].setMaximum(int(100000000000 / jerk_scale))  # Increase Jerk slider max limit
                self.inputs[2].setRange(1, 100000000000)  # Expand input range
                self.inputs[2].setValue(100000000000)  # Set Jerk to effectively infinite
                self.sliders[2].setValue(int(100000000000 / jerk_scale))
                self.sliders[2].setEnabled(False)  # Disable slider
                trapezoidal_button.setText("S-Curve Mode")
                self.default_j_max = 100000000000
                self.trapezoidal_mode = True
            else:
                self.sliders[2].setMaximum(int(10000000000 / jerk_scale))  # Restore original limit
                self.inputs[2].setRange(1, 10000000000)
                self.inputs[2].setValue(self.backup_default_jerk)  # Restore default Jerk
                self.sliders[2].setValue(int(self.backup_default_jerk / jerk_scale))
                self.sliders[2].setEnabled(True)  # Enable slider
                trapezoidal_button.setText("Trapezoidal Mode")
                self.default_j_max = self.backup_default_jerk
                self.trapezoidal_mode = False
            self.update_plot()

        trapezoidal_button.clicked.connect(toggle_mode)

        # Add both buttons to the spacer section
        button_layout = QHBoxLayout()
        button_layout.addWidget(reset_button)
        button_layout.addWidget(trapezoidal_button)
        spacer_section.addLayout(button_layout)


        # Build the right column
        right_column.addLayout(top_spacer_section)  # Blank section at the top
        right_column.addLayout(controls_layout)  # Sliders in the middle
        right_column.addLayout(spacer_section)  # Reset button at the bottom
        right_column.addStretch(1)  # Push everything else down

        # Main layout: Plots on the left, controls on the right
        main_layout.addLayout(plot_layout, 60)  # Left: Plot (takes 70% space)
        main_layout.addLayout(right_column, 40)  # Right: Controls (30% space)

        self.setLayout(main_layout)

        # Initial plot update
        self.update_plot()

    def update_plot(self):
        # Get values from UI
        self.q0 = self.inputs[3].value()
        self.q1 = self.inputs[4].value()
        self.v0 = self.inputs[5].value()
        self.v1 = self.inputs[6].value()
        self.v_max = self.inputs[0].value()
        self.a_max = self.inputs[1].value()
        self.j_max = self.inputs[2].value()

        try:
            # Plan trajectory
            tr = self.p.plan_trajectory(
                [self.q0], [self.q1], [self.v0], [self.v1], self.v_max, self.a_max, self.j_max
            )
            timesteps = 1000
            new_time = np.linspace(0, max(tr.time), timesteps)
            profiles = np.array([tr(t) for t in new_time])

            # Compute Jerk (numerical derivative of acceleration)
            jerk_values = np.gradient(profiles[:, 0, 0], new_time)  # d(Acceleration)/dt

            # Update plots
            self.position_line.set_xdata(new_time)
            self.position_line.set_ydata(profiles[:, 0, 2])  # Position

            self.speed_line.set_xdata(new_time)
            self.speed_line.set_ydata(profiles[:, 0, 1])  # Speed

            self.accel_line.set_xdata(new_time)
            self.accel_line.set_ydata(profiles[:, 0, 0])  # Acceleration

            self.jerk_line.set_xdata(new_time)
            self.jerk_line.set_ydata(jerk_values)  # Jerk

            # Adjust axes
            for ax in [self.ax_position, self.ax_speed, self.ax_accel, self.ax_jerk]:
                ax.relim()
                ax.autoscale_view()

            # Clear warning message (if any)
            self.warning_label.setText("")  

        except Exception as e:
            # Display a warning without crashing
            self.warning_label.setText(f"Warning: {str(e)}")
            self.warning_label.setStyleSheet("color: red; font-weight: bold;")

        # Redraw
        self.canvas.draw()
        
    def showEvent(self, event):
        super().showEvent(event)
        self.resize(self.width() + 1, self.height() + 1)
        self.resize(self.width() - 1, self.height() - 1)

if __name__ == "__main__":
    # Force Matplotlib to use Qt6
    os.environ["QT_API"] = "pyqt6"

    app = QApplication(sys.argv)
    window = TrajectoryPlanner()
    window.show()
    sys.exit(app.exec())