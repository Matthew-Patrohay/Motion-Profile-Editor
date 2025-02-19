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
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter


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
        self.ax_jerk.set_xlabel("milliseconds")

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
            
            # Add light gray grid lines
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="lightgray", alpha=0.5)

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet(f"background-color: {bg_color};")

        # Initial plot setup
        self.time = np.linspace(0, 1, 100)
        self.position_line, = self.ax_position.plot(self.time, np.zeros_like(self.time), label="Position", color='#bf5af2')
        self.speed_line, = self.ax_speed.plot(self.time, np.zeros_like(self.time), label="Velocity", color='#0b84ff')
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
                if i == 2:
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

        # Label to display calculated statistics
        self.statistics_label = QLabel("Trajectory Statistics:\n")
        self.statistics_label.setStyleSheet(f"""
            color: {text_color}; 
            font-size: 14px; 
            font-family: "Menlo", Menlo, monospace;
        """)
        
        # Add both buttons to the spacer section
        button_layout = QHBoxLayout()
        button_layout.addWidget(reset_button)
        button_layout.addWidget(trapezoidal_button)
        spacer_section.addLayout(button_layout)


        # Build the right column
        right_column.addLayout(top_spacer_section)  # Blank section at the top
        right_column.addLayout(controls_layout)  # Sliders in the middle
        right_column.addLayout(spacer_section)  # Reset button at the bottom
        right_column.addWidget(self.statistics_label)  # Add to layout
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


            # Define important y-tick positions
            position_ticks = [self.q0, self.q1]  # Min/Max position
            velocity_ticks = [0, self.v_max] if (self.q0 - self.q1) < 0 else [-self.v_max, 0] # Min/Max velocity
            acceleration_ticks = [self.a_max, -self.a_max]  # Min/Max acceleration
            jerk_ticks = [self.j_max, -self.j_max]  # Min/Max jerk

            # Function to add padding
            def add_padding(tick_values, padding_factor=0.15):
                """Adds padding to the y-axis limits while preserving auto ticks."""
                min_val, max_val = min(tick_values), max(tick_values)
                padding = (max_val - min_val) * padding_factor  # 10% extra space
                return min_val - padding, max_val + padding

            for ax, manual_ticks in zip(
                [self.ax_position, self.ax_speed, self.ax_accel, self.ax_jerk],
                [position_ticks, velocity_ticks, acceleration_ticks, jerk_ticks]
            ):
                # Remove previously added grid lines if they exist
                if hasattr(ax, "custom_grid_lines"):
                    for line in ax.custom_grid_lines:
                        line.remove()  # Properly remove the old grid lines
                ax.custom_grid_lines = []  # Reset the list

                # Disable default grid
                ax.grid(False)

                min_val, max_val = min(manual_ticks), max(manual_ticks)
                computed_ticks = np.linspace(min_val, max_val, num=6)

                ax.set_yticks(computed_ticks)
                ax.set_ylim(add_padding(computed_ticks))

                # Ensure minor ticks are used but not excessive
                # ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))

                # Draw new manual grid lines
                for tick in computed_ticks:
                    line_alpha = 0.9 if tick in manual_ticks else 0.3  # Min/max values are stronger
                    line_color = "lightgray" if tick in manual_ticks else "darkgray"
                    line_thickness = 0.5 if tick in manual_ticks else 0.3
                    line = ax.axhline(y=tick, linestyle="--", linewidth=line_thickness, color=line_color, alpha=line_alpha)
                    ax.custom_grid_lines.append(line)  # Store reference for future removal

            # Compute statistics
            estimationErrorPercentage = 0.001
            total_time = max(tr.time)
            time_step = new_time[1] - new_time[0]
            max_velocity_time = np.sum(profiles[:, 0, 1] >= self.v_max - (estimationErrorPercentage * self.v_max)) * time_step
            max_acceleration_time = np.sum(profiles[:, 0, 0] >= self.a_max - (estimationErrorPercentage * self.a_max)) * time_step
            min_acceleration_time = np.sum(profiles[:, 0, 0] <= -self.a_max + (estimationErrorPercentage * self.a_max)) * time_step
            max_jerk_time = np.sum(jerk_values >= self.j_max - (estimationErrorPercentage * self.j_max)) * time_step
            min_jerk_time = np.sum(jerk_values <= -self.j_max + (estimationErrorPercentage * self.j_max)) * time_step
            
            # Compute percentages
            max_velocity_percent = (max_velocity_time / total_time) * 100
            max_acceleration_percent = (max_acceleration_time / total_time) * 100
            min_acceleration_percent = (min_acceleration_time / total_time) * 100
            max_jerk_percent = (max_jerk_time / total_time) * 100
            min_jerk_percent = (min_jerk_time / total_time) * 100
            
            
            
            
            
            def nice_ticks(data_max, num_ticks=18):
                """Compute a 'nice' tick interval ensuring total_time is included, while preventing excessive tick density."""
                
                raw_interval = data_max / (num_ticks - 1)
                
                # Determine base spacing
                if raw_interval <= 0.5:
                    nice_interval = 0.5  # 0.5ms spacing
                elif raw_interval <= 1:
                    nice_interval = 1  # 1ms spacing
                elif raw_interval <= 2:
                    nice_interval = 2  # 2ms spacing
                elif raw_interval <= 5:
                    nice_interval = 5  # 5ms spacing
                else:
                    nice_interval = round(raw_interval / 10) * 10  # Default rounding to nearest 10ms

                # Generate tick values
                ticks = np.arange(0, data_max, nice_interval)

                # # Reduce tick density further if more than 30 ticks are generated
                # if len(ticks) > 30:
                #     nice_interval = 5 if nice_interval == 1 else nice_interval  # Switch to 5ms if we were using 1ms
                #     ticks = np.arange(0, data_max, nice_interval)

                # Ensure total_time is explicitly included
                if total_time * 1000 not in ticks:
                    ticks = np.append(ticks, total_time * 1000)

                return np.unique(ticks)  # Ensure unique values


            # Compute x-ticks in milliseconds and convert to seconds
            x_ticks = nice_ticks(total_time * 1000) / 1000  

            # Define a function to format x-ticks as milliseconds
            def format_ms(x, _):
                """Format x-ticks: Hide only the tick right before total_time, 2 decimals for total_time, 1 decimal for others."""
                sorted_ticks = np.sort(x_ticks)  # Ensure sorted ticks
                if len(sorted_ticks) > 1:
                    tick_before_total_time = sorted_ticks[np.where(sorted_ticks < total_time)[0][-1]]  # Get the closest tick before total_time
                else:
                    tick_before_total_time = None  # Fallback

                if np.isclose(x, total_time, atol=1e-6):  # Highlight total_time with 2 decimals
                    return f"{x * 1000:.2f}"
                elif tick_before_total_time is not None and np.isclose(x, tick_before_total_time, atol=1e-6):
                    return ""  # Hide only the tick immediately before total_time
                return f"{x * 1000:.1f}"  # 1 decimal for other ticks
            
            
            # Apply new x-ticks and formatting
            for ax in [self.ax_position, self.ax_speed, self.ax_accel, self.ax_jerk]:
                if hasattr(ax, "custom_x_grid_lines"):
                    for line in ax.custom_x_grid_lines:
                        line.remove()
                ax.custom_x_grid_lines = []  # Reset the list

                ax.set_xticks(x_ticks)
                ax.xaxis.set_major_formatter(FuncFormatter(format_ms))

                # Draw new manual grid lines for x-axis
                for tick in x_ticks:
                    line_alpha = 0.9 if tick == total_time else 0.3  # Highlight final tick
                    line_color = "lightgray" if tick == total_time else "darkgray"
                    line_thickness = 0.5 if tick == total_time else 0.3
                    line = ax.axvline(x=tick, linestyle="--", linewidth=line_thickness, color=line_color, alpha=line_alpha)
                    ax.custom_x_grid_lines.append(line)
                    
        
        
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
                
            leftAlignPadding = 18;

            self.statistics_label.setText(f"""
                {"Total Time:".ljust(leftAlignPadding)} {total_time * 1000:>6.2f} ms (100.00%)
                
                {"Time at Max Vel:".ljust(leftAlignPadding)} {max_velocity_time * 1000:>6.2f} ms ({max_velocity_percent:>4.2f}%)
                {"Time at Max Accel:".ljust(leftAlignPadding)} {max_acceleration_time * 1000:>6.2f} ms ({max_acceleration_percent:>4.2f}%)
                {"Time at Max Accel:".ljust(leftAlignPadding)} {min_acceleration_time * 1000:>6.2f} ms ({min_acceleration_percent:>4.2f}%)
                {"Time at Max Jerk:".ljust(leftAlignPadding)} {max_jerk_time * 1000:>6.2f} ms ({max_jerk_percent:>4.2f}%)
                {"Time at Min Jerk:".ljust(leftAlignPadding)} {min_jerk_time * 1000:>6.2f} ms ({min_jerk_percent:>4.2f}%)
            """)
                            
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