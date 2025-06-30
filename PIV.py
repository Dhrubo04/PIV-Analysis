#!/usr/bin/env python3
"""
PIVlab Python - Digital Particle Image Velocimetry Tool
A Python implementation inspired by PIVlab MATLAB tool
Analyzes two frames to generate vector maps and locate vortices

Features:
- Cross-correlation based PIV analysis
- Sub-pixel accuracy using Gaussian fitting
- Vector field visualization
- Vortex detection and localization
- Streamline plotting
- Comprehensive post-processing filters

Author: Python implementation of PIVlab concepts
Original PIVlab by Dr. William Thielicke and Prof. Dr. Eize J. Stamhuis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import cv2
from scipy import ndimage, signal
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import sys
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class PIVAnalyzer:

    def __init__(self):
        self.version = "1.0"
        self.window_size = 64
        self.overlap = 0.5
        self.search_area = 128
        self.subpixel_method = 'gaussian'
        self.dt = 1.0  # time step between frames
        self.scale = 1.0  # pixels per unit

    def preprocess_image(self, image):
        """Preprocess image for PIV analysis"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(image)

        # Apply Gaussian filter for noise reduction
        image = cv2.GaussianBlur(image, (3, 3), 0.5)

        return image.astype(np.float64)

    def cross_correlation_fft(self, window1, window2):
        """Perform cross-correlation using FFT"""
        # Normalize windows
        window1 = (window1 - np.mean(window1)) / np.std(window1)
        window2 = (window2 - np.mean(window2)) / np.std(window2)

        # Pad windows to avoid edge effects
        pad_size = max(window1.shape[0], window1.shape[1])
        window1_padded = np.zeros((pad_size * 2, pad_size * 2))
        window2_padded = np.zeros((pad_size * 2, pad_size * 2))

        h1, w1 = window1.shape
        h2, w2 = window2.shape

        window1_padded[:h1, :w1] = window1
        window2_padded[:h2, :w2] = window2

        # FFT-based cross-correlation
        f1 = np.fft.fft2(window1_padded)
        f2 = np.fft.fft2(window2_padded)
        correlation = np.fft.ifft2(f1 * np.conj(f2))
        correlation = np.fft.fftshift(np.abs(correlation))

        return correlation

    def gaussian_subpixel(self, correlation, peak_pos):
        """Sub-pixel accuracy using Gaussian fitting"""
        try:
            y, x = peak_pos
            if (y < 1 or y >= correlation.shape[0] - 1 or
                    x < 1 or x >= correlation.shape[1] - 1):
                return peak_pos

            # Extract 3x3 neighborhood around peak
            c = correlation[y-1:y+2, x-1:x+2]

            # Gaussian fitting
            def gaussian_2d(xy, A, x0, y0, sigma_x, sigma_y):
                x_data, y_data = xy
                return A * np.exp(-(((x_data - x0) ** 2) / (2 * sigma_x ** 2) +
                                    ((y_data - y0) ** 2) / (2 * sigma_y ** 2)))

            # Create coordinate grids
            x_grid, y_grid = np.meshgrid(np.arange(3), np.arange(3))

            # Initial guess
            p0 = [np.max(c), 1, 1, 0.5, 0.5]

            # Fit Gaussian
            popt, _ = curve_fit(gaussian_2d, (x_grid.ravel(), y_grid.ravel()),
                                c.ravel(), p0=p0, maxfev=1000)

            # Extract sub-pixel position
            sub_x = x - 1 + popt[1]
            sub_y = y - 1 + popt[2]

            return (sub_y, sub_x)

        except:
            return peak_pos

    def analyze_window_pair(self, args):
        """Analyze a single window pair"""
        window1, window2, i, j = args

        # Cross-correlation
        correlation = self.cross_correlation_fft(window1, window2)

        # Find peak
        peak_pos = np.unravel_index(np.argmax(correlation), correlation.shape)

        # Sub-pixel accuracy
        if self.subpixel_method == 'gaussian':
            peak_pos = self.gaussian_subpixel(correlation, peak_pos)

        # Calculate displacement
        center = (correlation.shape[0] // 2, correlation.shape[1] // 2)
        dy = peak_pos[0] - center[0]
        dx = peak_pos[1] - center[1]

        # Calculate velocity
        u = dx / self.dt * self.scale
        v = dy / self.dt * self.scale

        # Signal-to-noise ratio
        correlation_sorted = np.sort(correlation.ravel())
        snr = correlation_sorted[-1] / correlation_sorted[-2] if len(correlation_sorted) > 1 else 1.0

        return i, j, u, v, snr

    def analyze_frames(self, frame1, frame2):
        """Main PIV analysis function"""
        # Preprocess images
        img1 = self.preprocess_image(frame1)
        img2 = self.preprocess_image(frame2)

        # Calculate grid parameters
        step = int(self.window_size * (1 - self.overlap))

        # Create coordinate grids
        x_coords = np.arange(self.window_size // 2,
                             img1.shape[1] - self.window_size // 2, step)
        y_coords = np.arange(self.window_size // 2,
                             img1.shape[0] - self.window_size // 2, step)

        # Prepare window pairs for analysis
        window_pairs = []
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                # Extract windows
                y1, y2 = int(y - self.window_size // 2), int(y + self.window_size // 2)
                x1, x2 = int(x - self.window_size // 2), int(x + self.window_size // 2)

                if (y2 <= img1.shape[0] and x2 <= img1.shape[1] and
                        y1 >= 0 and x1 >= 0):
                    window1 = img1[y1:y2, x1:x2]
                    window2 = img2[y1:y2, x1:x2]
                    window_pairs.append((window1, window2, i, j))

        # Parallel processing
        try:
            with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
                results = list(executor.map(self.analyze_window_pair, window_pairs))
        except:
            # Fallback to serial processing
            results = [self.analyze_window_pair(pair) for pair in window_pairs]

        # Organize results
        ni, nj = len(y_coords), len(x_coords)
        X, Y = np.meshgrid(x_coords, y_coords)
        U = np.zeros((ni, nj))
        V = np.zeros((ni, nj))
        SNR = np.zeros((ni, nj))

        for i, j, u, v, snr in results:
            if i < ni and j < nj:
                U[i, j] = u
                V[i, j] = v
                SNR[i, j] = snr

        return X, Y, U, V, SNR

class VortexDetector:
    """Vortex detection and analysis"""

    def __init__(self):
        self.vorticity_threshold = 0.1
        self.min_vortex_size = 3

    def calculate_vorticity(self, X, Y, U, V):
        """Calculate vorticity field"""
        # Calculate gradients
        dudx = np.gradient(U, X[0, :], axis=1)
        dudy = np.gradient(U, Y[:, 0], axis=0)
        dvdx = np.gradient(V, X[0, :], axis=1)
        dvdy = np.gradient(V, Y[:, 0], axis=0)

        # Vorticity = dv/dx - du/dy
        vorticity = dvdx - dudy

        return vorticity

    def detect_vortices(self, X, Y, U, V, vorticity):
        """Detect vortex centers"""
        # Apply threshold
        vortex_mask = np.abs(vorticity) > self.vorticity_threshold

        # Find connected components
        labeled, num_features = ndimage.label(vortex_mask)

        vortex_centers = []
        vortex_strengths = []
        vortex_types = []

        for i in range(1, num_features + 1):
            mask = labeled == i
            if np.sum(mask) >= self.min_vortex_size:
                # Find center of mass
                y_center, x_center = ndimage.center_of_mass(np.abs(vorticity) * mask)

                # Interpolate to get exact coordinates
                if (0 <= y_center < X.shape[0] and 0 <= x_center < X.shape[1]):
                    x_pos = X[int(y_center), int(x_center)]
                    y_pos = Y[int(y_center), int(x_center)]

                    # Calculate strength
                    strength = np.mean(vorticity[mask])

                    # Determine type (clockwise or counterclockwise)
                    vortex_type = 'CCW' if strength > 0 else 'CW'

                    vortex_centers.append((x_pos, y_pos))
                    vortex_strengths.append(abs(strength))
                    vortex_types.append(vortex_type)

        return vortex_centers, vortex_strengths, vortex_types

class PIVGUIApp:
    """GUI Application for PIV analysis"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"PIV Dhrubo v{PIVAnalyzer().version}")
        self.root.geometry("1200x800")

        self.piv_analyzer = PIVAnalyzer()
        self.vortex_detector = VortexDetector()

        self.frame1 = None
        self.frame2 = None
        self.results = None

        self.setup_gui()

    def setup_gui(self):
        """Setup the GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side='left', fill='y', padx=(0, 10))

        # Load images
        ttk.Label(control_frame, text="Load Images:").pack(anchor='w')
        ttk.Button(control_frame, text="Load Frame 1",
                   command=self.load_frame1).pack(fill='x', pady=2)
        ttk.Button(control_frame, text="Load Frame 2",
                   command=self.load_frame2).pack(fill='x', pady=2)

        # Analysis parameters
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(control_frame, text="PIV Parameters:").pack(anchor='w')

        # Window size
        ttk.Label(control_frame, text="Window Size:").pack(anchor='w')
        self.window_size_var = tk.StringVar(value="64")
        window_size_combo = ttk.Combobox(control_frame, textvariable=self.window_size_var,
                                         values=["8","16", "32", "64", "128"], state="readonly")
        window_size_combo.pack(fill='x', pady=2)

        # Overlap
        ttk.Label(control_frame, text="Overlap:").pack(anchor='w')
        self.overlap_var = tk.DoubleVar(value=0.5)
        overlap_scale = ttk.Scale(control_frame, from_=0.0, to=0.8,
                                  variable=self.overlap_var, orient='horizontal')
        overlap_scale.pack(fill='x', pady=2)

        # Time step
        ttk.Label(control_frame, text="Time Step (dt):").pack(anchor='w')
        self.dt_var = tk.DoubleVar(value=1.0)
        dt_entry = ttk.Entry(control_frame, textvariable=self.dt_var)
        dt_entry.pack(fill='x', pady=2)

        # Scale
        ttk.Label(control_frame, text="Scale (px/unit):").pack(anchor='w')
        self.scale_var = tk.DoubleVar(value=1.0)
        scale_entry = ttk.Entry(control_frame, textvariable=self.scale_var)
        scale_entry.pack(fill='x', pady=2)

        # Vortex detection parameters
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(control_frame, text="Vortex Detection:").pack(anchor='w')

        ttk.Label(control_frame, text="Vorticity Threshold:").pack(anchor='w')
        self.vorticity_threshold_var = tk.DoubleVar(value=0.1)
        vorticity_entry = ttk.Entry(control_frame, textvariable=self.vorticity_threshold_var)
        vorticity_entry.pack(fill='x', pady=2)

        # Analysis button
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Button(control_frame, text="Analyze PIV",
                   command=self.analyze_piv).pack(fill='x', pady=5)

        # Visualization options
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(control_frame, text="Visualization:").pack(anchor='w')
        ttk.Button(control_frame, text="Show Vector Field",
                   command=self.show_vector_field).pack(fill='x', pady=2)
        ttk.Button(control_frame, text="Show Vorticity",
                   command=self.show_vorticity).pack(fill='x', pady=2)
        ttk.Button(control_frame, text="Show Streamlines",
                   command=self.show_streamlines).pack(fill='x', pady=2)

        # Export
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Button(control_frame, text="Export Results",
                   command=self.export_results).pack(fill='x', pady=2)

        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.pack(side='bottom', fill='x', pady=5)

        # Display area
        self.display_frame = ttk.Frame(main_frame)
        self.display_frame.pack(side='right', fill='both', expand=True)

        # Initialize matplotlib figure
        self.setup_matplotlib()

    def setup_matplotlib(self):
        """Setup matplotlib display"""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.fig = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, self.display_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def load_frame1(self):
        """Load first frame"""
        filename = filedialog.askopenfilename(
            title="Select Frame 1",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp")]
        )
        if filename:
            self.frame1 = cv2.imread(filename)
            self.status_var.set("Frame 1 loaded")
            self.display_frame_preview()

    def load_frame2(self):
        """Load second frame"""
        filename = filedialog.askopenfilename(
            title="Select Frame 2",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp")]
        )
        if filename:
            self.frame2 = cv2.imread(filename)
            self.status_var.set("Frame 2 loaded")
            self.display_frame_preview()

    def display_frame_preview(self):
        """Display frame preview"""
        if self.frame1 is not None and self.frame2 is not None:
            self.fig.clear()

            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)

            # Convert BGR to RGB for display
            frame1_rgb = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)
            frame2_rgb = cv2.cvtColor(self.frame2, cv2.COLOR_BGR2RGB)

            ax1.imshow(frame1_rgb)
            ax1.set_title("Frame 1")
            ax1.axis('off')

            ax2.imshow(frame2_rgb)
            ax2.set_title("Frame 2")
            ax2.axis('off')

            self.fig.tight_layout()
            self.canvas.draw()

    def analyze_piv(self):
        """Perform PIV analysis"""
        if self.frame1 is None or self.frame2 is None:
            messagebox.showerror("Error", "Please load both frames first")
            return

        # Update analyzer parameters
        self.piv_analyzer.window_size = int(self.window_size_var.get())
        self.piv_analyzer.overlap = self.overlap_var.get()
        self.piv_analyzer.dt = self.dt_var.get()
        self.piv_analyzer.scale = self.scale_var.get()

        self.vortex_detector.vorticity_threshold = self.vorticity_threshold_var.get()

        self.status_var.set("Analyzing...")
        self.root.update()

        try:
            # Perform PIV analysis
            X, Y, U, V, SNR = self.piv_analyzer.analyze_frames(self.frame1, self.frame2)

            # Calculate vorticity
            vorticity = self.vortex_detector.calculate_vorticity(X, Y, U, V)

            # Detect vortices
            vortex_centers, vortex_strengths, vortex_types = \
                self.vortex_detector.detect_vortices(X, Y, U, V, vorticity)

            # Store results
            self.results = {
                'X': X, 'Y': Y, 'U': U, 'V': V, 'SNR': SNR,
                'vorticity': vorticity,
                'vortex_centers': vortex_centers,
                'vortex_strengths': vortex_strengths,
                'vortex_types': vortex_types
            }

            self.status_var.set(f"Analysis complete. Found {len(vortex_centers)} vortices")
            self.show_vector_field()

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.status_var.set("Analysis failed")

    def show_vector_field(self):
        """Display vector field with vortex locations overlayed on original image"""
        if self.results is None:
            messagebox.showwarning("Warning", "No results to display")
            return

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Convert frame1 to RGB for matplotlib
        frame1_rgb = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)
        ax.imshow(frame1_rgb)

        # Extract results
        X, Y, U, V = self.results['X'], self.results['Y'], self.results['U'], self.results['V']
        vortex_centers = self.results['vortex_centers']
        vortex_types = self.results['vortex_types']
        vortex_strengths = self.results['vortex_strengths']

        # Subsample for cleaner display
        skip = max(1, len(X[0]) // 20)
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  U[::skip, ::skip], V[::skip, ::skip],
                  scale=50, color='yellow', alpha=0.9)

        # Plot vortex centers
        for i, (x, y) in enumerate(vortex_centers):
            color = 'red' if vortex_types[i] == 'CW' else 'green'
            strength = vortex_strengths[i]
            circle = Circle((x, y), radius=strength*20,
                            fill=False, color=color, linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, vortex_types[i],
                    ha='center', va='center', fontsize=8, color=color)

        # Calculate average flow velocity
        mag = np.sqrt(U**2 + V**2)
        mean_velocity = np.mean(mag)
        self.status_var.set(f"Vector field shown. Mean velocity: {mean_velocity:.2f} units/s")

        ax.set_title('Vector Field on Original Frame')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_aspect('equal')
        ax.invert_yaxis()

        self.fig.tight_layout()
        self.canvas.draw()


    def show_vorticity(self):
        """Display vorticity field"""
        if self.results is None:
            messagebox.showwarning("Warning", "No results to display")
            return

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        X, Y = self.results['X'], self.results['Y']
        vorticity = self.results['vorticity']

        # Plot vorticity
        im = ax.contourf(X, Y, vorticity, levels=20, cmap='RdBu_r')
        self.fig.colorbar(im, ax=ax, label='Vorticity')

        # Plot vortex centers
        for i, (x, y) in enumerate(self.results['vortex_centers']):
            ax.plot(x, y, 'ko', markersize=8)
            ax.text(x, y+20, self.results['vortex_types'][i],
                    ha='center', va='bottom', fontsize=10, color='black')

        ax.set_title('Vorticity Field')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_aspect('equal')
        ax.invert_yaxis()

        self.fig.tight_layout()
        self.canvas.draw()

    def show_streamlines(self):
        """Display streamlines overlayed on original image"""
        if self.results is None:
            messagebox.showwarning("Warning", "No results to display")
            return
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Show original image (frame 1) as background
        frame1_rgb = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)
        ax.imshow(frame1_rgb)

        # Get results
        X, Y, U, V = self.results['X'], self.results['Y'], self.results['U'], self.results['V']
        vortex_centers = self.results['vortex_centers']
        vortex_types = self.results['vortex_types']

        # Streamplot (no subsampling needed here)
        ax.streamplot(X, Y, U, V, density=2, color='cyan', linewidth=1.5, arrowsize=1)

        # Mark vortex centers
        for i, (x, y) in enumerate(vortex_centers):
            color = 'red' if vortex_types[i] == 'CW' else 'green'
            ax.plot(x, y, 'o', color=color, markersize=10)
            ax.text(x, y+20, vortex_types[i],
                    ha='center', va='bottom', fontsize=10, color=color)

        ax.set_title('Streamlines on Original Frame')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_aspect('equal')
        ax.invert_yaxis()

        self.fig.tight_layout()
        self.canvas.draw()


    def export_results(self):
        """Export analysis results"""
        if self.results is None:
            messagebox.showwarning("Warning", "No results to export")
            return

        filename = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".npz",
            filetypes=[("NumPy files", "*.npz"), ("CSV files", "*.csv")]
        )

        if filename:
            if filename.endswith('.npz'):
                np.savez(filename, **self.results)
            elif filename.endswith('.csv'):
                # Export vector field data
                X, Y, U, V = self.results['X'], self.results['Y'], self.results['U'], self.results['V']
                data = np.column_stack([X.ravel(), Y.ravel(), U.ravel(), V.ravel()])
                np.savetxt(filename, data, delimiter=',',
                           header='X,Y,U,V', comments='')

            self.status_var.set("Results exported")

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

def main():
    """Main entry point"""
    print("=" * 60)
    print("PIVlab Python - Digital Particle Image Velocimetry Tool")
    print("=" * 60)
    print("Version: 1.0")
    print("Features:")
    print("- Cross-correlation based PIV analysis")
    print("- Sub-pixel accuracy with Gaussian fitting")
    print("- Automatic vortex detection and classification")
    print("- Vector field visualization")
    print("- Streamline plotting")
    print("- Multi-core processing support")
    print("=" * 60)

    # Check dependencies
    try:
        import cv2
        import scipy
        import matplotlib
        print("✓ All dependencies found")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install opencv-python scipy matplotlib pillow")
        sys.exit(1)

    # Check multiprocessing support
    try:
        cores = mp.cpu_count()
        print(f"✓ Multi-core processing available ({cores} cores)")
    except:
        print("✗ Multi-core processing not available")

    print("=" * 60)
    print("Starting GUI...")

    # Start GUI application
    app = PIVGUIApp()
    app.run()

if __name__ == "__main__":
    main()
