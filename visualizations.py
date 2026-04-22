"""
Enhanced Visualization Module for Mathematical Expert System
visualizations.py

Professional visualizations with animations and real-life examples
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Arc
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
import numpy as np
import math

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Color scheme
COLORS = {
    'primary': '#3498db',
    'secondary': '#e74c3c',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'purple': '#9b59b6',
    'dark': '#2c3e50',
    'light': '#ecf0f1'
}


def setup_plot_style(ax, title, xlabel='x', ylabel='y'):
    """Apply consistent styling to plots"""
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.3)
    ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.3)


def format_polynomial_label(terms):
    """Format polynomial for labels"""
    if not terms:
        return "0"
    
    result = []
    for i, term in enumerate(terms):
        coef, power = term['coef'], term['power']
        
        if abs(coef) < 0.0001:
            continue
        
        if power == 0:
            result.append(f"{coef:.2g}" if i == 0 else f"{'+' if coef > 0 else ''}{coef:.2g}")
        elif power == 1:
            if abs(coef) == 1:
                result.append("x" if coef > 0 else "-x")
            else:
                result.append(f"{coef:.2g}x" if i == 0 else f"{'+' if coef > 0 else ''}{coef:.2g}x")
        else:
            if abs(coef) == 1:
                result.append(f"x^{int(power)}" if coef > 0 else f"-x^{int(power)}")
            else:
                result.append(f"{coef:.2g}x^{int(power)}" if i == 0 else f"{'+' if coef > 0 else ''}{coef:.2g}x^{int(power)}")
    
    return "".join(result) if result else "0"


def visualize_derivative(power=3):
    """Basic derivative visualization"""
    x = np.linspace(-3, 3, 1000)
    y = x ** power
    dy = power * x ** (power - 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Derivative Visualization', fontsize=16, fontweight='bold')
    
    # Original function
    ax1.plot(x, y, color=COLORS['primary'], linewidth=2.5, label=f'f(x) = x^{power}')
    setup_plot_style(ax1, 'Original Function', 'x', 'f(x)')
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.fill_between(x, y, alpha=0.2, color=COLORS['primary'])
    
    # Derivative
    ax2.plot(x, dy, color=COLORS['secondary'], linewidth=2.5, label=f"f'(x) = {power}x^{power-1}")
    setup_plot_style(ax2, 'Derivative (Rate of Change)', 'x', "f'(x)")
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.fill_between(x, dy, alpha=0.2, color=COLORS['secondary'])
    
    plt.tight_layout()
    plt.show()


def visualize_polynomial_derivative(original_terms, derivative_terms):
    """Enhanced polynomial derivative visualization with tangent lines"""
    x = np.linspace(-5, 5, 1000)
    
    # Calculate functions
    y_orig = np.zeros_like(x)
    for term in original_terms:
        y_orig += term['coef'] * (x ** term['power'])
    
    y_deriv = np.zeros_like(x)
    for term in derivative_terms:
        y_deriv += term['coef'] * (x ** term['power'])
    
    # Format labels
    orig_label = format_polynomial_label(original_terms)
    deriv_label = format_polynomial_label(derivative_terms)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Main plot - Original function
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x, y_orig, color=COLORS['primary'], linewidth=3, label=f'f(x) = {orig_label}', zorder=3)
    
    # Add tangent lines at key points
    tangent_points = [-2, 0, 2]
    for xp in tangent_points:
        # Calculate y and slope at point
        yp = sum(term['coef'] * (xp ** term['power']) for term in original_terms)
        slope = sum(term['coef'] * term['power'] * (xp ** (term['power']-1)) for term in original_terms if term['power'] > 0)
        
        # Tangent line
        x_tan = np.linspace(xp - 1.5, xp + 1.5, 100)
        y_tan = slope * (x_tan - xp) + yp
        
        ax1.plot(x_tan, y_tan, '--', color=COLORS['warning'], linewidth=1.5, alpha=0.7)
        ax1.plot(xp, yp, 'o', color=COLORS['secondary'], markersize=10, zorder=4)
        ax1.annotate(f'slope={slope:.2f}', xy=(xp, yp), xytext=(xp+0.5, yp+2),
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['warning'], alpha=0.7))
    
    setup_plot_style(ax1, f'Original Function: f(x) = {orig_label}')
    ax1.legend(fontsize=11, loc='best')
    ax1.fill_between(x, y_orig, alpha=0.15, color=COLORS['primary'])
    
    # Derivative plot
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x, y_deriv, color=COLORS['secondary'], linewidth=3, label=f"f'(x) = {deriv_label}")
    
    # Highlight zeros of derivative (critical points)
    for i in range(1, len(x)):
        if y_deriv[i-1] * y_deriv[i] < 0:  # Sign change
            ax2.plot(x[i], 0, 'o', color=COLORS['success'], markersize=12, zorder=4)
            ax2.annotate('Critical Point', xy=(x[i], 0), xytext=(x[i], -2),
                        fontsize=8, ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['success'], alpha=0.7))
    
    setup_plot_style(ax2, f"Derivative: f'(x) = {deriv_label}")
    ax2.legend(fontsize=11)
    ax2.fill_between(x, y_deriv, where=(y_deriv > 0), alpha=0.3, color=COLORS['success'], label='Increasing')
    ax2.fill_between(x, y_deriv, where=(y_deriv < 0), alpha=0.3, color=COLORS['secondary'], label='Decreasing')
    ax2.legend(fontsize=9)
    
    # Real-life application
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Simulate velocity and acceleration
    time = np.linspace(0, 10, 100)
    position = 0.5 * time ** 2  # s = 0.5at²
    velocity = time  # v = at
    
    ax3.plot(time, position, color=COLORS['primary'], linewidth=2, label='Position (m)', marker='o', markersize=3)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(time, velocity, color=COLORS['secondary'], linewidth=2, label='Velocity (m/s)', linestyle='--')
    
    ax3.set_title('Real-Life: Motion Analysis', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Position (m)', fontsize=11, fontweight='bold', color=COLORS['primary'])
    ax3_twin.set_ylabel('Velocity (m/s)', fontsize=11, fontweight='bold', color=COLORS['secondary'])
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', fontsize=9)
    ax3_twin.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('📊 Comprehensive Derivative Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.show()


def visualize_integration():
    """Basic integration visualization"""
    x = np.linspace(0, 5, 1000)
    y = x ** 2
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(x, y, color=COLORS['primary'], linewidth=3, label='f(x) = x²')
    
    # Fill area
    x_fill = np.linspace(1, 4, 100)
    y_fill = x_fill ** 2
    ax.fill_between(x_fill, y_fill, alpha=0.4, color=COLORS['success'], label='∫f(x)dx (Area)')
    
    # Add rectangles to show Riemann sum concept
    n_rects = 8
    dx = 3 / n_rects
    for i in range(n_rects):
        x_rect = 1 + i * dx
        y_rect = x_rect ** 2
        rect = plt.Rectangle((x_rect, 0), dx, y_rect, 
                            edgecolor=COLORS['dark'], facecolor=COLORS['warning'], 
                            alpha=0.3, linewidth=1)
        ax.add_patch(rect)
    
    setup_plot_style(ax, 'Integration: Area Under Curve', 'x', 'f(x)')
    ax.legend(fontsize=12)
    ax.text(2.5, 10, 'Riemann Sum\nApproximation', fontsize=11, ha='center',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def visualize_integration_enhanced(terms):
    """Enhanced integration with real-life context"""
    x = np.linspace(0, 5, 1000)
    
    # Calculate function
    y = np.zeros_like(x)
    for term in terms:
        y += term['coef'] * (x ** term['power'])
    
    # Calculate integral (add one to power, divide by new power)
    y_integral = np.zeros_like(x)
    for term in terms:
        new_power = term['power'] + 1
        y_integral += (term['coef'] / new_power) * (x ** new_power)
    
    func_label = format_polynomial_label(terms)
    
    fig = plt.figure(figsize=(16, 6))
    
    # Area under curve
    ax1 = plt.subplot(131)
    ax1.plot(x, y, color=COLORS['primary'], linewidth=3, label=f'f(x) = {func_label}')
    
    x_fill = x[(x >= 1) & (x <= 4)]
    y_fill = np.zeros_like(x_fill)
    for term in terms:
        y_fill += term['coef'] * (x_fill ** term['power'])
    
    ax1.fill_between(x_fill, y_fill, alpha=0.4, color=COLORS['success'])
    ax1.axvline(1, color=COLORS['secondary'], linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(4, color=COLORS['secondary'], linestyle='--', linewidth=2, alpha=0.7)
    
    area = sum((term['coef'] / (term['power'] + 1)) * (4**(term['power']+1) - 1**(term['power']+1)) for term in terms)
    ax1.text(2.5, max(y_fill)/2, f'Area ≈ {area:.2f}', fontsize=12, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    setup_plot_style(ax1, f'Area Under f(x) = {func_label}')
    ax1.legend()
    
    # Accumulation function
    ax2 = plt.subplot(132)
    ax2.plot(x, y_integral, color=COLORS['purple'], linewidth=3, label='F(x) = ∫f(x)dx')
    ax2.fill_between(x, y_integral, alpha=0.2, color=COLORS['purple'])
    setup_plot_style(ax2, 'Accumulation Function')
    ax2.legend()
    
    # Real-life: Distance from velocity
    ax3 = plt.subplot(133)
    time = np.linspace(0, 10, 100)
    velocity = 5 + 2 * time  # v = 5 + 2t
    distance = 5 * time + time**2  # s = 5t + t²
    
    ax3.fill_between(time, velocity, alpha=0.3, color=COLORS['info'], label='Velocity (area)')
    ax3.plot(time, velocity, color=COLORS['primary'], linewidth=2, label='Velocity (m/s)')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(time, distance, color=COLORS['secondary'], linewidth=2, label='Distance (m)', linestyle='--')
    
    ax3.set_title('Real-Life: Velocity → Distance', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (s)', fontweight='bold')
    ax3.set_ylabel('Velocity (m/s)', fontweight='bold', color=COLORS['primary'])
    ax3_twin.set_ylabel('Distance (m)', fontweight='bold', color=COLORS['secondary'])
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', fontsize=9)
    ax3_twin.legend(loc='lower right', fontsize=9)
    
    plt.suptitle('📊 Comprehensive Integration Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_dot_product_enhanced(v1, v2):
    """Enhanced dot product with geometric interpretation"""
    if len(v1) == 2 and len(v2) == 2:
        fig = plt.figure(figsize=(16, 6))
        
        # Vector plot
        ax1 = plt.subplot(131)
        ax1.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
                  color=COLORS['primary'], width=0.008, label=f'v₁ = {v1}', zorder=3)
        ax1.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
                  color=COLORS['secondary'], width=0.008, label=f'v₂ = {v2}', zorder=3)
        
        # Angle between vectors
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        arc = Arc((0, 0), 1, 1, angle=0, theta1=0, 
                 theta2=np.degrees(angle), color='green', linewidth=2)
        ax1.add_patch(arc)
        ax1.text(0.5, 0.3, f'θ={np.degrees(angle):.1f}°', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        dot = np.dot(v1, v2)
        max_val = max(abs(v1[0]), abs(v1[1]), abs(v2[0]), abs(v2[1])) + 1
        ax1.set_xlim(-1, max_val)
        ax1.set_ylim(-1, max_val)
        setup_plot_style(ax1, f'Dot Product = {dot:.2f}')
        ax1.legend(fontsize=10)
        ax1.set_aspect('equal')
        
        # Projection visualization
        ax2 = plt.subplot(132)
        # Project v2 onto v1
        proj = (np.dot(v1, v2) / np.dot(v1, v1)) * np.array(v1)
        
        ax2.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
                  color=COLORS['primary'], width=0.008, label='v₁', zorder=3)
        ax2.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
                  color=COLORS['secondary'], width=0.008, label='v₂', zorder=3)
        ax2.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1,
                  color=COLORS['success'], width=0.008, label='Projection', zorder=3)
        
        # Draw projection lines
        ax2.plot([v2[0], proj[0]], [v2[1], proj[1]], 'k--', linewidth=1, alpha=0.5)
        
        ax2.set_xlim(-1, max_val)
        ax2.set_ylim(-1, max_val)
        setup_plot_style(ax2, 'Geometric Interpretation')
        ax2.legend(fontsize=10)
        ax2.set_aspect('equal')
        
        # Real-life: Work calculation
        ax3 = plt.subplot(133)
        
        force = np.array([10, 0])
        displacement = np.array([5, 3])
        work = np.dot(force, displacement)
        
        ax3.quiver(0, 0, force[0], force[1], angles='xy', scale_units='xy', scale=1,
                  color='red', width=0.01, label=f'Force = {force} N', zorder=3)
        ax3.quiver(0, 0, displacement[0], displacement[1], angles='xy', scale_units='xy', scale=1,
                  color='blue', width=0.01, label=f'Displacement = {displacement} m', zorder=3)
        
        ax3.set_xlim(-1, 12)
        ax3.set_ylim(-1, 5)
        ax3.set_title(f'Work Done = F·d = {work:.1f} Joules', fontsize=12, fontweight='bold')
        ax3.set_xlabel('x', fontweight='bold')
        ax3.set_ylabel('y', fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        ax3.text(6, 4, 'Work = Force · Displacement\nOnly parallel component counts!',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.suptitle('📊 Dot Product: Complete Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    elif len(v1) == 3 and len(v2) == 3:
        fig = plt.figure(figsize=(15, 5))
        
        # 3D vector plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.quiver(0, 0, 0, v1[0], v1[1], v1[2], color=COLORS['primary'],
                  arrow_length_ratio=0.1, linewidth=3, label=f'v₁ = {v1}')
        ax1.quiver(0, 0, 0, v2[0], v2[1], v2[2], color=COLORS['secondary'],
                  arrow_length_ratio=0.1, linewidth=3, label=f'v₂ = {v2}')
        
        dot = np.dot(v1, v2)
        ax1.set_title(f'3D Dot Product = {dot:.2f}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X', fontweight='bold')
        ax1.set_ylabel('Y', fontweight='bold')
        ax1.set_zlabel('Z', fontweight='bold')
        ax1.legend()
        
        # Component-wise multiplication
        ax2 = fig.add_subplot(122)
        components = [v1[i] * v2[i] for i in range(3)]
        bars = ax2.bar(['X', 'Y', 'Z'], components, color=[COLORS['primary'], COLORS['secondary'], COLORS['success']])
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_title('Component-wise Products', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Value', fontweight='bold')
        
        for bar, val in zip(bars, components):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom' if height > 0 else 'top')
        
        ax2.text(1, max(components)*0.8, f'Sum = {dot:.2f}',
                fontsize=12, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        plt.suptitle('📊 3D Dot Product Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


def visualize_cross_product_enhanced(v1, v2):
    """Enhanced cross product with applications"""
    if len(v1) != 3 or len(v2) != 3:
        print("Cross product requires 3D vectors")
        return
    
    cross = np.cross(v1, v2)
    
    fig = plt.figure(figsize=(16, 6))
    
    # 3D visualization
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot vectors
    ax1.quiver(0, 0, 0, v1[0], v1[1], v1[2], color=COLORS['primary'],
              arrow_length_ratio=0.15, linewidth=4, label=f'v₁ = {v1}')
    ax1.quiver(0, 0, 0, v2[0], v2[1], v2[2], color=COLORS['secondary'],
              arrow_length_ratio=0.15, linewidth=4, label=f'v₂ = {v2}')
    ax1.quiver(0, 0, 0, cross[0], cross[1], cross[2], color=COLORS['success'],
              arrow_length_ratio=0.15, linewidth=4, label=f'v₁×v₂ = [{cross[0]:.2f}, {cross[1]:.2f}, {cross[2]:.2f}]')
    
    # Draw plane spanned by v1 and v2
    if np.linalg.norm(cross) > 0.01:
        scale = 2
        xx, yy = np.meshgrid(np.linspace(-scale, scale, 10), np.linspace(-scale, scale, 10))
        d = 0  # plane through origin
        zz = (-cross[0] * xx - cross[1] * yy - d) / (cross[2] + 0.001)
        ax1.plot_surface(xx, yy, zz, alpha=0.2, color=COLORS['warning'])
    
    ax1.set_xlabel('X', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Y', fontweight='bold', fontsize=11)
    ax1.set_zlabel('Z', fontweight='bold', fontsize=11)
    ax1.set_title('Cross Product: Perpendicular Vector', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    
    # Magnitude comparison
    ax2 = fig.add_subplot(132)
    magnitudes = {
        '|v₁|': np.linalg.norm(v1),
        '|v₂|': np.linalg.norm(v2),
        '|v₁×v₂|': np.linalg.norm(cross)
    }
    bars = ax2.bar(magnitudes.keys(), magnitudes.values(),
                   color=[COLORS['primary'], COLORS['secondary'], COLORS['success']])
    ax2.set_title('Vector Magnitudes', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Magnitude', fontweight='bold')
    
    for bar, (name, val) in zip(bars, magnitudes.items()):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Area of parallelogram
    area = np.linalg.norm(cross)
    ax2.text(1, max(magnitudes.values())*0.7,
            f'Area of parallelogram\nformed by v₁ and v₂:\n{area:.2f} units²',
            ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Real-life: Torque
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Position and force for torque calculation
    r = np.array([2, 0, 0])  # position vector (wrench)
    F = np.array([0, 3, 0])  # force vector
    torque = np.cross(r, F)
    
    ax3.quiver(0, 0, 0, r[0], r[1], r[2], color='brown',
              arrow_length_ratio=0.15, linewidth=3, label=f'Position r = {r} m')
    ax3.quiver(r[0], r[1], r[2], F[0], F[1], F[2], color='red',
              arrow_length_ratio=0.15, linewidth=3, label=f'Force F = {F} N')
    ax3.quiver(0, 0, 0, torque[0], torque[1], torque[2], color='purple',
              arrow_length_ratio=0.15, linewidth=3, label=f'Torque τ = {[f"{t:.1f}" for t in torque]} N·m')
    
    ax3.set_xlabel('X', fontweight='bold')
    ax3.set_ylabel('Y', fontweight='bold')
    ax3.set_zlabel('Z', fontweight='bold')
    ax3.set_title('Real-Life: Torque = r × F', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    
    plt.suptitle('📊 Cross Product: Complete Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_statistics_enhanced(data, stat_type='mean'):
    """Enhanced statistics visualization with distributions"""
    data = np.array(data)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_val = np.std(data)
    var_val = np.var(data)
    
    # Histogram with distribution
    ax1 = fig.add_subplot(gs[0, :2])
    n, bins, patches = ax1.hist(data, bins=20, color=COLORS['primary'], 
                                edgecolor='black', alpha=0.7, density=True)
    
    # Overlay normal distribution
    mu, sigma = mean_val, std_val
    x_dist = np.linspace(data.min(), data.max(), 100)
    y_dist = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_dist - mu)/sigma)**2)
    ax1.plot(x_dist, y_dist, 'r-', linewidth=3, label='Normal Distribution')
    
    ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2.5, label=f'Mean = {mean_val:.2f}')
    ax1.axvline(median_val, color='green', linestyle='--', linewidth=2.5, label=f'Median = {median_val:.2f}')
    ax1.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax1.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='±1 Std Dev')
    
    ax1.set_title('Distribution Analysis', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Value', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2 = fig.add_subplot(gs[0, 2])
    bp = ax2.boxplot(data, vert=True, patch_artist=True,
                     boxprops=dict(facecolor=COLORS['secondary'], alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    # Add statistics text
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    
    stats_text = f'Q1: {q1:.2f}\nMedian: {median_val:.2f}\nQ3: {q3:.2f}\nIQR: {iqr:.2f}'
    ax2.text(1.35, median_val, stats_text, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax2.set_title('Box Plot', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Value', fontweight='bold')
    ax2.set_xticklabels(['Data'])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Scatter plot with mean line
    ax3 = fig.add_subplot(gs[1, 0])
    indices = np.arange(len(data))
    ax3.scatter(indices, data, c=data, cmap='viridis', s=100, edgecolor='black', linewidth=1, alpha=0.7)
    ax3.axhline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.2f}')
    ax3.fill_between(indices, mean_val - std_val, mean_val + std_val, alpha=0.2, color='orange')
    
    ax3.set_title('Data Points with Mean', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Index', fontweight='bold')
    ax3.set_ylabel('Value', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Statistics summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    stats_summary = f"""
    📊 STATISTICAL SUMMARY
    
    Count:           {len(data)}
    Mean (μ):        {mean_val:.4f}
    Median:          {median_val:.4f}
    Mode:            {float(np.argmax(np.bincount(data.astype(int)))) if len(data) > 0 else 'N/A'}
    
    Std Dev (σ):     {std_val:.4f}
    Variance (σ²):   {var_val:.4f}
    Range:           {data.max() - data.min():.4f}
    
    Min:             {data.min():.4f}
    Q1:              {q1:.4f}
    Q3:              {q3:.4f}
    Max:             {data.max():.4f}
    IQR:             {iqr:.4f}
    
    Skewness:        {((data - mean_val)**3).mean() / std_val**3:.4f}
    """
    
    ax4.text(0.1, 0.5, stats_summary, fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor=COLORS['light'], alpha=0.8))
    
    # Real-life example
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    real_life = f"""
    🌍 REAL-LIFE APPLICATION
    
    Test Scores Example:
    
    Class Average: {mean_val:.1f}
    Typical Range: {mean_val-std_val:.1f} to {mean_val+std_val:.1f}
    
    Interpretation:
    • 68% of students score within 1σ
    • 95% within 2σ
    • 99.7% within 3σ
    
    This helps teachers understand:
    - Class performance
    - Score distribution
    - Outlier identification
    - Grade cutoffs
    """
    
    ax5.text(0.1, 0.5, real_life, fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f'📊 Comprehensive Statistical Analysis: {stat_type.upper()}', 
                fontsize=16, fontweight='bold')
    plt.show()


def visualize_quadratic_enhanced(a, b, c):
    """Enhanced quadratic visualization"""
    x = np.linspace(-10, 10, 1000)
    y = a * x**2 + b * x + c
    
    disc = b**2 - 4*a*c
    
    fig = plt.figure(figsize=(16, 6))
    
    # Parabola plot
    ax1 = plt.subplot(131)
    ax1.plot(x, y, color=COLORS['primary'], linewidth=3, label=f'y = {a:.2g}x² + {b:.2g}x + {c:.2g}')
    ax1.fill_between(x, y, 0, where=(y >= 0), alpha=0.2, color=COLORS['success'])
    ax1.fill_between(x, y, 0, where=(y < 0), alpha=0.2, color=COLORS['secondary'])
    
    # Vertex
    vertex_x = -b / (2*a)
    vertex_y = a * vertex_x**2 + b * vertex_x + c
    ax1.plot(vertex_x, vertex_y, 'ro', markersize=12, label=f'Vertex ({vertex_x:.2f}, {vertex_y:.2f})')
    
    # Roots
    if disc >= 0:
        root1 = (-b + np.sqrt(disc)) / (2*a)
        root2 = (-b - np.sqrt(disc)) / (2*a)
        ax1.plot([root1, root2], [0, 0], 'go', markersize=12, label='Roots')
        ax1.axvline(root1, color='green', linestyle='--', alpha=0.5)
        ax1.axvline(root2, color='green', linestyle='--', alpha=0.5)
    
    # Axis of symmetry
    ax1.axvline(vertex_x, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Axis of Symmetry')
    
    setup_plot_style(ax1, f'Quadratic Function: {a:.2g}x² + {b:.2g}x + {c:.2g}')
    ax1.legend(fontsize=9)
    ax1.set_ylim(vertex_y - 20, vertex_y + 20)
    
    # Discriminant analysis
    ax2 = plt.subplot(132)
    ax2.axis('off')
    
    if disc > 0:
        root_type = "Two Real Roots"
        root_color = COLORS['success']
        explanation = f"Δ = {disc:.2f} > 0\n\nRoots:\nx₁ = {root1:.4f}\nx₂ = {root2:.4f}"
    elif disc == 0:
        root_type = "One Repeated Root"
        root_color = COLORS['warning']
        explanation = f"Δ = 0\n\nRoot:\nx = {vertex_x:.4f}"
    else:
        root_type = "Complex Roots"
        root_color = COLORS['secondary']
        real_part = -b / (2*a)
        imag_part = np.sqrt(-disc) / (2*a)
        explanation = f"Δ = {disc:.2f} < 0\n\nComplex Roots:\nx = {real_part:.4f} ± {imag_part:.4f}i"
    
    disc_text = f"""
    📐 QUADRATIC ANALYSIS
    
    Standard Form:
    ax² + bx + c = 0
    
    Coefficients:
    a = {a:.4f}
    b = {b:.4f}
    c = {c:.4f}
    
    Discriminant:
    Δ = b² - 4ac
    
    {explanation}
    
    Type: {root_type}
    
    Vertex Form:
    y = {a:.2g}(x - {vertex_x:.2f})² + {vertex_y:.2f}
    """
    
    ax2.text(0.5, 0.5, disc_text, fontsize=11, family='monospace',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=1', facecolor=root_color, alpha=0.3))
    
    # Real-life: Projectile motion
    ax3 = plt.subplot(133)
    
    # Projectile example
    t = np.linspace(0, 10, 100)
    v0 = 50  # initial velocity
    h0 = 0   # initial height
    g = 9.8  # gravity
    
    height = h0 + v0*t - 0.5*g*t**2
    
    ax3.plot(t, height, color=COLORS['primary'], linewidth=3)
    ax3.fill_between(t, 0, height, where=(height >= 0), alpha=0.3, color=COLORS['success'])
    
    # Find when it hits ground
    t_land = 2 * v0 / g
    ax3.plot(t_land, 0, 'ro', markersize=12, label=f'Lands at t={t_land:.2f}s')
    
    # Maximum height
    t_max = v0 / g
    h_max = h0 + v0*t_max - 0.5*g*t_max**2
    ax3.plot(t_max, h_max, 'go', markersize=12, label=f'Max height={h_max:.2f}m')
    
    ax3.set_title('Real-Life: Projectile Motion', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (s)', fontweight='bold')
    ax3.set_ylabel('Height (m)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(0, t_land + 1)
    ax3.set_ylim(0, h_max + 10)
    
    plt.suptitle('📊 Comprehensive Quadratic Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_trigonometry_enhanced(func='sin'):
    """Enhanced trigonometry visualization with unit circle"""
    x = np.linspace(-2*np.pi, 2*np.pi, 1000)
    
    fig = plt.figure(figsize=(16, 6))
    
    # Function plot
    ax1 = plt.subplot(131)
    
    if func == 'sin':
        y = np.sin(x)
        dy = np.cos(x)
        ax1.plot(x, y, color=COLORS['primary'], linewidth=3, label='sin(x)')
        ax1.plot(x, dy, color=COLORS['secondary'], linewidth=2, linestyle='--', label="cos(x) = d/dx[sin(x)]")
        title = 'Sine Function'
    elif func == 'cos':
        y = np.cos(x)
        dy = -np.sin(x)
        ax1.plot(x, y, color=COLORS['primary'], linewidth=3, label='cos(x)')
        ax1.plot(x, dy, color=COLORS['secondary'], linewidth=2, linestyle='--', label="-sin(x) = d/dx[cos(x)]")
        title = 'Cosine Function'
    else:  # tan
        y = np.tan(x)
        y = np.where(np.abs(y) > 10, np.nan, y)
        ax1.plot(x, y, color=COLORS['primary'], linewidth=3, label='tan(x)')
        title = 'Tangent Function'
    
    # Mark important points
    important_x = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi, -np.pi/2, -np.pi, -3*np.pi/2, -2*np.pi]
    for xp in important_x:
        if func != 'tan' or abs(np.tan(xp)) < 10:
            if func == 'sin':
                yp = np.sin(xp)
            elif func == 'cos':
                yp = np.cos(xp)
            else:
                yp = np.tan(xp) if abs(np.tan(xp)) < 10 else np.nan
            
            if not np.isnan(yp):
                ax1.plot(xp, yp, 'o', color=COLORS['warning'], markersize=8)
    
    setup_plot_style(ax1, title)
    ax1.legend(fontsize=10)
    ax1.set_ylim(-3, 3)
    
    # Add period markers
    for i in range(-2, 3):
        ax1.axvline(i * np.pi, color='gray', linestyle=':', alpha=0.3)
    
    # Unit circle
    ax2 = plt.subplot(132, projection='polar')
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.ones_like(theta)
    ax2.plot(theta, r, color=COLORS['dark'], linewidth=2)
    
    # Mark angles
    angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6, np.pi]
    for angle in angles:
        ax2.plot([angle, angle], [0, 1], color=COLORS['primary'], linewidth=1, alpha=0.5)
        if func == 'sin':
            r_val = np.sin(angle)
        elif func == 'cos':
            r_val = np.cos(angle)
        else:
            r_val = np.tan(angle) if abs(np.tan(angle)) < 10 else 0
        
        ax2.plot(angle, abs(r_val), 'o', color=COLORS['secondary'], markersize=10)
    
    ax2.set_title(f'Unit Circle: {func.upper()}', fontsize=12, fontweight='bold')
    
    # Real-life application
    ax3 = plt.subplot(133)
    
    if func in ['sin', 'cos']:
        # Sound wave
        t = np.linspace(0, 0.01, 1000)
        frequency = 440  # A note
        amplitude = 1
        
        if func == 'sin':
            wave = amplitude * np.sin(2 * np.pi * frequency * t)
            ax3.plot(t * 1000, wave, color=COLORS['primary'], linewidth=2)
            ax3.set_title('Real-Life: Sound Wave (440 Hz)', fontsize=12, fontweight='bold')
        else:
            wave = amplitude * np.cos(2 * np.pi * frequency * t)
            ax3.plot(t * 1000, wave, color=COLORS['primary'], linewidth=2)
            ax3.set_title('Real-Life: AC Current', fontsize=12, fontweight='bold')
        
        ax3.set_xlabel('Time (ms)', fontweight='bold')
        ax3.set_ylabel('Amplitude', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color='black', linewidth=0.8)
        
        ax3.text(5, 0.7, f'Frequency: {frequency} Hz\nPeriod: {1/frequency*1000:.2f} ms',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    else:
        # Slope/Grade
        angles = np.linspace(0, 45, 100)
        slopes = np.tan(np.radians(angles))
        
        ax3.plot(angles, slopes * 100, color=COLORS['primary'], linewidth=3)
        ax3.set_title('Real-Life: Road Grade/Slope', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Angle (degrees)', fontweight='bold')
        ax3.set_ylabel('Grade (%)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Mark common slopes
        common_angles = [15, 30, 45]
        for angle in common_angles:
            grade = np.tan(np.radians(angle)) * 100
            ax3.plot(angle, grade, 'ro', markersize=10)
            ax3.annotate(f'{angle}°: {grade:.1f}%', xy=(angle, grade), 
                        xytext=(angle+2, grade+5), fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.suptitle(f'📊 Trigonometric Function Analysis: {func.upper()}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Additional helper functions for backward compatibility
def visualize_dot_product(v1, v2):
    """Backward compatible wrapper"""
    visualize_dot_product_enhanced(v1, v2)

def visualize_cross_product(v1, v2):
    """Backward compatible wrapper"""
    visualize_cross_product_enhanced(v1, v2)

def visualize_modulus(vector):
    """Simple modulus visualization"""
    visualize_dot_product_enhanced(vector, vector)

def visualize_statistics(data, stat_type):
    """Backward compatible wrapper"""
    visualize_statistics_enhanced(data, stat_type)

def visualize_floor_ceil_abs(value, func_type):
    """Visualize floor, ceiling, and absolute value"""
    x = np.linspace(-5, 5, 1000)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if func_type == 'floor':
        y = np.floor(x)
        result = math.floor(value)
        ax1.step(x, y, color=COLORS['primary'], linewidth=2.5, where='post', label='⌊x⌋')
        ax1.plot(value, result, 'ro', markersize=12, label=f'⌊{value}⌋ = {result}')
        title = f'Floor Function: ⌊{value}⌋ = {result}'
        
        # Real-life example
        ax2.axis('off')
        example = f"""
        🌍 REAL-LIFE APPLICATION
        
        Floor Function in Daily Life:
        
        Example: Pizza Slices
        You have {value:.2f} pizzas
        Each person gets whole pizzas only
        
        Result: {result} whole pizzas
        (Leftover: {value - result:.2f} pizza)
        
        Other Uses:
        • Integer division
        • Counting complete units
        • Rounding down prices
        • Grouping items
        """
        ax2.text(0.1, 0.5, example, fontsize=11, va='center',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))
        
    elif func_type == 'ceiling':
        y = np.ceil(x)
        result = math.ceil(value)
        ax1.step(x, y, color=COLORS['success'], linewidth=2.5, where='post', label='⌈x⌉')
        ax1.plot(value, result, 'ro', markersize=12, label=f'⌈{value}⌉ = {result}')
        title = f'Ceiling Function: ⌈{value}⌉ = {result}'
        
        # Real-life example
        ax2.axis('off')
        example = f"""
        🌍 REAL-LIFE APPLICATION
        
        Ceiling Function in Daily Life:
        
        Example: Taxis Needed
        You have {value:.2f} groups of people
        Each taxi holds 1 group
        
        Result: Need {result} taxis
        (Can't use partial taxis!)
        
        Other Uses:
        • Rounding up for resources
        • Minimum containers needed
        • Booking accommodations
        • Package requirements
        """
        ax2.text(0.1, 0.5, example, fontsize=11, va='center',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.8))
        
    else:  # absolute
        y = np.abs(x)
        result = abs(value)
        ax1.plot(x, y, color=COLORS['secondary'], linewidth=2.5, label='|x|')
        ax1.plot(value, result, 'bo', markersize=12, label=f'|{value}| = {result}')
        ax1.fill_between(x, y, alpha=0.2, color=COLORS['secondary'])
        title = f'Absolute Value: |{value}| = {result}'
        
        # Real-life example
        ax2.axis('off')
        example = f"""
        🌍 REAL-LIFE APPLICATION
        
        Absolute Value in Daily Life:
        
        Example: Temperature Difference
        Current temp: {value:.2f}°C
        Reference: 0°C
        
        Difference: {result:.2f}°C
        (Direction doesn't matter!)
        
        Other Uses:
        • Distance calculations
        • Error measurements  
        • Magnitude of change
        • Deviation from target
        • Signal strength
        """
        ax2.text(0.1, 0.5, example, fontsize=11, va='center',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightcoral', alpha=0.8))
    
    setup_plot_style(ax1, title)
    ax1.legend(fontsize=11)
    ax1.set_ylim(-6, 6)
    
    plt.suptitle(f'📊 {func_type.title()} Function Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def visualize_quadratic(a, b, c, roots):
    """Backward compatible wrapper"""
    visualize_quadratic_enhanced(a, b, c)

def visualize_trigonometry(func):
    """Backward compatible wrapper"""
    visualize_trigonometry_enhanced(func)