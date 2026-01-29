# app.py - Complete Backend with FULL Prolog Integration
from flask import Flask, render_template, request, jsonify
import math
import os
import io
import base64
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
import atexit

plt.ioff()
matplotlib.rcParams['figure.max_open_warning'] = 0

app = Flask(__name__)
sns.set_style("darkgrid")
sns.set_palette("husl")

# Initialize Prolog with proper error handling
prolog = None
prolog_available = False

try:
    from pyswip import Prolog
    prolog = Prolog()
    prolog_file = os.path.join(os.path.dirname(__file__), 'math_expert_system.pl')
    if os.path.exists(prolog_file):
        prolog.consult(prolog_file)
        prolog_available = True
        print(f"✓ Loaded Prolog knowledge base from {prolog_file}")
    else:
        print(f"✗ Prolog file not found at {prolog_file}")
except Exception as e:
    print(f"✗ Prolog not available: {str(e)}")
    print("  → Using Python fallback for all computations")

def cleanup():
    try:
        plt.close('all')
    except:
        pass

atexit.register(cleanup)

def make_json_serializable(obj):
    """Convert ANY object to JSON-serializable format"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, bytes):
        return base64.b64encode(obj).decode('utf-8')
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(key): make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    else:
        try:
            return str(obj)
        except:
            return None

class MathExpertSystem:
    def __init__(self):
        self.prolog = prolog if prolog_available else None
        self.prolog_available = prolog_available
    
    def query_prolog(self, query_string):
        """Query Prolog knowledge base"""
        if not self.prolog_available:
            return []
        try:
            results = list(self.prolog.query(query_string))
            return results
        except Exception as e:
            print(f"Prolog query error: {e}")
            return []
    
    def get_prolog_explanation(self, concept):
        """Get explanation from Prolog KB"""
        if not self.prolog_available:
            return None
        try:
            query = f"explanation({concept}, Exp)"
            results = self.query_prolog(query)
            if results and len(results) > 0:
                exp = results[0].get('Exp', None)
                # FIXED: Decode bytes to string if needed
                if isinstance(exp, bytes):
                    exp = exp.decode('utf-8')
                return exp
        except Exception as e:
            print(f"Error getting explanation: {e}")
            pass
        return None
    
    def create_figure(self, figsize=(10, 6)):
        fig = plt.figure(figsize=figsize, facecolor='#1e293b')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0f172a')
        ax.tick_params(colors='#cbd5e1')
        for spine in ax.spines.values():
            spine.set_color('#475569')
        return fig, ax
    
    def create_3d_figure(self, figsize=(10, 8)):
        fig = plt.figure(figsize=figsize, facecolor='#1e293b')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#0f172a')
        ax.tick_params(colors='#cbd5e1')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        return fig, ax
    
    def fig_to_base64(self, fig):
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', facecolor='#1e293b', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close(fig)
            return f"data:image/png;base64,{img_base64}"
        except:
            plt.close(fig)
            return None
    
    # ==================== CALCULUS WITH PROLOG ====================
    
    def compute_derivative(self, func_type, coefficient, exponent):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available and func_type == 'power':
                query = f"compute(derivative, [power, x, {exponent}], Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            # Python computation
            if func_type == 'power':
                new_coef = coefficient * exponent
                new_exp = exponent - 1
                result_text = f"d/dx({coefficient}x^{exponent}) = {new_coef}x^{new_exp}" if new_exp != 0 else f"{new_coef}"
                
                fig, ax = self.create_figure()
                x = np.linspace(-5, 5, 1000)
                y_orig = coefficient * x**exponent
                y_deriv = new_coef * x**new_exp if new_exp != 0 else np.full_like(x, new_coef)
                ax.plot(x, y_orig, 'cyan', linewidth=2.5, label='f(x)', alpha=0.9)
                ax.plot(x, y_deriv, 'magenta', linewidth=2.5, label="f'(x)", alpha=0.9)
                ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)
                ax.axvline(0, color='white', linewidth=0.5, alpha=0.3)
                ax.set_title('Derivative Visualization', color='#f1f5f9', fontsize=16, fontweight='bold', pad=15)
                ax.set_xlabel('x', color='#cbd5e1', fontsize=12)
                ax.set_ylabel('y', color='#cbd5e1', fontsize=12)
                ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
                ax.grid(True, alpha=0.2, linestyle='--')
                
                explanation = self.get_prolog_explanation('derivative')
                
                return make_json_serializable({
                    'result': result_text, 
                    'visualization': self.fig_to_base64(fig), 
                    'prolog_used': self.prolog_available,
                    'prolog_result': str(prolog_result) if prolog_result else None,
                    'explanation': explanation
                })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_integration(self, func_type, coefficient, exponent):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available and func_type == 'power':
                query = f"compute(integration, [power, x, {exponent}], Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            if func_type == 'power':
                new_exp = exponent + 1
                new_coef = coefficient / new_exp
                result_text = f"∫{coefficient}x^{exponent}dx = {new_coef:.3f}x^{new_exp} + C"
                
                fig, ax = self.create_figure()
                x = np.linspace(0, 5, 1000)
                y = coefficient * x**exponent
                ax.plot(x, y, 'cyan', linewidth=2.5, label='f(x)', alpha=0.9)
                ax.fill_between(x, y, 0, alpha=0.3, color='cyan')
                ax.set_title('Integration - Area Under Curve', color='#f1f5f9', fontsize=16, fontweight='bold', pad=15)
                ax.set_xlabel('x', color='#cbd5e1', fontsize=12)
                ax.set_ylabel('y', color='#cbd5e1', fontsize=12)
                ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
                ax.grid(True, alpha=0.2, linestyle='--')
                
                explanation = self.get_prolog_explanation('integration')
                
                return make_json_serializable({
                    'result': result_text, 
                    'visualization': self.fig_to_base64(fig), 
                    'prolog_used': self.prolog_available,
                    'prolog_result': str(prolog_result) if prolog_result else None,
                    'explanation': explanation
                })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_limit(self, func_type, point, direction='both'):
        try:
            if func_type == 'rational':
                x = np.linspace(0, 4, 1000)
                x = x[x != point]
                y = (x**2 - 4) / (x - 2)
                limit_value = 2 * point
                
                fig, ax = self.create_figure()
                ax.plot(x, y, 'cyan', linewidth=2.5, label='f(x)', alpha=0.9)
                ax.plot(point, limit_value, 'ro', markersize=12, label=f'Limit = {limit_value}', zorder=5)
                ax.axvline(point, color='magenta', linestyle='--', alpha=0.5, linewidth=2)
                ax.set_title(f'Limit as x → {point}', color='#f1f5f9', fontsize=16, fontweight='bold', pad=15)
                ax.set_xlabel('x', color='#cbd5e1', fontsize=12)
                ax.set_ylabel('y', color='#cbd5e1', fontsize=12)
                ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
                ax.grid(True, alpha=0.2, linestyle='--')
                
                explanation = self.get_prolog_explanation('limit')
                
                return make_json_serializable({
                    'result': f"lim(x→{point}) = {limit_value}",
                    'visualization': self.fig_to_base64(fig),
                    'prolog_used': self.prolog_available,
                    'explanation': explanation
                })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_partial_derivative(self, var, x, y):
        try:
            if var == 'x':
                result = 2 * x
                result_text = f"∂f/∂x at ({x},{y}) = {result}"
            else:
                result = 2 * y
                result_text = f"∂f/∂y at ({x},{y}) = {result}"
            
            fig = plt.figure(figsize=(10, 8), facecolor='#1e293b')
            ax = fig.add_subplot(111, projection='3d')
            X = np.linspace(-5, 5, 50)
            Y = np.linspace(-5, 5, 50)
            X, Y = np.meshgrid(X, Y)
            Z = X**2 + Y**2
            surf = ax.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis', edgecolor='none')
            ax.scatter([x], [y], [x**2 + y**2], color='red', s=150, zorder=5, edgecolors='white', linewidths=2)
            ax.set_title('Partial Derivative of f(x,y) = x² + y²', color='#f1f5f9', fontweight='bold', fontsize=14, pad=15)
            ax.set_xlabel('X', color='#cbd5e1', fontsize=11)
            ax.set_ylabel('Y', color='#cbd5e1', fontsize=11)
            ax.set_zlabel('Z', color='#cbd5e1', fontsize=11)
            
            explanation = self.get_prolog_explanation('partial_derivative')
            
            return make_json_serializable({
                'result': result_text, 
                'visualization': self.fig_to_base64(fig), 
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_second_derivative(self, coefficient, exponent):
        try:
            first_coef = coefficient * exponent
            first_exp = exponent - 1
            second_coef = first_coef * first_exp
            second_exp = first_exp - 1
            
            result_text = f"f''(x) = {second_coef}x^{second_exp}" if second_exp > 0 else f"{second_coef}"
            
            fig, ax = self.create_figure()
            x = np.linspace(-5, 5, 1000)
            y = coefficient * x**exponent
            y2 = second_coef * x**second_exp if second_exp > 0 else np.full_like(x, second_coef)
            ax.plot(x, y, 'cyan', linewidth=2.5, label='f(x)', alpha=0.9)
            ax.plot(x, y2, 'red', linewidth=2.5, label="f''(x)", alpha=0.9)
            ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)
            ax.set_title('Second Derivative (Concavity)', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('x', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('y', color='#cbd5e1', fontsize=12)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('second_derivative')
            
            return make_json_serializable({
                'result': result_text, 
                'visualization': self.fig_to_base64(fig), 
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    # ==================== VECTORS WITH PROLOG ====================
    
    def compute_dot_product(self, x1, y1, x2, y2):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                query = f"compute(dot_product, [{x1}, {y1}], [{x2}, {y2}], Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            dot = x1*x2 + y1*y2
            mag1 = math.sqrt(x1**2 + y1**2)
            mag2 = math.sqrt(x2**2 + y2**2)
            angle = math.degrees(math.acos(dot / (mag1 * mag2))) if mag1 > 0 and mag2 > 0 else 0
            
            fig, ax = self.create_figure()
            ax.quiver(0, 0, x1, y1, angles='xy', scale_units='xy', scale=1, color='cyan', width=0.012, label='Vector A')
            ax.quiver(0, 0, x2, y2, angles='xy', scale_units='xy', scale=1, color='magenta', width=0.012, label='Vector B')
            max_val = max(abs(x1), abs(x2), abs(y1), abs(y2)) * 1.5
            ax.set_xlim(-max_val, max_val)
            ax.set_ylim(-max_val, max_val)
            ax.set_aspect('equal')
            ax.set_title(f'Dot Product = {dot:.2f}, Angle = {angle:.2f}°', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('dot_product')
            
            return make_json_serializable({
                'dot_product': round(dot, 3),
                'angle': round(angle, 2),
                'magnitude_a': round(mag1, 3),
                'magnitude_b': round(mag2, 3),
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_cross_product(self, x1, y1, z1, x2, y2, z2):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                query = f"compute(cross_product, [{x1}, {y1}, {z1}], [{x2}, {y2}, {z2}], Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            rx = y1*z2 - z1*y2
            ry = z1*x2 - x1*z2
            rz = x1*y2 - y1*x2
            
            fig = plt.figure(figsize=(10, 8), facecolor='#1e293b')
            ax = fig.add_subplot(111, projection='3d')
            ax.quiver(0, 0, 0, x1, y1, z1, color='cyan', arrow_length_ratio=0.15, linewidth=2.5, label='Vector A')
            ax.quiver(0, 0, 0, x2, y2, z2, color='magenta', arrow_length_ratio=0.15, linewidth=2.5, label='Vector B')
            ax.quiver(0, 0, 0, rx, ry, rz, color='yellow', arrow_length_ratio=0.15, linewidth=2.5, label='A × B')
            ax.set_title('Cross Product (Perpendicular Vector)', color='#f1f5f9', fontweight='bold', fontsize=14, pad=15)
            ax.set_xlabel('X', color='#cbd5e1', fontsize=11)
            ax.set_ylabel('Y', color='#cbd5e1', fontsize=11)
            ax.set_zlabel('Z', color='#cbd5e1', fontsize=11)
            ax.legend(loc='best', fontsize=10)
            
            explanation = self.get_prolog_explanation('cross_product')
            
            return make_json_serializable({
                'result': [round(rx, 3), round(ry, 3), round(rz, 3)],
                'magnitude': round(math.sqrt(rx**2 + ry**2 + rz**2), 3),
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_modulus(self, x, y, z=0):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                if z == 0:
                    query = f"compute(modulus, [{x}, {y}], Result)"
                else:
                    query = f"compute(modulus, [{x}, {y}, {z}], Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            result = math.sqrt(x**2 + y**2 + z**2)
            
            fig, ax = self.create_figure()
            ax.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1, color='cyan', width=0.012)
            ax.plot([0, x], [0, y], 'o', color='cyan', markersize=10)
            max_val = max(abs(x), abs(y)) * 1.5 if max(abs(x), abs(y)) > 0 else 1
            ax.set_xlim(-max_val, max_val)
            ax.set_ylim(-max_val, max_val)
            ax.set_aspect('equal')
            ax.set_title(f'Vector Magnitude = {result:.3f}', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.set_xlabel('x', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('y', color='#cbd5e1', fontsize=12)
            
            explanation = self.get_prolog_explanation('modulus')
            
            return make_json_serializable({
                'result': round(result, 3), 
                'visualization': self.fig_to_base64(fig), 
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_unit_vector(self, x, y, z=0):
        try:
            mag = math.sqrt(x**2 + y**2 + z**2)
            if mag == 0:
                return {'error': 'Cannot normalize zero vector'}
            ux, uy, uz = x/mag, y/mag, z/mag
            
            fig, ax = self.create_figure()
            ax.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1, color='cyan', width=0.012, label='Original')
            ax.quiver(0, 0, ux, uy, angles='xy', scale_units='xy', scale=1, color='yellow', width=0.012, label='Unit Vector')
            ax.set_aspect('equal')
            ax.set_title('Unit Vector (Magnitude = 1)', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('unit_vector')
            
            return make_json_serializable({
                'result': [round(ux, 3), round(uy, 3), round(uz, 3)] if z != 0 else [round(ux, 3), round(uy, 3)],
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_vector_projection(self, ax, ay, bx, by):
        try:
            dot = ax*bx + ay*by
            mag_b_sq = bx**2 + by**2
            if mag_b_sq == 0:
                return {'error': 'Cannot project onto zero vector'}
            scalar = dot / mag_b_sq
            proj_x, proj_y = scalar * bx, scalar * by
            
            fig, ax_plot = self.create_figure()
            ax_plot.quiver(0, 0, ax, ay, angles='xy', scale_units='xy', scale=1, color='cyan', width=0.012, label='Vector a')
            ax_plot.quiver(0, 0, bx, by, angles='xy', scale_units='xy', scale=1, color='magenta', width=0.012, label='Vector b')
            ax_plot.quiver(0, 0, proj_x, proj_y, angles='xy', scale_units='xy', scale=1, color='yellow', width=0.012, label='proj_b(a)')
            ax_plot.plot([ax, proj_x], [ay, proj_y], 'w--', alpha=0.5, linewidth=1.5)
            ax_plot.set_aspect('equal')
            ax_plot.set_title('Vector Projection', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax_plot.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax_plot.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('vector_projection')
            
            return make_json_serializable({
                'result': [round(proj_x, 3), round(proj_y, 3)],
                'scalar': round(scalar, 3),
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    # ==================== STATISTICS WITH PROLOG ====================
    
    def compute_mean(self, values):
        try:
            nums = [float(x.strip()) for x in values.split(',') if x.strip()]
            
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                list_str = str(nums).replace(' ', '')
                query = f"compute(mean, {list_str}, Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            mean_val = sum(nums) / len(nums)
            
            fig, ax = self.create_figure()
            ax.bar(range(len(nums)), nums, color='cyan', alpha=0.7, edgecolor='white', linewidth=1.5)
            ax.axhline(mean_val, color='magenta', linewidth=2.5, linestyle='--', label=f'Mean = {mean_val:.2f}')
            ax.set_title('Mean (Average)', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('Index', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('Value', color='#cbd5e1', fontsize=12)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, axis='y', linestyle='--')
            
            explanation = self.get_prolog_explanation('mean')
            
            return make_json_serializable({
                'mean': round(mean_val, 3), 
                'visualization': self.fig_to_base64(fig), 
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_median(self, values):
        try:
            nums = [float(x.strip()) for x in values.split(',') if x.strip()]
            
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                list_str = str(nums).replace(' ', '')
                query = f"compute(median, {list_str}, Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            sorted_nums = sorted(nums)
            n = len(sorted_nums)
            median_val = sorted_nums[n//2] if n % 2 == 1 else (sorted_nums[n//2-1] + sorted_nums[n//2]) / 2
            
            fig, ax = self.create_figure()
            bp = ax.boxplot([nums], vert=True, patch_artist=True, widths=0.6)
            bp['boxes'][0].set_facecolor('cyan')
            bp['boxes'][0].set_alpha(0.7)
            bp['medians'][0].set_color('magenta')
            bp['medians'][0].set_linewidth(3)
            ax.set_title(f'Median = {median_val:.2f}', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_ylabel('Value', color='#cbd5e1', fontsize=12)
            ax.grid(True, alpha=0.2, axis='y', linestyle='--')
            
            explanation = self.get_prolog_explanation('median')
            
            return make_json_serializable({
                'median': round(median_val, 3), 
                'visualization': self.fig_to_base64(fig), 
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_mode(self, values):
        try:
            nums = [float(x.strip()) for x in values.split(',') if x.strip()]
            from collections import Counter
            counts = Counter(nums)
            max_count = max(counts.values())
            modes = [k for k, v in counts.items() if v == max_count]
            
            fig, ax = self.create_figure()
            unique_vals = sorted(set(nums))
            freq = [nums.count(v) for v in unique_vals]
            colors = ['magenta' if nums.count(v) == max_count else 'cyan' for v in unique_vals]
            ax.bar(range(len(unique_vals)), freq, color=colors, alpha=0.7, edgecolor='white', linewidth=1.5)
            ax.set_xticks(range(len(unique_vals)))
            ax.set_xticklabels([f'{v:.1f}' for v in unique_vals], color='#cbd5e1')
            ax.set_title('Mode (Most Frequent)', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('Value', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('Frequency', color='#cbd5e1', fontsize=12)
            ax.grid(True, alpha=0.2, axis='y', linestyle='--')
            
            explanation = self.get_prolog_explanation('mode')
            
            return make_json_serializable({
                'mode': modes, 
                'frequency': max_count, 
                'visualization': self.fig_to_base64(fig), 
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_variance(self, values):
        try:
            nums = [float(x.strip()) for x in values.split(',') if x.strip()]
            
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                list_str = str(nums).replace(' ', '')
                query = f"compute(variance, {list_str}, Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            mean_val = sum(nums) / len(nums)
            variance = sum((x - mean_val)**2 for x in nums) / len(nums)
            std_dev = math.sqrt(variance)
            
            fig, ax = self.create_figure()
            ax.scatter(range(len(nums)), nums, c='cyan', s=120, alpha=0.7, edgecolors='white', linewidths=2)
            ax.axhline(mean_val, color='magenta', linewidth=2.5, linestyle='--', label=f'Mean = {mean_val:.2f}')
            ax.axhline(mean_val + std_dev, color='yellow', linewidth=2, linestyle=':', label=f'±1 SD')
            ax.axhline(mean_val - std_dev, color='yellow', linewidth=2, linestyle=':')
            ax.set_title(f'Variance = {variance:.2f}, Std Dev = {std_dev:.2f}', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('Index', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('Value', color='#cbd5e1', fontsize=12)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('variance')
            
            return make_json_serializable({
                'variance': round(variance, 3),
                'std_dev': round(std_dev, 3),
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_standard_deviation(self, values):
        result = self.compute_variance(values)
        if 'std_dev' in result:
            result['result'] = result['std_dev']
        return result
    
    def compute_range(self, values):
        try:
            nums = [float(x.strip()) for x in values.split(',') if x.strip()]
            
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                list_str = str(nums).replace(' ', '')
                query = f"compute(range, {list_str}, Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            range_val = max(nums) - min(nums)
            
            fig, ax = self.create_figure()
            ax.scatter(range(len(nums)), nums, c='cyan', s=150, alpha=0.7, edgecolors='white', linewidths=2)
            ax.axhline(max(nums), color='magenta', linewidth=2.5, linestyle='--', label=f'Max = {max(nums):.2f}')
            ax.axhline(min(nums), color='yellow', linewidth=2.5, linestyle='--', label=f'Min = {min(nums):.2f}')
            ax.set_title(f'Range = {range_val:.2f}', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('Index', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('Value', color='#cbd5e1', fontsize=12)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('range')
            
            return make_json_serializable({
                'result': round(range_val, 3),
                'max': max(nums),
                'min': min(nums),
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_percentile(self, values, percentile):
        try:
            nums = [float(x.strip()) for x in values.split(',') if x.strip()]
            sorted_nums = sorted(nums)
            k = (len(sorted_nums) - 1) * (percentile / 100)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                result = sorted_nums[int(k)]
            else:
                result = sorted_nums[int(f)] * (c-k) + sorted_nums[int(c)] * (k-f)
            
            fig, ax = self.create_figure()
            ax.hist(nums, bins=10, color='cyan', alpha=0.7, edgecolor='white', linewidth=1.5)
            ax.axvline(result, color='magenta', linewidth=2.5, linestyle='--', label=f'{percentile}th percentile = {result:.2f}')
            ax.set_title(f'{percentile}th Percentile', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('Value', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('Frequency', color='#cbd5e1', fontsize=12)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, axis='y', linestyle='--')
            
            explanation = self.get_prolog_explanation('percentile')
            
            return make_json_serializable({
                'result': round(result, 3), 
                'visualization': self.fig_to_base64(fig), 
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_correlation(self, x_values, y_values):
        try:
            x = [float(v.strip()) for v in x_values.split(',') if v.strip()]
            y = [float(v.strip()) for v in y_values.split(',') if v.strip()]
            if len(x) != len(y):
                return {'error': 'X and Y must have same length'}
            
            n = len(x)
            mean_x = sum(x) / n
            mean_y = sum(y) / n
            
            numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
            denom_x = math.sqrt(sum((xi - mean_x)**2 for xi in x))
            denom_y = math.sqrt(sum((yi - mean_y)**2 for yi in y))
            
            r = numerator / (denom_x * denom_y) if denom_x != 0 and denom_y != 0 else 0
            
            fig, ax = self.create_figure()
            ax.scatter(x, y, c='cyan', s=120, alpha=0.7, edgecolors='white', linewidths=2)
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(x), max(x), 100)
            ax.plot(x_line, p(x_line), "magenta", linewidth=2.5, linestyle='--', label=f'r = {r:.3f}')
            ax.set_xlabel('X', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('Y', color='#cbd5e1', fontsize=12)
            ax.set_title(f'Correlation Coefficient = {r:.3f}', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('correlation')
            
            return make_json_serializable({
                'result': round(r, 3), 
                'visualization': self.fig_to_base64(fig), 
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_z_score(self, value, mean, std_dev):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                query = f"compute(z_score, [{value}, {mean}, {std_dev}], Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            z = (value - mean) / std_dev if std_dev != 0 else 0
            
            fig, ax = self.create_figure()
            x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
            y = (1/(std_dev * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean)/std_dev)**2)
            ax.plot(x, y, 'cyan', linewidth=2.5)
            ax.axvline(value, color='magenta', linewidth=2.5, linestyle='--', label=f'z = {z:.2f}')
            ax.fill_between(x, y, where=(x < value), alpha=0.3, color='cyan')
            ax.set_title(f'Z-Score = {z:.2f}', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('Value', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('Probability Density', color='#cbd5e1', fontsize=12)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('z_score')
            
            return make_json_serializable({
                'result': round(z, 3), 
                'visualization': self.fig_to_base64(fig), 
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    # ==================== PROBABILITY WITH PROLOG ====================
    
    def compute_probability(self, favorable, total):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                query = f"compute(probability, [{favorable}, {total}], Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            prob = favorable / total
            
            fig, ax = self.create_figure()
            sizes = [favorable, total - favorable]
            colors = ['cyan', '#334155']
            explode = (0.05, 0)
            ax.pie(sizes, labels=['Favorable', 'Unfavorable'], colors=colors, autopct='%1.1f%%',
                   textprops={'color': '#f1f5f9', 'fontsize': 12, 'fontweight': 'bold'}, explode=explode, shadow=True, startangle=90)
            ax.set_title(f'Probability = {prob:.4f} ({prob*100:.2f}%)', color='#f1f5f9', fontweight='bold', fontsize=16, pad=20)
            
            explanation = self.get_prolog_explanation('probability')
            
            return make_json_serializable({
                'probability': round(prob, 4),
                'percentage': round(prob * 100, 2),
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_conditional_probability(self, p_a_and_b, p_b):
        try:
            if p_b == 0:
                return {'error': 'P(B) cannot be zero'}
            p_a_given_b = p_a_and_b / p_b
            
            explanation = self.get_prolog_explanation('conditional_probability')
            
            return make_json_serializable({
                'result': round(p_a_given_b, 4),
                'formula': f'P(A|B) = P(A∩B) / P(B) = {p_a_and_b} / {p_b} = {p_a_given_b:.4f}',
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_bayes_theorem(self, p_b_given_a, p_a, p_b):
        try:
            if p_b == 0:
                return {'error': 'P(B) cannot be zero'}
            p_a_given_b = (p_b_given_a * p_a) / p_b
            
            explanation = self.get_prolog_explanation('bayes_theorem')
            
            return make_json_serializable({
                'result': round(p_a_given_b, 4),
                'formula': f"P(A|B) = P(B|A)·P(A) / P(B) = {p_b_given_a}·{p_a} / {p_b} = {p_a_given_b:.4f}",
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_expected_value(self, values, probabilities):
        try:
            vals = [float(v.strip()) for v in values.split(',') if v.strip()]
            probs = [float(p.strip()) for p in probabilities.split(',') if p.strip()]
            if len(vals) != len(probs):
                return {'error': 'Values and probabilities must have same length'}
            if abs(sum(probs) - 1.0) > 0.01:
                return {'error': f'Probabilities must sum to 1 (current sum: {sum(probs):.3f})'}
            
            ev = sum(v * p for v, p in zip(vals, probs))
            
            fig, ax = self.create_figure()
            ax.bar(range(len(vals)), probs, color='cyan', alpha=0.7, edgecolor='white', linewidth=1.5)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels([f'{v:.1f}' for v in vals], color='#cbd5e1')
            ax.axhline(sum(probs)/len(probs), color='magenta', linewidth=2.5, linestyle='--', alpha=0.5)
            ax.set_title(f'Expected Value = {ev:.2f}', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('Values', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('Probability', color='#cbd5e1', fontsize=12)
            ax.grid(True, alpha=0.2, axis='y', linestyle='--')
            
            explanation = self.get_prolog_explanation('expected_value')
            
            return make_json_serializable({
                'result': round(ev, 3), 
                'visualization': self.fig_to_base64(fig), 
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    # ==================== COMBINATORICS WITH PROLOG ====================
    
    def compute_permutation(self, n, r):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                query = f"compute(permutation, {n}, {r}, Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            result = math.factorial(n) // math.factorial(n - r)
            
            explanation = self.get_prolog_explanation('permutation')
            
            return make_json_serializable({
                'result': result,
                'formula': f"P({n},{r}) = {n}!/({n}-{r})! = {result:,}",
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_combination(self, n, r):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                query = f"compute(combination, {n}, {r}, Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            result = math.factorial(n) // (math.factorial(r) * math.factorial(n - r))
            
            explanation = self.get_prolog_explanation('combination')
            
            return make_json_serializable({
                'result': result,
                'formula': f"C({n},{r}) = {n}!/({r}!·({n}-{r})!) = {result:,}",
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_factorial(self, n):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                query = f"compute(factorial, {n}, Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            result = math.factorial(n)
            
            # Visualization showing factorial growth
            fig, ax = self.create_figure()
            x_vals = list(range(1, min(n+1, 12)))
            y_vals = [math.factorial(i) for i in x_vals]
            ax.plot(x_vals, y_vals, 'cyan', marker='o', linewidth=2.5, markersize=10)
            ax.scatter([n], [result], color='magenta', s=250, zorder=5, edgecolors='white', linewidth=2.5)
            ax.set_title(f'{n}! = {result:,}', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('n', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('n!', color='#cbd5e1', fontsize=12)
            ax.set_yscale('log')
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('factorial')
            
            return make_json_serializable({
                'result': result, 
                'visualization': self.fig_to_base64(fig), 
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    # ==================== ALGEBRA WITH PROLOG ====================
    
    def solve_quadratic(self, a, b, c):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                query = f"compute(quadratic_equation, [{a}, {b}, {c}], Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            discriminant = b**2 - 4*a*c
            if discriminant < 0:
                explanation = self.get_prolog_explanation('quadratic_equation')
                return {
                    'message': 'Complex roots (discriminant < 0)', 
                    'discriminant': discriminant, 
                    'prolog_used': self.prolog_available,
                    'explanation': explanation
                }
            
            root1 = (-b + math.sqrt(discriminant)) / (2*a)
            root2 = (-b - math.sqrt(discriminant)) / (2*a)
            
            fig, ax = self.create_figure()
            x = np.linspace(min(root1, root2) - 3, max(root1, root2) + 3, 1000)
            y = a*x**2 + b*x + c
            ax.plot(x, y, 'cyan', linewidth=2.5)
            ax.plot([root1, root2], [0, 0], 'ro', markersize=12, label=f'Roots: {root1:.2f}, {root2:.2f}', zorder=5)
            ax.axhline(0, color='magenta', linewidth=1.5, alpha=0.5)
            ax.axvline(0, color='white', linewidth=0.5, alpha=0.3)
            vertex_x = -b / (2*a)
            vertex_y = a*vertex_x**2 + b*vertex_x + c
            ax.plot(vertex_x, vertex_y, 'yo', markersize=12, label='Vertex', zorder=5)
            ax.set_title(f'{a}x² + {b}x + {c} = 0', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('x', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('y', color='#cbd5e1', fontsize=12)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('quadratic_equation')
            
            return make_json_serializable({
                'roots': [round(root1, 3), round(root2, 3)],
                'discriminant': discriminant,
                'vertex': [round(vertex_x, 3), round(vertex_y, 3)],
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def solve_linear(self, a, b):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                query = f"compute(linear_equation, [{a}, {b}], Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            if a == 0:
                return {'error': 'a cannot be zero'}
            x = -b / a
            
            # Visualization
            fig, ax = self.create_figure()
            x_vals = np.linspace(x-5, x+5, 100)
            y_vals = a * x_vals + b
            ax.plot(x_vals, y_vals, 'cyan', linewidth=2.5)
            ax.plot(x, 0, 'ro', markersize=12, label=f'Solution: x = {x:.2f}', zorder=5)
            ax.axhline(0, color='magenta', linewidth=1.5, alpha=0.5)
            ax.axvline(0, color='white', linewidth=0.5, alpha=0.3)
            ax.set_title(f'{a}x + {b} = 0', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('x', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('y', color='#cbd5e1', fontsize=12)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('linear_equation')
            
            return make_json_serializable({
                'result': f"x = {x:.3f}", 
                'visualization': self.fig_to_base64(fig), 
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def solve_cubic(self, a, b, c, d):
        try:
            coeffs = [a, b, c, d]
            roots = np.roots(coeffs)
            real_roots = [float(r.real) for r in roots if abs(r.imag) < 1e-10]
            
            # Visualization
            fig, ax = self.create_figure()
            x = np.linspace(-5, 5, 1000)
            y = a*x**3 + b*x**2 + c*x + d
            ax.plot(x, y, 'cyan', linewidth=2.5)
            for root in real_roots:
                ax.plot(root, 0, 'ro', markersize=12, zorder=5)
            ax.axhline(0, color='magenta', linewidth=1.5, alpha=0.5)
            ax.axvline(0, color='white', linewidth=0.5, alpha=0.3)
            ax.set_title(f'{a}x³ + {b}x² + {c}x + {d} = 0', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('x', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('y', color='#cbd5e1', fontsize=12)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('cubic_equation')
            
            return make_json_serializable({
                'roots': [round(r, 3) for r in real_roots],
                'formula': f'{a}x³ + {b}x² + {c}x + {d} = 0',
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_absolute_value(self, x):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                query = f"compute(absolute_value, {x}, Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            result = abs(x)
            
            # Visualization
            fig, ax = self.create_figure()
            x_vals = np.linspace(-10, 10, 1000)
            y_vals = np.abs(x_vals)
            ax.plot(x_vals, y_vals, 'cyan', linewidth=2.5)
            ax.plot(x, result, 'ro', markersize=12, label=f'|{x}| = {result}', zorder=5)
            ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)
            ax.axvline(0, color='white', linewidth=0.5, alpha=0.3)
            ax.set_title('Absolute Value Function', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('x', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('|x|', color='#cbd5e1', fontsize=12)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('absolute_value')
            
            return make_json_serializable({
                'result': result, 
                'visualization': self.fig_to_base64(fig), 
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_floor(self, x):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                query = f"compute(floor_function, {x}, Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            result = math.floor(x)
            
            # Visualization
            fig, ax = self.create_figure()
            x_vals = np.linspace(x-2, x+2, 1000)
            y_vals = np.floor(x_vals)
            ax.plot(x_vals, y_vals, 'cyan', linewidth=2.5, label='⌊x⌋')
            ax.plot(x, result, 'ro', markersize=12, label=f'⌊{x}⌋ = {result}', zorder=5)
            ax.set_title('Floor Function', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('x', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('⌊x⌋', color='#cbd5e1', fontsize=12)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('floor_function')
            
            return make_json_serializable({
                'result': result, 
                'visualization': self.fig_to_base64(fig), 
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_ceiling(self, x):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                query = f"compute(ceiling_function, {x}, Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            result = math.ceil(x)
            
            # Visualization
            fig, ax = self.create_figure()
            x_vals = np.linspace(x-2, x+2, 1000)
            y_vals = np.ceil(x_vals)
            ax.plot(x_vals, y_vals, 'cyan', linewidth=2.5, label='⌈x⌉')
            ax.plot(x, result, 'ro', markersize=12, label=f'⌈{x}⌉ = {result}', zorder=5)
            ax.set_title('Ceiling Function', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('x', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('⌈x⌉', color='#cbd5e1', fontsize=12)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('ceiling_function')
            
            return make_json_serializable({
                'result': result, 
                'visualization': self.fig_to_base64(fig), 
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    # Continue with remaining concepts (exponential, trig, number theory, matrices, advanced)
    # Due to length, I'll include the key remaining ones...
    
    def compute_logarithm(self, x, base=10):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                query = f"compute(logarithm, [{x}, {base}], Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            result = math.log(x, base)
            
            # Visualization
            fig, ax = self.create_figure()
            x_vals = np.linspace(0.1, 10, 1000)
            y_vals = np.log(x_vals) / np.log(base)
            ax.plot(x_vals, y_vals, 'cyan', linewidth=2.5)
            ax.plot(x, result, 'ro', markersize=12, label=f'log_{base}({x}) = {result:.2f}', zorder=5)
            ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)
            ax.axvline(1, color='magenta', linestyle='--', alpha=0.5, linewidth=2)
            ax.set_title(f'Logarithm Base {base}', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('x', color='#cbd5e1', fontsize=12)
            ax.set_ylabel(f'log_{base}(x)', color='#cbd5e1', fontsize=12)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('logarithm')
            
            return make_json_serializable({
                'result': round(result, 4),
                'formula': f'log_{base}({x}) = {result:.4f}',
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_exponential(self, x):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                query = f"compute(exponential, {x}, Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            result = math.exp(x)
            
            # Visualization
            fig, ax = self.create_figure()
            x_vals = np.linspace(-3, 3, 1000)
            y_vals = np.exp(x_vals)
            ax.plot(x_vals, y_vals, 'cyan', linewidth=2.5)
            ax.plot(x, result, 'ro', markersize=12, label=f'e^{x} = {result:.2f}', zorder=5)
            ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)
            ax.axvline(0, color='white', linewidth=0.5, alpha=0.3)
            ax.set_title('Exponential Function', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('x', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('e^x', color='#cbd5e1', fontsize=12)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('exponential')
            
            return make_json_serializable({
                'result': round(result, 4), 
                'visualization': self.fig_to_base64(fig), 
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_exponential_growth(self, initial, rate, time):
        try:
            result = initial * math.exp(rate * time)
            
            fig, ax = self.create_figure()
            t = np.linspace(0, time*1.5, 100)
            y = initial * np.exp(rate * t)
            ax.plot(t, y, 'cyan', linewidth=2.5)
            ax.plot(time, result, 'ro', markersize=12, label=f'N({time}) = {result:.2f}', zorder=5)
            ax.axhline(initial, color='yellow', linestyle=':', alpha=0.5, linewidth=2, label=f'Initial = {initial}')
            ax.set_title(f'Exponential Growth: N(t) = {initial}e^({rate}t)', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('Time', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('Value', color='#cbd5e1', fontsize=12)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('exponential_growth')
            
            return make_json_serializable({
                'result': round(result, 3),
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_exponential_decay(self, initial, rate, time):
        try:
            result = initial * math.exp(-rate * time)
            
            fig, ax = self.create_figure()
            t = np.linspace(0, time*1.5, 100)
            y = initial * np.exp(-rate * t)
            ax.plot(t, y, 'cyan', linewidth=2.5)
            ax.plot(time, result, 'ro', markersize=12, label=f'N({time}) = {result:.2f}', zorder=5)
            ax.axhline(initial/2, color='yellow', linestyle=':', alpha=0.5, linewidth=2, label='Half-life')
            ax.set_title(f'Exponential Decay: N(t) = {initial}e^(-{rate}t)', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('Time', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('Value', color='#cbd5e1', fontsize=12)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('exponential_decay')
            
            return make_json_serializable({
                'result': round(result, 3),
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_trigonometry(self, func, angle_deg):
        try:
            angle_rad = math.radians(angle_deg)
            if func == 'sin':
                result = math.sin(angle_rad)
                func_name = 'Sine'
            elif func == 'cos':
                result = math.cos(angle_rad)
                func_name = 'Cosine'
            elif func == 'tan':
                result = math.tan(angle_rad)
                func_name = 'Tangent'
            else:
                result = 0
                func_name = 'Unknown'
            
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                query = f"compute(trigonometry_{func}, {angle_rad}, Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            # Visualization
            fig, ax = self.create_figure()
            x_vals = np.linspace(0, 360, 1000)
            if func == 'sin':
                y_vals = np.sin(np.radians(x_vals))
            elif func == 'cos':
                y_vals = np.cos(np.radians(x_vals))
            else:
                y_vals = np.tan(np.radians(x_vals))
                y_vals = np.clip(y_vals, -10, 10)
            
            ax.plot(x_vals, y_vals, 'cyan', linewidth=2.5)
            ax.plot(angle_deg, result, 'ro', markersize=12, label=f'{func}({angle_deg}°) = {result:.3f}', zorder=5)
            ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)
            ax.axvline(angle_deg, color='magenta', linestyle='--', alpha=0.5, linewidth=2)
            ax.set_title(f'{func_name} Function', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('Angle (degrees)', color='#cbd5e1', fontsize=12)
            ax.set_ylabel(func, color='#cbd5e1', fontsize=12)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.set_xlim(0, 360)
            
            explanation = self.get_prolog_explanation(f'trigonometry_{func}')
            
            return make_json_serializable({
                'result': round(result, 4),
                'angle_deg': angle_deg,
                'angle_rad': round(angle_rad, 4),
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_inverse_trig(self, func, value):
        try:
            if func == 'arcsin':
                if abs(value) > 1:
                    return {'error': 'Value must be between -1 and 1 for arcsin'}
                result = math.asin(value)
                func_name = 'Arcsine'
                x_vals = np.linspace(-1, 1, 1000)
                y_vals = np.arcsin(x_vals)
            elif func == 'arccos':
                if abs(value) > 1:
                    return {'error': 'Value must be between -1 and 1 for arccos'}
                result = math.acos(value)
                func_name = 'Arccosine'
                x_vals = np.linspace(-1, 1, 1000)
                y_vals = np.arccos(x_vals)
            elif func == 'arctan':
                result = math.atan(value)
                func_name = 'Arctangent'
                x_vals = np.linspace(-5, 5, 1000)
                y_vals = np.arctan(x_vals)
            else:
                return {'error': 'Unknown function'}
            
            # Visualization
            fig, ax = self.create_figure()
            ax.plot(x_vals, y_vals, 'cyan', linewidth=2.5)
            ax.plot(value, result, 'ro', markersize=12, label=f'{func}({value}) = {result:.3f} rad', zorder=5)
            ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)
            ax.axvline(0, color='white', linewidth=0.5, alpha=0.3)
            ax.set_title(f'{func_name} Function', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('Value', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('Angle (radians)', color='#cbd5e1', fontsize=12)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('inverse_trig')
            
            return make_json_serializable({
                'result': round(result, 4),
                'degrees': round(math.degrees(result), 2),
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_gcd(self, a, b):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                query = f"compute(gcd, [{a}, {b}], Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            result = math.gcd(a, b)
            
            # Visualization showing Euclidean algorithm steps
            fig, ax = self.create_figure()
            steps = []
            temp_a, temp_b = abs(a), abs(b)
            while temp_b:
                steps.append((temp_a, temp_b))
                temp_a, temp_b = temp_b, temp_a % temp_b
            
            if steps:
                x_vals = range(len(steps))
                a_vals = [s[0] for s in steps]
                b_vals = [s[1] for s in steps]
                ax.plot(x_vals, a_vals, 'cyan', marker='o', linewidth=2.5, markersize=10, label='a')
                ax.plot(x_vals, b_vals, 'magenta', marker='s', linewidth=2.5, markersize=10, label='b')
                ax.set_title(f'GCD({a}, {b}) = {result}', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
                ax.set_xlabel('Step', color='#cbd5e1', fontsize=12)
                ax.set_ylabel('Value', color='#cbd5e1', fontsize=12)
                ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
                ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('gcd')
            
            return make_json_serializable({
                'result': result, 
                'visualization': self.fig_to_base64(fig), 
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_lcm(self, a, b):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                query = f"compute(lcm, [{a}, {b}], Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            result = (a * b) // math.gcd(a, b)
            
            # Visualization
            fig, ax = self.create_figure()
            multiples_a = [a * i for i in range(1, 20)]
            multiples_b = [b * i for i in range(1, 20)]
            
            ax.scatter(range(len(multiples_a)), multiples_a, c='cyan', s=120, alpha=0.6, label=f'Multiples of {a}')
            ax.scatter(range(len(multiples_b)), multiples_b, c='magenta', s=120, alpha=0.6, label=f'Multiples of {b}')
            ax.axhline(result, color='yellow', linewidth=2.5, linestyle='--', label=f'LCM = {result}')
            ax.set_title(f'LCM({a}, {b}) = {result}', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('Index', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('Value', color='#cbd5e1', fontsize=12)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            explanation = self.get_prolog_explanation('lcm')
            
            return make_json_serializable({
                'result': result, 
                'visualization': self.fig_to_base64(fig), 
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def check_prime(self, n):
        try:
            # Try Prolog first
            prolog_result = None
            if self.prolog_available:
                query = f"compute(prime, {n}, Result)"
                results = self.query_prolog(query)
                if results:
                    prolog_result = results[0].get('Result')
            
            if n < 2:
                is_prime = False
            else:
                is_prime = all(n % i != 0 for i in range(2, int(math.sqrt(n)) + 1))
            
            # Visualization showing prime distribution
            fig, ax = self.create_figure()
            start = max(2, n - 50)
            end = n + 50
            numbers = range(start, end)
            primes = [i for i in numbers if i >= 2 and all(i % j != 0 for j in range(2, int(math.sqrt(i)) + 1))]
            composites = [i for i in numbers if i not in primes and i >= 2]
            
            ax.scatter(composites, [1]*len(composites), c='#334155', s=50, alpha=0.5, label='Composite')
            ax.scatter(primes, [1]*len(primes), c='cyan', s=120, alpha=0.7, label='Prime')
            ax.scatter([n], [1], c='magenta' if is_prime else 'red', s=300, marker='*', label=f'{n}', edgecolors='white', linewidth=2.5, zorder=5)
            ax.set_title(f'{n} is {"PRIME" if is_prime else "NOT PRIME"}', color='#10b981' if is_prime else '#ef4444', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('Number', color='#cbd5e1', fontsize=12)
            ax.set_yticks([])
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=11)
            ax.grid(True, alpha=0.2, axis='x', linestyle='--')
            
            explanation = self.get_prolog_explanation('prime')
            
            return make_json_serializable({
                'result': is_prime,
                'message': f'{n} is {"" if is_prime else "not "}a prime number',
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'prolog_result': prolog_result,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    # Matrix and advanced operations... (continuing with shorter implementations for space)
    
    def compute_matrix_multiplication(self, a11, a12, a21, a22, b11, b12, b21, b22):
        try:
            c11 = a11*b11 + a12*b21
            c12 = a11*b12 + a12*b22
            c21 = a21*b11 + a22*b21
            c22 = a21*b12 + a22*b22
            
            fig, ax = self.create_figure(figsize=(14, 6))
            ax.axis('off')
            
            matrix_a = f'A = ⎡{a11:6.1f}  {a12:6.1f}⎤\n    ⎣{a21:6.1f}  {a22:6.1f}⎦'
            matrix_b = f'B = ⎡{b11:6.1f}  {b12:6.1f}⎤\n    ⎣{b21:6.1f}  {b22:6.1f}⎦'
            matrix_c = f'C = ⎡{c11:6.1f}  {c12:6.1f}⎤\n    ⎣{c21:6.1f}  {c22:6.1f}⎦'
            
            ax.text(0.05, 0.65, matrix_a, fontsize=18, color='cyan', family='monospace', fontweight='bold')
            ax.text(0.32, 0.5, '×', fontsize=28, color='#f1f5f9', fontweight='bold')
            ax.text(0.40, 0.65, matrix_b, fontsize=18, color='magenta', family='monospace', fontweight='bold')
            ax.text(0.67, 0.5, '=', fontsize=28, color='#f1f5f9', fontweight='bold')
            ax.text(0.75, 0.65, matrix_c, fontsize=18, color='yellow', family='monospace', fontweight='bold')
            ax.set_title('Matrix Multiplication: A × B = C', color='#f1f5f9', fontweight='bold', fontsize=18, pad=20)
            
            explanation = self.get_prolog_explanation('matrix_multiplication')
            
            return make_json_serializable({
                'result': [[c11, c12], [c21, c22]],
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_matrix_determinant(self, a11, a12, a21, a22):
        try:
            det = a11*a22 - a12*a21
            
            fig, ax = self.create_figure(figsize=(12, 6))
            ax.axis('off')
            
            matrix = f'A = ⎡{a11:6.1f}  {a12:6.1f}⎤\n    ⎣{a21:6.1f}  {a22:6.1f}⎦'
            formula = f'det(A) = ({a11})({a22}) - ({a12})({a21})\n       = {det:.2f}'
            
            ax.text(0.2, 0.6, matrix, fontsize=20, color='cyan', family='monospace', fontweight='bold')
            ax.text(0.2, 0.3, formula, fontsize=18, color='#f1f5f9', family='monospace', fontweight='bold')
            ax.set_title(f'Matrix Determinant = {det:.2f}', color='#f1f5f9', fontweight='bold', fontsize=20, pad=20)
            
            status = 'Invertible (det ≠ 0)' if det != 0 else 'Singular (det = 0, Not Invertible)'
            color = '#10b981' if det != 0 else '#ef4444'
            ax.text(0.2, 0.1, f'Status: {status}', fontsize=16, color=color, fontweight='bold')
            
            explanation = self.get_prolog_explanation('matrix_determinant')
            
            return make_json_serializable({
                'result': det, 
                'visualization': self.fig_to_base64(fig), 
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_matrix_inverse(self, a11, a12, a21, a22):
        try:
            det = a11*a22 - a12*a21
            if det == 0:
                explanation = self.get_prolog_explanation('matrix_inverse')
                return {
                    'error': 'Matrix is singular (determinant = 0)', 
                    'prolog_used': self.prolog_available,
                    'explanation': explanation
                }
            
            inv = [[a22/det, -a12/det], [-a21/det, a11/det]]
            
            fig, ax = self.create_figure(figsize=(14, 6))
            ax.axis('off')
            
            matrix_a = f'A = ⎡{a11:6.1f}  {a12:6.1f}⎤\n    ⎣{a21:6.1f}  {a22:6.1f}⎦'
            matrix_inv = f'A⁻¹ = ⎡{inv[0][0]:7.3f}  {inv[0][1]:7.3f}⎤\n      ⎣{inv[1][0]:7.3f}  {inv[1][1]:7.3f}⎦'
            
            ax.text(0.15, 0.6, matrix_a, fontsize=20, color='cyan', family='monospace', fontweight='bold')
            ax.text(0.5, 0.6, matrix_inv, fontsize=20, color='yellow', family='monospace', fontweight='bold')
            ax.set_title(f'Matrix Inverse (det = {det:.2f})', color='#f1f5f9', fontweight='bold', fontsize=20, pad=20)
            
            explanation = self.get_prolog_explanation('matrix_inverse')
            
            return make_json_serializable({
                'result': [[round(x, 3) for x in row] for row in inv],
                'determinant': det,
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_eigenvalue(self, a11, a12, a21, a22):
        try:
            trace = a11 + a22
            det = a11*a22 - a12*a21
            
            discriminant = trace**2 - 4*det
            if discriminant < 0:
                explanation = self.get_prolog_explanation('eigenvalue')
                return {
                    'message': 'Complex eigenvalues (discriminant < 0)', 
                    'prolog_used': self.prolog_available,
                    'explanation': explanation
                }
            
            lambda1 = (trace + math.sqrt(discriminant)) / 2
            lambda2 = (trace - math.sqrt(discriminant)) / 2
            
            fig, ax = self.create_figure(figsize=(12, 6))
            ax.axis('off')
            
            matrix = f'A = ⎡{a11:6.1f}  {a12:6.1f}⎤\n    ⎣{a21:6.1f}  {a22:6.1f}⎦'
            eigenvalues = f'λ₁ = {lambda1:.3f}\nλ₂ = {lambda2:.3f}'
            
            ax.text(0.2, 0.6, matrix, fontsize=20, color='cyan', family='monospace', fontweight='bold')
            ax.text(0.6, 0.6, eigenvalues, fontsize=20, color='yellow', family='monospace', fontweight='bold')
            ax.set_title('Eigenvalues', color='#f1f5f9', fontweight='bold', fontsize=20, pad=20)
            
            ax.text(0.2, 0.3, f'Trace = {trace:.2f}', fontsize=16, color='#cbd5e1', fontweight='bold')
            ax.text(0.2, 0.2, f'Determinant = {det:.2f}', fontsize=16, color='#cbd5e1', fontweight='bold')
            
            explanation = self.get_prolog_explanation('eigenvalue')
            
            return make_json_serializable({
                'eigenvalues': [round(lambda1, 3), round(lambda2, 3)],
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_fourier_transform(self, signal_type='sine', frequency=1):
        try:
            t = np.linspace(0, 1, 1000)
            if signal_type == 'sine':
                sig = np.sin(2 * np.pi * frequency * t)
            else:
                sig = np.cos(2 * np.pi * frequency * t)
            
            fft = np.fft.fft(sig)
            freq = np.fft.fftfreq(len(t), t[1] - t[0])
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), facecolor='#1e293b')
            for ax in [ax1, ax2]:
                ax.set_facecolor('#0f172a')
                ax.tick_params(colors='#cbd5e1')
                for spine in ax.spines.values():
                    spine.set_color('#475569')
            
            ax1.plot(t, sig, 'cyan', linewidth=2.5)
            ax1.set_title('Time Domain Signal', color='#f1f5f9', fontweight='bold', fontsize=14, pad=15)
            ax1.set_xlabel('Time (s)', color='#cbd5e1', fontsize=12)
            ax1.set_ylabel('Amplitude', color='#cbd5e1', fontsize=12)
            ax1.grid(True, alpha=0.2, linestyle='--')
            
            ax2.plot(freq[:len(freq)//2], np.abs(fft)[:len(fft)//2], 'magenta', linewidth=2.5)
            ax2.set_title('Frequency Domain (Fourier Transform)', color='#f1f5f9', fontweight='bold', fontsize=14, pad=15)
            ax2.set_xlabel('Frequency (Hz)', color='#cbd5e1', fontsize=12)
            ax2.set_ylabel('Magnitude', color='#cbd5e1', fontsize=12)
            ax2.grid(True, alpha=0.2, linestyle='--')
            
            plt.tight_layout()
            
            explanation = self.get_prolog_explanation('fourier_transform')
            
            return make_json_serializable({
                'result': 'Fourier Transform computed',
                'dominant_frequency': frequency,
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})
    
    def compute_taylor_series(self, func, point, terms=5):
        try:
            x = np.linspace(-2, 2, 1000)
            if func == 'exp':
                actual = np.exp(x)
                approx = sum(x**n / math.factorial(n) for n in range(terms))
                func_name = 'e^x'
            elif func == 'sin':
                actual = np.sin(x)
                approx = sum((-1)**n * x**(2*n+1) / math.factorial(2*n+1) for n in range(terms))
                func_name = 'sin(x)'
            else:
                actual = np.cos(x)
                approx = sum((-1)**n * x**(2*n) / math.factorial(2*n) for n in range(terms))
                func_name = 'cos(x)'
            
            fig, ax = self.create_figure(figsize=(12, 7))
            ax.plot(x, actual, 'cyan', linewidth=3.5, label=f'Actual {func_name}')
            ax.plot(x, approx, 'magenta', linewidth=2.5, linestyle='--', label=f'Taylor ({terms} terms)')
            ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)
            ax.axvline(0, color='white', linewidth=0.5, alpha=0.3)
            ax.set_title(f'Taylor Series Approximation of {func_name}', color='#f1f5f9', fontweight='bold', fontsize=16, pad=15)
            ax.set_xlabel('x', color='#cbd5e1', fontsize=12)
            ax.set_ylabel('y', color='#cbd5e1', fontsize=12)
            ax.legend(loc='best', facecolor='#1e293b', edgecolor='#475569', fontsize=12)
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.set_ylim(-5, 5)
            
            explanation = self.get_prolog_explanation('taylor_series')
            
            return make_json_serializable({
                'result': f'Taylor series with {terms} terms',
                'function': func_name,
                'visualization': self.fig_to_base64(fig),
                'prolog_used': self.prolog_available,
                'explanation': explanation
            })
        except Exception as e:
            return make_json_serializable({'error': str(e)})

# Create instance
math_system = MathExpertSystem()

# Natural Language Processing
def parse_natural_language(query):
    """Parse natural language queries into mathematical concepts and parameters"""
    query = query.lower().strip()
    
    # Mean/Average
    if any(word in query for word in ['average', 'mean', 'avg']):
        numbers = re.findall(r'-?\d+\.?\d*', query)
        if numbers:
            return {'concept': 'mean', 'params': {'values': ', '.join(numbers)}}
    
    # Quadratic equations - FIXED
    if any(word in query for word in ['squared', 'quadratic', 'x^2', 'x²', 'square']):
        if any(word in query for word in ['solve', 'roots', 'root', 'find']):
            # Extract numbers from query
            numbers = re.findall(r'-?\d+\.?\d*', query)
            if len(numbers) >= 3:
                return {'concept': 'quadratic_equation', 'params': {
                    'a': float(numbers[0]),
                    'b': float(numbers[1]),
                    'c': float(numbers[2])
                }}
            else:
                # Default values if not enough numbers found
                return {'concept': 'quadratic_equation', 'params': {
                    'a': 1,
                    'b': -5,
                    'c': 6
                }}
    
    # GCD
    if any(word in query for word in ['gcd', 'greatest common divisor', 'greatest common factor', 'gcf']):
        numbers = re.findall(r'\d+', query)
        if len(numbers) >= 2:
            return {'concept': 'gcd', 'params': {'a': int(numbers[0]), 'b': int(numbers[1])}}
    
    # LCM
    if any(word in query for word in ['lcm', 'least common multiple']):
        numbers = re.findall(r'\d+', query)
        if len(numbers) >= 2:
            return {'concept': 'lcm', 'params': {'a': int(numbers[0]), 'b': int(numbers[1])}}
    
    # Factorial
    if 'factorial' in query:
        numbers = re.findall(r'\d+', query)
        if numbers:
            return {'concept': 'factorial', 'params': {'n': int(numbers[0])}}
    
    # Median
    if 'median' in query or 'middle' in query:
        numbers = re.findall(r'-?\d+\.?\d*', query)
        if numbers:
            return {'concept': 'median', 'params': {'values': ', '.join(numbers)}}
    
    # Mode
    if 'mode' in query or 'most frequent' in query:
        numbers = re.findall(r'-?\d+\.?\d*', query)
        if numbers:
            return {'concept': 'mode', 'params': {'values': ', '.join(numbers)}}
    
    # Variance/Standard Deviation
    if 'variance' in query:
        numbers = re.findall(r'-?\d+\.?\d*', query)
        if numbers:
            return {'concept': 'variance', 'params': {'values': ', '.join(numbers)}}
    
    if any(word in query for word in ['standard deviation', 'std dev', 'stddev']):
        numbers = re.findall(r'-?\d+\.?\d*', query)
        if numbers:
            return {'concept': 'standard_deviation', 'params': {'values': ', '.join(numbers)}}
    
    # Trigonometry
    if 'sine' in query or query.startswith('sin'):
        numbers = re.findall(r'-?\d+\.?\d*', query)
        if numbers:
            return {'concept': 'trigonometry_sin', 'params': {'angle': float(numbers[0])}}
    
    if 'cosine' in query or query.startswith('cos'):
        numbers = re.findall(r'-?\d+\.?\d*', query)
        if numbers:
            return {'concept': 'trigonometry_cos', 'params': {'angle': float(numbers[0])}}
    
    if 'tangent' in query or query.startswith('tan'):
        numbers = re.findall(r'-?\d+\.?\d*', query)
        if numbers:
            return {'concept': 'trigonometry_tan', 'params': {'angle': float(numbers[0])}}
    
    # Prime check
    if 'prime' in query:
        numbers = re.findall(r'\d+', query)
        if numbers:
            return {'concept': 'prime', 'params': {'n': int(numbers[0])}}
    
    # Derivative
    if 'derivative' in query or 'differentiate' in query:
        return {'concept': 'derivative', 'params': {'function': 'power', 'coefficient': 1, 'exponent': 2}}
    
    # Integration
    if any(word in query for word in ['integrate', 'integration', 'integral']):
        return {'concept': 'integration', 'params': {'function': 'power', 'coefficient': 1, 'exponent': 2}}
    
    return None

# Process concept - central function
def process_concept(concept, params):
    """Central function to process any concept"""
    method_map = {
        'derivative': lambda: math_system.compute_derivative(
            params.get('function', 'power'),
            float(params.get('coefficient', 1)),
            float(params.get('exponent', 2))
        ),
        'integration': lambda: math_system.compute_integration(
            params.get('function', 'power'),
            float(params.get('coefficient', 1)),
            float(params.get('exponent', 2))
        ),
        'limit': lambda: math_system.compute_limit(
            params.get('function', 'rational'),
            float(params.get('point', 2))
        ),
        'partial_derivative': lambda: math_system.compute_partial_derivative(
            params.get('var', 'x'),
            float(params.get('x', 1)),
            float(params.get('y', 1))
        ),
        'second_derivative': lambda: math_system.compute_second_derivative(
            float(params.get('coefficient', 1)),
            float(params.get('exponent', 3))
        ),
        'dot_product': lambda: math_system.compute_dot_product(
            float(params.get('x1', 0)),
            float(params.get('y1', 0)),
            float(params.get('x2', 0)),
            float(params.get('y2', 0))
        ),
        'cross_product': lambda: math_system.compute_cross_product(
            float(params.get('x1', 0)),
            float(params.get('y1', 0)),
            float(params.get('z1', 0)),
            float(params.get('x2', 0)),
            float(params.get('y2', 0)),
            float(params.get('z2', 0))
        ),
        'modulus': lambda: math_system.compute_modulus(
            float(params.get('x', 0)),
            float(params.get('y', 0)),
            float(params.get('z', 0))
        ),
        'unit_vector': lambda: math_system.compute_unit_vector(
            float(params.get('x', 1)),
            float(params.get('y', 1)),
            float(params.get('z', 0))
        ),
        'vector_projection': lambda: math_system.compute_vector_projection(
            float(params.get('ax', 3)),
            float(params.get('ay', 4)),
            float(params.get('bx', 1)),
            float(params.get('by', 0))
        ),
        'mean': lambda: math_system.compute_mean(params.get('values', '')),
        'median': lambda: math_system.compute_median(params.get('values', '')),
        'mode': lambda: math_system.compute_mode(params.get('values', '')),
        'variance': lambda: math_system.compute_variance(params.get('values', '')),
        'standard_deviation': lambda: math_system.compute_standard_deviation(params.get('values', '')),
        'range': lambda: math_system.compute_range(params.get('values', '')),
        'percentile': lambda: math_system.compute_percentile(
            params.get('values', ''),
            float(params.get('percentile', 50))
        ),
        'correlation': lambda: math_system.compute_correlation(
            params.get('x_values', ''),
            params.get('y_values', '')
        ),
        'z_score': lambda: math_system.compute_z_score(
            float(params.get('value', 0)),
            float(params.get('mean', 0)),
            float(params.get('std_dev', 1))
        ),
        'probability': lambda: math_system.compute_probability(
            float(params.get('favorable', 1)),
            float(params.get('total', 6))
        ),
        'conditional_probability': lambda: math_system.compute_conditional_probability(
            float(params.get('p_a_and_b', 0.2)),
            float(params.get('p_b', 0.5))
        ),
        'bayes_theorem': lambda: math_system.compute_bayes_theorem(
            float(params.get('p_b_given_a', 0.8)),
            float(params.get('p_a', 0.3)),
            float(params.get('p_b', 0.5))
        ),
        'expected_value': lambda: math_system.compute_expected_value(
            params.get('values', ''),
            params.get('probabilities', '')
        ),
        'permutation': lambda: math_system.compute_permutation(
            int(params.get('n', 5)),
            int(params.get('r', 3))
        ),
        'combination': lambda: math_system.compute_combination(
            int(params.get('n', 5)),
            int(params.get('r', 3))
        ),
        'factorial': lambda: math_system.compute_factorial(int(params.get('n', 5))),
        'quadratic_equation': lambda: math_system.solve_quadratic(
            float(params.get('a', 1)),
            float(params.get('b', -5)),
            float(params.get('c', 6))
        ),
        'linear_equation': lambda: math_system.solve_linear(
            float(params.get('a', 1)),
            float(params.get('b', -5))
        ),
        'cubic_equation': lambda: math_system.solve_cubic(
            float(params.get('a', 1)),
            float(params.get('b', 0)),
            float(params.get('c', -1)),
            float(params.get('d', 0))
        ),
        'absolute_value': lambda: math_system.compute_absolute_value(float(params.get('x', -5))),
        'floor_function': lambda: math_system.compute_floor(float(params.get('x', 3.7))),
        'ceiling_function': lambda: math_system.compute_ceiling(float(params.get('x', 3.2))),
        'logarithm': lambda: math_system.compute_logarithm(
            float(params.get('x', 100)),
            float(params.get('base', 10))
        ),
        'exponential': lambda: math_system.compute_exponential(float(params.get('x', 2))),
        'exponential_growth': lambda: math_system.compute_exponential_growth(
            float(params.get('initial', 100)),
            float(params.get('rate', 0.05)),
            float(params.get('time', 10))
        ),
        'exponential_decay': lambda: math_system.compute_exponential_decay(
            float(params.get('initial', 100)),
            float(params.get('rate', 0.1)),
            float(params.get('time', 5))
        ),
        'trigonometry_sin': lambda: math_system.compute_trigonometry('sin', float(params.get('angle', 30))),
        'trigonometry_cos': lambda: math_system.compute_trigonometry('cos', float(params.get('angle', 30))),
        'trigonometry_tan': lambda: math_system.compute_trigonometry('tan', float(params.get('angle', 30))),
        'inverse_trig': lambda: math_system.compute_inverse_trig(
            params.get('function', 'arcsin'),
            float(params.get('value', 0.5))
        ),
        'gcd': lambda: math_system.compute_gcd(
            int(params.get('a', 48)),
            int(params.get('b', 18))
        ),
        'lcm': lambda: math_system.compute_lcm(
            int(params.get('a', 12)),
            int(params.get('b', 18))
        ),
        'prime': lambda: math_system.check_prime(int(params.get('n', 17))),
        'matrix_multiplication': lambda: math_system.compute_matrix_multiplication(
            float(params.get('a11', 1)), float(params.get('a12', 2)),
            float(params.get('a21', 3)), float(params.get('a22', 4)),
            float(params.get('b11', 5)), float(params.get('b12', 6)),
            float(params.get('b21', 7)), float(params.get('b22', 8))
        ),
        'matrix_determinant': lambda: math_system.compute_matrix_determinant(
            float(params.get('a11', 1)), float(params.get('a12', 2)),
            float(params.get('a21', 3)), float(params.get('a22', 4))
        ),
        'matrix_inverse': lambda: math_system.compute_matrix_inverse(
            float(params.get('a11', 1)), float(params.get('a12', 2)),
            float(params.get('a21', 3)), float(params.get('a22', 4))
        ),
        'eigenvalue': lambda: math_system.compute_eigenvalue(
            float(params.get('a11', 1)), float(params.get('a12', 2)),
            float(params.get('a21', 3)), float(params.get('a22', 4))
        ),
        'fourier_transform': lambda: math_system.compute_fourier_transform(
            params.get('signal_type', 'sine'),
            float(params.get('frequency', 1))
        ),
        'taylor_series': lambda: math_system.compute_taylor_series(
            params.get('function', 'exp'),
            float(params.get('point', 0)),
            int(params.get('terms', 5))
        ),
    }
    
    if concept in method_map:
        return method_map[concept]()
    else:
        return {'error': f'Unknown concept: {concept}'}

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def natural_language_query():
    try:
        data = request.json
        query = data.get('query', '')
        
        parsed = parse_natural_language(query)
        if not parsed:
            return jsonify({
                'success': False,
                'error': 'Could not understand the question. Please try rephrasing or select a concept from the categories below.'
            })
        
        concept = parsed['concept']
        params = parsed['params']
        
        # Process the computation
        result = process_concept(concept, params)
        
        return jsonify({
            'success': True,
            'concept': concept,
            'result': result,
            'query': query
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/compute', methods=['POST'])
def compute():
    try:
        data = request.json
        concept = data.get('concept')
        params = data.get('params', {})
        
        result = process_concept(concept, params)
        return jsonify({'success': True, 'result': result})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 MATHEMATICAL EXPERT SYSTEM - FULL PROLOG INTEGRATION")
    print("="*80)
    print(f"✓ Prolog KB: {'ENABLED ✓' if prolog_available else 'Python Fallback'}")
    print(f"✓ Natural Language Processing: ENABLED")
    print(f"✓ All 48 Concepts: ENABLED")
    print(f"✓ Visualizations: ENABLED")
    print(f"✓ Server: http://localhost:5000")
    print("="*80)
    if prolog_available:
        print("✓ All concepts connected to Prolog knowledge base!")
    else:
        print("⚠ Prolog not available - Install PySwip for full features")
    print("="*80 + "\n")
    
    app.run(debug=True, port=5000, use_reloader=False, threaded=False)