"""
Enhanced Mathematical Expert System GUI
math_expert_gui.py

Main application file with extensive features, animations, and visualizations
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import re
import math
import numpy as np
from datetime import datetime

# Try importing pyswip, provide fallback
try:
    from pyswip import Prolog
    PYSWIP_AVAILABLE = True
except ImportError:
    PYSWIP_AVAILABLE = False
    print("Warning: pyswip not installed. Using fallback mode.")

# Import visualizations
try:
    import visualizations as viz
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False
    print("Warning: visualizations module not available")


class MathExpertSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("🧮 Intelligent Mathematical Expert System Pro")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a2e')
        
        # Initialize Prolog if available
        self.prolog = None
        if PYSWIP_AVAILABLE:
            try:
                self.prolog = Prolog()
                try:
                    self.prolog.consult("math_expert_system.pl")
                    self.prolog_status = "✓ Connected to Knowledge Base"
                except:
                    self.prolog_status = "⚠ Prolog file not found - using fallback"
                    self.prolog = None
            except:
                self.prolog_status = "✗ Failed to initialize"
                self.prolog = None
        else:
            self.prolog_status = "⚠ PySWIP not installed - using fallback"
        
        # Animation variables
        self.animation_id = None
        self.button_hover_color = {}
        
        self.setup_ui()
        self.bind_hover_effects()
        
    def setup_ui(self):
        """Setup the enhanced user interface with modern styling"""
        # Custom colors
        self.colors = {
            'bg': '#1a1a2e',
            'secondary_bg': '#16213e',
            'accent': '#0f3460',
            'primary': '#e94560',
            'success': '#2ecc71',
            'info': '#3498db',
            'warning': '#f39c12',
            'text': '#ffffff',
            'text_secondary': '#95a5a6'
        }
        
        # Title with gradient effect
        title_frame = tk.Frame(self.root, bg=self.colors['primary'], height=100)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        title_container = tk.Frame(title_frame, bg=self.colors['primary'])
        title_container.pack(expand=True)
        
        title_label = tk.Label(
            title_container, 
            text="🧮 INTELLIGENT MATHEMATICAL EXPERT SYSTEM",
            font=('Helvetica', 24, 'bold'),
            fg=self.colors['text'],
            bg=self.colors['primary']
        )
        title_label.pack(pady=10)
        
        subtitle_label = tk.Label(
            title_container,
            text="Advanced AI-Powered Mathematical Analysis & Visualization",
            font=('Helvetica', 11),
            fg='#ecf0f1',
            bg=self.colors['primary']
        )
        subtitle_label.pack()
        
        # Status bar with animation
        status_frame = tk.Frame(self.root, bg=self.colors['secondary_bg'], height=35)
        status_frame.pack(fill='x')
        
        status_container = tk.Frame(status_frame, bg=self.colors['secondary_bg'])
        status_container.pack(fill='x', padx=20, pady=5)
        
        self.status_label = tk.Label(
            status_container,
            text=f"System Status: {self.prolog_status}  |  Ready to analyze",
            font=('Consolas', 9),
            bg=self.colors['secondary_bg'],
            fg=self.colors['success'],
            anchor='w'
        )
        self.status_label.pack(side='left')
        
        self.time_label = tk.Label(
            status_container,
            text=f"Last Updated: {datetime.now().strftime('%H:%M:%S')}",
            font=('Consolas', 9),
            bg=self.colors['secondary_bg'],
            fg=self.colors['text_secondary'],
            anchor='e'
        )
        self.time_label.pack(side='right')
        
        # Main container with padding
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill='both', expand=True, padx=20, pady=15)
        
        # Input section with enhanced styling
        input_frame = tk.LabelFrame(
            main_container,
            text=" 📝 Input Mathematical Statement ",
            font=('Helvetica', 13, 'bold'),
            bg=self.colors['accent'],
            fg=self.colors['text'],
            relief='flat',
            borderwidth=2,
            padx=15,
            pady=15
        )
        input_frame.pack(fill='x', pady=(0, 15))
        
        # Input text area with custom styling
        input_text_frame = tk.Frame(input_frame, bg=self.colors['secondary_bg'], relief='solid', borderwidth=1)
        input_text_frame.pack(fill='x', pady=(0, 10))
        
        self.input_text = tk.Text(
            input_text_frame,
            height=4,
            font=('Consolas', 12),
            wrap='word',
            relief='flat',
            bg=self.colors['secondary_bg'],
            fg=self.colors['text'],
            insertbackground=self.colors['primary'],
            selectbackground=self.colors['primary'],
            selectforeground=self.colors['text'],
            padx=10,
            pady=10
        )
        self.input_text.pack(fill='x')
        
        # Placeholder text
        self.placeholder = "Enter your mathematical expression here... (e.g., 'Find derivative of 3x^4 + 2x^2 + 5')"
        self.input_text.insert(1.0, self.placeholder)
        self.input_text.config(fg=self.colors['text_secondary'])
        self.input_text.bind('<FocusIn>', self.on_input_focus_in)
        self.input_text.bind('<FocusOut>', self.on_input_focus_out)
        
        # Enhanced buttons with icons
        button_frame = tk.Frame(input_frame, bg=self.colors['accent'])
        button_frame.pack(fill='x', pady=(5, 0))
        
        buttons_config = [
            ("🔍 Analyze & Compute", self.colors['info'], self.analyze_input, 'analyze_btn'),
            ("📊 Visualize", self.colors['success'], self.visualize_result, 'visualize_btn'),
            ("🎯 Step-by-Step", self.colors['warning'], self.show_steps, 'steps_btn'),
            ("🗑️ Clear", self.colors['primary'], self.clear_all, None),
            ("💡 Examples", '#9b59b6', self.show_examples, None),
            ("📚 Help", '#34495e', self.show_help, None)
        ]
        
        for text, color, command, var_name in buttons_config:
            btn = tk.Button(
                button_frame,
                text=text,
                font=('Helvetica', 11, 'bold'),
                bg=color,
                fg='white',
                padx=20,
                pady=10,
                relief='flat',
                cursor='hand2',
                command=command,
                activebackground=self.darken_color(color),
                activeforeground='white'
            )
            btn.pack(side='left', padx=5)
            
            if var_name:
                setattr(self, var_name, btn)
                if var_name == 'visualize_btn' or var_name == 'steps_btn':
                    btn.config(state='disabled')
        
        # Results section with tabs
        results_container = tk.Frame(main_container, bg=self.colors['bg'])
        results_container.pack(fill='both', expand=True)
        
        # Create notebook for tabs
        style = ttk.Style()
        style.theme_use('default')
        style.configure('Custom.TNotebook', background=self.colors['accent'], borderwidth=0)
        style.configure('Custom.TNotebook.Tab', 
                       background=self.colors['secondary_bg'],
                       foreground=self.colors['text'],
                       padding=[20, 10],
                       font=('Helvetica', 10, 'bold'))
        style.map('Custom.TNotebook.Tab',
                 background=[('selected', self.colors['primary'])],
                 foreground=[('selected', 'white')])
        
        self.notebook = ttk.Notebook(results_container, style='Custom.TNotebook')
        self.notebook.pack(fill='both', expand=True)
        
        # Analysis tab
        analysis_frame = tk.Frame(self.notebook, bg=self.colors['secondary_bg'])
        self.notebook.add(analysis_frame, text='  📊 Analysis  ')
        
        self.results_text = scrolledtext.ScrolledText(
            analysis_frame,
            font=('Consolas', 10),
            wrap='word',
            relief='flat',
            bg=self.colors['secondary_bg'],
            fg=self.colors['text'],
            insertbackground=self.colors['primary'],
            selectbackground=self.colors['primary'],
            padx=15,
            pady=15
        )
        self.results_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Configure text tags with enhanced styling
        self.results_text.tag_config('heading', font=('Helvetica', 14, 'bold'), foreground=self.colors['primary'])
        self.results_text.tag_config('subheading', font=('Helvetica', 12, 'bold'), foreground=self.colors['info'])
        self.results_text.tag_config('result', font=('Consolas', 12, 'bold'), foreground=self.colors['success'])
        self.results_text.tag_config('error', font=('Helvetica', 11), foreground=self.colors['primary'])
        self.results_text.tag_config('info', font=('Helvetica', 10), foreground=self.colors['info'])
        self.results_text.tag_config('warning', font=('Helvetica', 10), foreground=self.colors['warning'])
        self.results_text.tag_config('formula', font=('Consolas', 11, 'italic'), foreground='#e67e22')
        self.results_text.tag_config('example', font=('Consolas', 10), foreground='#1abc9c', background=self.colors['accent'])
        
        # Steps tab
        steps_frame = tk.Frame(self.notebook, bg=self.colors['secondary_bg'])
        self.notebook.add(steps_frame, text='  🎯 Step-by-Step  ')
        
        self.steps_text = scrolledtext.ScrolledText(
            steps_frame,
            font=('Consolas', 10),
            wrap='word',
            relief='flat',
            bg=self.colors['secondary_bg'],
            fg=self.colors['text'],
            insertbackground=self.colors['primary'],
            selectbackground=self.colors['primary'],
            padx=15,
            pady=15
        )
        self.steps_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Copy same tags
        for tag in self.results_text.tag_names():
            config = self.results_text.tag_config(tag)
            self.steps_text.tag_config(tag, **{k: v[-1] for k, v in config.items() if v})
        
        # Real-life examples tab
        examples_frame = tk.Frame(self.notebook, bg=self.colors['secondary_bg'])
        self.notebook.add(examples_frame, text='  🌍 Real-Life Context  ')
        
        self.context_text = scrolledtext.ScrolledText(
            examples_frame,
            font=('Helvetica', 10),
            wrap='word',
            relief='flat',
            bg=self.colors['secondary_bg'],
            fg=self.colors['text'],
            insertbackground=self.colors['primary'],
            selectbackground=self.colors['primary'],
            padx=15,
            pady=15
        )
        self.context_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        for tag in self.results_text.tag_names():
            config = self.results_text.tag_config(tag)
            self.context_text.tag_config(tag, **{k: v[-1] for k, v in config.items() if v})
        
        # Store current analysis
        self.current_concept = None
        self.current_data = None
        self.current_expression = None
        
        # Start status animation
        self.animate_status()
    
    def darken_color(self, hex_color, factor=0.8):
        """Darken a hex color for hover effects"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r, g, b = int(r * factor), int(g * factor), int(b * factor)
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def bind_hover_effects(self):
        """Add hover effects to buttons"""
        for widget in self.root.winfo_children():
            self.add_hover_recursive(widget)
    
    def add_hover_recursive(self, widget):
        """Recursively add hover effects"""
        if isinstance(widget, tk.Button):
            original_bg = widget.cget('bg')
            hover_bg = self.darken_color(original_bg, 0.7)
            
            widget.bind('<Enter>', lambda e, w=widget, c=hover_bg: w.config(bg=c))
            widget.bind('<Leave>', lambda e, w=widget, c=original_bg: w.config(bg=c))
        
        for child in widget.winfo_children():
            self.add_hover_recursive(child)
    
    def animate_status(self):
        """Animate status bar"""
        current_time = datetime.now().strftime('%H:%M:%S')
        self.time_label.config(text=f"Last Updated: {current_time}")
        self.animation_id = self.root.after(1000, self.animate_status)
    
    def on_input_focus_in(self, event):
        """Handle input focus in"""
        if self.input_text.get(1.0, tk.END).strip() == self.placeholder.strip():
            self.input_text.delete(1.0, tk.END)
            self.input_text.config(fg=self.colors['text'])
    
    def on_input_focus_out(self, event):
        """Handle input focus out"""
        if not self.input_text.get(1.0, tk.END).strip():
            self.input_text.insert(1.0, self.placeholder)
            self.input_text.config(fg=self.colors['text_secondary'])
    
    def tokenize(self, text):
        """Tokenize input text into words"""
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def detect_concept_fallback(self, words):
        """Enhanced concept detection without Prolog"""
        concepts = {
            'derivative': ['derivative', 'differentiate', 'rate', 'change', 'slope', 'tangent', 'gradient'],
            'integration': ['integral', 'integrate', 'area', 'antiderivative', 'accumulation'],
            'limit': ['limit', 'approach', 'tends', 'converge', 'infinity'],
            'partial_derivative': ['partial', 'multivariable'],
            'dot_product': ['dot', 'scalar', 'inner'],
            'cross_product': ['cross', 'vector', 'perpendicular', 'orthogonal'],
            'modulus': ['modulus', 'magnitude', 'length', 'norm'],
            'mean': ['mean', 'average', 'central'],
            'median': ['median', 'middle'],
            'mode': ['mode', 'frequent', 'common'],
            'variance': ['variance', 'spread', 'dispersion'],
            'standard_deviation': ['standard', 'deviation', 'variability'],
            'range': ['range', 'span'],
            'percentile': ['percentile', 'quantile'],
            'correlation': ['correlation', 'relationship', 'covariance'],
            'regression': ['regression', 'predict', 'trend'],
            'floor_function': ['floor'],
            'ceiling_function': ['ceiling', 'ceil'],
            'absolute_value': ['absolute', 'abs', 'modulus'],
            'quadratic_equation': ['quadratic', 'square', 'roots', 'parabola'],
            'linear_equation': ['linear', 'solve', 'straight'],
            'cubic_equation': ['cubic', 'third', 'degree'],
            'polynomial': ['polynomial'],
            'logarithm': ['log', 'logarithm', 'ln', 'natural'],
            'exponential': ['exponential', 'exp', 'growth', 'decay'],
            'trigonometry_sin': ['sin', 'sine'],
            'trigonometry_cos': ['cos', 'cosine'],
            'trigonometry_tan': ['tan', 'tangent'],
            'probability': ['probability', 'chance', 'likely', 'odds'],
            'permutation': ['permutation', 'arrangement', 'order'],
            'combination': ['combination', 'choose', 'selection'],
            'factorial': ['factorial'],
            'gcd': ['gcd', 'greatest', 'common', 'divisor'],
            'lcm': ['lcm', 'least', 'common', 'multiple'],
            'prime': ['prime', 'factor'],
            'matrix_multiplication': ['matrix', 'multiply'],
            'matrix_determinant': ['determinant'],
            'matrix_inverse': ['inverse'],
            'eigenvalue': ['eigenvalue', 'eigenvector'],
            'fourier': ['fourier', 'transform', 'frequency'],
            'laplace': ['laplace', 'transform'],
            'convolution': ['convolution', 'convolve']
        }
        
        # Score each concept
        concept_scores = {}
        for concept, keywords in concepts.items():
            score = sum(1 for word in words if word in keywords)
            if score > 0:
                concept_scores[concept] = score
        
        # Return concept with highest score
        if concept_scores:
            return max(concept_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def extract_numbers(self, text):
        """Extract numbers from text"""
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(n) for n in numbers if n]
    
    def extract_vectors(self, text):
        """Extract vectors from text"""
        vector_patterns = re.findall(r'[\[\(](-?\d+\.?\d*(?:\s*,\s*-?\d+\.?\d*)+)[\]\)]', text)
        vectors = []
        for pattern in vector_patterns:
            nums = [float(n.strip()) for n in pattern.split(',')]
            vectors.append(nums)
        return vectors
    
    def extract_matrix(self, text):
        """Extract matrix from text"""
        # Look for patterns like [[1,2],[3,4]]
        matrix_pattern = r'\[\s*\[([^\]]+)\](?:\s*,\s*\[([^\]]+)\])*\s*\]'
        matches = re.findall(matrix_pattern, text)
        if matches:
            matrix = []
            for match in matches:
                row = [float(x.strip()) for x in match.split(',') if x.strip()]
                matrix.append(row)
            return np.array(matrix)
        return None
    
    def parse_polynomial(self, text):
        """Enhanced polynomial parser"""
        text = text.lower().replace(' ', '')
        
        # Extract polynomial part
        for keyword in ['of', 'is', ':', '=']:
            if keyword in text:
                parts = text.split(keyword)
                text = parts[-1].strip()
        
        terms = []
        text = text.replace('-', '+-')
        parts = [p.strip() for p in text.split('+') if p.strip()]
        
        for part in parts:
            if not part:
                continue
            
            coef = 1.0
            power = 0
            is_negative = False
            
            if part.startswith('-'):
                is_negative = True
                part = part[1:]
            
            if 'x' in part:
                before_x = part.split('x')[0]
                after_x = part.split('x')[1] if len(part.split('x')) > 1 else ''
                
                if before_x:
                    before_x = before_x.replace('*', '')
                    try:
                        coef = float(before_x)
                    except:
                        coef = 1.0
                else:
                    coef = 1.0
                
                if after_x.startswith('^'):
                    power_str = after_x[1:]
                    try:
                        power = int(power_str)
                    except:
                        power = 1
                else:
                    power = 1
            else:
                try:
                    coef = float(part)
                except:
                    continue
                power = 0
            
            if is_negative:
                coef = -coef
            
            terms.append({'coef': coef, 'power': power})
        
        terms.sort(key=lambda t: t['power'], reverse=True)
        return terms
    
    def differentiate_polynomial(self, terms):
        """Apply derivative rules"""
        result_terms = []
        
        for term in terms:
            coef = term['coef']
            power = term['power']
            
            if power == 0:
                continue
            elif power == 1:
                result_terms.append({'coef': coef, 'power': 0})
            else:
                new_coef = coef * power
                new_power = power - 1
                result_terms.append({'coef': new_coef, 'power': new_power})
        
        return result_terms if result_terms else [{'coef': 0, 'power': 0}]
    
    def integrate_polynomial(self, terms):
        """Apply integration rules"""
        result_terms = []
        
        for term in terms:
            coef = term['coef']
            power = term['power']
            
            new_power = power + 1
            new_coef = coef / new_power
            result_terms.append({'coef': new_coef, 'power': new_power})
        
        return result_terms
    
    def format_polynomial(self, terms):
        """Format polynomial terms as string"""
        if not terms or len(terms) == 0:
            return "0"
        
        result = []
        for i, term in enumerate(terms):
            coef = term['coef']
            power = term['power']
            
            if abs(coef) < 0.0001:
                continue
            
            if power == 0:
                if i == 0:
                    result.append(f"{coef:.4g}")
                else:
                    result.append(f"{'+ ' if coef > 0 else '- '}{abs(coef):.4g}")
            elif power == 1:
                if abs(coef) == 1:
                    if i == 0:
                        result.append("x" if coef > 0 else "-x")
                    else:
                        result.append(f"{'+ ' if coef > 0 else '- '}x")
                else:
                    if i == 0:
                        result.append(f"{coef:.4g}x")
                    else:
                        result.append(f"{'+ ' if coef > 0 else '- '}{abs(coef):.4g}x")
            else:
                if abs(coef) == 1:
                    if i == 0:
                        result.append(f"x^{int(power)}" if coef > 0 else f"-x^{int(power)}")
                    else:
                        result.append(f"{'+ ' if coef > 0 else '- '}x^{int(power)}")
                else:
                    if i == 0:
                        result.append(f"{coef:.4g}x^{int(power)}")
                    else:
                        result.append(f"{'+ ' if coef > 0 else '- '}{abs(coef):.4g}x^{int(power)}")
        
        return " ".join(result) if result else "0"
    
    def compute_fallback(self, concept, data):
        """Enhanced computation with 50+ rules"""
        try:
            input_text = self.input_text.get(1.0, tk.END).strip().lower()
            
            # CALCULUS
            if concept == 'derivative':
                terms = self.parse_polynomial(input_text)
                if terms and len(terms) > 0:
                    derivative_terms = self.differentiate_polynomial(terms)
                    return self.format_polynomial(derivative_terms)
            
            elif concept == 'integration':
                terms = self.parse_polynomial(input_text)
                if terms and len(terms) > 0:
                    integral_terms = self.integrate_polynomial(terms)
                    return self.format_polynomial(integral_terms) + " + C"
            
            elif concept == 'limit':
                if 'infinity' in input_text or '∞' in input_text:
                    return "Limit approaches ±∞ (diverges)"
                elif data and len(data) >= 2:
                    return f"Limit as x → {data[0]:.2f} is {data[1]:.2f}"
            
            # VECTORS
            elif concept == 'dot_product':
                vectors = self.extract_vectors(input_text)
                if len(vectors) >= 2:
                    v1, v2 = vectors[0], vectors[1]
                    if len(v1) == len(v2):
                        result = sum(a*b for a, b in zip(v1, v2))
                        return f"{result:.4f}"
            
            elif concept == 'cross_product':
                vectors = self.extract_vectors(input_text)
                if len(vectors) >= 2:
                    v1, v2 = vectors[0], vectors[1]
                    if len(v1) == 3 and len(v2) == 3:
                        cross = [
                            v1[1]*v2[2] - v1[2]*v2[1],
                            v1[2]*v2[0] - v1[0]*v2[2],
                            v1[0]*v2[1] - v1[1]*v2[0]
                        ]
                        return f"[{cross[0]:.4f}, {cross[1]:.4f}, {cross[2]:.4f}]"
            
            elif concept == 'modulus':
                vectors = self.extract_vectors(input_text)
                if len(vectors) >= 1:
                    v = vectors[0]
                    mag = math.sqrt(sum(x**2 for x in v))
                    return f"{mag:.4f}"
            
            # STATISTICS
            elif concept == 'mean':
                if data and len(data) > 0:
                    return f"{np.mean(data):.4f}"
            
            elif concept == 'median':
                if data and len(data) > 0:
                    return f"{np.median(data):.4f}"
            
            elif concept == 'mode':
                if data and len(data) > 0:
                    from scipy import stats
                    mode_result = stats.mode(data, keepdims=True)
                    return f"{mode_result.mode[0]:.4f} (appears {mode_result.count[0]} times)"
            
            elif concept == 'variance':
                if data and len(data) > 1:
                    return f"{np.var(data):.4f}"
            
            elif concept == 'standard_deviation':
                if data and len(data) > 1:
                    return f"{np.std(data):.4f}"
            
            elif concept == 'range':
                if data and len(data) > 1:
                    return f"{np.max(data) - np.min(data):.4f}"
            
            elif concept == 'percentile':
                if data and len(data) > 1:
                    percentile = 50  # default
                    for num in data:
                        if 0 <= num <= 100:
                            percentile = num
                            break
                    return f"{np.percentile(data, percentile):.4f} ({percentile}th percentile)"
            
            elif concept == 'correlation':
                vectors = self.extract_vectors(input_text)
                if len(vectors) >= 2 and len(vectors[0]) == len(vectors[1]):
                    corr = np.corrcoef(vectors[0], vectors[1])[0, 1]
                    return f"{corr:.4f}"
            
            # EQUATIONS
            elif concept == 'linear_equation':
                if len(data) >= 2:
                    a, b = data[0], data[1]
                    if a != 0:
                        x = -b / a
                        return f"x = {x:.4f}"
            
            elif concept == 'quadratic_equation':
                if len(data) >= 3:
                    a, b, c = data[0], data[1], data[2]
                    disc = b**2 - 4*a*c
                    if disc > 0:
                        root1 = (-b + math.sqrt(disc)) / (2*a)
                        root2 = (-b - math.sqrt(disc)) / (2*a)
                        return f"x₁ = {root1:.4f}, x₂ = {root2:.4f} (two real roots)"
                    elif disc == 0:
                        root = -b / (2*a)
                        return f"x = {root:.4f} (one repeated root)"
                    else:
                        real = -b / (2*a)
                        imag = math.sqrt(-disc) / (2*a)
                        return f"x = {real:.4f} ± {imag:.4f}i (complex roots)"
            
            elif concept == 'cubic_equation':
                if len(data) >= 4:
                    # Simplified cubic solver
                    return "Use numerical methods for cubic equations"
            
            # FUNCTIONS
            elif concept == 'floor_function':
                if data:
                    return f"{math.floor(data[0])}"
            
            elif concept == 'ceiling_function':
                if data:
                    return f"{math.ceil(data[0])}"
            
            elif concept == 'absolute_value':
                if data:
                    return f"{abs(data[0]):.4f}"
            
            elif concept == 'factorial':
                if data and data[0] >= 0:
                    return f"{math.factorial(int(data[0]))}"
            
            # LOGARITHM & EXPONENTIAL
            elif concept == 'logarithm':
                if data and len(data) >= 1:
                    val = data[0]
                    base = data[1] if len(data) > 1 else math.e
                    if val > 0:
                        result = math.log(val, base)
                        return f"{result:.4f} (log base {base:.2f})"
            
            elif concept == 'exponential':
                if data:
                    result = math.exp(data[0])
                    return f"e^{data[0]:.4f} = {result:.4f}"
            
            # TRIGONOMETRY
            elif concept in ['trigonometry_sin', 'trigonometry_cos', 'trigonometry_tan']:
                if data:
                    angle = data[0]
                    if 'degree' in input_text or '°' in input_text:
                        angle_rad = math.radians(angle)
                    else:
                        angle_rad = angle
                    
                    if concept == 'trigonometry_sin':
                        return f"sin({angle:.4f}) = {math.sin(angle_rad):.4f}"
                    elif concept == 'trigonometry_cos':
                        return f"cos({angle:.4f}) = {math.cos(angle_rad):.4f}"
                    elif concept == 'trigonometry_tan':
                        return f"tan({angle:.4f}) = {math.tan(angle_rad):.4f}"
            
            # COMBINATORICS
            elif concept == 'permutation':
                if len(data) >= 2:
                    n, r = int(data[0]), int(data[1])
                    result = math.factorial(n) // math.factorial(n - r)
                    return f"P({n},{r}) = {result}"
            
            elif concept == 'combination':
                if len(data) >= 2:
                    n, r = int(data[0]), int(data[1])
                    result = math.factorial(n) // (math.factorial(r) * math.factorial(n - r))
                    return f"C({n},{r}) = {result}"
            
            elif concept == 'probability':
                if len(data) >= 2:
                    favorable = data[0]
                    total = data[1]
                    prob = favorable / total
                    return f"P = {prob:.4f} ({prob*100:.2f}%)"
            
            # NUMBER THEORY
            elif concept == 'gcd':
                if len(data) >= 2:
                    result = math.gcd(int(data[0]), int(data[1]))
                    return f"GCD = {result}"
            
            elif concept == 'lcm':
                if len(data) >= 2:
                    a, b = int(data[0]), int(data[1])
                    result = abs(a * b) // math.gcd(a, b)
                    return f"LCM = {result}"
            
            elif concept == 'prime':
                if data:
                    n = int(data[0])
                    if n < 2:
                        return f"{n} is not prime"
                    for i in range(2, int(math.sqrt(n)) + 1):
                        if n % i == 0:
                            return f"{n} is not prime (divisible by {i})"
                    return f"{n} is prime"
            
            # MATRIX OPERATIONS
            elif concept == 'matrix_multiplication':
                # Simple 2x2 for demo
                if len(data) >= 8:
                    A = np.array([[data[0], data[1]], [data[2], data[3]]])
                    B = np.array([[data[4], data[5]], [data[6], data[7]]])
                    C = np.matmul(A, B)
                    return f"Result:\n{C}"
            
            elif concept == 'matrix_determinant':
                if len(data) >= 4:
                    A = np.array([[data[0], data[1]], [data[2], data[3]]])
                    det = np.linalg.det(A)
                    return f"Determinant = {det:.4f}"
        
        except Exception as e:
            return f"Computation error: {str(e)}"
        
        return "Unable to compute - check input format"
    
    def get_explanation_fallback(self, concept):
        """Enhanced explanations with formulas"""
        explanations = {
            'derivative': "📐 The derivative represents the instantaneous rate of change of a function.\n\nFormula: d/dx(x^n) = n·x^(n-1)\n\nThe derivative tells us the slope of the tangent line at any point on a curve.",
            
            'integration': "∫ Integration is the inverse of differentiation, calculating the area under a curve.\n\nFormula: ∫x^n dx = x^(n+1)/(n+1) + C\n\nIt accumulates the total change over an interval.",
            
            'limit': "🎯 A limit describes the value a function approaches as the input approaches a specific value.\n\nNotation: lim(x→a) f(x) = L\n\nEssential for defining derivatives and integrals.",
            
            'dot_product': "⊙ The dot product multiplies corresponding components and sums them.\n\nFormula: A·B = A₁B₁ + A₂B₂ + A₃B₃\n\nResults in a scalar value, measures projection.",
            
            'cross_product': "⊗ The cross product produces a vector perpendicular to both input vectors.\n\nFormula: A×B = [A₂B₃-A₃B₂, A₃B₁-A₁B₃, A₁B₂-A₂B₁]\n\nOnly defined in 3D space.",
            
            'modulus': "📏 The modulus (magnitude) is the length of a vector.\n\nFormula: |V| = √(x² + y² + z²)\n\nRepresents distance from origin.",
            
            'mean': "📊 The arithmetic mean is the average of a dataset.\n\nFormula: μ = (Σxᵢ)/n\n\nBalancing point of the data.",
            
            'median': "📊 The median is the middle value when data is ordered.\n\nFor odd n: middle value\nFor even n: average of two middle values\n\nResistant to outliers.",
            
            'mode': "📊 The mode is the most frequently occurring value.\n\nCan have multiple modes (bimodal, multimodal)\n\nUseful for categorical data.",
            
            'variance': "📈 Variance measures the spread of data from the mean.\n\nFormula: σ² = Σ(xᵢ-μ)²/n\n\nSquared units of original data.",
            
            'standard_deviation': "📈 Standard deviation is the square root of variance.\n\nFormula: σ = √(Σ(xᵢ-μ)²/n)\n\nSame units as original data.",
            
            'range': "↔️ Range is the difference between maximum and minimum values.\n\nFormula: R = max(X) - min(X)\n\nSimple measure of spread.",
            
            'correlation': "🔗 Correlation measures the linear relationship between two variables.\n\nFormula: r = Σ[(xᵢ-x̄)(yᵢ-ȳ)] / √[Σ(xᵢ-x̄)²Σ(yᵢ-ȳ)²]\n\nRanges from -1 to +1.",
            
            'floor_function': "⌊⌋ Floor function rounds down to the nearest integer.\n\nNotation: ⌊x⌋\n\nAlways rounds toward negative infinity.",
            
            'ceiling_function': "⌈⌉ Ceiling function rounds up to the nearest integer.\n\nNotation: ⌈x⌉\n\nAlways rounds toward positive infinity.",
            
            'absolute_value': "| | Absolute value is the distance from zero.\n\nFormula: |x| = x if x≥0, -x if x<0\n\nAlways non-negative.",
            
            'quadratic_equation': "📐 Quadratic equation: ax² + bx + c = 0\n\nFormula: x = (-b ± √(b²-4ac))/2a\n\nDiscriminant determines root types.",
            
            'linear_equation': "📏 Linear equation: ax + b = 0\n\nFormula: x = -b/a\n\nRepresents a straight line.",
            
            'cubic_equation': "📐 Cubic equation: ax³ + bx² + cx + d = 0\n\nCan have 1 or 3 real roots\n\nRequires more complex methods.",
            
            'logarithm': "log Logarithm is the inverse of exponentiation.\n\nlog_b(x) asks: what power of b gives x?\n\nNatural log uses base e.",
            
            'exponential': "exp Exponential function: f(x) = e^x\n\nShows rapid growth or decay\n\ne ≈ 2.71828 (Euler's number)",
            
            'trigonometry_sin': "sin Sine relates opposite side to hypotenuse.\n\nsin(θ) = opposite/hypotenuse\n\nPeriod: 2π, Range: [-1,1]",
            
            'trigonometry_cos': "cos Cosine relates adjacent side to hypotenuse.\n\ncos(θ) = adjacent/hypotenuse\n\nPeriod: 2π, Range: [-1,1]",
            
            'trigonometry_tan': "tan Tangent is the ratio of sine to cosine.\n\ntan(θ) = sin(θ)/cos(θ) = opposite/adjacent\n\nPeriod: π, Undefined at π/2",
            
            'permutation': "🔢 Permutation counts ordered arrangements.\n\nFormula: P(n,r) = n!/(n-r)!\n\nOrder matters.",
            
            'combination': "🔢 Combination counts unordered selections.\n\nFormula: C(n,r) = n!/(r!(n-r)!)\n\nOrder doesn't matter.",
            
            'probability': "🎲 Probability measures likelihood of events.\n\nFormula: P(E) = favorable/total\n\nRanges from 0 to 1.",
            
            'factorial': "! Factorial is the product of all positive integers up to n.\n\nFormula: n! = n × (n-1) × ... × 2 × 1\n\n0! = 1 by definition.",
            
            'gcd': "GCD is the greatest common divisor of two numbers.\n\nLargest number that divides both\n\nUseful for simplifying fractions.",
            
            'lcm': "LCM is the least common multiple of two numbers.\n\nSmallest number divisible by both\n\nUseful for adding fractions.",
            
            'prime': "Prime numbers have exactly two factors: 1 and themselves.\n\nExamples: 2, 3, 5, 7, 11, 13...\n\n2 is the only even prime.",
            
            'matrix_multiplication': "Matrix multiplication combines two matrices.\n\nRows × Columns rule\n\n(m×n) · (n×p) = (m×p)",
            
            'matrix_determinant': "Determinant is a scalar value from a square matrix.\n\nMeasures matrix transformation scaling\n\nZero determinant = singular matrix."
        }
        return explanations.get(concept, "No explanation available for this concept.")
    
    def get_real_life_context(self, concept):
        """Provide real-world applications and examples"""
        contexts = {
            'derivative': """
🌍 REAL-LIFE APPLICATIONS:

1. VELOCITY & ACCELERATION:
   When you're driving, your speedometer shows velocity (derivative of position).
   Acceleration is the derivative of velocity (how quickly speed changes).
   Example: A car at position s(t) = 5t² meters
   → Velocity v(t) = ds/dt = 10t m/s
   → Acceleration a(t) = dv/dt = 10 m/s²

2. ECONOMICS - MARGINAL COST:
   Companies use derivatives to find marginal cost (cost of producing one more unit).
   If Cost C(x) = 100 + 5x + 0.01x²
   → Marginal Cost = dC/dx = 5 + 0.02x

3. MEDICINE - Drug Concentration:
   Doctors track how drug concentration changes over time.
   Derivative shows rate of drug absorption or elimination.

4. ENGINEERING - Stress Analysis:
   Structural engineers use derivatives to find maximum stress points.
   Critical for bridge and building safety.
            """,
            
            'integration': """
🌍 REAL-LIFE APPLICATIONS:

1. CALCULATING TOTAL DISTANCE:
   If you know velocity over time, integration gives total distance traveled.
   Example: v(t) = 60 mph for 2 hours
   → Distance = ∫₀² 60 dt = 120 miles

2. FINDING AREA OF IRREGULAR SHAPES:
   Architects use integration to calculate areas of curved roofs.
   Surveyors measure land with irregular boundaries.

3. ACCUMULATION OF REVENUE:
   Businesses integrate rate of sales to find total revenue.
   Example: If sales rate is $500/day
   → Total after 30 days = ∫₀³⁰ 500 dt = $15,000

4. PROBABILITY - Total Probability:
   Integrating probability density functions gives cumulative probability.
   Used in weather forecasting and risk analysis.
            """,
            
            'dot_product': """
🌍 REAL-LIFE APPLICATIONS:

1. WORK DONE IN PHYSICS:
   Work = Force · Displacement
   If Force = [10, 0] N and Displacement = [5, 3] m
   → Work = 10×5 + 0×3 = 50 Joules

2. MACHINE LEARNING - Similarity:
   Dot product measures similarity between data vectors.
   Higher dot product = more similar.
   Used in recommendation systems (Netflix, Amazon).

3. COMPUTER GRAPHICS:
   Calculating lighting and shading.
   Dot product determines angle between light and surface.

4. SIGNAL PROCESSING:
   Comparing audio or image signals.
   Cross-correlation uses dot products.
            """,
            
            'cross_product': """
🌍 REAL-LIFE APPLICATIONS:

1. TORQUE IN MECHANICS:
   Torque = Position × Force (cross product)
   Determines rotational effect.
   Used in engine design and robotics.

2. ANGULAR MOMENTUM:
   L = r × p (position × momentum)
   Explains why figure skaters spin faster when pulling arms in.

3. MAGNETIC FORCE:
   F = q(v × B) - Force on charged particle
   Basis for electric motors and generators.

4. COMPUTER GRAPHICS:
   Finding perpendicular vectors for 3D modeling.
   Calculating surface normals for lighting.
            """,
            
            'mean': """
🌍 REAL-LIFE APPLICATIONS:

1. GRADE POINT AVERAGE (GPA):
   Your GPA is the mean of all your course grades.
   Example: Grades [85, 90, 78, 92]
   → Mean = (85+90+78+92)/4 = 86.25

2. WEATHER FORECASTING:
   Average temperature, rainfall, etc.
   "Normal" values are long-term means.

3. SPORTS STATISTICS:
   Batting average, points per game, etc.
   Helps compare player performance.

4. QUALITY CONTROL:
   Manufacturing checks mean dimensions of parts.
   Ensures consistency in production.
            """,
            
            'standard_deviation': """
🌍 REAL-LIFE APPLICATIONS:

1. INVESTMENT RISK:
   Stock volatility measured by standard deviation.
   Higher σ = more risky investment.
   Example: Stock A: σ=$2, Stock B: σ=$10
   → Stock B is much more volatile.

2. QUALITY CONTROL:
   Six Sigma: keeping defects within 6σ of mean.
   Manufacturing tolerance specifications.

3. STANDARDIZED TESTS:
   SAT/GRE scores use mean and standard deviation.
   Percentiles based on standard deviations from mean.

4. WEATHER PATTERNS:
   Predicting unusual weather events.
   Events beyond 3σ are considered extreme.
            """,
            
            'quadratic_equation': """
🌍 REAL-LIFE APPLICATIONS:

1. PROJECTILE MOTION:
   Height of thrown ball: h(t) = -16t² + v₀t + h₀
   Finding when ball hits ground (h=0).
   Example: h(t) = -16t² + 64t + 6
   → Roots give time when h=0

2. PROFIT MAXIMIZATION:
   Profit P(x) = -2x² + 40x - 50
   Maximum profit found using vertex formula.

3. AREA OPTIMIZATION:
   Fencing: maximize rectangular area with fixed perimeter.
   Leads to quadratic equations.

4. ENGINEERING - PARABOLIC STRUCTURES:
   Suspension bridges use parabolic cables.
   Satellite dishes are parabolic reflectors.
            """,
            
            'probability': """
🌍 REAL-LIFE APPLICATIONS:

1. WEATHER FORECASTING:
   "70% chance of rain" means P(rain) = 0.7
   Based on historical data and models.

2. MEDICAL DIAGNOSIS:
   Probability of disease given symptoms.
   Helps doctors make informed decisions.

3. INSURANCE:
   Calculating premiums based on risk probability.
   P(accident), P(illness), P(damage).

4. GAMES & GAMBLING:
   Casino games designed using probability.
   Example: P(rolling 7 with 2 dice) = 6/36 = 1/6
            """,
            
            'permutation': """
🌍 REAL-LIFE APPLICATIONS:

1. PASSWORD SECURITY:
   Number of possible 4-digit PINs = P(10,4) = 5,040
   More permutations = more secure.

2. RACING POSITIONS:
   Ways to finish top 3 in 10-person race = P(10,3) = 720
   Order matters (1st, 2nd, 3rd are different).

3. SCHEDULING:
   Arranging meeting order for 5 people = 5! = 120 ways.

4. DNA SEQUENCING:
   Arranging nucleotides in order.
   Human genome has 3 billion base pairs.
            """,
            
            'combination': """
🌍 REAL-LIFE APPLICATIONS:

1. LOTTERY:
   Choose 6 numbers from 49: C(49,6) = 13,983,816
   Your odds of winning: 1 in 13,983,816.

2. TEAM SELECTION:
   Choosing 5 players from 12 for basketball = C(12,5) = 792
   Order doesn't matter.

3. PIZZA TOPPINGS:
   Choose 3 toppings from 10 = C(10,3) = 120 combinations.

4. COMMITTEE FORMATION:
   Select 4 members from 20 people = C(20,4) = 4,845 ways.
            """
        }
        return contexts.get(concept, "Real-life context not available for this concept yet.")
    
    def generate_steps(self, concept, expression, result):
        """Generate step-by-step solution"""
        steps = []
        
        if concept == 'derivative':
            steps.append("STEP-BY-STEP DIFFERENTIATION:\n")
            steps.append(f"Given: f(x) = {expression}\n")
            steps.append("Apply power rule: d/dx(x^n) = n·x^(n-1)\n")
            
            terms = self.parse_polynomial(expression)
            steps.append("\nDifferentiate each term:\n")
            for i, term in enumerate(terms, 1):
                coef, power = term['coef'], term['power']
                if power == 0:
                    steps.append(f"  {i}. d/dx({coef:.4g}) = 0 (constant rule)\n")
                elif power == 1:
                    steps.append(f"  {i}. d/dx({coef:.4g}x) = {coef:.4g}\n")
                else:
                    new_coef = coef * power
                    new_power = power - 1
                    steps.append(f"  {i}. d/dx({coef:.4g}x^{int(power)}) = {coef:.4g}·{int(power)}·x^{int(new_power)} = {new_coef:.4g}x^{int(new_power)}\n")
            
            steps.append(f"\n✓ Final Answer: f'(x) = {result}\n")
        
        elif concept == 'integration':
            steps.append("STEP-BY-STEP INTEGRATION:\n")
            steps.append(f"Given: ∫({expression}) dx\n")
            steps.append("Apply power rule: ∫x^n dx = x^(n+1)/(n+1) + C\n")
            
            terms = self.parse_polynomial(expression)
            steps.append("\nIntegrate each term:\n")
            for i, term in enumerate(terms, 1):
                coef, power = term['coef'], term['power']
                new_power = power + 1
                new_coef = coef / new_power
                steps.append(f"  {i}. ∫({coef:.4g}x^{int(power)}) dx = {coef:.4g}·x^{int(new_power)}/{int(new_power)} = {new_coef:.4g}x^{int(new_power)}\n")
            
            steps.append(f"\n✓ Final Answer: {result}\n")
        
        elif concept == 'quadratic_equation':
            numbers = self.extract_numbers(expression)
            if len(numbers) >= 3:
                a, b, c = numbers[0], numbers[1], numbers[2]
                steps.append("STEP-BY-STEP QUADRATIC SOLUTION:\n")
                steps.append(f"Given: {a:.4g}x² + {b:.4g}x + {c:.4g} = 0\n")
                steps.append("\nUsing Quadratic Formula: x = (-b ± √(b²-4ac))/2a\n")
                steps.append(f"\nStep 1: Identify coefficients")
                steps.append(f"\n  a = {a:.4g}, b = {b:.4g}, c = {c:.4g}\n")
                
                disc = b**2 - 4*a*c
                steps.append(f"\nStep 2: Calculate discriminant")
                steps.append(f"\n  Δ = b² - 4ac = ({b:.4g})² - 4({a:.4g})({c:.4g}) = {disc:.4g}\n")
                
                if disc > 0:
                    steps.append("\nStep 3: Discriminant > 0, so two real roots exist\n")
                    root1 = (-b + math.sqrt(disc)) / (2*a)
                    root2 = (-b - math.sqrt(disc)) / (2*a)
                    steps.append(f"  x₁ = ({-b:.4g} + √{disc:.4g})/(2·{a:.4g}) = {root1:.4f}\n")
                    steps.append(f"  x₂ = ({-b:.4g} - √{disc:.4g})/(2·{a:.4g}) = {root2:.4f}\n")
                elif disc == 0:
                    steps.append("\nStep 3: Discriminant = 0, so one repeated root\n")
                    root = -b / (2*a)
                    steps.append(f"  x = {-b:.4g}/(2·{a:.4g}) = {root:.4f}\n")
                else:
                    steps.append("\nStep 3: Discriminant < 0, so complex roots\n")
                    real = -b / (2*a)
                    imag = math.sqrt(-disc) / (2*a)
                    steps.append(f"  x = {real:.4f} ± {imag:.4f}i\n")
        
        elif concept == 'dot_product':
            vectors = self.extract_vectors(expression)
            if len(vectors) >= 2:
                v1, v2 = vectors[0], vectors[1]
                steps.append("STEP-BY-STEP DOT PRODUCT:\n")
                steps.append(f"Given: A = {v1}, B = {v2}\n")
                steps.append("\nFormula: A·B = A₁B₁ + A₂B₂ + ...\n")
                steps.append("\nCalculation:\n")
                for i in range(len(v1)):
                    steps.append(f"  Term {i+1}: {v1[i]:.4g} × {v2[i]:.4g} = {v1[i]*v2[i]:.4g}\n")
                total = sum(a*b for a, b in zip(v1, v2))
                steps.append(f"\nSum = {' + '.join(f'{v1[i]*v2[i]:.4g}' for i in range(len(v1)))}")
                steps.append(f"\n✓ Final Answer: A·B = {total:.4f}\n")
        
        elif concept == 'mean':
            numbers = self.extract_numbers(expression)
            if numbers:
                steps.append("STEP-BY-STEP MEAN CALCULATION:\n")
                steps.append(f"Given data: {numbers}\n")
                steps.append(f"\nFormula: μ = (Σxᵢ)/n\n")
                steps.append(f"\nStep 1: Sum all values")
                steps.append(f"\n  Σxᵢ = {' + '.join(f'{x:.4g}' for x in numbers)} = {sum(numbers):.4g}\n")
                steps.append(f"\nStep 2: Count values")
                steps.append(f"\n  n = {len(numbers)}\n")
                steps.append(f"\nStep 3: Divide sum by count")
                steps.append(f"\n  μ = {sum(numbers):.4g}/{len(numbers)} = {np.mean(numbers):.4f}\n")
        
        return "".join(steps)
    
    def analyze_input(self):
        """Main analysis function with enhanced output"""
        input_text = self.input_text.get(1.0, tk.END).strip()
        
        # Check if placeholder
        if input_text == self.placeholder.strip() or not input_text:
            messagebox.showwarning("Input Required", "Please enter a mathematical statement!")
            return
        
        # Clear all tabs
        self.results_text.delete(1.0, tk.END)
        self.steps_text.delete(1.0, tk.END)
        self.context_text.delete(1.0, tk.END)
        
        # Update status
        self.status_label.config(text=f"System Status: Analyzing... | Processing mathematical expression", fg=self.colors['warning'])
        self.root.update()
        
        # === ANALYSIS TAB ===
        self.results_text.insert(tk.END, "═" * 90 + "\n")
        self.results_text.insert(tk.END, "  🎯 MATHEMATICAL ANALYSIS REPORT\n", 'heading')
        self.results_text.insert(tk.END, "═" * 90 + "\n\n")
        
        # Display input
        self.results_text.insert(tk.END, "📝 INPUT EXPRESSION:\n", 'subheading')
        self.results_text.insert(tk.END, "─" * 90 + "\n")
        self.results_text.insert(tk.END, f"  {input_text}\n\n", 'formula')
        
        # Tokenize
        words = self.tokenize(input_text)
        self.results_text.insert(tk.END, "🔤 KEYWORD EXTRACTION:\n", 'subheading')
        self.results_text.insert(tk.END, "─" * 90 + "\n")
        self.results_text.insert(tk.END, f"  Keywords: {', '.join(set(words))}\n\n")
        
        # Detect concept
        concept = None
        if self.prolog:
            try:
                query = f"detect_concept({words}, Concept)"
                results = list(self.prolog.query(query))
                if results:
                    concept = results[0]['Concept']
            except:
                pass
        
        if not concept:
            concept = self.detect_concept_fallback(words)
        
        if not concept:
            self.results_text.insert(tk.END, "❌ ERROR: Could not detect mathematical concept!\n\n", 'error')
            self.results_text.insert(tk.END, "💡 SUGGESTIONS:\n", 'warning')
            self.results_text.insert(tk.END, "  • Use more specific mathematical keywords\n")
            self.results_text.insert(tk.END, "  • Check the Examples tab for proper format\n")
            self.results_text.insert(tk.END, "  • Ensure your expression is complete\n")
            self.status_label.config(text=f"System Status: Error - Concept not recognized", fg=self.colors['primary'])
            return
        
        self.current_concept = concept
        self.current_expression = input_text
        
        # Display concept
        self.results_text.insert(tk.END, "🎓 CONCEPT IDENTIFICATION:\n", 'subheading')
        self.results_text.insert(tk.END, "─" * 90 + "\n")
        self.results_text.insert(tk.END, f"  ✓ Detected: ", 'info')
        self.results_text.insert(tk.END, f"{concept.replace('_', ' ').upper()}\n\n", 'result')
        
        # Get explanation
        explanation = None
        if self.prolog:
            try:
                query = f"get_explanation({concept}, Explanation)"
                results = list(self.prolog.query(query))
                if results:
                    explanation = results[0]['Explanation']
                    if isinstance(explanation, bytes):
                        explanation = explanation.decode('utf-8')
            except:
                pass
        
        if not explanation:
            explanation = self.get_explanation_fallback(concept)
        
        self.results_text.insert(tk.END, "📚 MATHEMATICAL EXPLANATION:\n", 'subheading')
        self.results_text.insert(tk.END, "─" * 90 + "\n")
        self.results_text.insert(tk.END, f"{explanation}\n\n")
        
        # Extract and compute
        numbers = self.extract_numbers(input_text)
        self.current_data = numbers
        
        self.results_text.insert(tk.END, "🔢 COMPUTATION:\n", 'subheading')
        self.results_text.insert(tk.END, "─" * 90 + "\n")
        
        result = self.compute_fallback(concept, numbers)
        
        if result and result != "Unable to compute - check input format":
            self.results_text.insert(tk.END, "  ✓ Result: ", 'info')
            self.results_text.insert(tk.END, f"{result}\n\n", 'result')
            self.visualize_btn.config(state='normal')
            self.steps_btn.config(state='normal')
            
            # Generate steps
            steps_content = self.generate_steps(concept, input_text, result)
            self.steps_text.insert(tk.END, steps_content)
            
            # Add real-life context
            context = self.get_real_life_context(concept)
            self.context_text.insert(tk.END, context)
            
            self.status_label.config(text=f"System Status: Analysis complete ✓ | Concept: {concept}", fg=self.colors['success'])
        else:
            self.results_text.insert(tk.END, f"  ⚠ {result}\n\n", 'error')
            self.results_text.insert(tk.END, "💡 TIP: Check your input format and try again.\n", 'warning')
            self.status_label.config(text=f"System Status: Computation failed", fg=self.colors['primary'])
        
        self.results_text.insert(tk.END, "═" * 90 + "\n")
    
    def show_steps(self):
        """Switch to step-by-step tab"""
        if not self.current_concept:
            messagebox.showwarning("No Analysis", "Please analyze a statement first!")
            return
        self.notebook.select(1)  # Switch to steps tab
    
    def visualize_result(self):
        """Visualize with enhanced graphics"""
        if not self.current_concept:
            messagebox.showwarning("No Analysis", "Please analyze a statement first!")
            return
        
        if not VIZ_AVAILABLE:
            messagebox.showerror("Error", "Visualization module not available!")
            return
        
        try:
            input_text = self.input_text.get(1.0, tk.END).strip().lower()
            
            if self.current_concept == 'derivative':
                terms = self.parse_polynomial(input_text)
                if terms:
                    viz.visualize_polynomial_derivative(terms, self.differentiate_polynomial(terms))
                else:
                    viz.visualize_derivative()
            
            elif self.current_concept == 'integration':
                terms = self.parse_polynomial(input_text)
                if terms:
                    viz.visualize_integration_enhanced(terms)
                else:
                    viz.visualize_integration()
            
            elif self.current_concept == 'dot_product':
                vectors = self.extract_vectors(input_text)
                if len(vectors) >= 2:
                    viz.visualize_dot_product_enhanced(vectors[0], vectors[1])
                else:
                    viz.visualize_dot_product_enhanced([3, 4], [1, 2])
            
            elif self.current_concept == 'cross_product':
                vectors = self.extract_vectors(input_text)
                if len(vectors) >= 2:
                    viz.visualize_cross_product_enhanced(vectors[0], vectors[1])
                else:
                    viz.visualize_cross_product_enhanced([1, 0, 0], [0, 1, 0])
            
            elif self.current_concept in ['mean', 'variance', 'standard_deviation']:
                if self.current_data and len(self.current_data) > 1:
                    viz.visualize_statistics_enhanced(self.current_data, self.current_concept)
                else:
                    sample = np.random.normal(50, 10, 100)
                    viz.visualize_statistics_enhanced(sample, self.current_concept)
            
            elif self.current_concept == 'quadratic_equation':
                numbers = self.extract_numbers(input_text)
                if len(numbers) >= 3:
                    a, b, c = numbers[0], numbers[1], numbers[2]
                    viz.visualize_quadratic_enhanced(a, b, c)
                else:
                    viz.visualize_quadratic_enhanced(1, 0, -4)
            
            elif self.current_concept in ['trigonometry_sin', 'trigonometry_cos', 'trigonometry_tan']:
                func = self.current_concept.split('_')[1]
                viz.visualize_trigonometry_enhanced(func)
            
            else:
                messagebox.showinfo("Visualization", f"Advanced visualization for {self.current_concept.replace('_', ' ')} coming soon!")
        
        except Exception as e:
            messagebox.showerror("Visualization Error", f"An error occurred:\n{str(e)}")
    
    def clear_all(self):
        """Clear everything with animation"""
        self.input_text.delete(1.0, tk.END)
        self.input_text.insert(1.0, self.placeholder)
        self.input_text.config(fg=self.colors['text_secondary'])
        
        self.results_text.delete(1.0, tk.END)
        self.steps_text.delete(1.0, tk.END)
        self.context_text.delete(1.0, tk.END)
        
        self.current_concept = None
        self.current_data = None
        self.current_expression = None
        
        self.visualize_btn.config(state='disabled')
        self.steps_btn.config(state='disabled')
        
        self.status_label.config(text=f"System Status: {self.prolog_status}  |  Ready to analyze", fg=self.colors['success'])
        self.notebook.select(0)
    
    def show_examples(self):
        """Show comprehensive examples"""
        examples_window = tk.Toplevel(self.root)
        examples_window.title("📚 Example Mathematical Queries")
        examples_window.geometry("900x700")
        examples_window.configure(bg=self.colors['secondary_bg'])
        
        # Title
        title = tk.Label(
            examples_window,
            text="💡 EXAMPLE QUERIES & FORMATS",
            font=('Helvetica', 16, 'bold'),
            bg=self.colors['primary'],
            fg='white',
            pady=15
        )
        title.pack(fill='x')
        
        # Text widget
        text_widget = scrolledtext.ScrolledText(
            examples_window,
            font=('Consolas', 10),
            wrap='word',
            bg=self.colors['secondary_bg'],
            fg=self.colors['text'],
            padx=20,
            pady=20
        )
        text_widget.pack(fill='both', expand=True, padx=10, pady=10)
        
        examples = """
═══════════════════════════════════════════════════════════════════════════════

                        COMPREHENSIVE EXAMPLE QUERIES

═══════════════════════════════════════════════════════════════════════════════

🎯 CALCULUS - Derivatives
────────────────────────────────────────────────────────────────────────────
• "Find derivative of x^3"
• "Differentiate 3x^4 + 2x^2 + 5"
• "Rate of change of x^5 - 4x^2 + 7x - 1"
• "Calculate gradient of 2x^3 + x"

🎯 CALCULUS - Integration  
────────────────────────────────────────────────────────────────────────────
• "Integrate x^2"
• "Find integral of 4x^3 + 2x"
• "Area under x^4 - 3x^2"
• "Antiderivative of 5x^2 + 3x + 1"

🎯 VECTORS - Dot Product
────────────────────────────────────────────────────────────────────────────
• "Calculate dot product of [1,2,3] and [4,5,6]"
• "Find scalar product of [2,3] and [1,4]"
• "Inner product [5,2,1] [3,4,2]"

🎯 VECTORS - Cross Product
────────────────────────────────────────────────────────────────────────────
• "Calculate cross product of [1,0,0] and [0,1,0]"
• "Vector product of [1,2,3] and [4,5,6]"
• "Find perpendicular vector to [2,1,3] and [1,2,1]"

🎯 VECTORS - Magnitude
────────────────────────────────────────────────────────────────────────────
• "Find magnitude of vector [3,4]"
• "Calculate modulus of [1,2,3]"
• "Length of vector [5,12]"

🎯 STATISTICS - Central Tendency
────────────────────────────────────────────────────────────────────────────
• "Find mean of 10, 20, 30, 40, 50"
• "Calculate average of 5, 15, 25, 35"
• "Median of 3, 7, 2, 9, 5"
• "Mode of 1, 2, 2, 3, 3, 3, 4"

🎯 STATISTICS - Dispersion
────────────────────────────────────────────────────────────────────────────
• "Calculate variance of 5, 10, 15, 20"
• "Standard deviation of 100, 200, 300"
• "Range of 5, 12, 3, 19, 7"

🎯 EQUATIONS - Quadratic
────────────────────────────────────────────────────────────────────────────
• "Solve quadratic equation with a=1, b=0, c=-4"
• "Find roots of x^2 + 5x + 6"
• "Quadratic 2x^2 - 4x + 2"

🎯 EQUATIONS - Linear
────────────────────────────────────────────────────────────────────────────
• "Solve linear equation 2x + 6 = 0"
• "Find x in 5x - 15 = 0"
• "Linear solve 3x + 9"

🎯 FUNCTIONS - Rounding
────────────────────────────────────────────────────────────────────────────
• "Floor of 3.7"
• "Ceiling of 2.3"
• "Absolute value of -15"
• "Floor function 5.9"

🎯 TRIGONOMETRY
────────────────────────────────────────────────────────────────────────────
• "Sin of 30 degrees"
• "Cosine of 45"
• "Tangent of 60 degrees"

🎯 LOGARITHMS & EXPONENTIALS
────────────────────────────────────────────────────────────────────────────
• "Logarithm of 100 base 10"
• "Natural log of 20"
• "Exponential of 2"

🎯 COMBINATORICS
────────────────────────────────────────────────────────────────────────────
• "Permutation 10 choose 3"
• "Combination 5 choose 2"
• "Arrange 6 objects in order"
• "Select 4 from 12"

🎯 PROBABILITY
────────────────────────────────────────────────────────────────────────────
• "Probability 3 favorable out of 10 total"
• "Chance of 5 success in 20 trials"

🎯 NUMBER THEORY
────────────────────────────────────────────────────────────────────────────
• "GCD of 48 and 18"
• "LCM of 12 and 18"
• "Is 17 prime"
• "Factorial of 5"

═══════════════════════════════════════════════════════════════════════════════

💡 TIPS:
  • Be specific with mathematical terms
  • Include all necessary numbers and coefficients
  • Use brackets [x,y,z] for vectors
  • Separate numbers with spaces or commas

═══════════════════════════════════════════════════════════════════════════════
        """
        
        text_widget.insert(1.0, examples)
        text_widget.config(state='disabled')
        
        # Close button
        close_btn = tk.Button(
            examples_window,
            text="✓ Got It!",
            font=('Helvetica', 12, 'bold'),
            bg=self.colors['success'],
            fg='white',
            padx=30,
            pady=10,
            relief='flat',
            cursor='hand2',
            command=examples_window.destroy
        )
        close_btn.pack(pady=15)
    
    def show_help(self):
        """Show help documentation"""
        help_window = tk.Toplevel(self.root)
        help_window.title("❓ Help & Documentation")
        help_window.geometry("800x600")
        help_window.configure(bg=self.colors['secondary_bg'])
        
        title = tk.Label(
            help_window,
            text="❓ SYSTEM HELP & DOCUMENTATION",
            font=('Helvetica', 16, 'bold'),
            bg=self.colors['info'],
            fg='white',
            pady=15
        )
        title.pack(fill='x')
        
        text_widget = scrolledtext.ScrolledText(
            help_window,
            font=('Helvetica', 10),
            wrap='word',
            bg=self.colors['secondary_bg'],
            fg=self.colors['text'],
            padx=20,
            pady=20
        )
        text_widget.pack(fill='both', expand=True, padx=10, pady=10)
        
        help_text = """
🎓 WELCOME TO THE INTELLIGENT MATHEMATICAL EXPERT SYSTEM

This system uses AI and knowledge-based reasoning to analyze, compute, and visualize
mathematical concepts with real-world applications.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 HOW TO USE:

1. ENTER YOUR EXPRESSION:
   Type your mathematical problem in natural language or formula format
   
2. CLICK "ANALYZE & COMPUTE":
   The system will:
   • Detect the mathematical concept
   • Provide detailed explanation
   • Compute the result
   • Show step-by-step solution

3. VIEW RESULTS IN TABS:
   • Analysis: Main results and computation
   • Step-by-Step: Detailed solution process
   • Real-Life Context: Practical applications

4. VISUALIZE:
   Click "Visualize" to see interactive graphs and plots

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔧 SUPPORTED OPERATIONS:

✓ Calculus: Derivatives, Integrals, Limits
✓ Algebra: Linear & Quadratic Equations
✓ Vectors: Dot Product, Cross Product, Magnitude
✓ Statistics: Mean, Median, Variance, Standard Deviation
✓ Trigonometry: Sin, Cos, Tan
✓ Combinatorics: Permutations, Combinations
✓ Number Theory: GCD, LCM, Primes, Factorials
✓ And many more...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 INPUT FORMATS:

Polynomials: "x^3 + 2x^2 - 5x + 3"
Vectors: "[1,2,3]" or "(4,5,6)"
Numbers: "12.5" or "-7" or "3.14"
Natural Language: "find the average of 10, 20, 30"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎨 VISUALIZATIONS:

The system creates professional plots showing:
• Function graphs with derivatives/integrals
• Vector representations in 2D/3D
• Statistical distributions
• Equation solutions
• Real-world applications

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ KEYBOARD SHORTCUTS:

• Clear All: Ctrl+Delete
• Examples: F1
• Help: F2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📞 TECHNICAL DETAILS:

System Architecture:
• Python GUI (Tkinter)
• Prolog Knowledge Base (optional)
• NumPy for computations
• Matplotlib for visualizations

Version: 2.0 Pro
Concepts Supported: 50+
Last Updated: 2025

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Need more help? Click "Examples" button for sample queries!
        """
        
        text_widget.insert(1.0, help_text)
        text_widget.config(state='disabled')
        
        close_btn = tk.Button(
            help_window,
            text="Close",
            font=('Helvetica', 11, 'bold'),
            bg=self.colors['primary'],
            fg='white',
            padx=30,
            pady=8,
            relief='flat',
            cursor='hand2',
            command=help_window.destroy
        )
        close_btn.pack(pady=15)
    
    def __del__(self):
        """Cleanup"""
        if self.animation_id:
            self.root.after_cancel(self.animation_id)


def main():
    """Main function"""
    root = tk.Tk()
    app = MathExpertSystem(root)
    
    # Keyboard shortcuts
    root.bind('<Control-Delete>', lambda e: app.clear_all())
    root.bind('<F1>', lambda e: app.show_examples())
    root.bind('<F2>', lambda e: app.show_help())
    
    root.mainloop()


if __name__ == "__main__":
    main()