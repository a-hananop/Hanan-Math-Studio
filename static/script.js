// script.js - Complete Fully Functional Frontend with All Features

// ==================== CONCEPT DATA ====================
const conceptsData = {
    calculus: {
        color: 'blue',
        concepts: [
            {
                id: 'derivative',
                name: 'Derivative',
                description: 'Rate of change of a function',
                explanation: 'The derivative measures the instantaneous rate of change of a function at any point. It represents the slope of the tangent line to the curve at that point.',
                realWorld: 'Used in physics for calculating velocity and acceleration, in economics for marginal analysis, and in engineering for optimization problems.',
                inputs: [
                    { name: 'function', label: 'Function Type', type: 'select', options: ['power', 'sin', 'cos', 'exp'], default: 'power' },
                    { name: 'coefficient', label: 'Coefficient', type: 'number', default: 1, step: 0.1 },
                    { name: 'exponent', label: 'Exponent', type: 'number', default: 2, step: 0.1 }
                ]
            },
            {
                id: 'integration',
                name: 'Integration',
                description: 'Area under a curve / Antiderivative',
                explanation: 'Integration calculates the accumulated quantity or the area under a curve. It is the inverse operation of differentiation.',
                realWorld: 'Used to calculate work done by a force, total distance from velocity, accumulated profit, and fluid flow in engineering.',
                inputs: [
                    { name: 'function', label: 'Function Type', type: 'select', options: ['power', 'sin', 'cos'], default: 'power' },
                    { name: 'coefficient', label: 'Coefficient', type: 'number', default: 1, step: 0.1 },
                    { name: 'exponent', label: 'Exponent', type: 'number', default: 2, step: 0.1 }
                ]
            },
            {
                id: 'limit',
                name: 'Limit',
                description: 'Value a function approaches',
                explanation: 'Limits describe the behavior of a function as it approaches a particular point. They are fundamental to defining derivatives and integrals.',
                realWorld: 'Used in physics for instantaneous rates, in engineering for analyzing system behavior at boundaries, and in computer science for algorithm analysis.',
                inputs: [
                    { name: 'function', label: 'Function Type', type: 'select', options: ['rational'], default: 'rational' },
                    { name: 'point', label: 'Approach Point', type: 'number', default: 2, step: 0.1 }
                ]
            },
            {
                id: 'partial_derivative',
                name: 'Partial Derivative',
                description: 'Derivative with respect to one variable',
                explanation: 'In multivariable calculus, partial derivatives measure how a function changes when only one variable changes, keeping others constant.',
                realWorld: 'Used in thermodynamics, economics (marginal analysis), machine learning (gradient descent), and physics (heat equations).',
                inputs: [
                    { name: 'var', label: 'Variable', type: 'select', options: ['x', 'y'], default: 'x' },
                    { name: 'x', label: 'X value', type: 'number', default: 1, step: 0.1 },
                    { name: 'y', label: 'Y value', type: 'number', default: 1, step: 0.1 }
                ]
            },
            {
                id: 'second_derivative',
                name: 'Second Derivative',
                description: 'Rate of change of rate of change',
                explanation: 'The second derivative indicates concavity and acceleration. Positive values mean concave up, negative means concave down.',
                realWorld: 'Used in physics for acceleration, in economics for diminishing returns, and in optimization to find maxima and minima.',
                inputs: [
                    { name: 'coefficient', label: 'Coefficient', type: 'number', default: 1, step: 0.1 },
                    { name: 'exponent', label: 'Exponent', type: 'number', default: 3, step: 0.1 }
                ]
            }
        ]
    },
    
    vectors: {
        color: 'orange',
        concepts: [
            {
                id: 'dot_product',
                name: 'Dot Product',
                description: 'Scalar product of two vectors',
                explanation: 'The dot product measures how much two vectors align with each other. Result is a scalar (number), not a vector.',
                realWorld: 'Used in physics for work calculations, in computer graphics for lighting and projections, and in machine learning for similarity measures.',
                inputs: [
                    { name: 'x1', label: 'Vector A - X', type: 'number', default: 3, step: 0.1 },
                    { name: 'y1', label: 'Vector A - Y', type: 'number', default: 4, step: 0.1 },
                    { name: 'x2', label: 'Vector B - X', type: 'number', default: 2, step: 0.1 },
                    { name: 'y2', label: 'Vector B - Y', type: 'number', default: 1, step: 0.1 }
                ]
            },
            {
                id: 'cross_product',
                name: 'Cross Product',
                description: 'Vector perpendicular to both inputs',
                explanation: 'The cross product creates a vector perpendicular to both input vectors. Only defined in 3D space.',
                realWorld: 'Used in physics for torque and angular momentum, in computer graphics for surface normals, and in robotics for rotation calculations.',
                inputs: [
                    { name: 'x1', label: 'Vector A - X', type: 'number', default: 1, step: 0.1 },
                    { name: 'y1', label: 'Vector A - Y', type: 'number', default: 0, step: 0.1 },
                    { name: 'z1', label: 'Vector A - Z', type: 'number', default: 0, step: 0.1 },
                    { name: 'x2', label: 'Vector B - X', type: 'number', default: 0, step: 0.1 },
                    { name: 'y2', label: 'Vector B - Y', type: 'number', default: 1, step: 0.1 },
                    { name: 'z2', label: 'Vector B - Z', type: 'number', default: 0, step: 0.1 }
                ]
            },
            {
                id: 'modulus',
                name: 'Vector Magnitude',
                description: 'Length/size of a vector',
                explanation: 'The modulus (or magnitude) represents the length of a vector, calculated using the Pythagorean theorem.',
                realWorld: 'Used in physics for force magnitude, in navigation for distance, and in data science for Euclidean distance.',
                inputs: [
                    { name: 'x', label: 'X component', type: 'number', default: 3, step: 0.1 },
                    { name: 'y', label: 'Y component', type: 'number', default: 4, step: 0.1 },
                    { name: 'z', label: 'Z component (optional)', type: 'number', default: 0, step: 0.1 }
                ]
            },
            {
                id: 'unit_vector',
                name: 'Unit Vector',
                description: 'Normalized vector (magnitude = 1)',
                explanation: 'A unit vector has magnitude 1 and indicates direction only. Obtained by dividing a vector by its magnitude.',
                realWorld: 'Used in physics for direction fields, in computer graphics for normals, and in machine learning for feature normalization.',
                inputs: [
                    { name: 'x', label: 'X component', type: 'number', default: 3, step: 0.1 },
                    { name: 'y', label: 'Y component', type: 'number', default: 4, step: 0.1 },
                    { name: 'z', label: 'Z component (optional)', type: 'number', default: 0, step: 0.1 }
                ]
            },
            {
                id: 'vector_projection',
                name: 'Vector Projection',
                description: 'Component of one vector in direction of another',
                explanation: 'Vector projection finds how much of one vector lies in the direction of another vector.',
                realWorld: 'Used in physics for work calculations, in engineering for force decomposition, and in data science for dimensionality reduction.',
                inputs: [
                    { name: 'ax', label: 'Vector a - X', type: 'number', default: 3, step: 0.1 },
                    { name: 'ay', label: 'Vector a - Y', type: 'number', default: 4, step: 0.1 },
                    { name: 'bx', label: 'Vector b - X', type: 'number', default: 1, step: 0.1 },
                    { name: 'by', label: 'Vector b - Y', type: 'number', default: 0, step: 0.1 }
                ]
            }
        ]
    },
    
    statistics: {
        color: 'green',
        concepts: [
            {
                id: 'mean',
                name: 'Mean (Average)',
                description: 'Sum of values divided by count',
                explanation: 'The arithmetic mean represents the central value of a dataset. It is the balancing point of the distribution.',
                realWorld: 'Used in education (GPA calculation), business (average sales), weather (temperature averages), and economics (average income).',
                inputs: [{ name: 'values', label: 'Values (comma-separated)', type: 'text', default: '10, 20, 30, 40, 50' }]
            },
            {
                id: 'median',
                name: 'Median',
                description: 'Middle value when sorted',
                explanation: 'The median is the middle value in a sorted dataset. It is resistant to outliers and better for skewed distributions.',
                realWorld: 'Used in real estate (median home prices), income statistics, and any data with outliers where mean would be misleading.',
                inputs: [{ name: 'values', label: 'Values (comma-separated)', type: 'text', default: '10, 20, 30, 40, 50' }]
            },
            {
                id: 'mode',
                name: 'Mode',
                description: 'Most frequently occurring value',
                explanation: 'The mode is the value that appears most often in a dataset. A dataset can have multiple modes or no mode.',
                realWorld: 'Used in market research (most popular product), fashion (trending styles), and categorical data analysis.',
                inputs: [{ name: 'values', label: 'Values (comma-separated)', type: 'text', default: '1, 2, 2, 3, 3, 3, 4' }]
            },
            {
                id: 'variance',
                name: 'Variance',
                description: 'Average squared deviation from mean',
                explanation: 'Variance measures how spread out the data is from the mean. Larger variance indicates more variability.',
                realWorld: 'Used in finance for risk assessment, quality control for consistency, and scientific research for data reliability.',
                inputs: [{ name: 'values', label: 'Values (comma-separated)', type: 'text', default: '10, 20, 30, 40, 50' }]
            },
            {
                id: 'standard_deviation',
                name: 'Standard Deviation',
                description: 'Square root of variance',
                explanation: 'Standard deviation measures typical distance from the mean, in the same units as the original data.',
                realWorld: 'Used in finance (volatility), manufacturing (quality control), education (test score spread), and healthcare (normal ranges).',
                inputs: [{ name: 'values', label: 'Values (comma-separated)', type: 'text', default: '10, 20, 30, 40, 50' }]
            },
            {
                id: 'range',
                name: 'Range',
                description: 'Difference between max and min',
                explanation: 'Range is the simplest measure of spread, but it is sensitive to outliers.',
                realWorld: 'Used in weather reporting (temperature range), sports statistics, and data exploration for initial understanding.',
                inputs: [{ name: 'values', label: 'Values (comma-separated)', type: 'text', default: '5, 12, 8, 20, 15' }]
            },
            {
                id: 'percentile',
                name: 'Percentile',
                description: 'Value below which a percentage falls',
                explanation: 'The pth percentile is a value below which p% of the data falls. Quartiles divide data into 4 equal parts.',
                realWorld: 'Used in standardized testing (SAT scores), growth charts (pediatrics), and performance benchmarking.',
                inputs: [
                    { name: 'values', label: 'Values (comma-separated)', type: 'text', default: '10, 20, 30, 40, 50, 60, 70, 80, 90' },
                    { name: 'percentile', label: 'Percentile (0-100)', type: 'number', default: 75, min: 0, max: 100, step: 1 }
                ]
            },
            {
                id: 'correlation',
                name: 'Correlation',
                description: 'Linear relationship strength (-1 to 1)',
                explanation: 'Correlation measures the strength and direction of linear relationship. r=1 (perfect positive), r=-1 (perfect negative), r=0 (no linear relationship).',
                realWorld: 'Used in finance (portfolio diversification), social sciences (research studies), and data science (feature selection).',
                inputs: [
                    { name: 'x_values', label: 'X Values (comma-separated)', type: 'text', default: '1, 2, 3, 4, 5' },
                    { name: 'y_values', label: 'Y Values (comma-separated)', type: 'text', default: '2, 4, 5, 4, 5' }
                ]
            },
            {
                id: 'z_score',
                name: 'Z-Score',
                description: 'Standard deviations from mean',
                explanation: 'Z-score standardizes values by showing how many standard deviations they are from the mean. Used to compare different datasets.',
                realWorld: 'Used in testing (comparing scores across different exams), finance (comparing investments), and quality control (detecting anomalies).',
                inputs: [
                    { name: 'value', label: 'Value', type: 'number', default: 85, step: 0.1 },
                    { name: 'mean', label: 'Mean', type: 'number', default: 75, step: 0.1 },
                    { name: 'std_dev', label: 'Standard Deviation', type: 'number', default: 10, step: 0.1 }
                ]
            }
        ]
    },
    
    probability: {
        color: 'yellow',
        concepts: [
            {
                id: 'probability',
                name: 'Basic Probability',
                description: 'Likelihood of an event (0 to 1)',
                explanation: 'Probability measures the likelihood of an event occurring, ranging from 0 (impossible) to 1 (certain).',
                realWorld: 'Used in gambling, weather forecasting, insurance pricing, medical diagnosis, and risk assessment.',
                inputs: [
                    { name: 'favorable', label: 'Favorable Outcomes', type: 'number', default: 1, min: 0, step: 1 },
                    { name: 'total', label: 'Total Outcomes', type: 'number', default: 6, min: 1, step: 1 }
                ]
            },
            {
                id: 'conditional_probability',
                name: 'Conditional Probability',
                description: 'P(A|B) - probability of A given B',
                explanation: 'Conditional probability is the probability of event A occurring given that event B has already occurred.',
                realWorld: 'Used in medical testing (disease probability given symptoms), spam filtering, and predictive maintenance.',
                inputs: [
                    { name: 'p_a_and_b', label: 'P(A and B)', type: 'number', default: 0.2, min: 0, max: 1, step: 0.01 },
                    { name: 'p_b', label: 'P(B)', type: 'number', default: 0.5, min: 0.01, max: 1, step: 0.01 }
                ]
            },
            {
                id: 'bayes_theorem',
                name: "Bayes' Theorem",
                description: 'Update probabilities with new evidence',
                explanation: "Bayes' Theorem allows us to update probability estimates as we gain new information. Foundation of Bayesian statistics.",
                realWorld: 'Used in machine learning, spam filtering, medical diagnosis, search engines, and artificial intelligence.',
                inputs: [
                    { name: 'p_b_given_a', label: 'P(B|A)', type: 'number', default: 0.8, min: 0, max: 1, step: 0.01 },
                    { name: 'p_a', label: 'P(A)', type: 'number', default: 0.3, min: 0, max: 1, step: 0.01 },
                    { name: 'p_b', label: 'P(B)', type: 'number', default: 0.5, min: 0.01, max: 1, step: 0.01 }
                ]
            },
            {
                id: 'expected_value',
                name: 'Expected Value',
                description: 'Long-run average outcome',
                explanation: 'Expected value is the weighted average of all possible outcomes, where weights are the probabilities.',
                realWorld: 'Used in gambling strategy, insurance pricing, investment decisions, and game theory.',
                inputs: [
                    { name: 'values', label: 'Values (comma-separated)', type: 'text', default: '10, 20, 30' },
                    { name: 'probabilities', label: 'Probabilities (must sum to 1)', type: 'text', default: '0.2, 0.5, 0.3' }
                ]
            }
        ]
    },
    
    combinatorics: {
        color: 'pink',
        concepts: [
            {
                id: 'permutation',
                name: 'Permutation',
                description: 'Ordered arrangements',
                explanation: 'Permutations count arrangements where order matters. P(n,r) = n!/(n-r)!',
                realWorld: 'Used in password combinations, race rankings, scheduling, and cryptography.',
                inputs: [
                    { name: 'n', label: 'Total Items (n)', type: 'number', default: 5, min: 0, step: 1 },
                    { name: 'r', label: 'Items to Arrange (r)', type: 'number', default: 3, min: 0, step: 1 }
                ]
            },
            {
                id: 'combination',
                name: 'Combination',
                description: 'Unordered selections',
                explanation: 'Combinations count selections where order does not matter. C(n,r) = n!/(r!(n-r)!)',
                realWorld: 'Used in lottery calculations, team selection, sampling, and card games.',
                inputs: [
                    { name: 'n', label: 'Total Items (n)', type: 'number', default: 5, min: 0, step: 1 },
                    { name: 'r', label: 'Items to Choose (r)', type: 'number', default: 3, min: 0, step: 1 }
                ]
            },
            {
                id: 'factorial',
                name: 'Factorial',
                description: 'Product of all positive integers up to n',
                explanation: 'Factorial n! = n × (n-1) × ... × 2 × 1. Grows extremely rapidly.',
                realWorld: 'Used in permutations, combinations, probability, and Taylor series approximations.',
                inputs: [{ name: 'n', label: 'Number (n)', type: 'number', default: 5, min: 0, max: 20, step: 1 }]
            }
        ]
    },
    
    algebra: {
        color: 'purple',
        concepts: [
            {
                id: 'quadratic_equation',
                name: 'Quadratic Equation',
                description: 'ax² + bx + c = 0',
                explanation: 'Quadratic equations form parabolas. Solutions found using the quadratic formula: x = (-b ± √(b²-4ac))/2a',
                realWorld: 'Used in projectile motion, optimization problems, economics (profit/loss), and engineering.',
                inputs: [
                    { name: 'a', label: 'a (coefficient of x²)', type: 'number', default: 1, step: 0.1 },
                    { name: 'b', label: 'b (coefficient of x)', type: 'number', default: -5, step: 0.1 },
                    { name: 'c', label: 'c (constant)', type: 'number', default: 6, step: 0.1 }
                ]
            },
            {
                id: 'linear_equation',
                name: 'Linear Equation',
                description: 'ax + b = 0',
                explanation: 'Linear equations represent straight lines. Solution: x = -b/a',
                realWorld: 'Used in break-even analysis, linear programming, supply-demand models, and simple motion problems.',
                inputs: [
                    { name: 'a', label: 'a (coefficient)', type: 'number', default: 2, step: 0.1 },
                    { name: 'b', label: 'b (constant)', type: 'number', default: -10, step: 0.1 }
                ]
            },
            {
                id: 'cubic_equation',
                name: 'Cubic Equation',
                description: 'ax³ + bx² + cx + d = 0',
                explanation: 'Third-degree polynomial equations. Can have 1 or 3 real roots.',
                realWorld: 'Used in volume calculations, curve fitting, and advanced physics problems.',
                inputs: [
                    { name: 'a', label: 'a (coefficient of x³)', type: 'number', default: 1, step: 0.1 },
                    { name: 'b', label: 'b (coefficient of x²)', type: 'number', default: 0, step: 0.1 },
                    { name: 'c', label: 'c (coefficient of x)', type: 'number', default: -1, step: 0.1 },
                    { name: 'd', label: 'd (constant)', type: 'number', default: 0, step: 0.1 }
                ]
            },
            {
                id: 'absolute_value',
                name: 'Absolute Value',
                description: 'Distance from zero',
                explanation: 'Absolute value |x| is always non-negative, representing magnitude without direction.',
                realWorld: 'Used in error calculations, distance measurements, and signal processing.',
                inputs: [{ name: 'x', label: 'Value', type: 'number', default: -7.5, step: 0.1 }]
            },
            {
                id: 'floor_function',
                name: 'Floor Function',
                description: 'Largest integer ≤ x',
                explanation: 'Floor function ⌊x⌋ rounds down to the nearest integer.',
                realWorld: 'Used in computer science (integer division), pricing (rounding down), and discrete mathematics.',
                inputs: [{ name: 'x', label: 'Value', type: 'number', default: 3.7, step: 0.1 }]
            },
            {
                id: 'ceiling_function',
                name: 'Ceiling Function',
                description: 'Smallest integer ≥ x',
                explanation: 'Ceiling function ⌈x⌉ rounds up to the nearest integer.',
                realWorld: 'Used in resource allocation (rounding up needed units), pagination, and scheduling.',
                inputs: [{ name: 'x', label: 'Value', type: 'number', default: 3.2, step: 0.1 }]
            }
        ]
    },
    
    exponential: {
        color: 'red',
        concepts: [
            {
                id: 'logarithm',
                name: 'Logarithm',
                description: 'Inverse of exponentiation',
                explanation: 'Logarithm log_b(x) answers: "what power of b gives x?". log_b(x) = y means b^y = x',
                realWorld: 'Used in pH scale, Richter scale, decibels, compound interest, and computer science (time complexity).',
                inputs: [
                    { name: 'x', label: 'Value (x)', type: 'number', default: 100, min: 0.01, step: 0.1 },
                    { name: 'base', label: 'Base', type: 'number', default: 10, min: 1.01, step: 0.1 }
                ]
            },
            {
                id: 'exponential',
                name: 'Exponential (e^x)',
                description: 'Natural exponential function',
                explanation: 'Exponential function e^x represents continuous growth. e ≈ 2.71828 is Euler\'s number.',
                realWorld: 'Used in compound interest, population growth, radioactive decay, and natural phenomena modeling.',
                inputs: [{ name: 'x', label: 'Exponent (x)', type: 'number', default: 2, step: 0.1 }]
            },
            {
                id: 'exponential_growth',
                name: 'Exponential Growth',
                description: 'N(t) = N₀e^(rt)',
                explanation: 'Continuous exponential growth model where r > 0 is the growth rate.',
                realWorld: 'Used in population dynamics, viral spreading, compound interest, and bacterial growth.',
                inputs: [
                    { name: 'initial', label: 'Initial Value (N₀)', type: 'number', default: 100, min: 0, step: 1 },
                    { name: 'rate', label: 'Growth Rate (r)', type: 'number', default: 0.05, step: 0.01 },
                    { name: 'time', label: 'Time (t)', type: 'number', default: 10, min: 0, step: 0.1 }
                ]
            },
            {
                id: 'exponential_decay',
                name: 'Exponential Decay',
                description: 'N(t) = N₀e^(-rt)',
                explanation: 'Continuous exponential decay model where r > 0 is the decay rate. Half-life is constant.',
                realWorld: 'Used in radioactive decay, drug elimination from body, cooling (Newton\'s law), and depreciation.',
                inputs: [
                    { name: 'initial', label: 'Initial Value (N₀)', type: 'number', default: 100, min: 0, step: 1 },
                    { name: 'rate', label: 'Decay Rate (r)', type: 'number', default: 0.1, min: 0, step: 0.01 },
                    { name: 'time', label: 'Time (t)', type: 'number', default: 5, min: 0, step: 0.1 }
                ]
            }
        ]
    },
    
    trigonometry: {
        color: 'indigo',
        concepts: [
            {
                id: 'trigonometry_sin',
                name: 'Sine Function',
                description: 'sin(θ) = opposite/hypotenuse',
                explanation: 'Sine represents the vertical component or y-coordinate on the unit circle.',
                realWorld: 'Used in sound waves, AC electricity, signal processing, and harmonic motion.',
                inputs: [{ name: 'angle', label: 'Angle (degrees)', type: 'number', default: 30, step: 1 }]
            },
            {
                id: 'trigonometry_cos',
                name: 'Cosine Function',
                description: 'cos(θ) = adjacent/hypotenuse',
                explanation: 'Cosine represents the horizontal component or x-coordinate on the unit circle.',
                realWorld: 'Used in physics (work calculation), navigation, computer graphics, and wave analysis.',
                inputs: [{ name: 'angle', label: 'Angle (degrees)', type: 'number', default: 30, step: 1 }]
            },
            {
                id: 'trigonometry_tan',
                name: 'Tangent Function',
                description: 'tan(θ) = opposite/adjacent = sin/cos',
                explanation: 'Tangent represents the slope of the angle. It has vertical asymptotes at 90°, 270°, etc.',
                realWorld: 'Used in surveying, navigation, calculating slopes, and perspective in computer graphics.',
                inputs: [{ name: 'angle', label: 'Angle (degrees)', type: 'number', default: 30, step: 1 }]
            },
            {
                id: 'inverse_trig',
                name: 'Inverse Trigonometric Functions',
                description: 'arcsin, arccos, arctan',
                explanation: 'Inverse trig functions find angles from ratios. Domain restrictions ensure unique values.',
                realWorld: 'Used in robotics (inverse kinematics), navigation, solving triangles, and angle calculations.',
                inputs: [
                    { name: 'function', label: 'Function', type: 'select', options: ['arcsin', 'arccos', 'arctan'], default: 'arcsin' },
                    { name: 'value', label: 'Value', type: 'number', default: 0.5, step: 0.01 }
                ]
            }
        ]
    },
    
    numberTheory: {
        color: 'cyan',
        concepts: [
            {
                id: 'gcd',
                name: 'GCD (Greatest Common Divisor)',
                description: 'Largest number dividing both',
                explanation: 'GCD is found using the Euclidean algorithm. Essential for simplifying fractions.',
                realWorld: 'Used in cryptography (RSA), music theory (rhythm), gear ratios, and simplifying fractions.',
                inputs: [
                    { name: 'a', label: 'Number A', type: 'number', default: 48, min: 1, step: 1 },
                    { name: 'b', label: 'Number B', type: 'number', default: 18, min: 1, step: 1 }
                ]
            },
            {
                id: 'lcm',
                name: 'LCM (Least Common Multiple)',
                description: 'Smallest number divisible by both',
                explanation: 'LCM is used for adding fractions and synchronization problems. LCM(a,b) = (a×b)/GCD(a,b)',
                realWorld: 'Used in scheduling, signal processing, music (finding common beats), and adding fractions.',
                inputs: [
                    { name: 'a', label: 'Number A', type: 'number', default: 12, min: 1, step: 1 },
                    { name: 'b', label: 'Number B', type: 'number', default: 18, min: 1, step: 1 }
                ]
            },
            {
                id: 'prime',
                name: 'Prime Number Check',
                description: 'Is the number prime?',
                explanation: 'Prime numbers have exactly two factors: 1 and themselves. 2 is the only even prime.',
                realWorld: 'Used in cryptography (RSA encryption), hash tables, random number generation, and computer security.',
                inputs: [{ name: 'n', label: 'Number', type: 'number', default: 17, min: 1, step: 1 }]
            }
        ]
    },
    
    matrices: {
        color: 'teal',
        concepts: [
            {
                id: 'matrix_multiplication',
                name: 'Matrix Multiplication',
                description: '2×2 matrix product',
                explanation: 'Matrix multiplication combines transformations. Not commutative (AB ≠ BA generally).',
                realWorld: 'Used in computer graphics (transformations), machine learning, quantum mechanics, and economics (input-output models).',
                inputs: [
                    { name: 'a11', label: 'A[1,1]', type: 'number', default: 1, step: 0.1 },
                    { name: 'a12', label: 'A[1,2]', type: 'number', default: 2, step: 0.1 },
                    { name: 'a21', label: 'A[2,1]', type: 'number', default: 3, step: 0.1 },
                    { name: 'a22', label: 'A[2,2]', type: 'number', default: 4, step: 0.1 },
                    { name: 'b11', label: 'B[1,1]', type: 'number', default: 5, step: 0.1 },
                    { name: 'b12', label: 'B[1,2]', type: 'number', default: 6, step: 0.1 },
                    { name: 'b21', label: 'B[2,1]', type: 'number', default: 7, step: 0.1 },
                    { name: 'b22', label: 'B[2,2]', type: 'number', default: 8, step: 0.1 }
                ]
            },
            {
                id: 'matrix_determinant',
                name: 'Matrix Determinant',
                description: 'Scaling factor of transformation',
                explanation: 'Determinant measures how much a transformation scales areas. Zero determinant means the matrix is singular.',
                realWorld: 'Used in solving linear systems, calculating volumes, computer graphics, and determining invertibility.',
                inputs: [
                    { name: 'a11', label: 'A[1,1]', type: 'number', default: 1, step: 0.1 },
                    { name: 'a12', label: 'A[1,2]', type: 'number', default: 2, step: 0.1 },
                    { name: 'a21', label: 'A[2,1]', type: 'number', default: 3, step: 0.1 },
                    { name: 'a22', label: 'A[2,2]', type: 'number', default: 4, step: 0.1 }
                ]
            },
            {
                id: 'matrix_inverse',
                name: 'Matrix Inverse',
                description: 'Reverse transformation',
                explanation: 'Matrix inverse A⁻¹ satisfies AA⁻¹ = I. Only exists if determinant ≠ 0.',
                realWorld: 'Used in solving systems of equations, computer graphics (reversing transformations), and cryptography.',
                inputs: [
                    { name: 'a11', label: 'A[1,1]', type: 'number', default: 1, step: 0.1 },
                    { name: 'a12', label: 'A[1,2]', type: 'number', default: 2, step: 0.1 },
                    { name: 'a21', label: 'A[2,1]', type: 'number', default: 3, step: 0.1 },
                    { name: 'a22', label: 'A[2,2]', type: 'number', default: 4, step: 0.1 }
                ]
            },
            {
                id: 'eigenvalue',
                name: 'Eigenvalues',
                description: 'Characteristic values of matrix',
                explanation: 'Eigenvalues λ satisfy Av = λv. They reveal fundamental properties of the transformation.',
                realWorld: 'Used in stability analysis, principal component analysis (PCA), quantum mechanics, and vibration analysis.',
                inputs: [
                    { name: 'a11', label: 'A[1,1]', type: 'number', default: 3, step: 0.1 },
                    { name: 'a12', label: 'A[1,2]', type: 'number', default: 1, step: 0.1 },
                    { name: 'a21', label: 'A[2,1]', type: 'number', default: 1, step: 0.1 },
                    { name: 'a22', label: 'A[2,2]', type: 'number', default: 3, step: 0.1 }
                ]
            }
        ]
    },
    
    advanced: {
        color: 'violet',
        concepts: [
            {
                id: 'fourier_transform',
                name: 'Fourier Transform',
                description: 'Frequency domain analysis',
                explanation: 'Decomposes signals into constituent frequencies. Time domain → Frequency domain.',
                realWorld: 'Used in audio processing, image compression (JPEG), telecommunications, and signal analysis.',
                inputs: [
                    { name: 'signal_type', label: 'Signal Type', type: 'select', options: ['sine', 'cosine'], default: 'sine' },
                    { name: 'frequency', label: 'Frequency (Hz)', type: 'number', default: 1, min: 0.1, step: 0.1 }
                ]
            },
            {
                id: 'taylor_series',
                name: 'Taylor Series',
                description: 'Polynomial approximation of functions',
                explanation: 'Taylor series approximates functions using polynomials. More terms = better approximation.',
                realWorld: 'Used in numerical methods, calculator algorithms, physics approximations, and engineering calculations.',
                inputs: [
                    { name: 'function', label: 'Function', type: 'select', options: ['exp', 'sin', 'cos'], default: 'exp' },
                    { name: 'point', label: 'Expansion Point', type: 'number', default: 0, step: 0.1 },
                    { name: 'terms', label: 'Number of Terms', type: 'number', default: 5, min: 1, max: 10, step: 1 }
                ]
            }
        ]
    }
};

// ==================== STATE MANAGEMENT ====================
let currentCategory = null;
let currentConcept = null;

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Mathematical Expert System - Initializing...');
    setupEventListeners();
    setupExampleQueries();
    console.log('✅ All event listeners attached successfully');
});

// ==================== EVENT LISTENERS SETUP ====================
function setupEventListeners() {
    // Category selection
    const categoryCards = document.querySelectorAll('.category-card');
    console.log(`Found ${categoryCards.length} category cards`);
    categoryCards.forEach(card => {
        card.addEventListener('click', function() {
            const category = this.dataset.category;
            console.log(`Category selected: ${category}`);
            selectCategory(category);
        });
    });
    
    // Calculate button
    const calculateBtn = document.getElementById('calculateBtn');
    if (calculateBtn) {
        calculateBtn.addEventListener('click', handleCalculate);
        console.log('✓ Calculate button listener attached');
    }
    
    // Natural language query
    const askButton = document.getElementById('askButton');
    const nlQueryInput = document.getElementById('nlQueryInput');
    
    if (askButton) {
        askButton.addEventListener('click', handleNaturalLanguageQuery);
        console.log('✓ Ask button listener attached');
    }
    
    if (nlQueryInput) {
        nlQueryInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                handleNaturalLanguageQuery();
            }
        });
        console.log('✓ Query input listener attached');
    }
}

function setupExampleQueries() {
    const exampleBtns = document.querySelectorAll('.example-query');
    console.log(`Found ${exampleBtns.length} example query buttons`);
    exampleBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const query = this.dataset.query;
            const inputField = document.getElementById('nlQueryInput');
            if (inputField) {
                inputField.value = query;
                handleNaturalLanguageQuery();
            }
        });
    });
}

// ==================== CATEGORY SELECTION ====================
function selectCategory(category) {
    console.log(`Selecting category: ${category}`);
    currentCategory = category;
    
    // Update UI - remove active from all
    document.querySelectorAll('.category-card').forEach(c => c.classList.remove('active'));
    
    // Add active to selected
    const selectedCard = document.querySelector(`[data-category="${category}"]`);
    if (selectedCard) {
        selectedCard.classList.add('active');
        console.log(`✓ Category card marked as active`);
    }
    
    // Load concepts
    loadConcepts(category);
    
    // Show welcome screen
    const welcomeScreen = document.getElementById('welcomeScreen');
    const conceptContent = document.getElementById('conceptContent');
    
    if (welcomeScreen) welcomeScreen.style.display = 'block';
    if (conceptContent) conceptContent.style.display = 'none';
    
    console.log(`✓ Category ${category} selected successfully`);
}

// ==================== LOAD CONCEPTS ====================
function loadConcepts(category) {
    const list = document.getElementById('conceptsList');
    const data = conceptsData[category];
    
    if (!list) {
        console.error('Concepts list element not found');
        return;
    }
    
    if (!data) {
        console.error(`No data found for category: ${category}`);
        return;
    }
    
    console.log(`Loading ${data.concepts.length} concepts for ${category}`);
    
    list.innerHTML = '';
    data.concepts.forEach((concept, index) => {
        const btn = document.createElement('button');
        btn.className = 'concept-btn';
        btn.textContent = concept.name;
        btn.setAttribute('data-concept-id', concept.id);
        btn.addEventListener('click', function() {
            selectConcept(category, concept);
        });
        list.appendChild(btn);
    });
    
    console.log(`✓ Loaded ${data.concepts.length} concepts`);
}

// ==================== CONCEPT SELECTION ====================
function selectConcept(category, concept) {
    console.log(`Selecting concept: ${concept.name} (${concept.id})`);
    currentConcept = concept;
    
    // Update UI - remove active from all concept buttons
    document.querySelectorAll('.concept-btn').forEach(b => b.classList.remove('active'));
    
    // Add active to selected
    const selectedBtn = document.querySelector(`[data-concept-id="${concept.id}"]`);
    if (selectedBtn) {
        selectedBtn.classList.add('active');
    }
    
    // Show concept content
    const welcomeScreen = document.getElementById('welcomeScreen');
    const conceptContent = document.getElementById('conceptContent');
    
    if (welcomeScreen) welcomeScreen.style.display = 'none';
    if (conceptContent) conceptContent.style.display = 'block';
    
    // Update content
    const titleEl = document.getElementById('conceptTitle');
    const descEl = document.getElementById('conceptDescription');
    const explainEl = document.getElementById('conceptExplanation');
    const realWorldEl = document.getElementById('conceptRealWorld');
    
    if (titleEl) titleEl.textContent = concept.name;
    if (descEl) descEl.textContent = concept.description;
    if (explainEl) explainEl.textContent = concept.explanation;
    if (realWorldEl) realWorldEl.textContent = concept.realWorld;
    
    // Build form
    buildForm(concept);
    
    // Hide previous results
    const resultBox = document.getElementById('resultBox');
    const vizCard = document.getElementById('visualizationCard');
    if (resultBox) resultBox.style.display = 'none';
    if (vizCard) vizCard.style.display = 'none';
    
    // Apply gradient color
    const color = conceptsData[category].color;
    if (conceptContent) {
        conceptContent.style.setProperty('--gradient-color', color);
    }
    
    console.log(`✓ Concept ${concept.name} loaded successfully`);
}

// ==================== BUILD FORM ====================
function buildForm(concept) {
    const form = document.getElementById('calculatorForm');
    if (!form) {
        console.error('Calculator form not found');
        return;
    }
    
    console.log(`Building form with ${concept.inputs.length} inputs`);
    form.innerHTML = '';
    
    concept.inputs.forEach(input => {
        const group = document.createElement('div');
        group.className = 'form-group';
        
        const label = document.createElement('label');
        label.textContent = input.label;
        label.setAttribute('for', input.name);
        group.appendChild(label);
        
        if (input.type === 'select') {
            const select = document.createElement('select');
            select.name = input.name;
            select.id = input.name;
            input.options.forEach(opt => {
                const option = document.createElement('option');
                option.value = opt;
                option.textContent = opt;
                if (opt === input.default) option.selected = true;
                select.appendChild(option);
            });
            group.appendChild(select);
        } else {
            const inputEl = document.createElement('input');
            inputEl.type = input.type;
            inputEl.name = input.name;
            inputEl.id = input.name;
            inputEl.value = input.default;
            if (input.type === 'number') {
                inputEl.step = input.step || '0.01';
                if (input.min !== undefined) inputEl.min = input.min;
                if (input.max !== undefined) inputEl.max = input.max;
            }
            group.appendChild(inputEl);
        }
        
        form.appendChild(group);
    });
    
    console.log(`✓ Form built successfully`);
}

// ==================== HANDLE CALCULATE ====================
async function handleCalculate() {
    if (!currentConcept) {
        console.error('No concept selected');
        alert('Please select a concept first');
        return;
    }
    
    console.log(`Calculating for concept: ${currentConcept.id}`);
    
    const form = document.getElementById('calculatorForm');
    if (!form) {
        console.error('Form not found');
        return;
    }
    
    const formData = new FormData(form);
    const params = {};
    for (let [key, value] of formData.entries()) {
        params[key] = value;
    }
    
    console.log('Parameters:', params);
    
    // Show loading state
    const calculateBtn = document.getElementById('calculateBtn');
    const originalText = calculateBtn.innerHTML;
    calculateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Computing...';
    calculateBtn.disabled = true;
    
    try {
        console.log('Sending request to /api/compute...');
        const response = await fetch('/api/compute', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({concept: currentConcept.id, params})
        });
        
        console.log('Response status:', response.status);
        const data = await response.json();
        console.log('Response data:', data);
        
        if (data.success) {
            displayResult(data.result);
        } else {
            alert('Error: ' + (data.error || 'Unknown error occurred'));
            console.error('Computation error:', data.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Request error:', error);
    } finally {
        calculateBtn.innerHTML = originalText;
        calculateBtn.disabled = false;
    }
}

// ==================== DISPLAY RESULT ====================
function displayResult(result) {
    console.log('Displaying result:', result);
    
    const box = document.getElementById('resultBox');
    const content = document.getElementById('resultContent');
    const vizCard = document.getElementById('visualizationCard');
    const vizContainer = document.getElementById('visualizationContainer');
    
    if (!box || !content) {
        console.error('Result display elements not found');
        return;
    }
    
    box.style.display = 'block';
    
    // Format result
    let html = '';
    
    if (result.error) {
        html = `<div style="color: #ef4444; font-weight: 700; font-size: 1.2rem;">
            <i class="fas fa-exclamation-triangle"></i> Error: ${result.error}
        </div>`;
    } else {
        // Display main result
        if (result.result !== undefined) {
            html += `<div class="result-main">${formatResult(result.result)}</div>`;
        }
        
        // Display additional details
        const detailKeys = Object.keys(result).filter(k => 
            !['result', 'visualization', 'prolog_used', 'error', 'explanation', 'prolog_result'].includes(k)
        );
        
        if (detailKeys.length > 0) {
            html += '<div class="result-details">';
            detailKeys.forEach(key => {
                html += `<div class="result-detail">
                    <strong>${formatKey(key)}:</strong> ${formatResult(result[key])}
                </div>`;
            });
            html += '</div>';
        }
        
        // Prolog indicator
        if (result.prolog_used !== undefined) {
            const engine = result.prolog_used ? 'Prolog AI ✓' : 'Python Fallback';
            const icon = result.prolog_used ? 'robot' : 'code';
            const color = result.prolog_used ? '#a78bfa' : '#f59e0b';
            html += `<div style="color: ${color}; margin-top: 15px; padding-top: 15px; border-top: 1px solid #475569; font-weight: 600;">
                <i class="fas fa-${icon}"></i> Computed using ${engine}
            </div>`;
        }
        
        // Display explanation from Prolog
        if (result.explanation) {
            html += `<div style="margin-top: 15px; padding: 15px; background: rgba(96, 165, 250, 0.1); border-left: 3px solid #60a5fa; border-radius: 8px;">
                <strong style="color: #60a5fa;"><i class="fas fa-info-circle"></i> Prolog Explanation:</strong><br>
                <span style="color: #cbd5e1; font-size: 0.95rem;">${result.explanation}</span>
            </div>`;
        }
    }
    
    content.innerHTML = html;
    
    // Handle visualization
    if (result.visualization && vizCard && vizContainer) {
        vizCard.style.display = 'block';
        vizContainer.innerHTML = `<img src="${result.visualization}" alt="Visualization" style="max-width: 100%; height: auto; border-radius: 12px; border: 2px solid #475569; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);">`;
    } else if (vizCard) {
        vizCard.style.display = 'none';
    }
    
    // Smooth scroll to result
    box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    console.log('✓ Result displayed successfully');
}

// ==================== NATURAL LANGUAGE QUERY ====================
async function handleNaturalLanguageQuery() {
    const inputField = document.getElementById('nlQueryInput');
    if (!inputField) {
        console.error('Query input field not found');
        return;
    }
    
    const query = inputField.value.trim();
    if (!query) {
        showError('Please enter a question');
        return;
    }
    
    console.log(`Processing natural language query: "${query}"`);
    
    const resultDiv = document.getElementById('queryResult');
    if (!resultDiv) {
        console.error('Query result div not found');
        return;
    }
    
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> Processing your question...</div>';
    resultDiv.classList.remove('error');
    
    try {
        console.log('Sending request to /api/query...');
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query: query})
        });
        
        console.log('Response status:', response.status);
        const data = await response.json();
        console.log('Response data:', data);
        
        if (data.success) {
            displayNLQueryResult(data);
        } else {
            showError(data.error || 'Could not process your question. Please try rephrasing or select a concept from the categories.');
        }
    } catch (error) {
        console.error('Query error:', error);
        showError('Error: ' + error.message);
    }
}

function displayNLQueryResult(data) {
    console.log('Displaying NL query result:', data);
    const resultDiv = document.getElementById('queryResult');
    if (!resultDiv) return;
    
    const result = data.result;
    
    let html = '<h3><i class="fas fa-check-circle"></i> Answer</h3>';
    
    // Display the main result
    if (result.error) {
        html += `<p style="color: #ef4444; font-weight: 600;"><i class="fas fa-exclamation-triangle"></i> ${result.error}</p>`;
    } else {
        // Format the result based on type
        if (result.result !== undefined) {
            html += `<p style="font-size: 1.2rem; font-weight: 700; color: #10b981; margin: 15px 0;">
                <strong>Result:</strong> ${formatResult(result.result)}
            </p>`;
        }
        
        // Additional information
        Object.keys(result).forEach(key => {
            if (!['result', 'visualization', 'prolog_used', 'error', 'explanation', 'prolog_result'].includes(key)) {
                html += `<p><strong>${formatKey(key)}:</strong> ${formatResult(result[key])}</p>`;
            }
        });
        
        // Prolog indicator
        if (result.prolog_used !== undefined) {
            const engine = result.prolog_used ? 'Prolog AI ✓' : 'Python';
            const color = result.prolog_used ? '#a78bfa' : '#f59e0b';
            html += `<p style="color: ${color}; margin-top: 15px; font-weight: 600;">
                <i class="fas fa-${result.prolog_used ? 'robot' : 'code'}"></i> Computed using ${engine}
            </p>`;
        }
        
        // Display explanation
        if (result.explanation) {
            html += `<div style="margin-top: 20px; padding: 15px; background: rgba(96, 165, 250, 0.1); border-left: 3px solid #60a5fa; border-radius: 8px;">
                <strong style="color: #60a5fa;"><i class="fas fa-info-circle"></i> Explanation:</strong><br>
                <span style="color: #cbd5e1; margin-top: 8px; display: block;">${result.explanation}</span>
            </div>`;
        }
        
        // Visualization
        if (result.visualization) {
            html += `<div class="visualization-container" style="margin-top: 20px;">
                <img src="${result.visualization}" alt="Visualization" style="max-width: 100%; border-radius: 12px; margin-top: 15px; border: 2px solid #475569; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);">
            </div>`;
        }
    }
    
    // Show related concept info if available
    if (data.concept && conceptsData) {
        const conceptInfo = findConceptInfo(data.concept);
        if (conceptInfo) {
            html += `<div class="description" style="margin-top: 25px; padding: 20px; background: rgba(15, 23, 42, 0.6); border-radius: 10px; border-left: 4px solid #10b981;">
                <strong style="color: #10b981; font-size: 1.1rem;">About ${conceptInfo.name}:</strong><br>
                <span style="display: block; margin-top: 10px; color: #cbd5e1;">${conceptInfo.description}</span><br>
                <em style="display: block; margin-top: 10px; color: #94a3b8;">${conceptInfo.explanation}</em>
            </div>`;
        }
    }
    
    resultDiv.innerHTML = html;
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    console.log('✓ NL query result displayed');
}

function findConceptInfo(conceptId) {
    for (let category in conceptsData) {
        const concept = conceptsData[category].concepts.find(c => c.id === conceptId);
        if (concept) {
            console.log(`Found concept info for ${conceptId}`);
            return concept;
        }
    }
    console.log(`No concept info found for ${conceptId}`);
    return null;
}

function showError(message) {
    console.error('Showing error:', message);
    const resultDiv = document.getElementById('queryResult');
    if (!resultDiv) return;
    
    resultDiv.style.display = 'block';
    resultDiv.classList.add('error');
    resultDiv.innerHTML = `<h3><i class="fas fa-exclamation-circle"></i> Error</h3><p>${message}</p>`;
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ==================== UTILITY FUNCTIONS ====================
function formatKey(key) {
    return key
        .replace(/_/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase());
}

function formatResult(value) {
    if (value === null || value === undefined) {
        return 'N/A';
    }
    
    if (Array.isArray(value)) {
        if (value.length === 0) return '[]';
        
        // Check if it's an array of arrays (matrix)
        if (Array.isArray(value[0])) {
            return '<pre style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 6px; margin: 10px 0;">' + 
                   JSON.stringify(value, null, 2) + '</pre>';
        }
        
        // Regular array
        const formatted = value.map(v => 
            typeof v === 'number' ? v.toFixed(3) : v
        ).join(', ');
        return `[${formatted}]`;
    } else if (typeof value === 'object') {
        return '<pre style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 6px; margin: 10px 0;">' + 
               JSON.stringify(value, null, 2) + '</pre>';
    } else if (typeof value === 'number') {
        // Format numbers nicely
        if (Number.isInteger(value)) {
            return value.toLocaleString();
        }
        return value.toFixed(4);
    } else if (typeof value === 'boolean') {
        return value ? '<span style="color: #10b981;">✓ True</span>' : '<span style="color: #ef4444;">✗ False</span>';
    } else {
        return String(value);
    }
}

// ==================== CONSOLE LOGGING ====================
console.log(`
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║     🧮 MATHEMATICAL EXPERT SYSTEM - Frontend Initialized      ║
║                                                                ║
║  ✓ Concept Data Loaded: ${Object.keys(conceptsData).length} categories                         ║
║  ✓ Event Listeners: Ready                                     ║
║  ✓ Natural Language: Ready                                    ║
║  ✓ Visualizations: Ready                                      ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
`);

// Log all available concepts
let totalConcepts = 0;
Object.keys(conceptsData).forEach(cat => {
    totalConcepts += conceptsData[cat].concepts.length;
});
console.log(`📊 Total Concepts Available: ${totalConcepts}`);
console.log('🎨 Categories:', Object.keys(conceptsData).join(', '));
console.log('✅ JavaScript fully loaded and ready!');

// Export for debugging
window.mathExpertSystem = {
    conceptsData,
    selectCategory,
    selectConcept,
    handleCalculate,
    handleNaturalLanguageQuery,
    currentCategory: () => currentCategory,
    currentConcept: () => currentConcept
};

console.log('🔧 Debug tools available in window.mathExpertSystem');
