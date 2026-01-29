% ============================================================================
% ENHANCED MATHEMATICAL EXPERT SYSTEM KNOWLEDGE BASE
% math_expert_system.pl (FULLY CORRECTED - No Warnings)
% 
% Comprehensive Prolog knowledge base with 50+ mathematical concepts
% ============================================================================

% Declare discontiguous predicates
:- discontiguous related_concept/2.
:- discontiguous application/2.
:- discontiguous rate_of_change/1.
:- discontiguous slope_related/1.

:- discontiguous compute/3.
:- discontiguous compute/4.
:- discontiguous keyword/2.
:- discontiguous explanation/2.

% ============================================================================
% MATHEMATICAL CONCEPTS (50+ Concepts)
% ============================================================================

% Calculus
concept(derivative).
concept(integration).
concept(limit).
concept(partial_derivative).
concept(second_derivative).
concept(implicit_differentiation).
concept(chain_rule).
concept(product_rule).
concept(quotient_rule).

% Algebra
concept(linear_equation).
concept(quadratic_equation).
concept(cubic_equation).
concept(polynomial).
concept(system_of_equations).
concept(inequality).
concept(absolute_value).

% Exponential & Logarithmic
concept(logarithm).
concept(natural_log).
concept(exponential).
concept(exponential_growth).
concept(exponential_decay).

% Trigonometry
concept(trigonometry_sin).
concept(trigonometry_cos).
concept(trigonometry_tan).
concept(inverse_trig).
concept(trigonometric_identity).

% Vectors
concept(dot_product).
concept(cross_product).
concept(modulus).
concept(unit_vector).
concept(vector_projection).
concept(vector_angle).

% Statistics
concept(mean).
concept(median).
concept(mode).
concept(variance).
concept(standard_deviation).
concept(range).
concept(percentile).
concept(correlation).
concept(covariance).
concept(regression).
concept(z_score).

% Probability
concept(probability).
concept(conditional_probability).
concept(bayes_theorem).
concept(expected_value).

% Combinatorics
concept(permutation).
concept(combination).
concept(factorial).
concept(binomial_coefficient).

% Number Theory
concept(gcd).
concept(lcm).
concept(prime).
concept(prime_factorization).
concept(modular_arithmetic).

% Functions
concept(floor_function).
concept(ceiling_function).
concept(round_function).
concept(sign_function).

% Matrix Operations
concept(matrix_multiplication).
concept(matrix_addition).
concept(matrix_determinant).
concept(matrix_inverse).
concept(matrix_transpose).
concept(eigenvalue).
concept(eigenvector).

% Advanced Topics
concept(fourier_transform).
concept(laplace_transform).
concept(convolution).
concept(taylor_series).
concept(maclaurin_series).

% ============================================================================
% KEYWORDS FOR CONCEPT DETECTION
% ============================================================================

% Derivative keywords
keyword(derivative, derivative).
keyword(derivative, differentiate).
keyword(derivative, rate).
keyword(derivative, change).
keyword(derivative, slope).
keyword(derivative, tangent).
keyword(derivative, gradient).
keyword(derivative, instantaneous).
keyword(derivative, velocity).
keyword(derivative, acceleration).

% Integration keywords
keyword(integration, integral).
keyword(integration, integrate).
keyword(integration, area).
keyword(integration, antiderivative).
keyword(integration, accumulation).
keyword(integration, sum).
keyword(integration, total).

% Limit keywords
keyword(limit, limit).
keyword(limit, approach).
keyword(limit, tends).
keyword(limit, converge).
keyword(limit, infinity).
keyword(limit, continuous).

% Partial Derivative keywords
keyword(partial_derivative, partial).
keyword(partial_derivative, multivariable).
keyword(partial_derivative, respect).

% Second Derivative keywords
keyword(second_derivative, second).
keyword(second_derivative, concavity).
keyword(second_derivative, inflection).

% Linear Equation keywords
keyword(linear_equation, linear).
keyword(linear_equation, solve).
keyword(linear_equation, straight).
keyword(linear_equation, line).
keyword(linear_equation, equation).

% Quadratic Equation keywords
keyword(quadratic_equation, quadratic).
keyword(quadratic_equation, square).
keyword(quadratic_equation, squared).
keyword(quadratic_equation, parabola).
keyword(quadratic_equation, roots).
keyword(quadratic_equation, discriminant).

% Cubic keywords
keyword(cubic_equation, cubic).
keyword(cubic_equation, third).
keyword(cubic_equation, degree).

% Polynomial keywords
keyword(polynomial, polynomial).
keyword(polynomial, quartic).
keyword(polynomial, quintic).

% Logarithm keywords
keyword(logarithm, log).
keyword(logarithm, logarithm).
keyword(logarithm, logarithmic).
keyword(natural_log, ln).
keyword(natural_log, natural).

% Exponential keywords
keyword(exponential, exponential).
keyword(exponential, exp).
keyword(exponential_growth, growth).
keyword(exponential_decay, decay).
keyword(exponential, power).

% Trigonometry keywords
keyword(trigonometry_sin, sin).
keyword(trigonometry_sin, sine).
keyword(trigonometry_cos, cos).
keyword(trigonometry_cos, cosine).
keyword(trigonometry_tan, tan).
keyword(trigonometry_tan, tangent).
keyword(inverse_trig, arcsin).
keyword(inverse_trig, arccos).
keyword(inverse_trig, arctan).
keyword(inverse_trig, inverse).
keyword(trigonometric_identity, identity).
keyword(trigonometric_identity, pythagorean).

% Dot Product keywords
keyword(dot_product, dot).
keyword(dot_product, scalar).
keyword(dot_product, inner).

% Cross Product keywords
keyword(cross_product, cross).
keyword(cross_product, vector).
keyword(cross_product, perpendicular).
keyword(cross_product, orthogonal).

% Modulus keywords
keyword(modulus, modulus).
keyword(modulus, magnitude).
keyword(modulus, length).
keyword(modulus, norm).

% Unit Vector keywords
keyword(unit_vector, unit).
keyword(unit_vector, normalize).
keyword(unit_vector, direction).

% Vector Projection keywords
keyword(vector_projection, projection).
keyword(vector_projection, project).
keyword(vector_projection, component).

% Mean keywords
keyword(mean, mean).
keyword(mean, average).
keyword(mean, avg).
keyword(mean, central).

% Median keywords
keyword(median, median).
keyword(median, middle).
keyword(median, center).

% Mode keywords
keyword(mode, mode).
keyword(mode, frequent).
keyword(mode, common).

% Variance keywords
keyword(variance, variance).
keyword(variance, spread).
keyword(variance, dispersion).

% Standard Deviation keywords
keyword(standard_deviation, standard).
keyword(standard_deviation, deviation).
keyword(standard_deviation, variability).

% Range keywords
keyword(range, range).
keyword(range, span).
keyword(range, extent).

% Percentile keywords
keyword(percentile, percentile).
keyword(percentile, quantile).
keyword(percentile, quartile).

% Correlation keywords
keyword(correlation, correlation).
keyword(correlation, relationship).
keyword(correlation, association).

% Covariance keywords
keyword(covariance, covariance).
keyword(covariance, joint).

% Regression keywords
keyword(regression, regression).
keyword(regression, predict).
keyword(regression, trend).
keyword(regression, linear_regression).

% Z-Score keywords
keyword(z_score, zscore).
keyword(z_score, standardize).
keyword(z_score, normalize).

% Probability keywords
keyword(probability, probability).
keyword(probability, chance).
keyword(probability, likely).
keyword(probability, odds).
keyword(probability, random).

% Conditional Probability keywords
keyword(conditional_probability, conditional).
keyword(conditional_probability, given).

% Bayes Theorem keywords
keyword(bayes_theorem, bayes).
keyword(bayes_theorem, bayesian).

% Expected Value keywords
keyword(expected_value, expected).
keyword(expected_value, expectation).
keyword(expected_value, mean_value).

% Permutation keywords
keyword(permutation, permutation).
keyword(permutation, arrangement).
keyword(permutation, order).
keyword(permutation, sequence).

% Combination keywords
keyword(combination, combination).
keyword(combination, choose).
keyword(combination, selection).
keyword(combination, subset).

% Factorial keywords
keyword(factorial, factorial).
keyword(factorial, product).

% Binomial Coefficient keywords
keyword(binomial_coefficient, binomial).
keyword(binomial_coefficient, coefficient).

% GCD keywords
keyword(gcd, gcd).
keyword(gcd, greatest).
keyword(gcd, common).
keyword(gcd, divisor).

% LCM keywords
keyword(lcm, lcm).
keyword(lcm, least).
keyword(lcm, multiple).

% Prime keywords
keyword(prime, prime).
keyword(prime, factor).
keyword(prime_factorization, factorization).
keyword(prime_factorization, decomposition).

% Modular Arithmetic keywords
keyword(modular_arithmetic, modulo).
keyword(modular_arithmetic, mod).
keyword(modular_arithmetic, remainder).

% Floor Function keywords
keyword(floor_function, floor).
keyword(floor_function, round).
keyword(floor_function, down).

% Ceiling Function keywords
keyword(ceiling_function, ceiling).
keyword(ceiling_function, ceil).
keyword(ceiling_function, up).

% Round keywords
keyword(round_function, round).
keyword(round_function, nearest).

% Absolute Value keywords
keyword(absolute_value, absolute).
keyword(absolute_value, abs).
keyword(absolute_value, magnitude).

% Sign Function keywords
keyword(sign_function, sign).
keyword(sign_function, signum).

% Matrix keywords
keyword(matrix_multiplication, matrix).
keyword(matrix_multiplication, multiply).
keyword(matrix_addition, add).
keyword(matrix_determinant, determinant).
keyword(matrix_determinant, det).
keyword(matrix_inverse, inverse).
keyword(matrix_transpose, transpose).
keyword(eigenvalue, eigenvalue).
keyword(eigenvalue, characteristic).
keyword(eigenvector, eigenvector).

% Advanced topics keywords
keyword(fourier_transform, fourier).
keyword(fourier_transform, frequency).
keyword(laplace_transform, laplace).
keyword(convolution, convolution).
keyword(convolution, convolve).
keyword(taylor_series, taylor).
keyword(taylor_series, series).
keyword(maclaurin_series, maclaurin).

% ============================================================================
% CONCEPT DETECTION RULES - FIXED
% ============================================================================

% Detect concept from list of words
detect_concept(Words, Concept) :-
    concept(Concept),
    member(Word, Words),
    keyword(Concept, Word).

% Get all matching concepts
get_all_concepts(Words, Concepts) :-
    findall(C, detect_concept(Words, C), AllConcepts),
    list_to_set(AllConcepts, Concepts).

% Get best matching concept (most keyword matches) - FULLY CORRECTED
get_best_concept(Words, BestConcept) :-
    findall(C-Count, 
            (concept(C), count_matches(Words, C, Count), Count > 0),
            ConceptCounts),
    ConceptCounts \= [],
    max_count(ConceptCounts, BestConcept).

count_matches(Words, Concept, Count) :-
    findall(W, (member(W, Words), keyword(Concept, W)), Matches),
    length(Matches, Count).

% CORRECTED: No singleton variables - using underscore for unused variables
max_count([C-_], C) :- !.
max_count([C1-Count1, _-Count2|Rest], Best) :-
    Count1 >= Count2,
    !,
    max_count([C1-Count1|Rest], Best).
max_count([_|Rest], Best) :-
    max_count(Rest, Best).

% ============================================================================
% HELPER FUNCTIONS
% ============================================================================

% Factorial
factorial(0, 1) :- !.
factorial(N, F) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, F1),
    F is N * F1.

% GCD (Euclidean algorithm)
gcd_compute(A, 0, A) :- !.
gcd_compute(A, B, G) :-
    R is A mod B,
    gcd_compute(B, R, G).

% LCM
lcm_compute(A, B, L) :-
    gcd_compute(A, B, G),
    L is (A * B) // G.

% Check if prime
is_prime(N) :-
    N > 1,
    \+ has_factor(N, 2).

has_factor(N, F) :-
    F * F =< N,
    (   N mod F =:= 0
    ;   F1 is F + 1,
        has_factor(N, F1)
    ).

% Square difference for variance
square_diff(Mean, X, Diff) :-
    Diff is (X - Mean) * (X - Mean).

% correlation
% Sum of (X - MeanX) * (Y - MeanY)
sum_xy([], [], _, _, 0).
sum_xy([X|Xs], [Y|Ys], MX, MY, Sum) :-
    sum_xy(Xs, Ys, MX, MY, Rest),
    Sum is Rest + (X - MX) * (Y - MY).

% Sum of (X - Mean)^2
sum_sq([], _, 0).
sum_sq([X|Xs], Mean, Sum) :-
    sum_sq(Xs, Mean, Rest),
    Diff is X - Mean,
    Sum is Rest + Diff * Diff.

% Newton-Raphson helper for cubic
newton_cubic(_, _, _, _, X, 0, X) :- !.
newton_cubic(A, B, C, D, X, N, Root) :-
    N > 0,
    FX is A*X*X*X + B*X*X + C*X + D,
    FPrimeX is 3*A*X*X + 2*B*X + C,
    FPrimeX =\= 0,
    XNext is X - FX / FPrimeX,
    abs(XNext - X) < 0.0001,
    !,
    Root is XNext.
newton_cubic(A, B, C, D, X, N, Root) :-
    N > 0,
    FX is A*X*X*X + B*X*X + C*X + D,
    FPrimeX is 3*A*X*X + 2*B*X + C,
    FPrimeX =\= 0,
    XNext is X - FX / FPrimeX,
    N1 is N - 1,
    newton_cubic(A, B, C, D, XNext, N1, Root).

% ============================================================================
% COMPUTATION RULES (50+ Rules)
% ============================================================================

% Association rules
related_concept(X, Y) :- keyword(X, Y).

% --------- RULES ---------
% Rule 1: related concept
related_concept(X, Y) :- keyword(X, Y).

% Rule 2: derivative represents rate of change
rate_of_change(X) :- 
    keyword(X, rate);
    keyword(X, change);
    keyword(X, instantaneous).

% Rule 3: derivative related to slope/tangent/gradient
slope_related(X) :- 
    keyword(X, slope);
    keyword(X, tangent);
    keyword(X, gradient).

% Rule 4: derivative in physics applications
application(derivative, velocity) :- keyword(derivative, velocity).
application(derivative, acceleration) :- keyword(derivative, acceleration).


% ===== CALCULUS =====

% Derivative - related concepts
related_concept(derivative, differentiation).
rate_of_change(derivative).
slope_related(derivative).
application(derivative, velocity).
application(derivative, acceleration).

% Integration - related concepts
related_concept(integration, antiderivative).
application(integration, area_under_curve).

% Limit - related concepts
related_concept(limit, approaching).
application(limit, continuity).
application(limit, instantaneous_rate).

% Partial Derivative
related_concept(partial_derivative, multivariable_calculus).
application(partial_derivative, gradient).
application(partial_derivative, tangent_plane).

% Second Derivative
related_concept(second_derivative, concavity).
application(second_derivative, acceleration).

% ===== ALGEBRA =====

% Linear Equation
related_concept(linear_equation, straight_line).
application(linear_equation, slope_intercept_form).

% Quadratic Equation
related_concept(quadratic_equation, parabola).
application(quadratic_equation, projectile_motion).

% Cubic Equation
related_concept(cubic_equation, polynomial).
application(cubic_equation, curve_fitting).

% System of Linear Equations
related_concept(system_of_equations, simultaneous_equations).
application(system_of_equations, intersection_points).

% ===== FUNCTIONS =====

% Floor Function
related_concept(floor_function, greatest_integer).
application(floor_function, rounding_down).

% Ceiling Function
related_concept(ceiling_function, least_integer).
application(ceiling_function, rounding_up).

% Round Function
related_concept(round_function, approximation).
application(round_function, nearest_integer).

% Absolute Value
related_concept(absolute_value, distance).
application(absolute_value, magnitude).

% Sign Function
related_concept(sign_function, positive_negative).
application(sign_function, direction).

% ===== STATISTICS =====

% Mean
related_concept(mean, average).
application(mean, central_tendency).

% Median
related_concept(median, middle_value).
application(median, central_tendency).

% Mode
related_concept(mode, most_frequent).
application(mode, central_tendency).

% Percentile
related_concept(percentile, rank).
application(percentile, score_comparison).

% Correlation Coefficient
related_concept(correlation, relationship).
application(correlation, linear_trend).

% Variance
related_concept(variance, dispersion).
application(variance, risk_measure).

% Standard Deviation
related_concept(standard_deviation, spread).
application(standard_deviation, uncertainty).

% Range
related_concept(range, difference).
application(range, spread).

% Z-Score
related_concept(z_score, standardization).
application(z_score, comparison).

% ===== VECTORS =====

% Dot Product
related_concept(dot_product, scalar_product).
application(dot_product, projection).

% Cross Product
related_concept(cross_product, vector_product).
application(cross_product, perpendicular_vector).

% Vector Modulus
related_concept(modulus, magnitude).
application(modulus, length).

% Vector Angle
related_concept(vector_angle, direction).
application(vector_angle, orientation).

% Unit Vector
related_concept(unit_vector, normalized_vector).
application(unit_vector, direction_only).

% Vector Projection
related_concept(vector_projection, shadow_vector).
application(vector_projection, component_along).

% ===== COMBINATORICS =====

% Permutation
related_concept(permutation, arrangement).
application(permutation, order_matters).

% Combination
related_concept(combination, selection).
application(combination, order_does_not_matter).

% Binomial Coefficient
related_concept(binomial_coefficient, combination).
application(binomial_coefficient, pascal_triangle).

% Factorial
related_concept(factorial, product_sequence).
application(factorial, counting_permutations).

% ===== NUMBER THEORY =====

% GCD
related_concept(gcd, greatest_common_divisor).
application(gcd, simplification).

% LCM
related_concept(lcm, least_common_multiple).
application(lcm, scheduling).

% Prime Check
related_concept(prime, indivisible_number).
application(prime, cryptography).

% Modular Arithmetic
related_concept(modular_arithmetic, remainder).
application(modular_arithmetic, clock_arithmetic).

% ===== PROBABILITY =====

% Basic Probability
related_concept(probability, chance).
application(probability, prediction).

% Conditional Probability
related_concept(conditional_probability, dependent_events).
application(conditional_probability, bayes_theorem).

% Bayes Theorem
related_concept(bayes_theorem, conditional_probability).
application(bayes_theorem, diagnostic).

% Expected Value
related_concept(expected_value, average_outcome).
application(expected_value, decision_making).

% ===== LOGARITHM & EXPONENTIAL =====

% Natural Logarithm
related_concept(natural_log, ln).
application(natural_log, continuous_growth).

% Logarithm with Base
related_concept(logarithm, log_base).
application(logarithm, scaling).

% Exponential
related_concept(exponential, exp).
application(exponential, growth_decay).

% Exponential Growth
related_concept(exponential_growth, population_growth).
application(exponential_growth, finance).

% Exponential Decay
related_concept(exponential_decay, radioactive_decay).
application(exponential_decay, half_life).

% Power
related_concept(power, exponentiation).
application(power, scaling).

% ===== TRIGONOMETRY =====

% Sine
related_concept(trigonometry_sin, ratio).
application(trigonometry_sin, waves).

% Cosine
related_concept(trigonometry_cos, ratio).
application(trigonometry_cos, waves).

% Tangent
related_concept(trigonometry_tan, ratio).
application(trigonometry_tan, slope).

% Inverse Trigonometric
related_concept(inverse_trig, arcsin_arccos_arctan).
application(inverse_trig, angle_calculation).

% ===== MATRICES =====

% Matrix Multiplication
related_concept(matrix_multiplication, product).
application(matrix_multiplication, transformation).

% Matrix Determinant
related_concept(matrix_determinant, scalar_measure).
application(matrix_determinant, invertibility).

% Matrix Inverse
related_concept(matrix_inverse, reciprocal_matrix).
application(matrix_inverse, solving_linear_systems).

% Eigenvalues
related_concept(eigenvalue, characteristic_value).
application(eigenvalue, stability_analysis).

% ===== ADVANCED =====

% Fourier Transform
related_concept(fourier_transform, frequency_domain).
application(fourier_transform, signal_analysis).

% Taylor Series
related_concept(taylor_series, series_expansion).
application(taylor_series, approximation).


% ===== CALCULUS =====

% Derivative - Power Rule
compute(derivative, [power, Base, N], [multiply, N, [power, Base, N1]]) :-
    N > 0,
    N1 is N - 1.

% Derivative - Constant Rule
compute(derivative, [constant, _], [constant, 0]).

% Derivative - Trigonometric
compute(derivative, [sin, X], [cos, X]).
compute(derivative, [cos, X], [negative, [sin, X]]).
compute(derivative, [tan, X], [power, [sec, X], 2]).

% Derivative - Exponential and Logarithmic
compute(derivative, [exp, X], [exp, X]).
compute(derivative, [ln, X], [divide, 1, X]).
compute(derivative, [log, Base, X], [divide, 1, [multiply, X, [ln, Base]]]).

% Integration - Power Rule
compute(integration, [power, Base, N], [divide, [power, Base, N1], N1]) :-
    N1 is N + 1.

% Integration - Constant
compute(integration, [constant, C], [multiply, C, x]).

% Integration - Trigonometric
compute(integration, [sin, X], [negative, [cos, X]]).
compute(integration, [cos, X], [sin, X]).

% Limit 
f(X, Y) :- Y is X*X + 5*X + 2.
compute(limit, [Function, Point], Result) :-
    call(Function, Point, Result).

% partial_derivative
% with respect to x
compute(partial_derivative, [x, x, _Y], 1).
compute(partial_derivative, [x, _X, _Y], 0).

compute(partial_derivative, [pow(x, N), X, _Y], Result) :-
    Result is N * X ** (N - 1).

% with respect to y
compute(partial_derivative, [y, _X, y], 1).
compute(partial_derivative, [y, _X, _Y], 0).

compute(partial_derivative, [pow(y, N), _X, Y], Result) :-
    Result is N * Y ** (N - 1).

% Second Derivative
function(fx, [add, [power, x, 3], [sin, x]]).
compute(second_derivative, Function, SecondDeriv) :-
    compute(derivative, Function, FirstDeriv),
    compute(derivative, FirstDeriv, SecondDeriv).

% ===== ALGEBRA =====

% Linear Equation: ax + b = 0
compute(linear_equation, [A, B], X) :-
    A =\= 0,
    X is -B / A.

% Quadratic Equation: ax² + bx + c = 0
compute(quadratic_equation, [A, B, C], [Root1, Root2]) :-
    Discriminant is B*B - 4*A*C,
    Discriminant >= 0,
    SqrtDisc is sqrt(Discriminant),
    Root1 is (-B + SqrtDisc) / (2*A),
    Root2 is (-B - SqrtDisc) / (2*A).

% Cubic Equation (numerical approximation for one real root)
compute(cubic_equation, [A, B, C, D], Root) :-
    A =\= 0,
    % Simple case: find one real root using Newton-Raphson
    % Starting guess
    X0 = 0,
    newton_cubic(A, B, C, D, X0, 20, Root).

% System of 2x2 Linear Equations (Cramers Rule)
compute(system_of_equations, [[A1,B1,C1], [A2,B2,C2]], [X, Y]) :-
    Det is A1*B2 - A2*B1,
    Det =\= 0,
    X is (C1*B2 - C2*B1) / Det,
    Y is (A1*C2 - A2*C1) / Det.

% ===== FUNCTIONS =====

% Floor Function
compute(floor_function, X, Result) :-
    Result is floor(X).

% Ceiling Function
compute(ceiling_function, X, Result) :-
    Result is ceiling(X).

% Round Function
compute(round_function, X, Result) :-
    Result is round(X).

% Absolute Value
compute(absolute_value, X, Result) :-
    Result is abs(X).

% Sign Function
compute(sign_function, X, Result) :-
    (X > 0 -> Result = 1 ;
     X < 0 -> Result = -1 ;
     Result = 0).

% ===== STATISTICS =====

% Mean
compute(mean, List, Result) :-
    sum_list(List, Sum),
    length(List, N),
    N > 0,
    Result is Sum / N.

% Median
compute(median, List, Result) :-
    msort(List, Sorted),
    length(Sorted, N),
    (   N mod 2 =:= 1
    ->  Idx is N // 2,
        nth0(Idx, Sorted, Result)
    ;   Idx1 is N // 2 - 1,
        Idx2 is N // 2,
        nth0(Idx1, Sorted, Val1),
        nth0(Idx2, Sorted, Val2),
        Result is (Val1 + Val2) / 2
    ).

% Mode - most frequent value
compute(mode, List, Mode) :-
    findall(Count-Value, 
            (member(Value, List), 
             findall(V, (member(V, List), V =:= Value), Matches),
             length(Matches, Count)),
            Pairs),
    max_member(_MaxCount-Mode, Pairs).

% Percentile calculation
compute(percentile, [List, P], Result) :-
    msort(List, Sorted),
    length(Sorted, N),
    N > 0,
    Index is (N - 1) * (P / 100),
    Floor is floor(Index),
    Ceil is ceiling(Index),
    (   Floor =:= Ceil
    ->  nth0(Floor, Sorted, Result)
    ;   nth0(Floor, Sorted, V1),
        nth0(Ceil, Sorted, V2),
        Result is V1 * (Ceil - Index) + V2 * (Index - Floor)
    ).

% Correlation coefficient (Pearsons r)
compute(correlation, [ListX, ListY], R) :-
    length(ListX, N),
    length(ListY, N),
    N > 0,
    compute(mean, ListX, MeanX),
    compute(mean, ListY, MeanY),
    sum_xy(ListX, ListY, MeanX, MeanY, SumXY),
    sum_sq(ListX, MeanX, SumXX),
    sum_sq(ListY, MeanY, SumYY),
    SumXX > 0,
    SumYY > 0,
    R is SumXY / sqrt(SumXX * SumYY).

% Variance
compute(variance, List, Result) :-
    compute(mean, List, Mean),
    maplist(square_diff(Mean), List, Diffs),
    sum_list(Diffs, SumDiffs),
    length(List, N),
    N > 0,
    Result is SumDiffs / N.

% Standard Deviation
compute(standard_deviation, List, Result) :-
    compute(variance, List, Var),
    Result is sqrt(Var).

% Range
compute(range, List, Result) :-
    max_list(List, Max),
    min_list(List, Min),
    Result is Max - Min.

% Z-Score
compute(z_score, [Value, Mean, StdDev], Result) :-
    StdDev > 0,
    Result is (Value - Mean) / StdDev.

% ===== VECTORS (compute/4 for two arguments) =====

% Dot Product (2D)
compute(dot_product, [X1, Y1], [X2, Y2], Result) :-
    Result is X1*X2 + Y1*Y2.

% Dot Product (3D)
compute(dot_product, [X1, Y1, Z1], [X2, Y2, Z2], Result) :-
    Result is X1*X2 + Y1*Y2 + Z1*Z2.

% Cross Product (3D only)
compute(cross_product, [X1, Y1, Z1], [X2, Y2, Z2], [Rx, Ry, Rz]) :-
    Rx is Y1*Z2 - Z1*Y2,
    Ry is Z1*X2 - X1*Z2,
    Rz is X1*Y2 - Y1*X2.

% Vector Modulus (2D)
compute(modulus, [X, Y], Result) :-
    Result is sqrt(X*X + Y*Y).

% Vector Modulus (3D)
compute(modulus, [X, Y, Z], Result) :-
    Result is sqrt(X*X + Y*Y + Z*Z).

% Vector Angle (2D)
compute(vector_angle, [X1, Y1], [X2, Y2], Angle) :-
    compute(dot_product, [X1, Y1], [X2, Y2], Dot),
    compute(modulus, [X1, Y1], Mag1),
    compute(modulus, [X2, Y2], Mag2),
    Mag1 > 0, Mag2 > 0,
    CosAngle is Dot / (Mag1 * Mag2),
    Angle is acos(CosAngle).

% Unit Vector
compute(unit_vector, [X, Y], [Ux, Uy]) :-
    compute(modulus, [X, Y], Mag),
    Mag > 0,
    Ux is X / Mag,
    Uy is Y / Mag.

compute(unit_vector, [X, Y, Z], [Ux, Uy, Uz]) :-
    compute(modulus, [X, Y, Z], Mag),
    Mag > 0,
    Ux is X / Mag,
    Uy is Y / Mag,
    Uz is Z / Mag.

% Vector Projection: proj_b(a) = (a·b/|b|²)b
compute(vector_projection, [Ax, Ay], [Bx, By], [ProjX, ProjY]) :-
    Dot is Ax*Bx + Ay*By,
    MagBSq is Bx*Bx + By*By,
    MagBSq > 0,
    Scalar is Dot / MagBSq,
    ProjX is Scalar * Bx,
    ProjY is Scalar * By.


% ===== COMBINATORICS =====

% Permutation: P(n,r) = n!/(n-r)!
compute(permutation, N, R, Result) :-
    N >= R, R >= 0,
    factorial(N, FN),
    NR is N - R,
    factorial(NR, FNR),
    Result is FN / FNR.

% Combination: C(n,r) = n!/(r!(n-r)!)
compute(combination, N, R, Result) :-
    N >= R, R >= 0,
    factorial(N, FN),
    factorial(R, FR),
    NR is N - R,
    factorial(NR, FNR),
    Result is FN / (FR * FNR).

% Binomial Coefficient (same as combination)
compute(binomial_coefficient, N, R, Result) :-
    compute(combination, N, R, Result).

% Factorial
compute(factorial, N, Result) :-
    N >= 0,
    factorial(N, Result).

% ===== NUMBER THEORY =====

% GCD
compute(gcd, [A, B], Result) :-
    gcd_compute(A, B, Result).

% LCM
compute(lcm, [A, B], Result) :-
    lcm_compute(A, B, Result).

% Prime Check
compute(prime, N, Result) :-
    (is_prime(N) -> Result = true ; Result = false).

% Modular Arithmetic
compute(modular_arithmetic, [A, B, M], Result) :-
    M > 0,
    Result is (A + B) mod M.

% ===== PROBABILITY =====

% Basic Probability
compute(probability, [Favorable, Total], Result) :-
    Total > 0,
    Result is Favorable / Total.

% Conditional Probability: P(A|B) = P(A∩B) / P(B)
compute(conditional_probability, [PAandB, PB], Result) :-
    PB > 0,
    Result is PAandB / PB.

% Bayes Theorem: P(A|B) = P(B|A) * P(A) / P(B)
compute(bayes_theorem, [PBgivenA, PA, PB], Result) :-
    PB > 0,
    Result is (PBgivenA * PA) / PB.

% Expected Value
compute(expected_value, ValueProbPairs, Result) :-
    expected_value_sum(ValueProbPairs, Result).

expected_value_sum([], 0).
expected_value_sum([[Value, Prob]|Rest], Result) :-
    expected_value_sum(Rest, RestSum),
    Result is Value * Prob + RestSum.

% ===== LOGARITHM & EXPONENTIAL =====

% Natural Logarithm
compute(natural_log, X, Result) :-
    X > 0,
    Result is log(X).

% Logarithm with base
compute(logarithm, [X, Base], Result) :-
    X > 0, Base > 0, Base =\= 1,
    Result is log(X) / log(Base).

% Exponential
compute(exponential, X, Result) :-
    Result is exp(X).

% Exponential Growth: N(t) = N₀e^(rt)
compute(exponential_growth, [N0, R, T], Result) :-
    Result is N0 * exp(R * T).

% Exponential Decay: N(t) = N₀e^(-rt)
compute(exponential_decay, [N0, R, T], Result) :-
    Result is N0 * exp(-R * T).

% Power
compute(power, [Base, Exponent], Result) :-
    Result is Base ** Exponent.

% ===== TRIGONOMETRY =====

% Sine
compute(trigonometry_sin, AngleInRadians, Result) :-
    Result is sin(AngleInRadians).

% Cosine
compute(trigonometry_cos, AngleInRadians, Result) :-
    Result is cos(AngleInRadians).

% Tangent
compute(trigonometry_tan, AngleInRadians, Result) :-
    Result is tan(AngleInRadians).

% Inverse Trigonometric
compute(inverse_trig, [sin, Value], Result) :-
    Value >= -1, Value =< 1,
    Result is asin(Value).

compute(inverse_trig, [cos, Value], Result) :-
    Value >= -1, Value =< 1,
    Result is acos(Value).

compute(inverse_trig, [tan, Value], Result) :-
    Result is atan(Value).

% ===== MATRICES (Missing Rules) =====

% Matrix Multiplication (2x2 matrices)
compute(matrix_multiplication, [[A11, A12], [A21, A22]], [[B11, B12], [B21, B22]], 
        [[C11, C12], [C21, C22]]) :-
    C11 is A11*B11 + A12*B21,
    C12 is A11*B12 + A12*B22,
    C21 is A21*B11 + A22*B21,
    C22 is A21*B12 + A22*B22.

% Matrix Determinant (2x2)
compute(matrix_determinant, [[A11, A12], [A21, A22]], Det) :-
    Det is A11*A22 - A12*A21.

% Matrix Inverse (2x2)
compute(matrix_inverse, [[A11, A12], [A21, A22]], [[I11, I12], [I21, I22]]) :-
    Det is A11*A22 - A12*A21,
    Det =\= 0,
    I11 is A22 / Det,
    I12 is -A12 / Det,
    I21 is -A21 / Det,
    I22 is A11 / Det.

% Eigenvalues (2x2 matrix)
compute(eigenvalue, [[A11, A12], [A21, A22]], [Lambda1, Lambda2]) :-
    Trace is A11 + A22,
    Det is A11*A22 - A12*A21,
    Discriminant is Trace*Trace - 4*Det,
    Discriminant >= 0,
    SqrtDisc is sqrt(Discriminant),
    Lambda1 is (Trace + SqrtDisc) / 2,
    Lambda2 is (Trace - SqrtDisc) / 2.

% ===== ADVANCED (Missing Rules) =====

% Fourier Transform (symbolic representation)
compute(fourier_transform, [SignalType, Freq], Result) :-
    atom_string(SignalType, TypeStr),
    atom_string(Freq, FreqStr),
    string_concat("FFT of ", TypeStr, Temp),
    string_concat(Temp, " at ", Temp2),
    string_concat(Temp2, FreqStr, Temp3),
    string_concat(Temp3, " Hz", Result).

% Taylor Series (symbolic representation)
compute(taylor_series, [Function, _Point, Terms], Result) :-
    atom_string(Function, FuncStr),
    atom_string(Terms, TermsStr),
    string_concat("Taylor expansion of ", FuncStr, Temp),
    string_concat(Temp, " with ", Temp2),
    string_concat(Temp2, TermsStr, Temp3),
    string_concat(Temp3, " terms", Result).

% ============================================================================
% EXPLANATIONS (50+ Explanations)
% ============================================================================

explanation(derivative, "The derivative represents the instantaneous rate of change of a function. For power functions, we use the power rule: d/dx(x^n) = n*x^(n-1). The derivative is fundamental in optimization, physics, and engineering.").

explanation(integration, "Integration is the inverse of differentiation. It calculates the area under a curve or the antiderivative of a function. Used extensively in physics for calculating work, energy, and accumulated quantities.").

explanation(limit, "A limit describes the value that a function approaches as the input approaches some value. Limits are the foundation of calculus, defining derivatives and integrals.").

explanation(partial_derivative, "A partial derivative measures how a multivariable function changes with respect to one variable while keeping others constant. Essential in optimization and physics.").

explanation(second_derivative, "The second derivative measures the rate of change of the rate of change. It indicates concavity and helps find inflection points. Used in acceleration and optimization.").

explanation(dot_product, "The dot product of two vectors is calculated by multiplying corresponding components and summing them: A·B = A₁B₁ + A₂B₂ + A₃B₃. It results in a scalar and measures projection.").

explanation(cross_product, "The cross product produces a vector perpendicular to both input vectors. Only defined in 3D space: A×B = [A₂B₃-A₃B₂, A₃B₁-A₁B₃, A₁B₂-A₂B₁]. Used in torque and rotational mechanics.").

explanation(modulus, "The modulus (or magnitude) of a vector is its length, calculated using the Pythagorean theorem: |V| = sqrt(x² + y² + z²). Represents distance from origin.").

explanation(unit_vector, "A unit vector has magnitude 1 and indicates direction. Obtained by dividing a vector by its magnitude: û = v/|v|. Used in directional derivatives.").

explanation(vector_projection, "Vector projection finds the component of one vector in the direction of another. proj_b(a) = (a·b/|b|²)b. Used in work calculations and decomposition.").

explanation(mean, "The arithmetic mean is the average of a dataset: μ = (Σxᵢ)/n. It represents the center of the data and is the balancing point.").

explanation(median, "The median is the middle value when data is ordered. For even n, it's the average of the two middle values. Resistant to outliers, better than mean for skewed data.").

explanation(mode, "The mode is the most frequently occurring value in a dataset. A dataset can have multiple modes (bimodal, multimodal) or no mode. Useful for categorical data.").

explanation(variance, "Variance measures the spread of data from the mean: σ² = Σ(xᵢ-μ)²/n. Larger variance means more spread. In squared units of original data.").

explanation(standard_deviation, "Standard deviation is the square root of variance: σ = √(Σ(xᵢ-μ)²/n). Same units as original data, measures typical deviation from mean.").

explanation(range, "Range is the difference between maximum and minimum values: R = max(X) - min(X). Simple measure of spread but sensitive to outliers.").

explanation(percentile, "The pth percentile is a value below which p% of the data falls. Q1 (25th), Q2 (median, 50th), Q3 (75th) are quartiles. Used in standardized testing.").

explanation(correlation, "Correlation measures the linear relationship between two variables: r ∈ [-1, 1]. r=1: perfect positive, r=-1: perfect negative, r=0: no linear relationship.").

explanation(z_score, "Z-score standardizes data: z = (x-μ)/σ. Indicates how many standard deviations a value is from the mean. Used in comparing different datasets.").

explanation(probability, "Probability measures the likelihood of events: P(E) = favorable/total. Ranges from 0 (impossible) to 1 (certain). Foundation of statistics and decision theory.").

explanation(conditional_probability, "Conditional probability is the probability of A given B has occurred: P(A|B) = P(A∩B)/P(B). Used in Bayesian inference.").

explanation(bayes_theorem, "Bayes' Theorem relates conditional probabilities: P(A|B) = P(B|A)P(A)/P(B). Used in machine learning, medical diagnosis, and updating beliefs.").

explanation(expected_value, "Expected value is the long-run average outcome: E(X) = Σ[xᵢ·P(xᵢ)]. Used in decision theory, game theory, and insurance.").

explanation(permutation, "Permutation counts ordered arrangements: P(n,r) = n!/(n-r)!. Order matters. Used in passwords, rankings, and sequences.").

explanation(combination, "Combination counts unordered selections: C(n,r) = n!/(r!(n-r)!). Order doesn't matter. Used in lottery, team selection, and sampling.").

explanation(factorial, "Factorial is the product of all positive integers up to n: n! = nx(n-1)x...x2x1. 0! = 1 by definition. Grows extremely fast.").

explanation(floor_function, "Floor function rounds down to the nearest integer: ⌊x⌋. Always rounds toward negative infinity. Used in integer division.").

explanation(ceiling_function, "Ceiling function rounds up to the nearest integer: ⌈x⌉. Always rounds toward positive infinity. Used in resource allocation.").

explanation(absolute_value, "Absolute value is the distance from zero: |x| = x if x≥0, -x if x<0. Always non-negative. Represents magnitude without direction.").

explanation(quadratic_equation, "Quadratic equation ax²+bx+c=0 solved using: x = (-b ± √(b²-4ac))/2a. Discriminant Δ=b²-4ac determines root type: Δ>0 (two real), Δ=0 (one), Δ<0 (complex).").

explanation(linear_equation, "Linear equation ax+b=0 solved by: x = -b/a (a≠0). Represents a straight line. Simplest algebraic equation.").

explanation(cubic_equation, "Cubic equation ax³+bx²+cx+d=0 can have 1 or 3 real roots. Requires Cardano's formula or numerical methods. Degree 3 polynomial.").

explanation(logarithm, "Logarithm is the inverse of exponentiation: log_b(x) answers 'what power of b gives x?'. Natural log uses base e. Properties: log(xy)=log(x)+log(y).").

explanation(exponential, "Exponential function f(x)=e^x shows rapid growth. e≈2.71828 (Euler's number). Appears in compound interest, population growth, radioactive decay.").

explanation(exponential_growth, "Exponential growth: N(t) = N₀e^(rt) where r>0. Population doubles in constant time. Unsustainable in limited resources.").

explanation(exponential_decay, "Exponential decay: N(t) = N₀e^(-rt) where r>0. Half-life is constant. Models radioactivity, drug elimination, cooling.").

explanation(trigonometry_sin, "Sine function sin(θ) = opposite/hypotenuse in right triangle. Period 2π, range [-1,1]. Wave-like pattern, phase shifted from cosine by π/2.").

explanation(trigonometry_cos, "Cosine function cos(θ) = adjacent/hypotenuse. Period 2π, range [-1,1]. Derivative of sine. Used in projections and wave analysis.").

explanation(trigonometry_tan, "Tangent function tan(θ) = sin(θ)/cos(θ) = opposite/adjacent. Period π, undefined at π/2. Models slopes and angles.").

explanation(inverse_trig, "Inverse trigonometric functions: arcsin, arccos, arctan. Find angles from ratios. Domains restricted for uniqueness. Used in angle calculations.").

explanation(gcd, "Greatest Common Divisor (GCD) is the largest number that divides both numbers. Found using Euclidean algorithm. Used in fraction simplification.").

explanation(lcm, "Least Common Multiple (LCM) is the smallest number divisible by both. LCM(a,b) = (axb)/GCD(a,b). Used in adding fractions.").

explanation(prime, "Prime numbers have exactly two factors: 1 and themselves. 2 is the only even prime. Fundamental in cryptography and number theory.").

explanation(matrix_multiplication, "Matrix multiplication combines two matrices: (AB)ᵢⱼ = Σₖ AᵢₖBₖⱼ. Rows of A x columns of B. Non-commutative: AB≠BA usually.").

explanation(matrix_determinant, "Determinant is a scalar value from a square matrix. Measures scaling of transformation. det(A)=0 means singular (non-invertible) matrix.").

explanation(matrix_inverse, "Matrix inverse A⁻¹ satisfies AA⁻¹=I. Exists only if det(A)≠0. Used in solving linear systems: Ax=b → x=A⁻¹b.").

explanation(eigenvalue, "Eigenvalues λ satisfy Av=λv for eigenvector v. Indicates scaling along eigenvector direction. Used in stability analysis, PCA.").

explanation(fourier_transform, "Fourier transform decomposes signals into frequency components. F(ω) = ∫f(t)e^(-iωt)dt. Fundamental in signal processing and physics.").

explanation(taylor_series, "Taylor series expands function around point: f(x) = Σ[f^(n)(a)/n!]·(x-a)^n. Approximates functions with polynomials.").

% Get explanation for a concept
get_explanation(Concept, Explanation) :-
    explanation(Concept, Explanation).

% ============================================================================
% END OF KNOWLEDGE BASE
% ============================================================================